import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
pytest.importorskip('onnx')
pytest.importorskip('fastapi')
import cv2
from fastapi.testclient import TestClient
from scipy.io import savemat

from activity_data import file_hash
from replay_data import ReplayStore, prepare
from replay_server import create_replay_app


@pytest.fixture
def replay(tmp_path, monkeypatch):
    video, sensor = tmp_path / 'clip.avi', tmp_path / 'sensor.mat'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 15, (32, 32))
    for index in range(12):
        writer.write(np.full((32, 32, 3), index * 10, np.uint8))
    writer.release()
    savemat(sensor, {'d_iner': np.arange(120).reshape(20, 6)})
    rgb, inertial = tmp_path / 'RGB.zip', tmp_path / 'Inertial.zip'
    with ZipFile(rgb, 'w') as bundle:
        bundle.write(video, 'RGB/a1_s2_t1_color.avi')
    with ZipFile(inertial, 'w') as bundle:
        bundle.write(sensor, 'Inertial/a1_s2_t1_inertial.mat')
    monkeypatch.setattr('replay_data.HASHES', {'RGB.zip': file_hash(rgb), 'Inertial.zip': file_hash(inertial)})
    folder = tmp_path / 'replay'
    prepare(rgb, inertial, folder, ['a1_s2_t1'])
    return folder


def test_pairing_frames_and_integrity(replay):
    store = ReplayStore(replay)
    meta = store.trial('a1_s2_t1')
    assert meta['frame_count'] == 12 and len(meta['imu']) == 20
    assert store.frame('a1_s2_t1', 11).is_file()
    with pytest.raises(KeyError):
        store.frame('a1_s2_t1', 12)
    with pytest.raises(KeyError):
        store.inputs('../outside', 'both')
    (replay / 'a1_s2_t1' / 'inertial.mat').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        ReplayStore(replay)


def test_manifest_cannot_escape_root(replay):
    path = replay / 'index.json'
    index = json.loads(path.read_text())
    index['trials'][0]['files']['../secret'] = 'unused'
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match='Unsafe'):
        ReplayStore(replay)


def test_replay_service_modes_and_unknown_inputs(replay, tmp_path, monkeypatch):
    class Runtime:
        def __init__(self, *args):
            pass

        def predict(self, **kwargs):
            assert kwargs['video'] is None and Path(kwargs['inertial']).is_file()
            return {'fraction': kwargs['fraction'], 'modality': 'imu', 'top_action': 1}

    monkeypatch.setattr('inference_api.ModelRuntime', Runtime)
    ui = tmp_path / 'ui'
    (ui / 'assets').mkdir(parents=True)
    (ui / 'index.html').write_text('<html>replay</html>')
    with TestClient(create_replay_app({'robust': 'fixture'}, replay, ui)) as client:
        assert client.get('/').status_code == 200
        assert client.get('/replay').json()[0]['id'] == 'a1_s2_t1'
        assert client.get('/replay/a1_s2_t1/frames/0').headers['content-type'] == 'image/jpeg'
        result = client.post('/replay/a1_s2_t1/predict/robust?fraction=0.25&modality=imu')
        assert result.json() == {'trial': 'a1_s2_t1', 'fraction': .25, 'modality': 'imu', 'top_action': 1}
        for url in ['/replay/missing', '/replay/a1_s2_t1/frames/12']:
            assert client.get(url).status_code == 404
        for suffix in ['?fraction=0.3', '?modality=none', '?mode=invalid']:
            assert client.post('/replay/a1_s2_t1/predict/robust' + suffix).status_code == 422
        assert client.post('/replay/missing/predict/robust').status_code == 404
        assert client.post('/replay/a1_s2_t1/predict/missing').status_code == 404


def test_tracking_is_local_idempotent_and_keeps_source_identity(tmp_path, tmp_path_factory, monkeypatch):
    monkeypatch.setenv('MLFLOW_DISABLE_TELEMETRY', 'true')
    monkeypatch.setenv('DO_NOT_TRACK', 'true')
    pytest.importorskip('mlflow')
    from track_results import import_results
    from mlflow import MlflowClient
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'https://invalid.example')
    source = tmp_path / 'summary.json'
    source.write_text(json.dumps({'completed_at_utc': '2020-01-01T00:00:00Z', 'config': {'epochs': 2},
                      'runs': {'fixture-42': {'seed': 42, 'metrics': {'accuracy': .75, 'missing': None}}}}))
    store = tmp_path_factory.mktemp('m')
    first, second = import_results(source, store), import_results(source, store)
    assert not first['runs'][0]['reused'] and second['runs'][0]['reused']
    assert first['runs'][0]['run_id'] == second['runs'][0]['run_id']
    uri = 'sqlite:///' + (store / 'tracking.db').as_posix()
    run = MlflowClient(tracking_uri=uri).get_run(first['runs'][0]['run_id'])
    assert run.data.metrics['metrics.accuracy'] == .75
    assert run.data.tags['kind'] == 'result_import'
    assert run.data.tags['original_completed_at'] == '2020-01-01T00:00:00Z'
    assert run.info.start_time > 1700000000000
    assert run.info.artifact_uri.startswith('file:')
