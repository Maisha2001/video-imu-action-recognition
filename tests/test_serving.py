import asyncio
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest
pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')
pytest.importorskip('fastapi')
import torch
from fastapi.testclient import TestClient
from scipy.io import savemat

from attention_training import configure
from export_serving import export_package
from inference_api import BoundedUpload, create_app
from prefix_data import PREFIX_SPEC
from robust_model import RobustEncoder
from serving_runtime import ModelRuntime
from transfer_model import TransferFusion
from video_features import FEATURE_SPEC


@pytest.fixture(scope='module')
def packages(tmp_path_factory):
    directory = tmp_path_factory.mktemp('serving')
    result = {}
    configure(42)
    for family in ['robust', 'transfer']:
        norm = {'mean': torch.arange(6).float(), 'std': torch.arange(1, 7).float()}
        calibration = {'temperature': 1.2, 'threshold': 0.7}
        if family == 'robust':
            artifact = {'format_version': 1, 'architecture': 'masked-paired-v1', 'preprocess': PREFIX_SPEC,
                        'model': RobustEncoder(2).state_dict(), 'normalization': norm, 'calibration': calibration}
        else:
            artifact = {'format_version': 1, 'architecture': 'frozen-r3d-fusion-v1', 'feature_spec': FEATURE_SPEC,
                        'model': TransferFusion(2).state_dict(),
                        'normalization': {'imu': norm, 'video': {'mean': torch.zeros(512), 'std': torch.ones(512)}},
                        'calibration': {'pooled': calibration, 'conditional': {f'f{f}-{m}-n0': calibration for f in [25,50,75,100] for m in ['both','video','imu']}}}
        artifact.update(labels=[1, 2], seed=42, config={'epochs':1,'batch_size':1,'learning_rate':.001,'weight_decay':.0001,'dropout':.1})
        model = directory / f'{family}.pt'
        torch.save(artifact, model)
        with pytest.MonkeyPatch.context() as patch:
            import export_serving
            tiny = torch.nn.Sequential(torch.nn.AdaptiveAvgPool3d(1), torch.nn.Flatten(), torch.nn.Linear(3,512)).eval()
            patch.setattr(export_serving, 'backbone', lambda *args: tiny)
            export_package(model, directory / family, family, 'fixture')
        result[family] = directory / family
    return result


@pytest.mark.parametrize('family', ['robust', 'transfer'])
def test_export_parity_masks_and_input_contract(packages, family):
    runtimes = [ModelRuntime(packages[family], b) for b in ['onnx','torch']]
    rng = np.random.default_rng(2)
    values = {k: rng.normal(size=s).astype(np.float32) for k,s in runtimes[0].shapes.items()}
    for mask in [[1,1],[0,1],[1,0]]:
        values['available'] = np.array([mask], np.float32)
        a, b = [r.logits(values) for r in runtimes]
        np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)
        ignored = {k:v.copy() for k,v in values.items()}
        ignored['video' if mask[0] == 0 else 'imu'].fill(1000)
        if mask != [1,1]:
            np.testing.assert_array_equal(runtimes[0].logits(ignored), a)
    values['available'].fill(0)
    with pytest.raises(ValueError, match='sensor mask'):
        runtimes[0].logits(values)
    values['available'] = np.ones((2,2),np.float32)
    with pytest.raises(ValueError, match='shape'):
        runtimes[0].logits(values)


def test_package_integrity_and_calibration_match(packages, tmp_path):
    destination = tmp_path / 'package'
    shutil.copytree(packages['robust'], destination)
    meta_path = destination / 'manifest.json'
    meta = json.loads(meta_path.read_text())
    meta['calibration']['threshold'] = .01
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='metadata'):
        ModelRuntime(destination)
    with (destination / 'model.onnx').open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        ModelRuntime(destination)


def imu_bytes():
    buffer = BytesIO()
    savemat(buffer, {'d_iner': np.arange(384).reshape(64,6).astype(float)})
    return buffer.getvalue()


def test_api_real_sensor_prediction_and_errors(packages):
    with TestClient(create_app(packages)) as client:
        assert client.get('/health').json()['models'] == ['robust','transfer']
        assert client.get('/models').json()['transfer']['fractions'] == [.25,.5,.75,1.]
        for family in packages:
            response = client.post(f'/predict/{family}?fraction=0.25', files={'inertial': ('../../sensor.mat', imu_bytes())})
            assert response.status_code == 200
            r=response.json()
            assert r['modality']=='imu' and not r['rejection_is_unknown_action_proof']
            assert len(r['probabilities'])==2 and r['model_sha256']
        assert client.post('/predict/missing', files={'inertial': ('a.mat', imu_bytes())}).status_code==404
        assert client.post('/predict/robust?fraction=0.2', files={'inertial': ('a.mat', imu_bytes())}).status_code==422
        assert client.post('/predict/robust', files={'inertial': ('a.mat', b'invalid')}).status_code==422
        assert client.post('/predict/robust', files={'video': ('a.avi', b'invalid')}).status_code==422
        assert client.post('/predict/robust', files=[('inertial', ('a',imu_bytes())),('inertial',('b',imu_bytes()))]).status_code==422
        assert client.post('/predict/robust', files={'unexpected': ('a',imu_bytes())}).status_code==422
        assert client.post('/predict/robust').status_code==422


def test_upload_limit_without_content_length(packages):
    with TestClient(create_app(packages, body_limit=100)) as client:
        response=client.post('/predict/robust', content=iter([b'a'*60,b'b'*60]))
        assert response.status_code==413
        assert client.get('/health').status_code==200


def test_busy_request_rejected_and_slot_released():
    async def check():
        entered, release = asyncio.Event(), asyncio.Event()
        async def app(scope, receive, send):
            entered.set()
            await release.wait()
        bounded=BoundedUpload(app)
        async def receive():
            return {'type':'http.request','body':b'','more_body':False}
        sent=[]
        async def send(message):
            sent.append(message)
        scope={'type':'http','method':'POST'}
        first=asyncio.create_task(bounded(scope,receive,send))
        await entered.wait()
        await bounded(scope,receive,send)
        assert sent[0]['status']==503
        release.set()
        await first
        assert bounded.slots.acquire(blocking=False)
        bounded.slots.release()
    asyncio.run(check())


def test_native_runner_parity_and_malformed_files(packages,tmp_path):
    native=os.environ.get('ACTIVITY_NATIVE')
    if not native:
        pytest.skip('Set ACTIVITY_NATIVE to a built native runner')
    runtime=ModelRuntime(packages['robust'])
    values={k:np.zeros(s,np.float32) for k,s in runtime.shapes.items()}
    values['available'][:]=[0,1]
    file=tmp_path/'input.bin'; output=tmp_path/'output.bin'
    file.write_bytes(b'VIM1'+struct.pack('<I',1)+b''.join(values[k].astype('<f4').tobytes() for k in ['video','imu','available']))
    result=subprocess.run([native,str(packages['robust']/'model.onnx'),str(file),str(output)],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['records']==1
    np.testing.assert_allclose(np.fromfile(output,dtype='<f4'),runtime.logits(values)[0],atol=1e-5,rtol=1e-5)
    file.write_bytes(b'VIM1'+struct.pack('<I',1)+b'bad')
    assert subprocess.run([native,str(packages['robust']/'model.onnx'),str(file),str(output)],capture_output=True).returncode!=0
