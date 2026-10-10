import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import httpx
import numpy as np

from activity_data import file_hash
from attention_training import atomic_json
from replay_data import ReplayStore
from serving_runtime import ModelRuntime


def verify(url, replay, packages, output):
    if url.rstrip('/') != 'http://127.0.0.1:8017':
        raise ValueError('Verifier targets the documented local test port8017')
    store = ReplayStore(replay)
    models = {f: ModelRuntime(Path(packages) / f) for f in ('robust', 'transfer')}
    started = time.perf_counter()
    checks = []
    with httpx.Client(base_url=url, timeout=90, trust_env=False) as client:
        assert client.get('/').status_code == 200
        assert {t['id'] for t in client.get('/replay').json()} == set(store.metadata)
        for key in store.metadata:
            metadata = client.get('/replay/' + key).json()
            assert metadata == store.trial(key)
            for frame in (0, metadata['frame_count'] - 1):
                response = client.get(f'/replay/{key}/frames/{frame}')
                assert response.content == store.frame(key, frame).read_bytes()
            for family, runtime in models.items():
                for fraction in (.25, .5, .75, 1.):
                    for modality in ('both', 'video', 'imu'):
                        response = client.post(f'/replay/{key}/predict/{family}', params={'fraction': fraction, 'modality': modality})
                        response.raise_for_status()
                        actual = response.json()
                        expected = runtime.predict(**store.inputs(key, modality), fraction=fraction)
                        assert actual['model_sha256'] == expected['model_sha256']
                        assert actual['trial'] == key and actual['top_action'] == expected['top_action'] and actual['accepted'] == expected['accepted']
                        error = float(np.abs(np.asarray(actual['probabilities']) - expected['probabilities']).max())
                        assert error < 1e-6
                        checks.append({'trial':key, 'model':family, 'fraction':fraction, 'modality':modality,
                                       'top_action':actual['top_action'], 'accepted':actual['accepted'], 'max_probability_difference':error})
            print(f'{key}: replay metadata, first/last frames and24 predictions verified', flush=True)
        assert client.get('/replay/missing').status_code == 404
        assert client.post('/replay/a1_s2_t1/predict/robust?fraction=0.33').status_code == 422
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'seconds':time.perf_counter()-started,
              'checks':checks, 'scope':'Four fixed trials,4 fractions,3 modalities,2 models; real HTTP versus direct checked runtime. Demo samples are not evaluation estimates.',
              'replay_index_sha256':file_hash(Path(replay)/'index.json'),
              'source_sha256':{name:file_hash(Path(__file__).parent / name) for name in ['replay_data.py','replay_server.py','verify_replay.py']}}
    atomic_json(output,result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify a running local replay service')
    parser.add_argument('--url', default='http://127.0.0.1:8017')
    for key in ('replay','packages','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    verify(**vars(args))
