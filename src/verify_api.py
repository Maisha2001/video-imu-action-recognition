import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from zipfile import ZipFile

import httpx
import numpy as np

from activity_data import file_hash
from attention_training import atomic_json
from prefix_data import FRACTIONS
from robustness import predict as robust_predict
from serving_runtime import ModelRuntime
from transfer import predict as transfer_predict
from verify_serving import native_run
from video_features import decode_frames, prefix_tensor


def verify(packages, rgb, inertial, weights, native, work, output):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    command = [sys.executable, str(Path(__file__).with_name('inference_api.py')), '--package', f'robust={packages / "robust"}',
               '--package', f'transfer={packages / "transfer"}', '--port', str(port)]
    checks, backbone_records = [], []
    with (work / 'server.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=60, trust_env=False) as client:
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError('API exited during startup; inspect server log')
                    try:
                        if client.get('/health').status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(.1)
                else:
                    raise RuntimeError('API startup timed out')
                with ZipFile(rgb) as videos, ZipFile(inertial) as sensors:
                    for trial in ['a1_s2_t1','a22_s2_t1']:
                        files={}
                        for key, archive, suffix in [('video',videos,'_color.avi'),('inertial',sensors,'_inertial.mat')]:
                            member=next(n for n in archive.namelist() if Path(n).name == trial+suffix)
                            path=work/(trial+suffix)
                            path.write_bytes(archive.read(member))
                            files[key]=path
                        for clip in prefix_tensor(decode_frames(files['video'])).numpy():
                            backbone_records.append({'video':clip[None]})
                        for family in ['robust','transfer']:
                            for fraction in FRACTIONS:
                                for modality in ['both','video','imu']:
                                    paths={k:p for k,p in files.items() if modality=='both' or k==('video' if modality=='video' else 'inertial')}
                                    payload={k:(p.name,p.read_bytes()) for k,p in paths.items()}
                                    started=time.perf_counter()
                                    response=client.post(f'/predict/{family}?fraction={fraction}',files=payload)
                                    elapsed=(time.perf_counter()-started)*1000
                                    response.raise_for_status()
                                    result=response.json()
                                    if family=='robust':
                                        expected=robust_predict(packages/family/'source.pt',fraction=fraction,**paths)
                                    else:
                                        expected=transfer_predict(packages/family/'source.pt',weights=weights,fraction=fraction,**paths)
                                    assert result['top_action']==expected['top_action'] and result['accepted']==expected['accepted']
                                    error=float(np.abs(np.array(result['probabilities'])-expected['probabilities']).max())
                                    assert error<1e-5
                                    checks.append({'trial':trial,'family':family,'fraction':fraction,'modality':modality,
                                                   'action':result['top_action'],'accepted':result['accepted'],
                                                   'max_probability_difference':error,'http_roundtrip_ms':elapsed,
                                                   'timing_ms':result['timing_ms'],'input_sha256':{k:file_hash(p) for k,p in paths.items()}})
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    runtime=ModelRuntime(packages/'transfer')
    expected=np.concatenate([runtime.encoder.run(None,r)[0] for r in backbone_records])
    native_values,timing=native_run(native,packages/'transfer/backbone.onnx',backbone_records,work/'native-backbone',3,names=('video',))
    np.testing.assert_allclose(native_values,expected,atol=1e-5,rtol=1e-5)
    result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'transport':'actual HTTP over127.0.0.1; server stopped after checks',
            'checks':checks,'native_backbone':{'records':len(backbone_records),'max_feature_difference':float(np.abs(native_values-expected).max()),'timing':timing},
            'scope':'Two fixed known/downstream-held-out trials, all4 fractions and all3 sensor masks per family. HTTP timings include preprocessing; one request per condition, not a latency percentile benchmark. Native backbone timings use8 prepared clips, one thread, five warmups, three passes; decoding excluded.',
            'source_sha256':{name:file_hash(Path(__file__).parent / ('../' + name if name.startswith('native/') else name)) for name in ['inference_api.py','serving_runtime.py','verify_api.py','native/infer.cpp']}}
    atomic_json(output,result)
    print(json.dumps({'http_checks':len(checks),'native_backbone':result['native_backbone']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Check actual HTTP predictions against original file inference')
    for arg in ['packages','rgb','inertial','weights','native','work','output']:
        parser.add_argument('--'+arg,required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output')
    verify(**vars(args))
