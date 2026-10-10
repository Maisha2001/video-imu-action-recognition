import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

from activity_data import file_hash
from attention_training import atomic_json, configure
from prefix_data import load_prefix_cache
from serving_runtime import ModelRuntime
from verify_serving import latencies, native_run
from video_features import backbone, decode_frames, load_features, prefix_tensor


def benchmark(prefix_cache, features, packages, native, rgb, weights, work, output):
    configure(0)
    trials, clips, imu, _ = load_prefix_cache(prefix_cache)
    feature_trials, visual, _ = load_features(features)
    assert trials == feature_trials
    indices=[i for i,t in enumerate(trials) if t.subject in (2,4,6,8) and t.action in (1,22) and t.repetition==1]
    report={'measured_at_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
            'processor':platform.processor(),'native_sha256':file_hash(native),'models':{},
            'trials':[trials[i].key for i in indices]}
    for family in ['robust','transfer']:
        ort,pytorch=[ModelRuntime(packages/family,b) for b in ['onnx','torch']]
        records=[{'video':clips[i,-1][None].astype(np.float32)/255 if family=='robust' else visual[i,-1][None].copy(),
                  'imu':imu[i,-1][None].copy(),'available':np.ones((1,2),np.float32)} for i in indices]
        torch_records=[tuple(torch.from_numpy(r[k]) for k in ['video','imu','available']) for r in records]
        with torch.inference_mode():
            torch_timing=latencies(lambda args:pytorch.model(*args),torch_records)
        ort_timing=latencies(lambda r:ort.model.run(None,r),records)
        _,cpp_timing=native_run(native,packages/family/'model.onnx',records,work/family,10)
        report['models'][family]={'pytorch_cpu':torch_timing,'python_onnx_cpu':ort_timing,'cpp_onnx_cpu':cpp_timing}
        print(f'{family} classifier benchmark finished',flush=True)
    records=[]
    with ZipFile(rgb) as archive, TemporaryDirectory(dir=work) as temporary:
        clip=Path(temporary)/'clip.avi'
        for trial in ['a1_s2_t1','a22_s2_t1']:
            member=next(n for n in archive.namelist() if Path(n).name==trial+'_color.avi')
            clip.write_bytes(archive.read(member))
            records.extend({'video':p[None]} for p in prefix_tensor(decode_frames(clip)).numpy())
    encoder=backbone(weights)
    torch_records=[torch.from_numpy(r['video']) for r in records]
    with torch.inference_mode():
        torch_timing=latencies(encoder,torch_records,3)
    runtime=ModelRuntime(packages/'transfer')
    ort_timing=latencies(lambda r:runtime.encoder.run(None,r),records,3)
    _,cpp_timing=native_run(native,packages/'transfer/backbone.onnx',records,work/'backbone',3,names=('video',))
    report['backbone']={'pytorch_cpu':torch_timing,'python_onnx_cpu':ort_timing,'cpp_onnx_cpu':cpp_timing}
    report['scope']='Sequential warmed CPU graph measurements, one thread, batch1. Classifiers:8 fixed full-paired trials,5 warmups,10 passes. Backbone:2 fixed trials at4 fractions,5 warmups,3 passes. Excludes load, decoding, HTTP and validation. C++ includes tensor-view setup. Other applications and OS load are uncontrolled; no statistical speedup or real-time guarantee.'
    report['source_sha256']={name:file_hash(Path(__file__).parent / ('../' + name if name.startswith('native/') else name)) for name in ['benchmark_serving.py','verify_serving.py','serving_runtime.py','native/infer.cpp']}
    report['package_model_sha256']={f:ModelRuntime(packages/f).manifest['files'] for f in ['robust','transfer']}
    atomic_json(output,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Measure resident CPU classifier and backbone graph latency')
    for arg in ['prefix-cache','features','packages','native','rgb','weights','work','output']:
        parser.add_argument('--'+arg,required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output')
    args.work.mkdir(parents=True,exist_ok=True)
    benchmark(**vars(args))
