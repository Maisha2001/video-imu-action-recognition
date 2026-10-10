import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import struct
import subprocess
import time

import numpy as np
import onnxruntime
import torch

from activity_data import file_hash
from attention_training import atomic_json, configure
from prefix_data import FRACTIONS, load_prefix_cache
from robust_training import MASKS, condition_key, conditions
from serving_runtime import ModelRuntime
from video_features import load_features


def write_native_inputs(path, records, names=('video', 'imu', 'available')):
    with Path(path).open('wb') as stream:
        stream.write(b'VIM1' + struct.pack('<I', len(records)))
        for record in records:
            for name in names:
                stream.write(record[name].astype('<f4').tobytes())


def native_run(executable, model, records, directory, repeats=1, names=('video','imu','available')):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source, target = directory / 'inputs.bin', directory / 'outputs.bin'
    write_native_inputs(source, records, names)
    result = subprocess.run([str(executable), str(model), str(source), str(target), str(repeats)],
                            capture_output=True, text=True, check=True, timeout=1200)
    timing = json.loads(result.stdout)
    values = np.fromfile(target, dtype='<f4').reshape(len(records), timing['output_values_per_record'])
    return values, timing


def latencies(call, records, repeats=10):
    for _ in range(5):
        call(records[0])
    values = []
    for _ in range(repeats):
        for record in records:
            started = time.perf_counter()
            call(record)
            values.append((time.perf_counter() - started) * 1000)
    return {'samples': len(values), 'warmup': 5, 'repeats': repeats, 'threads': 1,
            'p50_ms': float(np.percentile(values, 50, method='inverted_cdf')),
            'p95_ms': float(np.percentile(values, 95, method='inverted_cdf')), 'mean_ms': float(np.mean(values))}


def verify(prefix, features, packages, native, work, output, stop=None):
    from robust_training import check_stop
    configure(0)
    started = time.perf_counter()
    trials, clips, sensors, cache_meta = load_prefix_cache(prefix)
    feature_trials, visual, feature_meta = load_features(features)
    if trials != feature_trials:
        raise ValueError('Feature trial identities differ')
    indices = [i for i,t in enumerate(trials) if t.subject % 2 == 0]
    subset = [i for i in indices if trials[i].action in (1,22) and trials[i].repetition == 1]
    assert len(indices)==430 and len(subset)==8
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'torch': str(torch.__version__),
              'onnxruntime': onnxruntime.__version__, 'platform': platform.platform(), 'processor': platform.processor(),
              'native_sha256': file_hash(native), 'archives': cache_meta['archive_sha256'],
              'prefix_cache_sha256': cache_meta['array_sha256'], 'feature_array_sha256': feature_meta['array_sha256'],
              'native_trials': [trials[i].key for i in subset], 'models': {}}
    for family in ['robust', 'transfer']:
        check_stop(stop)
        ort, pytorch = [ModelRuntime(Path(packages) / family, b) for b in ['onnx','torch']]
        manifest = ort.manifest
        per_condition, native_inputs, native_expected = {}, [], []
        benchmark_inputs = []
        for condition in conditions():
            check_stop(stop)
            key = condition_key(condition)
            max_logits, max_confidence, actions, rejections = 0., 0., 0, 0
            for i in indices:
                j = FRACTIONS.index(condition['fraction'])
                video = clips[i,j][None].astype(np.float32)/255 if family=='robust' else visual[i,j][None].copy()
                imu = sensors[i,j][None].copy()
                if condition['noise']:
                    norm = pytorch.model.imu_std.numpy()
                    rng = np.random.default_rng(np.random.SeedSequence([2026,i,round(condition['fraction']*100),round(condition['noise']*100)]))
                    imu += rng.normal(0,condition['noise'],imu.shape[1:]).astype(np.float32)[None]*norm
                mask = np.array([MASKS[condition['modality']]],np.float32)
                values = {'video':video,'imu':imu,'available':mask}
                a,b = ort.logits(values),pytorch.logits(values)
                np.testing.assert_allclose(a,b,atol=1e-4,rtol=1e-4)
                max_logits=max(max_logits,float(np.abs(a-b).max()))
                for mode in (['pooled','conditional'] if family=='transfer' else ['pooled']):
                    ra,rb=[ort.result(v,condition['fraction'],condition['modality'],mode) for v in [a,b]]
                    max_confidence=max(max_confidence,abs(ra['confidence']-rb['confidence']))
                    actions += ra['top_action']==rb['top_action']
                    rejections += ra['accepted']==rb['accepted']
                if i in subset:
                    native_inputs.append(values)
                    native_expected.append(a[0])
                    if key=='f100-both-n0':
                        benchmark_inputs.append(values)
            comparisons=len(indices)*(2 if family=='transfer' else 1)
            per_condition[key]={'trials':len(indices),'calibrated_comparisons':comparisons,'matching_actions':actions,
                                'matching_acceptance':rejections,'max_logit_difference':max_logits,'max_confidence_difference':max_confidence}
            print(f'{family} {key}: {actions}/{comparisons} actions, {rejections}/{comparisons} decisions',flush=True)
        native_values,native_timing=native_run(native,Path(packages)/family/'model.onnx',native_inputs,Path(work)/family/'parity')
        expected=np.asarray(native_expected)
        np.testing.assert_allclose(native_values,expected,atol=1e-5,rtol=1e-5)
        native_calibration=[]
        for c in conditions():
            for _ in subset:
                native_calibration.append(c)
        native_decisions=0
        for a,b,c in zip(native_values,expected,native_calibration):
            ra,rb=[ort.result(v[None],c['fraction'],c['modality']) for v in [a,b]]
            native_decisions += (ra['top_action']==rb['top_action'] and ra['accepted']==rb['accepted'])
        check_stop(stop)
        torch_inputs=[tuple(torch.from_numpy(v[k]) for k in ['video','imu','available']) for v in benchmark_inputs]
        with torch.inference_mode():
            torch_timing=latencies(lambda args:pytorch.model(*args),torch_inputs)
        ort_timing=latencies(lambda v:ort.model.run(None,v),benchmark_inputs)
        _,cpp_timing=native_run(native,Path(packages)/family/'model.onnx',benchmark_inputs,Path(work)/family/'benchmark',10)
        report['models'][family]={'source_model_sha256':manifest['source_model_sha256'],'files':manifest['files'],
                                  'conditions':per_condition,'native':{'records':len(native_inputs),'matching_decisions':native_decisions,
                                  'max_logit_difference':float(np.abs(native_values-expected).max())},
                                  'latency':{'pytorch_cpu':torch_timing,'python_onnx_cpu':ort_timing,'cpp_onnx_cpu':cpp_timing}}
    report['latency_scope']='One CPU thread, batch1, resident models and preprocessed inputs; eight fixed complete paired trials, five warmups, ten passes. Excludes model loading, file decoding, network, and frozen video backbone. Python bypasses input validation for graph timing; C++ includes tensor-view setup. No statistical speedup guarantee.'
    report['native_scope']='160 inputs per model: actions1/22, people2/4/6/8, repetition1, all20 conditions. Full PyTorch/ONNX classifier parity covers430 held-out trials and all20 conditions; cached video features used for transfer.'
    report['source_sha256']={name:file_hash(name) for name in ['export_serving.py','serving_runtime.py','inference_api.py','verify_serving.py','native/infer.cpp']}
    report['seconds']=time.perf_counter()-started
    atomic_json(output,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Verify held-out inference parity and CPU graph latency')
    parser.add_argument('--prefix-cache',required=True,type=Path)
    parser.add_argument('--features',required=True,type=Path)
    parser.add_argument('--packages',required=True,type=Path)
    parser.add_argument('--native',required=True,type=Path)
    parser.add_argument('--work',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--stop-at-utc')
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output')
    stop=datetime.fromisoformat(args.stop_at_utc) if args.stop_at_utc else None
    if stop is not None and stop.tzinfo is None:
        parser.error('Stop time needs an offset')
    verify(args.prefix_cache,args.features,args.packages,args.native,args.work,args.output,stop)
