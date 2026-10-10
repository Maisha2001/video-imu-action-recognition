import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from activity_data import file_hash
from attention_training import atomic_json, configure
from robust_experiment import load_model as load_robust
from transfer_experiment import load_model as load_transfer
from video_features import FEATURE_SPEC, backbone


class ServingGraph(nn.Module):
    def __init__(self, model, artifact, family):
        super().__init__()
        self.model, self.family = model, family
        norms = artifact['normalization']
        if family == 'transfer':
            self.register_buffer('video_mean', norms['video']['mean'])
            self.register_buffer('video_std', norms['video']['std'])
            norms = norms['imu']
        self.register_buffer('imu_mean', norms['mean'][None, :, None])
        self.register_buffer('imu_std', norms['std'][None, :, None])

    def forward(self, video, imu, available):
        if self.family == 'transfer':
            video = (video - self.video_mean) / self.video_std
        return self.model(video, (imu - self.imu_mean) / self.imu_std, available)


def session(path, threads=1):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def export_graph(model, inputs, names, path):
    torch.onnx.export(model.eval(), inputs, str(path), input_names=names, output_names=['logits'],
                      opset_version=18, dynamo=False, external_data=False)
    onnx.checker.check_model(str(path))
    runner = session(path)
    with torch.inference_mode():
        expected = model(*inputs).numpy()
    actual = runner.run(None, {k: v.numpy() for k, v in zip(names, inputs)})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    return float(np.max(np.abs(actual - expected)))


def export_package(model_path, output, family, weights=None):
    if family not in ('robust', 'transfer'):
        raise ValueError('Unsupported model family')
    configure(0)
    model, artifact = (load_robust if family == 'robust' else load_transfer)(model_path)
    if family == 'transfer' and weights is None:
        raise ValueError('Transfer export requires pinned backbone weights')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    visual_shape = (1, 3, 16, 96, 96) if family == 'robust' else (1, 512)
    inputs = (torch.zeros(visual_shape), torch.zeros(1, 6, 128), torch.ones(1, 2))
    graph = ServingGraph(model, artifact, family).eval()
    errors = {'classifier': export_graph(graph, inputs, ['video', 'imu', 'available'], output / 'model.onnx')}
    shutil.copyfile(model_path, output / 'source.pt')
    files = ['model.onnx', 'source.pt']
    if family == 'transfer':
        encoder = backbone(weights)
        errors['backbone'] = export_graph(encoder, (torch.zeros(1, 3, 16, 112, 112),), ['video'], output / 'backbone.onnx')
        files.append('backbone.onnx')
    manifest = {'format_version': 1, 'family': family, 'labels': artifact['labels'],
                'seed': artifact['seed'], 'calibration': artifact['calibration'],
                'source_model_sha256': file_hash(model_path), 'files': {n: file_hash(output / n) for n in files},
                'inputs': {n: list(v.shape) for n, v in zip(['video', 'imu', 'available'], inputs)},
                'feature_spec': FEATURE_SPEC if family == 'transfer' else None,
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'export': {'torch': str(torch.__version__), 'onnx': onnx.__version__, 'onnxruntime': ort.__version__,
                           'opset': 18, 'batch_size': 1, 'smoke_max_logit_difference': errors},
                'limitations': ['Offline segmented recordings; fraction alignment is not exact timestamp synchronization.',
                                'Confidence thresholds do not reliably identify unfamiliar actions.',
                                'Experimental models; no safety-critical or real-time performance claim.']}
    atomic_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a checked inference package')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=['robust', 'transfer'], required=True)
    parser.add_argument('--weights', type=Path)
    args = parser.parse_args()
    result = export_package(args.model, args.output, args.family, args.weights)
    print(json.dumps({'family': result['family'], 'checks': result['export']['smoke_max_logit_difference']}))
