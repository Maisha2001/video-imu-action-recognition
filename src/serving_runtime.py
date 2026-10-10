import json
from pathlib import Path
import time

import numpy as np
from scipy.io import loadmat

from activity_data import file_hash
from deep_data import resample_signal
from export_serving import ServingGraph, session
from prefix_data import FRACTIONS, decoded_frames
from robust_experiment import load_model as load_robust
from transfer_experiment import calibration_for, load_model as load_transfer
from uncertainty import probabilities
from video_features import FEATURE_SPEC, decode_frames, prefix_tensor


class ModelRuntime:
    def __init__(self, package, backend='onnx', threads=1):
        import torch
        if backend not in ('onnx', 'torch') or threads != 1:
            raise ValueError('Use onnx or torch with one CPU thread')
        self.package, self.backend = Path(package), backend
        self.manifest = json.loads((self.package / 'manifest.json').read_text())
        meta = self.manifest
        if meta['format_version'] != 1 or meta['family'] not in ('robust', 'transfer'):
            raise ValueError('Unsupported serving package')
        self.family = meta['family']
        required = {'model.onnx', 'source.pt'} | ({'backbone.onnx'} if self.family == 'transfer' else set())
        if set(meta['files']) != required:
            raise ValueError('Unexpected package files')
        for name, digest in meta['files'].items():
            if file_hash(self.package / name) != digest:
                raise ValueError('Package checksum mismatch')
        loader = load_robust if self.family == 'robust' else load_transfer
        model, artifact = loader(self.package / 'source.pt')
        if (artifact['labels'] != meta['labels'] or artifact['calibration'] != meta['calibration'] or
                file_hash(self.package / 'source.pt') != meta['source_model_sha256']):
            raise ValueError('Package metadata differs from fitted model')
        if self.family == 'transfer' and meta['feature_spec'] != FEATURE_SPEC:
            raise ValueError('Unsupported visual preprocessing')
        self.shapes = {'video': (1, 3, 16, 96, 96) if self.family == 'robust' else (1, 512),
                       'imu': (1, 6, 128), 'available': (1, 2)}
        if meta['inputs'] != {k: list(v) for k, v in self.shapes.items()}:
            raise ValueError('Unexpected input contract')
        torch.set_num_threads(1)
        self.model = session(self.package / 'model.onnx') if backend == 'onnx' else ServingGraph(model, artifact, self.family).eval()
        self.encoder = session(self.package / 'backbone.onnx') if self.family == 'transfer' else None

    def logits(self, inputs):
        import torch
        if set(inputs) != set(self.shapes):
            raise ValueError('Missing or unexpected model input')
        for key, value in inputs.items():
            if value.shape != self.shapes[key] or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('Invalid model input shape, type, or values')
        mask = inputs['available']
        if not np.isin(mask, [0, 1]).all() or not mask.any():
            raise ValueError('At least one binary sensor mask is required')
        if self.backend == 'onnx':
            return self.model.run(None, inputs)[0]
        with torch.inference_mode():
            return self.model(*(torch.from_numpy(inputs[k]) for k in ('video', 'imu', 'available'))).numpy()

    def inputs_from_files(self, video=None, inertial=None, fraction=1.0):
        if fraction not in FRACTIONS or (video is None and inertial is None):
            raise ValueError('Supply a sensor and supported fraction')
        inputs = {key: np.zeros(shape, np.float32) for key, shape in self.shapes.items()}
        if video is not None:
            if Path(video).stat().st_size > 64 * 1024 ** 2:
                raise ValueError('Video exceeds 64 MiB')
            if self.family == 'transfer':
                clip = prefix_tensor(decode_frames(video), (fraction,)).numpy()
                inputs['video'] = self.encoder.run(None, {'video': clip})[0]
            else:
                frames, _ = decoded_frames(video)
                count = int(np.floor(frames.shape[1] * fraction))
                if count < 2:
                    raise ValueError('Video prefix is too short')
                indices = np.rint(np.linspace(0, count - 1, 16)).astype(int)
                inputs['video'] = frames[:, indices][None].astype(np.float32) / 255
            inputs['available'][0, 0] = 1
        if inertial is not None:
            if Path(inertial).stat().st_size > 10 * 1024 ** 2:
                raise ValueError('IMU exceeds 10 MiB')
            raw = loadmat(inertial)['d_iner']
            if raw.ndim != 2 or raw.shape[0] > 100000:
                raise ValueError('Invalid IMU dimensions')
            inputs['imu'] = resample_signal(raw[:int(np.floor(len(raw) * fraction))])[None]
            inputs['available'][0, 1] = 1
        return inputs

    def result(self, logits, fraction, modality, mode='conditional'):
        if fraction not in FRACTIONS or modality not in ('both', 'video', 'imu') or mode not in ('pooled', 'conditional'):
            raise ValueError('Unsupported inference condition')
        calibration = self.manifest['calibration']
        effective = 'pooled' if self.family == 'robust' else mode
        if self.family == 'transfer':
            calibration = calibration_for(self.manifest, {'fraction': fraction, 'modality': modality, 'noise': 0.0}, mode)
        scores = probabilities(logits, calibration['temperature'])[0]
        confidence = float(scores.max())
        top = self.manifest['labels'][int(scores.argmax())]
        accepted = confidence >= calibration['threshold']
        return {'action': top if accepted else None, 'top_action': top, 'accepted': bool(accepted),
                'confidence': confidence, 'threshold': calibration['threshold'], 'probabilities': scores.tolist(),
                'labels': self.manifest['labels'], 'fraction': fraction, 'modality': modality,
                'calibration_mode': effective, 'model_family': self.family, 'backend': self.backend,
                'model_sha256': self.manifest['source_model_sha256'],
                'rejection_is_unknown_action_proof': False, 'limitations': self.manifest['limitations']}

    def predict(self, video=None, inertial=None, fraction=1.0, mode='conditional'):
        started = time.perf_counter()
        inputs = self.inputs_from_files(video, inertial, fraction)
        prepared = time.perf_counter()
        logits = self.logits(inputs)
        inferred = time.perf_counter()
        modality = 'both' if video is not None and inertial is not None else 'video' if video is not None else 'imu'
        result = self.result(logits, fraction, modality, mode)
        result['timing_ms'] = {'preprocessing': (prepared - started) * 1000, 'classifier': (inferred - prepared) * 1000}
        return result
