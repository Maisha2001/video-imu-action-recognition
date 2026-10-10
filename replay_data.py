import argparse
import json
from pathlib import Path
import re
from zipfile import ZipFile

import cv2
import numpy as np

from activity_data import archive_index, file_hash, read_signal, validate_signal
from attention_training import atomic_json
from fetch_data import HASHES


TRIAL_KEY = re.compile(r'a([1-9]|1[0-9]|2[0-7])_s([1-8])_t([1-4])')
DEFAULT_TRIALS = ['a1_s2_t1', 'a2_s4_t1', 'a22_s2_t1', 'a27_s8_t1']
ALIGNMENT = 'Normalized trial fraction; shared frame/sample timestamps are unavailable. This is not exact synchronization.'


def prepare(rgb, inertial, output, trial_keys):
    if not trial_keys or len(set(trial_keys)) != len(trial_keys) or any(not TRIAL_KEY.fullmatch(k) for k in trial_keys):
        raise ValueError('Provide unique UTD-MHAD trial IDs')
    for name, path in [('RGB.zip', rgb), ('Inertial.zip', inertial)]:
        if file_hash(path) != HASHES[name]:
            raise ValueError('Archive checksum mismatch')
    videos, sensors = archive_index(rgb, 'color'), archive_index(inertial, 'inertial')
    selected = {t.key: t for t in videos if t in sensors}
    if any(key not in selected for key in trial_keys):
        raise ValueError('Requested paired trial is missing')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    with ZipFile(rgb) as video_zip, ZipFile(inertial) as sensor_zip:
        for key in trial_keys:
            trial = selected[key]
            folder = output / key
            (folder / 'frames').mkdir(parents=True)
            (folder / 'video.avi').write_bytes(video_zip.read(videos[trial]))
            (folder / 'inertial.mat').write_bytes(sensor_zip.read(sensors[trial]))
            signal = read_signal(sensor_zip, sensors[trial])
            capture = cv2.VideoCapture(str(folder / 'video.avi'))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = 0
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frames >= 10000 or not cv2.imwrite(str(folder / 'frames' / f'{frames:05}.jpg'), frame):
                        raise ValueError('Cannot prepare video frames')
                    frames += 1
            finally:
                capture.release()
            if frames < 2 or not np.isfinite(fps) or fps <= 0:
                raise ValueError('Invalid video')
            metadata = {'id': key, 'action': trial.action, 'subject': trial.subject,
                        'repetition': trial.repetition, 'frame_count': frames, 'fps': fps,
                        'duration_seconds': frames / fps, 'imu': signal.tolist(),
                        'sensor_placement': 'wrist' if trial.action <= 21 else 'thigh',
                        'downstream_known_action': trial.action <= 21, 'alignment': ALIGNMENT}
            atomic_json(folder / 'metadata.json', metadata)
            files = {p.relative_to(folder).as_posix(): file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()}
            entries.append({'id': key, 'action': trial.action, 'subject': trial.subject, 'files': files})
    atomic_json(output / 'index.json', {'version': 1, 'alignment': ALIGNMENT,
                'source': 'UTD-MHAD; see DATA.md for access and citation. Recordings remain local.',
                'selection': 'Explicit trial IDs, not a representative evaluation sample.', 'trials': entries})
    return entries


class ReplayStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        index = json.loads((self.root / 'index.json').read_text())
        if index['version'] != 1 or not index['trials']:
            raise ValueError('Unsupported replay index')
        self.metadata = {}
        for entry in index['trials']:
            key = entry['id']
            if not TRIAL_KEY.fullmatch(key) or key in self.metadata:
                raise ValueError('Invalid or duplicate replay ID')
            folder = self.root / key
            if not {'video.avi', 'inertial.mat', 'metadata.json'} <= set(entry['files']):
                raise ValueError('Incomplete replay')
            for name, digest in entry['files'].items():
                if name not in ('video.avi', 'inertial.mat', 'metadata.json') and not re.fullmatch(r'frames/[0-9]{5}\.jpg', name):
                    raise ValueError('Unsafe replay path')
                path = (folder / name).resolve()
                if not path.is_relative_to(self.root) or file_hash(path) != digest:
                    raise ValueError('Replay checksum mismatch')
            meta = json.loads((folder / 'metadata.json').read_text())
            if meta['id'] != key or meta['frame_count'] < 2 or meta['frame_count'] > 10000:
                raise ValueError('Invalid replay metadata')
            identity = tuple(map(int, TRIAL_KEY.fullmatch(key).groups()))
            if (meta['action'], meta['subject'], meta['repetition']) != identity:
                raise ValueError('Replay identity differs from trial ID')
            validate_signal(meta['imu'])
            if not np.isfinite(meta['fps']) or meta['fps'] <= 0 or not np.isclose(meta['duration_seconds'], meta['frame_count'] / meta['fps']):
                raise ValueError('Invalid replay timing')
            if any(f'frames/{i:05}.jpg' not in entry['files'] for i in range(meta['frame_count'])):
                raise ValueError('Missing replay frame')
            self.metadata[key] = meta

    def trial(self, key):
        if key not in self.metadata:
            raise KeyError('Unknown replay trial')
        return self.metadata[key]

    def frame(self, key, number):
        meta = self.trial(key)
        if not 0 <= number < meta['frame_count']:
            raise KeyError('Unknown replay frame')
        return self.root / key / 'frames' / f'{number:05}.jpg'

    def inputs(self, key, modality):
        self.trial(key)
        if modality not in ('both', 'video', 'imu'):
            raise ValueError('Unsupported sensor mode')
        folder = self.root / key
        return {'video': folder / 'video.avi' if modality != 'imu' else None,
                'inertial': folder / 'inertial.mat' if modality != 'video' else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare local paired recordings for replay')
    parser.add_argument('--rgb', type=Path, required=True)
    parser.add_argument('--inertial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', nargs='+', default=DEFAULT_TRIALS)
    args = parser.parse_args()
    entries = prepare(args.rgb, args.inertial, args.output, args.trials)
    print(json.dumps({'trials': [e['id'] for e in entries]}))
