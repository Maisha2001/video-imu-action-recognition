import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import cv2
import numpy as np

from activity_data import Trial, file_hash, load_trials
from attention_training import atomic_json
from deep_data import decode_video, resample_signal


FRACTIONS = (0.25, 0.5, 0.75, 1.0)
PREFIX_SPEC = {"fractions": list(FRACTIONS), "frames": 16, "size": 96, "imu_samples": 128,
               "crop": "floor(fraction * original length), before resampling",
               "timing": "relative segmented-trial duration; exact shared timestamps unavailable"}


def prefix_pair(decoded, signal, fraction):
    if not 0 < fraction <= 1 or decoded.ndim != 4 or decoded.shape[0] != 3:
        raise ValueError("Expected decoded RGB frames and a fraction in (0, 1]")
    frame_count = int(np.floor(decoded.shape[1] * fraction))
    sample_count = int(np.floor(len(signal) * fraction))
    if frame_count < 2 or sample_count < 8:
        raise ValueError("Prefix needs at least two frames and eight IMU samples")
    indices = np.rint(np.linspace(0, frame_count - 1, 16)).astype(int)
    return decoded[:, indices].copy(), resample_signal(signal[:sample_count])


def decoded_frames(path):
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if not 2 <= count <= 1000:
        raise ValueError("Invalid original video frame count")
    return decode_video(path, frames=count)


def prepare_prefix_cache(inertial, rgb, output, require_complete=True):
    trials, signals, rows = load_trials(inertial, rgb, require_complete)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    video = np.lib.format.open_memmap(output / "video.npy", mode="w+", dtype=np.uint8,
                                     shape=(len(trials), 4, 3, 16, 96, 96))
    imu = np.empty((len(trials), 4, 6, 128), dtype=np.float32)
    with ZipFile(rgb) as bundle, TemporaryDirectory(prefix=".decode-", dir=output) as temporary:
        path = Path(temporary) / "clip.avi"
        for i, row in enumerate(rows):
            member = bundle.getinfo(row["video_member"])
            if member.file_size > 64 * 1024 * 1024:
                raise ValueError("Video exceeds the 64 MiB limit")
            path.write_bytes(bundle.read(member))
            decoded, details = decoded_frames(path)
            row["decoded_frames"], row["fps"] = details["decoded_frames"], details["fps"]
            for j, fraction in enumerate(FRACTIONS):
                video[i, j], imu[i, j] = prefix_pair(decoded, signals[i], fraction)
            if (i + 1) % 100 == 0 or i + 1 == len(rows):
                print(f"Prepared prefixes {i + 1}/{len(rows)}", flush=True)
    video.flush()
    del video
    np.save(output / "imu.npy", imu)
    metadata = {"format_version": 1, "preprocess": PREFIX_SPEC, "trials": rows,
                "archive_sha256": {"inertial": file_hash(inertial), "rgb": file_hash(rgb)},
                "array_sha256": {name: file_hash(output / name) for name in ("video.npy", "imu.npy")},
                "source_sha256": {name: file_hash(Path(__file__).parent / name)
                                  for name in ("prefix_data.py", "deep_data.py", "activity_data.py")}}
    atomic_json(output / "metadata.json", metadata)
    return metadata


def load_prefix_cache(path):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["format_version"] != 1 or metadata["preprocess"] != PREFIX_SPEC:
        raise ValueError("Unsupported prefix cache")
    for name, digest in metadata["array_sha256"].items():
        if name not in {"video.npy", "imu.npy"} or file_hash(path / name) != digest:
            raise ValueError("Prefix cache checksum mismatch")
    if set(metadata["array_sha256"]) != {"video.npy", "imu.npy"}:
        raise ValueError("Incomplete prefix cache")
    trials = [Trial(row["action"], row["subject"], row["repetition"]) for row in metadata["trials"]]
    if trials != sorted(set(trials)) or any(t.key != row["trial"] or not (1 <= t.action <= 27 and
           1 <= t.subject <= 8 and 1 <= t.repetition <= 4) for t, row in zip(trials, metadata["trials"])):
        raise ValueError("Invalid prefix trial identities")
    video = np.load(path / "video.npy", mmap_mode="r", allow_pickle=False)
    imu = np.load(path / "imu.npy", mmap_mode="r", allow_pickle=False)
    if video.shape != (len(trials), 4, 3, 16, 96, 96) or video.dtype != np.uint8:
        raise ValueError("Invalid prefix video array")
    if imu.shape != (len(trials), 4, 6, 128) or imu.dtype != np.float32 or not np.isfinite(imu).all():
        raise ValueError("Invalid prefix IMU array")
    return trials, video, imu, metadata
