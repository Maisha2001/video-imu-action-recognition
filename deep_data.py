import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import cv2
import numpy as np

from activity_data import Trial, file_hash, load_trials, validate_signal


PREPROCESS = {"frames": 16, "size": 96, "imu_samples": 128,
              "video": "uniform decoded frames; aspect-preserving resize and center crop; RGB",
              "imu": "linear interpolation over relative trial duration; six original channels",
              "alignment": "paired whole trials; no exact timestamp alignment assumed"}


def resample_signal(signal, samples=128):
    signal = validate_signal(signal)
    if samples < 8:
        raise ValueError("At least eight output samples are required")
    old = np.linspace(0, 1, len(signal))
    new = np.linspace(0, 1, samples)
    return np.stack([np.interp(new, old, column) for column in signal.T]).astype(np.float32)


def decode_video(path, frames=16, size=96):
    if frames < 2 or size < 16:
        raise ValueError("Invalid video sampling dimensions")
    capture = cv2.VideoCapture(str(path))
    decoded = []
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    advertised = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        if not capture.isOpened() or advertised > 1000:
            raise ValueError("Unreadable video or more than 1000 frames")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if max(height, width) > 2048 or min(height, width) < 16 or len(decoded) >= 1000:
                raise ValueError("Video exceeds input limits")
            scale = size / min(height, width)
            resized = cv2.resize(frame, (round(width * scale), round(height * scale)),
                                 interpolation=cv2.INTER_AREA)
            top, left = (resized.shape[0] - size) // 2, (resized.shape[1] - size) // 2
            decoded.append(cv2.cvtColor(resized[top:top + size, left:left + size], cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(decoded) < 2 or (advertised > 0 and advertised != len(decoded)):
        raise ValueError("Video is too short or did not decode completely")
    indices = np.rint(np.linspace(0, len(decoded) - 1, frames)).astype(int)
    clip = np.stack([decoded[i] for i in indices]).transpose(3, 0, 1, 2)
    return clip, {"decoded_frames": len(decoded), "sampled_frame_indices": indices.tolist(),
                  "fps": fps if np.isfinite(fps) and fps > 0 else None}


def prepare_cache(inertial, rgb, output, require_complete=True):
    trials, signals, rows = load_trials(inertial, rgb, require_complete)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    shape = (len(trials), 3, PREPROCESS["frames"], PREPROCESS["size"], PREPROCESS["size"])
    video = np.lib.format.open_memmap(output / "video.npy", mode="w+", dtype=np.uint8, shape=shape)
    with ZipFile(rgb) as bundle, TemporaryDirectory(prefix=".decode-", dir=output) as temporary:
        clip_path = Path(temporary) / "clip.avi"
        for i, row in enumerate(rows):
            member = bundle.getinfo(row["video_member"])
            if member.file_size > 64 * 1024 * 1024:
                raise ValueError("Video exceeds the 64 MiB limit")
            clip_path.write_bytes(bundle.read(member))
            video[i], details = decode_video(clip_path)
            row.update(details)
            if (i + 1) % 100 == 0 or i + 1 == len(rows):
                print(f"Decoded {i + 1}/{len(rows)} paired clips", flush=True)
    video.flush()
    del video
    np.save(output / "imu.npy", np.stack([resample_signal(signal) for signal in signals]))
    metadata = {"format_version": 1, "preprocess": PREPROCESS, "trials": rows,
                "archive_sha256": {"inertial": file_hash(inertial), "rgb": file_hash(rgb)},
                "array_sha256": {name: file_hash(output / name) for name in ("video.npy", "imu.npy")},
                "preprocess_source_sha256": file_hash(__file__), "opencv": cv2.__version__}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def load_cache(path):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["format_version"] != 1 or metadata["preprocess"] != PREPROCESS:
        raise ValueError("Unsupported cache format or preprocessing")
    for name in ("video.npy", "imu.npy"):
        if file_hash(path / name) != metadata["array_sha256"][name]:
            raise ValueError(f"Cache checksum mismatch: {name}")
    rows = metadata["trials"]
    trials = [Trial(row["action"], row["subject"], row["repetition"]) for row in rows]
    if len(set(trials)) != len(trials) or trials != sorted(trials):
        raise ValueError("Cache trials must be unique and sorted")
    if any(trial.key != row["trial"] or not (1 <= trial.action <= 27 and 1 <= trial.subject <= 8
           and 1 <= trial.repetition <= 4) for trial, row in zip(trials, rows)):
        raise ValueError("Invalid cache trial IDs")
    video = np.load(path / "video.npy", mmap_mode="r", allow_pickle=False)
    imu = np.load(path / "imu.npy", mmap_mode="r", allow_pickle=False)
    if video.shape != (len(trials), 3, 16, 96, 96) or video.dtype != np.uint8:
        raise ValueError("Invalid video cache shape or dtype")
    if imu.shape != (len(trials), 6, 128) or imu.dtype != np.float32 or not np.isfinite(imu).all():
        raise ValueError("Invalid IMU cache")
    return trials, video, imu, metadata
