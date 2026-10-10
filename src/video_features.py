import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from zipfile import ZipFile

import cv2
import numpy as np
import torch

from activity_data import Trial, file_hash, load_trials
from attention_training import atomic_json, configure
from prefix_data import FRACTIONS
from robust_training import check_stop


WEIGHTS_URL = "https://download.pytorch.org/models/r3d_18-b3b3357e.pth"
WEIGHTS_SHA256 = "b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d"
FEATURE_SPEC = {"backbone": "torchvision.r3d_18.KINETICS400_V1", "torchvision": "0.26.0",
                "weights_sha256": WEIGHTS_SHA256, "fractions": list(FRACTIONS), "frames": 16,
                "resize": [128, 171], "crop": [112, 112], "interpolation": "OpenCV INTER_LINEAR",
                "mean": [0.43216, 0.394666, 0.37645], "std": [0.22803, 0.22145, 0.216989],
                "features": 512, "mode": "frozen evaluation; float32; TF32 disabled"}


def backbone(weights, device="cpu"):
    import torchvision
    from torchvision.models.video import r3d_18
    if torchvision.__version__.split("+")[0] != FEATURE_SPEC["torchvision"]:
        raise ValueError("Use the pinned torchvision version")
    if file_hash(weights) != WEIGHTS_SHA256:
        raise ValueError("Pretrained weights checksum mismatch")
    model = r3d_18(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    return model.requires_grad_(False).eval().to(device)


def decode_frames(path):
    path = Path(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Video exceeds 64 MiB")
    capture = cv2.VideoCapture(str(path))
    advertised = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        if not capture.isOpened() or advertised > 1000:
            raise ValueError("Unreadable video or too many frames")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max(frame.shape[:2]) > 2048 or min(frame.shape[:2]) < 16 or len(frames) >= 1000:
                raise ValueError("Video exceeds frame limits")
            frame = cv2.resize(frame, (171, 128), interpolation=cv2.INTER_LINEAR)
            frames.append(cv2.cvtColor(frame[8:120, 30:142], cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) < 2 or (advertised > 0 and advertised != len(frames)):
        raise ValueError("Video did not decode completely")
    return np.stack(frames)


def prefix_tensor(frames, fractions=FRACTIONS):
    if frames.ndim != 4 or frames.shape[1:] != (112, 112, 3) or frames.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB frames at 112 by 112")
    clips = []
    for fraction in fractions:
        if fraction not in FRACTIONS:
            raise ValueError("Unsupported fraction")
        count = int(np.floor(len(frames) * fraction))
        if count < 2:
            raise ValueError("Prefix needs two frames")
        indices = np.rint(np.linspace(0, count - 1, 16)).astype(int)
        clips.append(frames[indices].transpose(3, 0, 1, 2))
    values = torch.from_numpy(np.stack(clips)).float() / 255
    mean = torch.tensor(FEATURE_SPEC["mean"])[None, :, None, None, None]
    std = torch.tensor(FEATURE_SPEC["std"])[None, :, None, None, None]
    return (values - mean) / std


def extract_file(model, path, device="cpu", fractions=FRACTIONS):
    with torch.inference_mode():
        values = model(prefix_tensor(decode_frames(path), fractions).to(device)).cpu().numpy()
    if values.shape != (len(fractions), 512) or not np.isfinite(values).all():
        raise ValueError("Invalid frozen embeddings")
    return values


def prepare_features(inertial, rgb, weights, output, device="cpu", resume=False, stop=None,
                     require_complete=True):
    configure(0)
    check_stop(stop)
    trials, _, rows = load_trials(inertial, rgb, require_complete)
    identity = {"spec": FEATURE_SPEC, "archives": {"inertial": file_hash(inertial), "rgb": file_hash(rgb)},
                "trials": [t.key for t in trials], "source_sha256": file_hash(__file__),
                "device": device, "torch": str(torch.__version__), "opencv": cv2.__version__}
    output = Path(output)
    if output.exists():
        if not resume or json.loads((output / "identity.json").read_text()) != identity:
            raise ValueError("Feature cache requires --resume with unchanged source, data, and settings")
        if (output / "metadata.json").exists():
            load_features(output)
            return json.loads((output / "metadata.json").read_text())
    else:
        output.mkdir(parents=True)
        atomic_json(output / "identity.json", identity)
    model = backbone(weights, device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    (output / "trials").mkdir(exist_ok=True)
    with ZipFile(rgb) as bundle, TemporaryDirectory(prefix=".video-", dir=output) as temporary:
        clip = Path(temporary) / "clip.avi"
        for i, row in enumerate(rows):
            check_stop(stop)
            target = output / "trials" / f"{row['trial']}.npy"
            if not target.exists():
                member = bundle.getinfo(row["video_member"])
                if member.file_size > 64 * 1024 * 1024:
                    raise ValueError("Video exceeds 64 MiB")
                clip.write_bytes(bundle.read(member))
                embedding = extract_file(model, clip, device)
                temporary_file = target.with_suffix(".partial")
                with temporary_file.open("wb") as stream:
                    np.save(stream, embedding, allow_pickle=False)
                temporary_file.replace(target)
            value = np.load(target, allow_pickle=False)
            if value.shape != (4, 512) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Invalid cached trial embedding")
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                print(f"Frozen video features {i + 1}/{len(rows)}", flush=True)
    values = np.stack([np.load(output / "trials" / f"{t.key}.npy", allow_pickle=False) for t in trials])
    np.save(output / "features.npy", values, allow_pickle=False)
    metadata = {"identity": identity, "trials": rows, "array_sha256": file_hash(output / "features.npy"),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": {"invocation_seconds": time.perf_counter() - started, "device": device,
                            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0}}
    atomic_json(output / "metadata.json", metadata)
    return metadata


def load_features(path):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["identity"]["spec"] != FEATURE_SPEC or file_hash(path / "features.npy") != metadata["array_sha256"]:
        raise ValueError("Frozen feature identity or checksum changed")
    trials = [Trial(r["action"], r["subject"], r["repetition"]) for r in metadata["trials"]]
    if trials != sorted(set(trials)) or [t.key for t in trials] != metadata["identity"]["trials"]:
        raise ValueError("Frozen feature trial identities changed")
    values = np.load(path / "features.npy", allow_pickle=False)
    if values.shape != (len(trials), 4, 512) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Invalid frozen feature array")
    return trials, values, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pinned frozen video features")
    parser.add_argument("--inertial", required=True, type=Path)
    parser.add_argument("--rgb", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/video-features"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-at-utc")
    args = parser.parse_args()
    stop = datetime.fromisoformat(args.stop_at_utc) if args.stop_at_utc else None
    if stop is not None and stop.tzinfo is None:
        parser.error("Stop time needs an offset")
    prepare_features(args.inertial, args.rgb, args.weights, args.output, args.device, args.resume, stop)
