import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch

from attention_training import CheckpointPause, configure
from deep_data import resample_signal
from prefix_data import FRACTIONS, decoded_frames, load_prefix_cache, prepare_prefix_cache
from robust_experiment import load_model, run_study
from robust_training import RobustConfig, check_stop
from uncertainty import probabilities


def predict(model_path, video=None, inertial=None, fraction=1.0):
    if fraction not in FRACTIONS or (video is None and inertial is None):
        raise ValueError("Choose a supported fraction and supply at least one modality")
    configure(0)
    model, artifact = load_model(model_path)
    clip = np.zeros((3, 16, 96, 96), dtype=np.uint8)
    signal = np.zeros((6, 128), dtype=np.float32)
    if video is not None:
        path = Path(video)
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Video exceeds the 64 MiB limit")
        decoded, _ = decoded_frames(path)
        frame_count = int(np.floor(decoded.shape[1] * fraction))
        if frame_count < 2:
            raise ValueError("Video prefix is too short")
        indices = np.rint(np.linspace(0, frame_count - 1, 16)).astype(int)
        clip = decoded[:, indices]
    if inertial is not None:
        path = Path(inertial)
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("IMU input exceeds the 10 MiB limit")
        raw = loadmat(path)["d_iner"]
        signal = resample_signal(raw[:int(np.floor(len(raw) * fraction))])
    norm = artifact["normalization"]
    normalized = (torch.from_numpy(signal) - norm["mean"][:, None]) / norm["std"][:, None]
    mask = torch.tensor([[float(video is not None), float(inertial is not None)]])
    with torch.inference_mode():
        logits = model(torch.from_numpy(clip.copy()).float()[None] / 255, normalized[None], mask).numpy()
    calibration = artifact["calibration"]
    scores = probabilities(logits, calibration["temperature"])[0]
    confidence = float(scores.max())
    top = artifact["labels"][int(scores.argmax())]
    accepted = confidence >= calibration["threshold"]
    return {"action": top if accepted else None, "top_action": top, "accepted": bool(accepted),
            "confidence": confidence, "threshold": calibration["threshold"], "labels": artifact["labels"],
            "probabilities": scores.tolist(), "fraction": fraction,
            "available": {"video": video is not None, "imu": inertial is not None},
            "rejection_is_unknown_action_proof": False}


def main():
    parser = argparse.ArgumentParser(description="Early, missing-modality, and open-set activity evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--inertial", type=Path, required=True)
    prepare.add_argument("--rgb", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=Path("data/prefix-cache"))
    train = commands.add_parser("train")
    train.add_argument("--cache", type=Path, default=Path("data/prefix-cache"))
    train.add_argument("--output", type=Path, default=Path("runs/robustness"))
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    train.add_argument("--resume", action="store_true")
    train.add_argument("--stop-at-utc")
    infer = commands.add_parser("predict")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--video", type=Path)
    infer.add_argument("--inertial", type=Path)
    infer.add_argument("--fraction", type=float, choices=FRACTIONS, default=1.0)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_prefix_cache(args.inertial, args.rgb, args.output)
        elif args.command == "train":
            stop = datetime.fromisoformat(args.stop_at_utc) if args.stop_at_utc else None
            if stop is not None and stop.tzinfo is None:
                raise ValueError("Stop time must include an offset")
            check_stop(stop)
            if args.device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable")
            data = load_prefix_cache(args.cache)
            report = run_study(*data, args.output, RobustConfig(epochs=args.epochs), tuple(args.seeds),
                               device=args.device, resume=args.resume, stop=stop)
            print(json.dumps({key: run["evaluations"]["f100-both-n0"]["metrics"] for key, run in report["runs"].items()}, indent=2))
        else:
            print(json.dumps(predict(args.model, args.video, args.inertial, args.fraction), indent=2, allow_nan=False))
    except CheckpointPause as error:
        parser.exit(75, f"Paused: {error}\n")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
