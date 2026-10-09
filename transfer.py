import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch

from attention_training import CheckpointPause, configure
from deep_data import resample_signal
from prefix_data import FRACTIONS
from robust_training import RobustConfig
from transfer_experiment import calibration_for, load_inputs, load_model, run_study
from uncertainty import probabilities
from video_features import backbone, extract_file


def predict(model_path, weights=None, video=None, inertial=None, fraction=1.0, mode="conditional", device="cpu"):
    if fraction not in FRACTIONS or (video is None and inertial is None):
        raise ValueError("Supply an available sensor and a supported fraction")
    configure(0)
    model, artifact = load_model(model_path)
    visual = np.zeros(512, dtype=np.float32)
    sensor = np.zeros((6, 128), dtype=np.float32)
    if video is not None:
        if weights is None:
            raise ValueError("Video input needs the pinned backbone weights")
        encoder = backbone(weights, device)
        visual = extract_file(encoder, video, device, (fraction,))[0]
        del encoder
    if inertial is not None:
        path = Path(inertial)
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("IMU input exceeds 10 MiB")
        raw = loadmat(path)["d_iner"]
        sensor = resample_signal(raw[:int(np.floor(len(raw) * fraction))])
    norm = artifact["normalization"]
    visual = (torch.from_numpy(visual) - norm["video"]["mean"]) / norm["video"]["std"]
    sensor = (torch.from_numpy(sensor) - norm["imu"]["mean"][:, None]) / norm["imu"]["std"][:, None]
    mask = torch.tensor([[float(video is not None), float(inertial is not None)]])
    modality = "both" if video is not None and inertial is not None else "video" if video is not None else "imu"
    calibration = calibration_for(artifact, {"fraction": fraction, "modality": modality, "noise": 0.0}, mode)
    with torch.inference_mode():
        logits = model.to(device)(visual[None].to(device), sensor[None].to(device), mask.to(device)).cpu().numpy()
    scores = probabilities(logits, calibration["temperature"])[0]
    confidence = float(scores.max())
    top = artifact["labels"][int(scores.argmax())]
    accepted = confidence >= calibration["threshold"]
    return {"action": top if accepted else None, "top_action": top, "accepted": bool(accepted),
            "confidence": confidence, "threshold": calibration["threshold"], "probabilities": scores.tolist(),
            "labels": artifact["labels"], "fraction": fraction, "modality": modality, "calibration_mode": mode,
            "rejection_is_unknown_action_proof": False}


def main():
    parser = argparse.ArgumentParser(description="Frozen video transfer with masked inertial fusion")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--features", type=Path, default=Path("data/video-features"))
    train.add_argument("--prefix-cache", type=Path, default=Path("data/prefix-cache"))
    train.add_argument("--output", type=Path, default=Path("runs/transfer"))
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--stop-at-utc")
    infer = commands.add_parser("predict")
    infer.add_argument("--model", required=True, type=Path)
    infer.add_argument("--weights", type=Path)
    infer.add_argument("--video", type=Path)
    infer.add_argument("--inertial", type=Path)
    infer.add_argument("--fraction", type=float, choices=FRACTIONS, default=1.0)
    infer.add_argument("--mode", choices=("pooled", "conditional"), default="conditional")
    infer.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        if args.command == "train":
            stop = datetime.fromisoformat(args.stop_at_utc) if args.stop_at_utc else None
            if stop is not None and stop.tzinfo is None:
                raise ValueError("Stop time needs an offset")
            report = run_study(*load_inputs(args.features, args.prefix_cache), args.output,
                               RobustConfig(epochs=args.epochs, batch_size=32), tuple(args.seeds),
                               device=args.device, resume=args.resume, stop=stop)
            print(json.dumps({key: value["selected_epoch"] for key, value in report["runs"].items()}))
        else:
            print(json.dumps(predict(args.model, args.weights, args.video, args.inertial, args.fraction, args.mode, args.device), allow_nan=False))
    except CheckpointPause as error:
        parser.exit(75, f"Paused: {error}\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
