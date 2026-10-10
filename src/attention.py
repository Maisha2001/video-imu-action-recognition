import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scipy.io import loadmat
import torch

from attention_experiment import predict_pair, run_study
from attention_training import AttentionConfig, CheckpointPause
from deep_data import decode_video, load_cache, resample_signal


def main():
    parser = argparse.ArgumentParser(description="Compare concatenation, cross-modal attention, and paired contrastive pretraining")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--cache", type=Path, default=Path("data/paired-cache"))
    train.add_argument("--output", type=Path, default=Path("runs/attention"))
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--pretrain-epochs", type=int, default=20)
    train.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    train.add_argument("--resume", action="store_true")
    train.add_argument("--stop-at-utc", help="ISO timestamp to pause before starting another epoch")
    predict = commands.add_parser("predict")
    predict.add_argument("--model", required=True, type=Path)
    predict.add_argument("--video", required=True, type=Path)
    predict.add_argument("--inertial", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "train":
            if args.device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable")
            stop = datetime.fromisoformat(args.stop_at_utc) if args.stop_at_utc else None
            if stop is not None and stop.tzinfo is None:
                raise ValueError("Stop time must include a UTC offset")
            if stop is not None and stop <= datetime.now(timezone.utc):
                raise CheckpointPause("Stop time has been reached")
            trials, video, imu, metadata = load_cache(args.cache)
            config = AttentionConfig(epochs=args.epochs, pretrain_epochs=args.pretrain_epochs)
            report = run_study(trials, video, imu, metadata, args.output, config, tuple(args.seeds),
                               device=args.device, resume=args.resume, stop_at_utc=stop)
            print(json.dumps(report["summary"], indent=2))
        else:
            if args.video.stat().st_size > 64 * 1024 * 1024 or args.inertial.stat().st_size > 10 * 1024 * 1024:
                raise ValueError("Input exceeds the video or IMU size limit")
            fields = loadmat(args.inertial)
            if "d_iner" not in fields:
                raise ValueError("IMU file must contain d_iner")
            clip, _ = decode_video(args.video)
            result = predict_pair(args.model, clip, resample_signal(fields["d_iner"]))
            print(json.dumps(result, indent=2, allow_nan=False))
    except CheckpointPause as error:
        parser.exit(75, f"Paused: {error}\n")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
