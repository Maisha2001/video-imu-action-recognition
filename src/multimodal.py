import argparse
import json
from pathlib import Path

from scipy.io import loadmat
import torch

from deep_data import decode_video, load_cache, prepare_cache, resample_signal
from deep_experiment import predict_pair, run_experiment
from deep_model import TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Train and compare paired video and IMU classifiers")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Validate pairs and decode whole-trial inputs")
    prepare.add_argument("--inertial", required=True, type=Path)
    prepare.add_argument("--rgb", required=True, type=Path)
    prepare.add_argument("--output", type=Path, default=Path("data/paired-cache"))
    train = commands.add_parser("train", help="Select on subject 7, refit on odd subjects, evaluate on even subjects")
    train.add_argument("--cache", type=Path, default=Path("data/paired-cache"))
    train.add_argument("--output", type=Path, default=Path("runs/multimodal"))
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--seed", type=int, default=42)
    predict = commands.add_parser("predict", help="Predict a matching video and IMU trial")
    predict.add_argument("--model", required=True, type=Path)
    predict.add_argument("--video", required=True, type=Path)
    predict.add_argument("--inertial", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            metadata = prepare_cache(args.inertial, args.rgb, args.output)
            print(json.dumps({"paired_trials": len(metadata["trials"])}))
        elif args.command == "train":
            if args.device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable; use --device cpu")
            trials, video, imu, metadata = load_cache(args.cache)
            config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
            report = run_experiment(trials, video, imu, metadata, args.output, config, args.device)
            print(json.dumps({name: {key: value[key] for key in ("accuracy", "macro_f1")}
                              for name, value in report["test"].items()}, indent=2))
        else:
            if args.video.stat().st_size > 64 * 1024 * 1024 or args.inertial.stat().st_size > 10 * 1024 * 1024:
                raise ValueError("Input exceeds the video or IMU size limit")
            fields = loadmat(args.inertial)
            if "d_iner" not in fields:
                raise ValueError("IMU file must contain d_iner")
            video, _ = decode_video(args.video)
            imu = resample_signal(fields["d_iner"])
            print(json.dumps(predict_pair(args.model, video, imu), indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
