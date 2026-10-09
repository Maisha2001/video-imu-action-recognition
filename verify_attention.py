import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from activity_data import file_hash, subject_split
from attention_experiment import load_model
from attention_model import VARIANTS
from attention_training import AttentionConfig, PairedDataset, configure, predict_probabilities
from deep_data import load_cache
from deep_experiment import write_json


def verify(cache, run, device):
    run = Path(run)
    report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    trials, video, imu, metadata = load_cache(cache)
    for key in ("archive_sha256", "array_sha256"):
        if metadata[key] != report["dataset"][key]:
            raise ValueError("Cache differs from the evaluated data")
    _, _, test = subject_split(trials)
    config = AttentionConfig(**report["config"])
    config.validate()
    configure(0)
    checks = {}
    for key, evaluation in report["runs"].items():
        if (evaluation["variant"] not in VARIANTS or type(evaluation["seed"]) is not int or
                not 0 <= evaluation["seed"] < 2 ** 32 or key != f"{evaluation['variant']}-{evaluation['seed']}"):
            raise ValueError("Invalid result identity")
        path = run / key / "model.pt"
        expected_hash = report["refits"][key]["model_sha256"]
        if file_hash(path) != expected_hash:
            raise ValueError("Model checksum differs from the evaluated model")
        if [trials[i].key for i in test] != [row["trial"] for row in evaluation["predictions"]]:
            raise ValueError("Evaluation trial order differs")
        model, labels, normalization = load_model(path)
        if labels != report["labels"]:
            raise ValueError("Model labels differ from evaluation labels")
        y = np.array([labels.index(trial.action) for trial in trials])
        dataset = PairedDataset(video, imu, y, test, normalization)
        probability = predict_probabilities(model.to(device), dataset, config, device)
        original = np.array([row["probabilities"] for row in evaluation["predictions"]])
        checks[key] = {"model_sha256": expected_hash, "trials": len(test),
                       "matching_actions": int((probability.argmax(1) == original.argmax(1)).sum()),
                       "max_probability_difference": float(np.max(np.abs(probability - original)))}
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "device": device,
            "torch": str(torch.__version__), "cuda_tf32": torch.backends.cudnn.allow_tf32,
            "verification_source_sha256": file_hash(__file__), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description="Verify every saved paired model against its measured predictions")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new verification output path")
    result = verify(args.cache, args.run, args.device)
    write_json(args.output, result)
    print(json.dumps(result["checks"], indent=2))


if __name__ == "__main__":
    main()
