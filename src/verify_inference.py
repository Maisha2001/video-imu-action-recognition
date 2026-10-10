import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from activity_data import file_hash, subject_split
from activity_model import scores
from deep_data import load_cache
from deep_experiment import load_model, write_json
from deep_model import TrainConfig, fuse, loader, probabilities, seed_training


def verify(cache, run, device):
    run = Path(run)
    report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if file_hash(run / "model.pt") != report["model_sha256"]:
        raise ValueError("Model checksum differs from the evaluated model")
    trials, video, imu, metadata = load_cache(cache)
    for key in ("array_sha256", "archive_sha256"):
        if metadata[key] != report["dataset"][key]:
            raise ValueError("Cache differs from the evaluated data")
    _, _, test = subject_split(trials)
    if [trials[i].key for i in test] != [row["trial"] for row in report["predictions"]]:
        raise ValueError("Evaluation trial order differs")
    seed_training(report["protocol"]["seed"])
    models, labels, normalization, weight = load_model(run / "model.pt")
    if labels != report["labels"]:
        raise ValueError("Model labels differ from evaluation labels")
    y = np.array([labels.index(trial.action) for trial in trials])
    actual = np.array([trials[i].action for i in test])
    predicted = {name: probabilities(models[name].to(device), loader(values, y, test, name, normalization,
                  TrainConfig(batch_size=report["selection"]["encoders"][name]["config"]["batch_size"])), device)
                 for name, values in (("video", video), ("imu", imu))}
    predicted["fusion"] = fuse(predicted["video"], predicted["imu"], weight)
    checks = {}
    for name, probability in predicted.items():
        original = np.array([row[name]["probabilities"] for row in report["predictions"]])
        action = np.array(labels)[probability.argmax(axis=1)]
        checks[name] = {"max_probability_difference": float(np.max(np.abs(probability - original))),
                        "matching_actions": int((probability.argmax(axis=1) == original.argmax(axis=1)).sum()),
                        "trials": len(test), "scores": scores(actual, action, labels)}
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "device": device,
            "model_sha256": report["model_sha256"], "torch": str(torch.__version__),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description="Compare saved-model inference with recorded evaluation")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new verification output path")
    result = verify(args.cache, args.run, args.device)
    write_json(args.output, result)
    print(json.dumps({name: {key: value[key] for key in ("matching_actions", "trials", "max_probability_difference")}
                      for name, value in result["checks"].items()}, indent=2))


if __name__ == "__main__":
    main()
