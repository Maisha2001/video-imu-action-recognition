import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from activity_data import file_hash
from attention_training import atomic_json, configure
from prefix_data import load_prefix_cache
from robust_experiment import load_model, partitions, VARIANTS
from robust_training import PrefixDataset, RobustConfig, condition_key, conditions, predict_logits
from uncertainty import probabilities


def verify(cache, run, device="cpu"):
    configure(0)
    run = Path(run)
    report = json.loads((run / "metrics.json").read_text())
    trials, video, imu, metadata = load_prefix_cache(cache)
    if {k: metadata[k] for k in report["data"]} != report["data"]:
        raise ValueError("Data identity changed")
    labels = report["labels"]
    indices = partitions(trials, labels)["test"]
    targets = np.array([labels.index(t.action) if t.action in labels else -1 for t in trials])
    config = RobustConfig(**report["config"])
    checked = {}
    for key, item in report["runs"].items():
        if item["variant"] not in VARIANTS or type(item["seed"]) is not int or key != f"{item['variant']}-{item['seed']}":
            raise ValueError("Invalid model identity")
        if file_hash(run / key / "model.pt") != item["model_sha256"]:
            raise ValueError("Model checksum changed")
        model, artifact = load_model(run / key / "model.pt")
        model.to(device)
        norm = {k: v.numpy() for k, v in artifact["normalization"].items()}
        checks = {}
        for condition in conditions():
            name = condition_key(condition)
            rows = json.loads((run / key / f"{name}.json").read_text())["predictions"]
            if [trials[i].key for i in indices] != [row["trial"] for row in rows]:
                raise ValueError("Trial order changed")
            dataset = PrefixDataset(video, imu, targets, indices, norm, condition)
            logits = predict_logits(model, dataset, config, device)
            probability = probabilities(logits, artifact["calibration"]["temperature"])
            confidence = probability.max(1)
            actions = np.array(labels)[probability.argmax(1)]
            rejection = confidence < artifact["calibration"]["threshold"]
            checks[name] = {"trials": len(rows), "matching_actions": int(np.sum(actions == [r["predicted"] for r in rows])),
                            "matching_rejections": int(np.sum(rejection == [r["rejected"] for r in rows])),
                            "max_confidence_difference": float(np.max(np.abs(confidence - [r["confidence"] for r in rows])))}
        checked[key] = checks
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "device": device,
            "verification_source_sha256": file_hash(__file__), "checks": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recheck all robustness predictions from saved models")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new verification output")
    result = verify(args.cache, args.run, args.device)
    atomic_json(args.output, result)
    print(json.dumps({key: {"trials": sum(c["trials"] for c in values.values()),
                            "matching_actions": sum(c["matching_actions"] for c in values.values()),
                            "matching_rejections": sum(c["matching_rejections"] for c in values.values())}
                      for key, values in result["checks"].items()}, indent=2))
