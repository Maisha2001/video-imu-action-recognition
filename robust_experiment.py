from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from activity_data import file_hash
from attention_training import atomic_json, atomic_save, configure, copy_state, fingerprint
from deep_model import fit_normalization
from prefix_data import PREFIX_SPEC
from robust_model import RobustEncoder
from robust_training import RobustConfig, PrefixDataset, check_stop, condition_key, conditions, fit_phase, predict_logits
from uncertainty import evaluate_logits, fit_calibration


KNOWN_ACTIONS = tuple(range(1, 22))
VARIANTS = ("full_only", "augmented")
SOURCE_FILES = ("activity_data.py", "deep_data.py", "deep_model.py", "attention_model.py", "attention_training.py",
                "prefix_data.py", "robust_model.py", "robust_training.py", "uncertainty.py", "robust_experiment.py", "robustness.py")


def partitions(trials, labels):
    groups = {"train": (1, 3), "validation": (5,), "refit": (1, 3, 5), "calibration": (7,), "test": (2, 4, 6, 8)}
    result = {name: np.array([i for i, trial in enumerate(trials) if trial.subject in subjects and
                            (name == "test" or trial.action in labels)], dtype=int) for name, subjects in groups.items()}
    for name, indices in result.items():
        if not len(indices) or not set(labels).issubset({trials[i].action for i in indices}):
            raise ValueError(f"Missing known actions in {name}")
    if not any(trials[i].action not in labels for i in result["test"]):
        raise ValueError("Unknown actions are required for the test protocol")
    return result


def run_study(trials, video, imu, metadata, output, config=RobustConfig(), seeds=(42, 43, 44),
              variants=VARIANTS, labels=KNOWN_ACTIONS, device="cpu", resume=False, stop=None):
    check_stop(stop)
    config.validate()
    labels = list(labels)
    if labels != sorted(set(labels)) or len(labels) < 2 or any(type(x) is not int or not 1 <= x <= 27 for x in labels):
        raise ValueError("Invalid known action vocabulary")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2 ** 32 for s in seeds):
        raise ValueError("Invalid seeds")
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError("Invalid variants")
    split = partitions(trials, labels)
    targets = np.array([labels.index(t.action) if t.action in labels else -1 for t in trials])
    source = {name: file_hash(Path(__file__).parent / name) for name in SOURCE_FILES}
    identity = {"data": metadata, "trials": [t.key for t in trials], "config": asdict(config), "seeds": list(seeds), "variants": list(variants),
                "labels": labels, "source_sha256": source, "device": device, "torch": str(torch.__version__)}
    output = Path(output)
    if output.exists():
        if not resume or json.loads((output / "protocol.json").read_text())["identity"] != identity:
            raise ValueError("Existing run requires --resume with identical inputs, source, and settings")
    else:
        output.mkdir(parents=True)
        atomic_json(output / "protocol.json", {"identity": identity, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "counts": {k: len(v) for k, v in split.items()},
                    "policy": "Train 1,3; select on 5; refit 1,3,5; known-only calibration on 7; test even subjects. Unknown actions excluded from all fitting."})
    if (output / "metrics.json").exists():
        report = json.loads((output / "metrics.json").read_text())
        expected = {f"{v}-{s}" for v in variants for s in seeds}
        if set(report["runs"]) != expected:
            raise ValueError("Incomplete completed report")
        for key, run in report["runs"].items():
            if file_hash(output / key / "model.pt") != run["model_sha256"]:
                raise ValueError("Completed model checksum changed")
        return report
    started = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    selected_conditions = [c for c in conditions() if c["noise"] == 0 and
                           (c["fraction"] == 1 or (c["fraction"] == 0.25 and c["modality"] == "both"))]
    clean_conditions = [c for c in conditions() if c["noise"] == 0]
    norm = fit_normalization(imu[:, -1], split["train"])
    validation = [PrefixDataset(video, imu, targets, split["validation"], norm, c) for c in selected_conditions]
    choices = {}
    for variant in variants:
        for seed in seeds:
            check_stop(stop)
            key = f"{variant}-{seed}"
            configure(seed)
            model = RobustEncoder(len(labels), config.dropout)
            training = PrefixDataset(video, imu, targets, split["train"], norm, variant=variant, seed=seed)
            choices[key] = fit_phase(model, training, validation, config, seed, config.epochs,
                                     output / key / "selection.pt", {"run": fingerprint(identity), "phase": "selection", "key": key}, device, stop)
            del model
    locked = output / "selection.json"
    if locked.exists():
        if json.loads(locked.read_text())["choices"] != choices:
            raise ValueError("Resumed selection changed")
    else:
        atomic_json(locked, {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "choices": choices})
    norm = fit_normalization(imu[:, -1], split["refit"])
    refits = {}
    for variant in variants:
        for seed in seeds:
            check_stop(stop)
            key = f"{variant}-{seed}"
            configure(seed)
            model = RobustEncoder(len(labels), config.dropout)
            training = PrefixDataset(video, imu, targets, split["refit"], norm, variant=variant, seed=seed)
            fitted = fit_phase(model, training, [], config, seed, choices[key]["best_epoch"],
                                output / key / "refit.pt", {"run": fingerprint(identity), "phase": "refit", "key": key}, device, stop)
            atomic_save({"format_version": 1, "architecture": "masked-paired-v1", "preprocess": PREFIX_SPEC,
                         "labels": labels, "variant": variant, "seed": seed, "config": asdict(config),
                         "normalization": {k: torch.from_numpy(v) for k, v in norm.items()},
                         "model": copy_state(model)}, output / key / "uncalibrated.pt")
            refits[key] = fitted
            del model
    calibrations = {}
    for key in choices:
        check_stop(stop)
        model, artifact = load_model(output / key / "uncalibrated.pt", require_calibration=False)
        model.to(device)
        logits = [predict_logits(model, PrefixDataset(video, imu, targets, split["calibration"], norm, c), config, device, stop)
                  for c in clean_conditions]
        calibration = fit_calibration(np.concatenate(logits), np.tile(targets[split["calibration"]], len(clean_conditions)))
        calibration["unique_trials"] = len(split["calibration"])
        artifact["calibration"] = calibration
        atomic_save(artifact, output / key / "model.pt")
        calibrations[key] = calibration
        del model
    atomic_json(output / "calibration.json", calibrations)
    runs = {}
    for key in choices:
        check_stop(stop)
        model, artifact = load_model(output / key / "model.pt")
        model.to(device)
        evaluations = {}
        for condition in conditions():
            dataset = PrefixDataset(video, imu, targets, split["test"], norm, condition)
            logits = predict_logits(model, dataset, config, device, stop)
            actions = np.array([trials[i].action for i in split["test"]])
            metrics, predictions = evaluate_logits(logits, actions, labels, artifact["calibration"])
            for row, i in zip(predictions, split["test"]):
                row["trial"] = trials[i].key
            name = condition_key(condition)
            atomic_json(output / key / f"{name}.json", {"condition": condition, "metrics": metrics, "predictions": predictions})
            evaluations[name] = {"condition": condition, "metrics": metrics}
        runs[key] = {"variant": artifact["variant"], "seed": artifact["seed"], "model_sha256": file_hash(output / key / "model.pt"),
                     "calibration": artifact["calibration"], "selection": choices[key], "refit": refits[key], "evaluations": evaluations}
        del model
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "labels": labels,
              "unknown_actions": sorted({t.action for t in trials} - set(labels)), "seeds": list(seeds),
              "config": asdict(config), "counts": {k: len(v) for k, v in split.items()}, "runs": runs,
              "data": {k: metadata[k] for k in ("archive_sha256", "array_sha256")}, "source_sha256": source,
              "runtime": {"invocation_seconds": time.perf_counter() - started, "device": device,
                          "torch": str(torch.__version__), "hardware": torch.cuda.get_device_name() if device == "cuda" else "CPU",
                          "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0}}
    atomic_json(output / "metrics.json", report)
    return report


def load_model(path, require_calibration=True):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact["format_version"] != 1 or artifact["architecture"] != "masked-paired-v1" or artifact["preprocess"] != PREFIX_SPEC:
        raise ValueError("Unsupported robustness model")
    labels = artifact["labels"]
    if len(labels) < 2 or labels != sorted(set(labels)) or any(type(x) is not int or not 1 <= x <= 27 for x in labels):
        raise ValueError("Invalid known labels")
    config = RobustConfig(**artifact["config"])
    config.validate()
    norm = artifact["normalization"]
    if set(norm) != {"mean", "std"} or any(v.shape != (6,) or not torch.isfinite(v).all() for v in norm.values()) or (norm["std"] <= 0).any():
        raise ValueError("Invalid fitted normalization")
    if require_calibration:
        c = artifact["calibration"]
        if not np.isfinite(c["temperature"]) or c["temperature"] <= 0 or not 0 <= c["threshold"] <= 1:
            raise ValueError("Invalid calibration")
    model = RobustEncoder(len(labels), config.dropout)
    model.load_state_dict(artifact["model"])
    return model.eval(), artifact
