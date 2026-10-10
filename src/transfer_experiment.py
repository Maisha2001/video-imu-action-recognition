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
from prefix_data import load_prefix_cache
from robust_experiment import KNOWN_ACTIONS, partitions
from robust_training import RobustConfig, check_stop, condition_key, conditions, fit_phase, predict_logits
from transfer_model import FeatureDataset, TransferFusion, fit_feature_normalization
from uncertainty import evaluate_logits, fit_calibration
from video_features import FEATURE_SPEC, load_features


SOURCE_FILES = ("activity_data.py", "deep_data.py", "deep_model.py", "attention_model.py", "attention_training.py",
                "prefix_data.py", "robust_model.py", "robust_training.py", "robust_experiment.py", "uncertainty.py",
                "video_features.py", "transfer_model.py", "transfer_experiment.py", "transfer.py")
CALIBRATION_MODES = ("pooled", "conditional")


def clean_key(condition):
    return condition_key(dict(condition, noise=0.0))


def load_inputs(features_path, prefix_path):
    trials, features, feature_meta = load_features(features_path)
    imu_trials, _, imu, prefix_meta = load_prefix_cache(prefix_path)
    if trials != imu_trials or feature_meta["identity"]["archives"] != prefix_meta["archive_sha256"]:
        raise ValueError("Video features and IMU cache must identify the same source trials")
    identity = {"feature_array_sha256": feature_meta["array_sha256"], "feature_identity": feature_meta["identity"],
                "prefix_arrays": prefix_meta["array_sha256"], "archives": prefix_meta["archive_sha256"]}
    return trials, features, imu, identity


def normalization(features, imu, indices):
    return {"video": fit_feature_normalization(features, indices), "imu": fit_normalization(imu[:, -1], indices)}


def calibrate_conditions(logits, targets):
    expected = {condition_key(c) for c in conditions() if c["noise"] == 0}
    if set(logits) != expected:
        raise ValueError("Calibration requires all twelve clean conditions")
    ordered = sorted(logits)
    return {"pooled": fit_calibration(np.concatenate([logits[k] for k in ordered]), np.tile(targets, len(ordered))),
            "conditional": {key: fit_calibration(logits[key], targets) for key in ordered},
            "unique_trials": len(targets)}


def calibration_for(artifact, condition, mode="conditional"):
    if mode not in CALIBRATION_MODES:
        raise ValueError("Unknown calibration mode")
    calibration = artifact["calibration"]
    return calibration["pooled"] if mode == "pooled" else calibration["conditional"][clean_key(condition)]


def run_study(trials, features, imu, metadata, output, config=RobustConfig(batch_size=32),
              seeds=(42, 43, 44), labels=KNOWN_ACTIONS, device="cpu", resume=False, stop=None):
    check_stop(stop)
    config.validate()
    labels = list(labels)
    if labels != sorted(set(labels)) or len(labels) < 2 or any(type(x) is not int or not 1 <= x <= 27 for x in labels):
        raise ValueError("Invalid known labels")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2 ** 32 for s in seeds):
        raise ValueError("Invalid seeds")
    split = partitions(trials, labels)
    targets = np.array([labels.index(t.action) if t.action in labels else -1 for t in trials])
    source = {name: file_hash(Path(__file__).parent / name) for name in SOURCE_FILES}
    identity = {"data": metadata, "trials": [t.key for t in trials], "config": asdict(config), "seeds": list(seeds),
                "labels": labels, "source_sha256": source, "device": device, "torch": str(torch.__version__)}
    output = Path(output)
    if output.exists():
        if not resume or json.loads((output / "protocol.json").read_text())["identity"] != identity:
            raise ValueError("Use --resume with identical data, sources, and settings")
    else:
        output.mkdir(parents=True)
        atomic_json(output / "protocol.json", {"identity": identity, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    if (output / "metrics.json").exists():
        report = json.loads((output / "metrics.json").read_text())
        if set(report["runs"]) != {f"transfer-{s}" for s in seeds}:
            raise ValueError("Incomplete report")
        for key, item in report["runs"].items():
            if file_hash(output / key / "model.pt") != item["model_sha256"]:
                raise ValueError("Saved model checksum changed")
        return report
    started = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    norm = normalization(features, imu, split["train"])
    selected = [c for c in conditions() if c["noise"] == 0 and
                (c["fraction"] == 1 or (c["fraction"] == 0.25 and c["modality"] == "both"))]
    validation = [FeatureDataset(features, imu, targets, split["validation"], norm, c) for c in selected]
    choices = {}
    for seed in seeds:
        configure(seed)
        model = TransferFusion(len(labels), config.dropout)
        training = FeatureDataset(features, imu, targets, split["train"], norm, augment=True, seed=seed)
        key = f"transfer-{seed}"
        choices[key] = fit_phase(model, training, validation, config, seed, config.epochs,
                                 output / key / "selection.pt", {"run": fingerprint(identity), "phase": "selection", "key": key}, device, stop)
    locked = output / "selection.json"
    if locked.exists() and json.loads(locked.read_text())["choices"] != choices:
        raise ValueError("Resumed selection changed")
    if not locked.exists():
        atomic_json(locked, {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "choices": choices})
    norm = normalization(features, imu, split["refit"])
    refits = {}
    for seed in seeds:
        check_stop(stop)
        configure(seed)
        key = f"transfer-{seed}"
        model = TransferFusion(len(labels), config.dropout)
        training = FeatureDataset(features, imu, targets, split["refit"], norm, augment=True, seed=seed)
        refits[key] = fit_phase(model, training, [], config, seed, choices[key]["best_epoch"],
                                output / key / "refit.pt", {"run": fingerprint(identity), "phase": "refit", "key": key}, device, stop)
        atomic_save({"format_version": 1, "architecture": "frozen-r3d-fusion-v1", "feature_spec": FEATURE_SPEC,
                     "labels": labels, "seed": seed, "config": asdict(config), "model": copy_state(model),
                     "normalization": {stream: {k: torch.from_numpy(v) for k, v in values.items()} for stream, values in norm.items()}},
                    output / key / "uncalibrated.pt")
    for key in choices:
        check_stop(stop)
        model, artifact = load_model(output / key / "uncalibrated.pt", require_calibration=False)
        model.to(device)
        logits = {condition_key(c): predict_logits(model, FeatureDataset(features, imu, targets, split["calibration"], norm, c), config, device, stop)
                  for c in conditions() if c["noise"] == 0}
        artifact["calibration"] = calibrate_conditions(logits, targets[split["calibration"]])
        atomic_save(artifact, output / key / "model.pt")
    atomic_json(output / "calibration-complete.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat()})
    runs = {}
    for key in choices:
        check_stop(stop)
        model, artifact = load_model(output / key / "model.pt")
        model.to(device)
        evaluations = {}
        for condition in conditions():
            dataset = FeatureDataset(features, imu, targets, split["test"], norm, condition)
            logits = predict_logits(model, dataset, config, device, stop)
            actions = np.array([trials[i].action for i in split["test"]])
            for mode in CALIBRATION_MODES:
                metrics, rows = evaluate_logits(logits, actions, labels, calibration_for(artifact, condition, mode))
                for row, i in zip(rows, split["test"]):
                    row["trial"] = trials[i].key
                name = f"{condition_key(condition)}-{mode}"
                result = {"condition": condition, "calibration_mode": mode, "metrics": metrics}
                atomic_json(output / key / f"{name}.json", dict(result, predictions=rows))
                evaluations[name] = result
        runs[key] = {"seed": artifact["seed"], "selected_epoch": choices[key]["best_epoch"],
                     "selection": choices[key], "refit": refits[key], "calibration": artifact["calibration"],
                     "model_sha256": file_hash(output / key / "model.pt"), "evaluations": evaluations}
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "labels": labels,
              "unknown_actions": sorted({t.action for t in trials} - set(labels)), "seeds": list(seeds),
              "config": asdict(config), "counts": {k: len(v) for k, v in split.items()}, "data": metadata,
              "source_sha256": source, "runs": runs,
              "runtime": {"invocation_seconds": time.perf_counter() - started, "device": device,
                          "torch": str(torch.__version__), "hardware": torch.cuda.get_device_name() if device == "cuda" else "CPU",
                          "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0}}
    atomic_json(output / "metrics.json", report)
    return report


def load_model(path, require_calibration=True):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact["format_version"] != 1 or artifact["architecture"] != "frozen-r3d-fusion-v1" or artifact["feature_spec"] != FEATURE_SPEC:
        raise ValueError("Unsupported transfer model")
    labels = artifact["labels"]
    if len(labels) < 2 or labels != sorted(set(labels)) or any(type(x) is not int or not 1 <= x <= 27 for x in labels):
        raise ValueError("Invalid known labels")
    config = RobustConfig(**artifact["config"])
    config.validate()
    for stream, size in (("video", 512), ("imu", 6)):
        norm = artifact["normalization"][stream]
        if set(norm) != {"mean", "std"} or any(v.shape != (size,) or not torch.isfinite(v).all() for v in norm.values()) or (norm["std"] <= 0).any():
            raise ValueError("Invalid saved normalization")
    if require_calibration:
        calibration = artifact["calibration"]
        if set(calibration["conditional"]) != {condition_key(c) for c in conditions() if c["noise"] == 0}:
            raise ValueError("Incomplete condition calibration")
        for c in [calibration["pooled"], *calibration["conditional"].values()]:
            if not np.isfinite(c["temperature"]) or c["temperature"] <= 0 or not 0 <= c["threshold"] <= 1:
                raise ValueError("Invalid calibration values")
    model = TransferFusion(len(labels), config.dropout)
    model.load_state_dict(artifact["model"])
    return model.eval(), artifact
