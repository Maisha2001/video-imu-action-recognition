from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from activity_data import file_hash, subject_split
from activity_model import scores
from attention_model import PairedEncoder, VARIANTS
from attention_training import (AttentionConfig, PairedDataset, atomic_save, configure,
                                copy_state, fingerprint, predict_probabilities, train_phase)
from attention_training import atomic_json as write_json
from deep_data import PREPROCESS
from deep_model import fit_normalization


SOURCE_FILES = ("activity_data.py", "activity_model.py", "deep_data.py", "deep_model.py",
                "deep_experiment.py", "attention_model.py", "attention_training.py",
                "attention_experiment.py", "attention.py")


def run_study(trials, video, imu, metadata, output, config=AttentionConfig(), seeds=(42, 43, 44),
              variants=VARIANTS, device="cpu", resume=False, stop_at_utc=None):
    config.validate()
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2 ** 32 for seed in seeds):
        raise ValueError("Seeds must be unique unsigned 32-bit integers")
    if not variants or len(set(variants)) != len(variants) or any(variant not in VARIANTS for variant in variants):
        raise ValueError("Invalid comparison variants")
    if "contrastive_attention" in variants and config.pretrain_epochs < 1:
        raise ValueError("The contrastive variant requires pretraining")
    if len(set(trials)) != len(trials) or len(video) != len(trials) or len(imu) != len(trials):
        raise ValueError("Expected one video/IMU pair per unique trial")
    train, validation, test = subject_split(trials)
    labels = sorted({trial.action for trial in trials})
    y = np.array([labels.index(trial.action) for trial in trials])
    source_hashes = {name: file_hash(Path(__file__).parent / name) for name in SOURCE_FILES}
    identity = {"dataset": {key: metadata[key] for key in ("archive_sha256", "array_sha256")},
                "trials": [trial.key for trial in trials], "source_sha256": source_hashes,
                "config": asdict(config), "seeds": list(seeds), "variants": list(variants),
                "device": device, "torch": str(torch.__version__)}
    output = Path(output)
    if output.exists():
        if not resume:
            raise ValueError("Run exists; use --resume with unchanged inputs, source, and configuration")
        saved = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("Run identity, source, device, or configuration differs")
        if (output / "metrics.json").exists():
            report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            expected = {f"{variant}-{seed}" for variant in variants for seed in seeds}
            if set(report["runs"]) != expected:
                raise ValueError("Completed report does not contain every requested run")
            for key in expected:
                model_path = output / key / "model.pt"
                if not model_path.exists() or file_hash(model_path) != report["refits"][key]["model_sha256"]:
                    raise ValueError("Completed model is missing or differs from its recorded checksum")
            return report
    else:
        output.mkdir(parents=True)
        saved = {"identity": identity, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                 "protocol": {"train_subjects": [1, 3, 5], "validation_subjects": [7],
                              "refit_subjects": [1, 3, 5, 7], "test_subjects": [2, 4, 6, 8],
                              "selection": "Best validation macro-F1, earliest epoch on ties; every seed reported",
                              "pretraining": "Unlabeled paired training trials only; fixed epoch budget",
                              "test_used_for_selection": False,
                              "counts": {"train": len(train), "validation": len(validation),
                                         "refit": len(train) + len(validation), "test": len(test)}}}
        write_json(output / "protocol.json", saved)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    norm = fit_normalization(imu, train)
    training = PairedDataset(video, imu, y, train, norm)
    validating = PairedDataset(video, imu, y, validation, norm)
    selections = {}
    for variant in variants:
        for seed in seeds:
            key = f"{variant}-{seed}"
            configure(seed)
            model = PairedEncoder(len(labels), variant, config.dropout)
            base = {"run": fingerprint(identity), "variant": variant, "seed": seed, "split": "selection"}
            pretraining = None
            if variant == "contrastive_attention":
                pretraining = train_phase(model, training, None, config, seed, config.pretrain_epochs, "contrastive",
                                          output / key / "selection-pretrain.pt", base, device, stop_at_utc=stop_at_utc)
            selected = train_phase(model, training, validating, config, seed, config.epochs, "classification",
                                   output / key / "selection-fit.pt", base, device, stop_at_utc=stop_at_utc)
            selections[key] = {"variant": variant, "seed": seed, "pretraining": pretraining, "supervised": selected}
            del model
    locked_path = output / "selection.json"
    if not locked_path.exists():
        write_json(locked_path, {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "runs": selections})
    else:
        if json.loads(locked_path.read_text(encoding="utf-8"))["runs"] != selections:
            raise ValueError("Saved selection differs from resumed checkpoints")
    development = np.concatenate((train, validation))
    norm = fit_normalization(imu, development)
    refitting = PairedDataset(video, imu, y, development, norm)
    refits = {}
    for key, selection in selections.items():
        variant, seed = selection["variant"], selection["seed"]
        configure(seed)
        model = PairedEncoder(len(labels), variant, config.dropout)
        base = {"run": fingerprint(identity), "variant": variant, "seed": seed, "split": "refit"}
        pretraining = None
        if variant == "contrastive_attention":
            pretraining = train_phase(model, refitting, None, config, seed, config.pretrain_epochs, "contrastive",
                                      output / key / "refit-pretrain.pt", base, device, stop_at_utc=stop_at_utc)
        fitted = train_phase(model, refitting, None, config, seed, selection["supervised"]["best_epoch"], "classification",
                             output / key / "refit-fit.pt", base, device, stop_at_utc=stop_at_utc)
        model_path = output / key / "model.pt"
        atomic_save({"format_version": 1, "architecture": "paired-token-v1", "variant": variant, "seed": seed,
                     "labels": labels, "preprocess": PREPROCESS, "config": asdict(config),
                     "normalization": {name: torch.from_numpy(value) for name, value in norm.items()},
                     "model": copy_state(model)}, model_path)
        refits[key] = {"pretraining": pretraining, "supervised": fitted, "model_sha256": file_hash(model_path),
                       "parameters": sum(parameter.numel() for parameter in model.parameters())}
        del model
    # Every variant's epoch choice and refit finish before any test prediction.
    test_data = PairedDataset(video, imu, y, test, norm)
    actual = np.array([trials[i].action for i in test])
    evaluations = {}
    for key, selection in selections.items():
        model, _, _ = load_model(output / key / "model.pt")
        probability = predict_probabilities(model.to(device), test_data, config, device)
        predicted = np.array(labels)[probability.argmax(1)]
        subjects = {}
        for subject in sorted({trials[i].subject for i in test}):
            positions = [j for j, i in enumerate(test) if trials[i].subject == subject]
            subjects[str(subject)] = scores(actual[positions], predicted[positions], labels)
        evaluation = {"variant": selection["variant"], "seed": selection["seed"],
                      "test": scores(actual, predicted, labels), "per_test_subject": subjects,
                      "predictions": [{"trial": trials[i].key, "actual": int(actual[j]), "predicted": int(predicted[j]),
                                       "probabilities": probability[j].tolist()} for j, i in enumerate(test)]}
        evaluations[key] = evaluation
        write_json(output / key / "evaluation.json", evaluation)
        del model
    summary = {}
    for variant in variants:
        runs = [values for values in evaluations.values() if values["variant"] == variant]
        summary[variant] = {metric: {"mean": float(np.mean([values["test"][metric] for values in runs])),
                                    "std": float(np.std([values["test"][metric] for values in runs], ddof=1)) if len(runs) > 1 else None,
                                    "values": [values["test"][metric] for values in runs]}
                            for metric in ("accuracy", "macro_f1")}
    report = {"created_at_utc": saved["created_at_utc"], "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": saved["protocol"], "config": asdict(config), "seeds": list(seeds), "labels": labels,
              "preprocess": PREPROCESS, "summary": summary, "selections": selections, "refits": refits,
              "runs": evaluations, "dataset": identity["dataset"], "source_sha256": source_hashes,
              "runtime": {"invocation_seconds": time.perf_counter() - run_started, "device": device,
                          "hardware": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
                          "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
                          "torch": str(torch.__version__), "python": platform.python_version(), "numpy": np.__version__,
                          "cuda_tf32": False, "deterministic_algorithms": True},
              "limitations": ["Three seeds measure initialization variability, not participant-level statistical confidence.",
                              "Repeated evaluation on a previously observed benchmark; no new blind test is claimed.",
                              "Different trials of the same action can be false negatives during label-free contrastive training.",
                              "Attention weights are internal associations, not causal explanations or exact temporal alignment.",
                              "Whole segmented trials only; confidence is not calibrated; no external pretrained weights."]}
    write_json(output / "metrics.json", report)
    return report


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1 or checkpoint["architecture"] != "paired-token-v1" or checkpoint["preprocess"] != PREPROCESS:
        raise ValueError("Unsupported paired model format")
    labels = checkpoint["labels"]
    if not labels or labels != sorted(set(labels)) or any(type(label) is not int or not 1 <= label <= 27 for label in labels):
        raise ValueError("Invalid class labels")
    config = AttentionConfig(**checkpoint["config"])
    config.validate()
    normalization = {name: value.numpy() for name, value in checkpoint["normalization"].items()}
    if (set(normalization) != {"mean", "std"} or any(value.shape != (6,) or not np.isfinite(value).all()
            for value in normalization.values()) or (normalization["std"] <= 0).any()):
        raise ValueError("Invalid normalization")
    model = PairedEncoder(len(labels), checkpoint["variant"], config.dropout)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), labels, normalization


def predict_pair(path, video, imu):
    if video.shape != (3, 16, 96, 96) or video.dtype != np.uint8:
        raise ValueError("Expected 16 RGB frames at 96 by 96")
    if imu.shape != (6, 128) or not np.isfinite(imu).all():
        raise ValueError("Expected 128 finite six-channel IMU samples")
    configure(0)
    model, labels, norm = load_model(path)
    data = PairedDataset(video[None], imu[None], [0], [0], norm)
    v, s, _ = data[0]
    with torch.inference_mode():
        logits, attention = model(v[None], s[None], return_attention=True)
    probability = logits.softmax(dim=-1)[0].numpy()
    return {"labels": labels, "action": labels[int(probability.argmax())], "probabilities": probability.tolist(),
            "scores_are_calibrated": False,
            "attention": {name: weights[0].mean(dim=0).tolist() for name, weights in attention.items()},
            "attention_is_causal_explanation": False}
