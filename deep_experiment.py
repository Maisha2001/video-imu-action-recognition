from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from activity_data import file_hash, subject_split
from activity_model import scores
from deep_data import PREPROCESS
from deep_model import (TrainConfig, fit_normalization, fuse, loader, make_model,
                        probabilities, select_fusion, train_encoder)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_experiment(trials, video, imu, metadata, output, config=TrainConfig(), device="cpu"):
    config.validate()
    if len(trials) != len(video) or len(trials) != len(imu) or len(set(trials)) != len(trials):
        raise ValueError("Expected one paired input per unique trial")
    train, validation, test = subject_split(trials)
    labels = sorted({trial.action for trial in trials})
    y = np.array([labels.index(trial.action) for trial in trials])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start, started_at = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    values = {"video": video, "imu": imu}
    normalization = fit_normalization(imu, train)
    selected = {}
    for modality in values:
        model, selected[modality] = train_encoder(values[modality], y, train, validation, modality,
                                                normalization, len(labels), config, device)
        del model
    weight, candidates = select_fusion(np.array(selected["video"]["validation_probabilities"]),
                                       np.array(selected["imu"]["validation_probabilities"]), y[validation])
    selection = {"encoders": selected, "video_weight": weight, "fusion_candidates": candidates,
                 "validation_trials": [trials[i].key for i in validation],
                 "validation_labels": [trials[i].action for i in validation],
                 "normalization": {key: value.tolist() for key, value in normalization.items()},
                 "locked_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "selection.json", selection)
    development = np.concatenate([train, validation])
    normalization = fit_normalization(imu, development)
    models, refit = {}, {}
    for modality in values:
        refit_config = replace(config, epochs=selected[modality]["selected_epoch"])
        models[modality], refit[modality] = train_encoder(values[modality], y, development, [], modality,
                                                        normalization, len(labels), refit_config, device)
    checkpoint = {"format_version": 1, "architecture": "compact-conv-v1", "labels": labels,
                  "preprocess": PREPROCESS, "video_weight": weight,
                  "normalization": {key: torch.from_numpy(value) for key, value in normalization.items()},
                  "models": {modality: {key: value.detach().cpu() for key, value in model.state_dict().items()}
                             for modality, model in models.items()}, "config": asdict(config)}
    torch.save(checkpoint, output / "model.pt")
    # Selection and refitting finish before any held-out prediction is computed.
    predicted = {modality: probabilities(model, loader(values[modality], y, test, modality,
                                                      normalization, config), device)
                 for modality, model in models.items()}
    predicted["fusion"] = fuse(predicted["video"], predicted["imu"], weight)
    actual = np.array([trials[i].action for i in test])
    classes = {modality: np.array(labels)[prob.argmax(axis=1)] for modality, prob in predicted.items()}
    per_subject = {}
    for subject in sorted({trials[i].subject for i in test}):
        positions = [j for j, i in enumerate(test) if trials[i].subject == subject]
        per_subject[str(subject)] = {modality: scores(actual[positions], p[positions], labels)
                                     for modality, p in classes.items()}
    source_names = ("activity_data.py", "activity_model.py", "deep_data.py", "deep_model.py",
                    "deep_experiment.py", "multimodal.py")
    report = {"started_at_utc": started_at, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": {"development_subjects": [1, 3, 5], "validation_subjects": [7],
                           "final_training_subjects": [1, 3, 5, 7], "test_subjects": [2, 4, 6, 8],
                           "test_used_for_selection": False, "seed": config.seed,
                           "counts": {"development": len(train), "validation": len(validation),
                                      "final_training": len(development), "test": len(test)},
                           "selection": "Highest validation macro-F1; earliest epoch on ties. Fusion ties prefer 0.5, then smaller video weight.",
                           "refit": "Reset each encoder to the same seed; train on odd subjects for its selected epoch count."},
              "labels": labels, "preprocess": PREPROCESS, "selection": selection, "refit": refit,
              "test": {modality: scores(actual, p, labels) for modality, p in classes.items()},
              "per_test_subject": per_subject,
              "predictions": [{"trial": trials[i].key, "actual": int(actual[j]),
                               **{modality: {"action": int(classes[modality][j]), "probabilities": prob[j].tolist()}
                                  for modality, prob in predicted.items()}} for j, i in enumerate(test)],
              "dataset": {key: metadata[key] for key in ("archive_sha256", "array_sha256", "preprocess_source_sha256", "opencv")},
              "runtime": {"seconds": time.perf_counter() - start, "device": device,
                          "hardware": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
                          "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
                          "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                          "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "threads": torch.get_num_threads()},
              "source_sha256": {name: file_hash(Path(__file__).parent / name) for name in source_names},
              "model_sha256": file_hash(output / "model.pt"),
              "limitations": ["Single-seed evaluation on a small segmented benchmark; no live-stream or new-environment claim.",
                              "Softmax and fused scores are not calibrated confidence estimates.",
                              "Whole-trial relative resampling does not establish exact cross-modal time alignment.",
                              "Sensor placement differs between action groups; no pretrained weights are used."]}
    write_json(output / "metrics.json", report)
    (output / "report.html").write_text(report_html(report), encoding="utf-8")
    return report


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["format_version"] != 1 or checkpoint["architecture"] != "compact-conv-v1"
            or checkpoint["preprocess"] != PREPROCESS):
        raise ValueError("Unsupported model format")
    labels = checkpoint["labels"]
    if not labels or sorted(set(labels)) != labels or not all(type(label) is int and 1 <= label <= 27 for label in labels):
        raise ValueError("Invalid model labels")
    normalization = {key: value.numpy() for key, value in checkpoint["normalization"].items()}
    if (set(normalization) != {"mean", "std"} or any(value.shape != (6,) or not np.isfinite(value).all()
            for value in normalization.values()) or (normalization["std"] <= 0).any()):
        raise ValueError("Invalid model normalization")
    if not 0 <= checkpoint["video_weight"] <= 1:
        raise ValueError("Invalid model fusion weight")
    models = {}
    for modality in ("video", "imu"):
        models[modality] = make_model(modality, len(labels))
        models[modality].load_state_dict(checkpoint["models"][modality])
        models[modality].eval()
    return models, labels, normalization, checkpoint["video_weight"]


def predict_pair(path, video, imu):
    if video.shape != (3, 16, 96, 96) or video.dtype != np.uint8:
        raise ValueError("Expected a sampled RGB clip")
    if imu.shape != (6, 128) or not np.isfinite(imu).all():
        raise ValueError("Expected a resampled six-channel IMU trial")
    models, labels, normalization, weight = load_model(path)
    predicted = {modality: probabilities(model, loader(values[None], np.array([0]), [0], modality,
                                                       normalization, TrainConfig()), "cpu")
                 for modality, model, values in (("video", models["video"], video), ("imu", models["imu"], imu))}
    predicted["fusion"] = fuse(predicted["video"], predicted["imu"], weight)
    return {"labels": labels, "video_weight": weight,
            "scores_are_calibrated": False,
            **{modality: {"action": labels[int(prob.argmax(axis=1)[0])], "probabilities": prob[0].tolist()}
               for modality, prob in predicted.items()}}


def report_html(report):
    rows = "".join(f"<tr><td>{name}</td><td>{value['accuracy']:.2%}</td><td>{value['macro_f1']:.3f}</td></tr>"
                   for name, value in report["test"].items())
    subjects = "".join(f"<tr><td>{subject}</td>" + "".join(f"<td>{values[name]['accuracy']:.2%}</td>"
                        for name in ("video", "imu", "fusion")) + "</tr>"
                        for subject, values in report["per_test_subject"].items())
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Video and IMU comparison</title><style>body{{font:17px system-ui;max-width:850px;margin:48px auto;padding:0 20px;color:#172c40}}
table{{border-collapse:collapse;width:100%;margin:24px 0}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #cbd5df}}
small{{color:#526171}}</style><h1>Video and IMU comparison</h1>
<p>Held-out subjects 2, 4, 6, and 8: {report['protocol']['counts']['test']} paired trials.</p>
<table><tr><th>Model</th><th>Accuracy</th><th>Macro-F1</th></tr>{rows}</table>
<p>Fusion uses {report['selection']['video_weight']:.0%} video weight, selected using subject 7 only.</p>
<h2>Accuracy by participant</h2><table><tr><th>Subject</th><th>Video</th><th>IMU</th><th>Fusion</th></tr>{subjects}</table>
<p>Models are trained from scratch. Epochs and fusion weight are selected on validation data, then models are refit on odd subjects.</p>
<p>These are segmented recordings. Scores do not establish live-stream performance or calibrated confidence.</p>
<p><a href="metrics.json">Predictions, per-class scores, selection, and runtime evidence</a></p></html>"""
