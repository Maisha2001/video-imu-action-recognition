import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from activity_data import subject_split, validate_signal


def features(signal):
    signal = validate_signal(signal)
    chunks = [signal, *np.array_split(signal, 4)]
    values = []
    for chunk in chunks:
        values.extend([chunk.mean(axis=0), chunk.std(axis=0), chunk.min(axis=0),
                       chunk.max(axis=0), np.sqrt(np.mean(chunk ** 2, axis=0)),
                       np.mean(np.abs(np.diff(chunk, axis=0)), axis=0)])
    vector = np.concatenate(values)
    if not np.isfinite(vector).all():
        raise ValueError("Non-finite model features")
    return vector


def scores(y, predicted, labels):
    return {"accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=labels,
                                      average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, predicted, labels=labels).tolist(),
            "per_class": classification_report(y, predicted, labels=labels,
                                                output_dict=True, zero_division=0)}


def fit_baseline(trials, signals, c_values=(0.1, 1.0, 10.0)):
    if len(trials) != len(signals):
        raise ValueError("Each trial must have exactly one signal")
    if len(set(trials)) != len(trials):
        raise ValueError("Trial IDs must be unique")
    train, validation, test = subject_split(trials)
    x = np.stack([features(signal) for signal in signals])
    y = np.array([trial.action for trial in trials])
    labels = sorted(set(y.tolist()))
    if not c_values or any(c <= 0 or not np.isfinite(c) for c in c_values):
        raise ValueError("C values must be finite and positive")
    candidates = []
    for c in c_values:
        model = make_pipeline(StandardScaler(), SVC(C=c, kernel="rbf", gamma="scale"))
        model.fit(x[train], y[train])
        prediction = model.predict(x[validation])
        candidates.append({"C": c, "macro_f1": float(f1_score(
            y[validation], prediction, labels=labels, average="macro", zero_division=0))})
    chosen = max(candidates, key=lambda candidate: (candidate["macro_f1"], -candidate["C"]))
    # Refit only on development subjects after choosing C on subject 7.
    development = np.concatenate([train, validation])
    model = make_pipeline(StandardScaler(), SVC(C=chosen["C"], kernel="rbf", gamma="scale"))
    model.fit(x[development], y[development])
    predicted = model.predict(x[test])
    dummy = DummyClassifier(strategy="most_frequent").fit(x[development], y[development])
    per_subject = {}
    for subject in sorted({trials[i].subject for i in test}):
        positions = [j for j, i in enumerate(test) if trials[i].subject == subject]
        per_subject[str(subject)] = scores(y[test][positions], predicted[positions], labels)
    report = {
        "protocol": {"development_subjects": [1, 3, 5], "validation_subjects": [7],
                     "final_training_subjects": [1, 3, 5, 7], "test_subjects": [2, 4, 6, 8],
                     "test_used_for_selection": False,
                     "counts": {"development": len(train), "validation": len(validation),
                                "final_training": len(development), "test": len(test)}},
        "features": {"count": x.shape[1], "windows": "whole trial plus four equal sample ranges",
                     "statistics": ["mean", "std", "min", "max", "rms", "mean absolute difference"]},
        "validation_candidates": candidates,
        "selected_C": chosen["C"],
        "labels": labels,
        "svm": scores(y[test], predicted, labels),
        "majority": scores(y[test], dummy.predict(x[test]), labels),
        "per_test_subject": per_subject,
        "predictions": [{"trial": trials[i].key, "actual": int(y[i]),
                         "predicted": int(predicted[j])} for j, i in enumerate(test)]}
    return model, report
