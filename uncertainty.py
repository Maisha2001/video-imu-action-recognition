import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def checked_logits(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not len(logits) or logits.shape[1] < 2 or not np.isfinite(logits).all():
        raise ValueError("Expected finite class logits")
    return logits


def probabilities(logits, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    return softmax(checked_logits(logits) / temperature, axis=1)


def fit_calibration(logits, targets):
    logits = checked_logits(logits)
    targets = np.asarray(targets, dtype=int)
    if targets.shape != (len(logits),) or (targets < 0).any() or (targets >= logits.shape[1]).any():
        raise ValueError("Calibration requires known labels only")
    def loss(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(targets)), targets]))
    fitted = minimize_scalar(loss, bounds=(np.log(0.05), np.log(20)), method="bounded")
    if not fitted.success:
        raise ValueError("Temperature fitting failed")
    temperature = float(np.exp(fitted.x))
    confidence = probabilities(logits, temperature).max(1)
    return {"temperature": temperature, "threshold": float(np.quantile(confidence, 0.05, method="lower")),
            "observations": len(logits), "target_known_acceptance": 0.95,
            "calibration_nll_before": loss(0), "calibration_nll_after": loss(fitted.x)}


def calibration_scores(probability, targets, bins=15):
    targets = np.asarray(targets)
    confidence, predicted = probability.max(1), probability.argmax(1)
    correct = predicted == targets
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = sum(np.mean(indices == b) * abs(float(correct[indices == b].mean()) - float(confidence[indices == b].mean()))
              for b in range(bins) if np.any(indices == b))
    actual = probability[np.arange(len(targets)), targets]
    brier = np.sum(probability ** 2, axis=1) - 2 * actual + 1
    return {"nll": float(-np.log(np.clip(actual, 1e-15, 1)).mean()),
            "brier": float(brier.mean()), "ece_15_bins": float(ece)}


def evaluate_logits(logits, actions, labels, calibration):
    logits = checked_logits(logits)
    actions = np.asarray(actions)
    probability = probabilities(logits, calibration["temperature"])
    known = np.isin(actions, labels)
    if not known.any() or known.all():
        raise ValueError("Evaluation requires known and unknown examples")
    targets = np.array([labels.index(int(action)) for action in actions[known]])
    confidence = probability.max(1)
    predicted = np.array(labels)[probability.argmax(1)]
    accepted = confidence >= calibration["threshold"]
    known_accepted = known & accepted
    unknown = ~known
    metrics = {"known_trials": int(known.sum()), "unknown_trials": int(unknown.sum()),
               "known_accuracy": float((predicted[known] == actions[known]).mean()),
               "known_macro_f1": float(f1_score(actions[known], predicted[known], labels=labels, average="macro", zero_division=0)),
               "known_acceptance": float(accepted[known].mean()),
               "accepted_known_accuracy": float((predicted[known_accepted] == actions[known_accepted]).mean()) if known_accepted.any() else None,
               "unknown_rejection_recall": float((~accepted[unknown]).mean()),
               "unknown_auroc": float(roc_auc_score(unknown, 1 - confidence)),
               "unknown_average_precision": float(average_precision_score(unknown, 1 - confidence)),
               "uncalibrated": calibration_scores(probabilities(logits)[known], targets),
               "calibrated": calibration_scores(probability[known], targets)}
    rows = []
    for i, action in enumerate(actions):
        actual_probability = float(probability[i, labels.index(int(action))]) if known[i] else None
        rows.append({"actual": int(action), "known": bool(known[i]), "predicted": int(predicted[i]),
                     "confidence": float(confidence[i]), "rejected": bool(not accepted[i]),
                     "actual_probability": actual_probability})
    return metrics, rows
