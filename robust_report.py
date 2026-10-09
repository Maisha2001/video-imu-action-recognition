import argparse
from html import escape
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from attention_training import atomic_json
from robust_experiment import VARIANTS
from robust_training import condition_key, conditions


def export_report(run, output):
    run, output = Path(run), Path(output)
    report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    for key, item in report["runs"].items():
        if item["variant"] not in VARIANTS or type(item["seed"]) is not int or key != f"{item['variant']}-{item['seed']}":
            raise ValueError("Invalid run identity")
    output.mkdir(parents=True, exist_ok=False)
    summary = {key: value for key, value in report.items() if key != "runs"}
    summary["runs"], training = {}, {}
    for key, item in report["runs"].items():
        summary["runs"][key] = {k: v for k, v in item.items() if k not in {"selection", "refit"}}
        summary["runs"][key]["selected_epoch"] = item["selection"]["best_epoch"]
        training[key] = {"selection": item["selection"], "refit": item["refit"]}
        with (output / f"{key}.jsonl").open("w", encoding="utf-8", newline="\n") as target:
            for condition in conditions():
                name = condition_key(condition)
                result = json.loads((run / key / f"{name}.json").read_text())
                per_subject = {}
                for subject in (2, 4, 6, 8):
                    rows = [row for row in result["predictions"] if f"_s{subject}_" in row["trial"]]
                    known = [row for row in rows if row["known"]]
                    unknown = [row for row in rows if not row["known"]]
                    per_subject[str(subject)] = {
                        "known_accuracy": float(np.mean([r["actual"] == r["predicted"] for r in known])),
                        "known_macro_f1": float(f1_score([r["actual"] for r in known], [r["predicted"] for r in known],
                                                          labels=report["labels"], average="macro", zero_division=0)),
                        "known_acceptance": float(np.mean([not r["rejected"] for r in known])),
                        "unknown_rejection_recall": float(np.mean([r["rejected"] for r in unknown])),
                        "unknown_auroc": float(roc_auc_score([not r["known"] for r in rows], [1 - r["confidence"] for r in rows]))}
                summary["runs"][key]["evaluations"][name]["per_test_subject"] = per_subject
                for row in result["predictions"]:
                    target.write(json.dumps(dict(row, condition=name), allow_nan=False, separators=(",", ":")) + "\n")
    comparisons = {}
    for variant in VARIANTS:
        runs = [item for item in summary["runs"].values() if item["variant"] == variant]
        if not runs:
            continue
        comparisons[variant] = {}
        for condition in conditions():
            key = condition_key(condition)
            items = [run["evaluations"][key]["metrics"] for run in runs]
            aggregate = {}
            for name in ("known_accuracy", "known_macro_f1", "known_acceptance", "unknown_auroc",
                         "unknown_average_precision", "unknown_rejection_recall"):
                values = [item[name] for item in items]
                aggregate[name] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
            for name in ("nll", "brier", "ece_15_bins"):
                for calibrated in ("uncalibrated", "calibrated"):
                    values = [item[calibrated][name] for item in items]
                    aggregate[f"{calibrated}_{name}"] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
            comparisons[variant][key] = aggregate
    summary["comparisons"] = comparisons
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "training.json", training)
    rows = ""
    for variant, entries in comparisons.items():
        for key, values in entries.items():
            rows += (f"<tr><td>{escape(variant)}</td><td>{escape(key)}</td>"
                     f"<td>{values['known_accuracy']['mean']:.1%}</td><td>{values['known_macro_f1']['mean']:.3f}</td>"
                     f"<td>{values['known_acceptance']['mean']:.1%}</td><td>{values['unknown_auroc']['mean']:.3f}</td>"
                     f"<td>{values['unknown_rejection_recall']['mean']:.1%}</td><td>{values['calibrated_ece_15_bins']['mean']:.3f}</td></tr>")
    links = " · ".join(f"<a href='{escape(key)}.jsonl'>{escape(key)}</a>" for key in summary["runs"])
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Early and missing-modality evaluation</title><style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 18px;color:#172c40}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #cbd5df}}.scroll{{overflow-x:auto}}a{{color:#175ca4}}</style>
<h1>Early and missing-modality evaluation</h1><p>Mean results across {len(report['seeds'])} seeds. f25/f50/f75/f100 denote the observed fraction;
video/imu denote the available sensor; n10/n30 denote noise in normalized IMU units.</p>
<p>Known actions: {len(report['labels'])}; held-out unknown actions: {len(report['unknown_actions'])}. Unknowns never enter training or calibration.
Fractions use known trial boundaries and do not demonstrate online early detection. Confidence thresholds can reject known actions and accept unknown ones.</p>
<div class="scroll"><table><tr><th>Training</th><th>Condition</th><th>Known accuracy</th><th>Known F1</th><th>Known acceptance</th><th>Unknown AUROC</th><th>Unknown rejection</th><th>Calibrated ECE</th></tr>{rows}</table></div>
<p><a href="summary.json">Every seed, sample SD, calibration, and runtime</a> · <a href="training.json">Training histories</a></p><p>Predictions: {links}</p></html>"""
    (output / "report.html").write_text(html, encoding="utf-8", newline="\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export every robustness seed and condition")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export_report(args.run, args.output)
