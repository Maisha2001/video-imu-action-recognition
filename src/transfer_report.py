import argparse
from html import escape
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from attention_training import atomic_json


def export_report(run, output):
    run, output = Path(run), Path(output)
    report = json.loads((run / "metrics.json").read_text())
    for key, item in report["runs"].items():
        if type(item["seed"]) is not int or key != f"transfer-{item['seed']}":
            raise ValueError("Invalid run identity")
    output.mkdir(parents=True, exist_ok=False)
    summary = {k: v for k, v in report.items() if k != "runs"}
    summary["runs"], training = {}, {}
    for key, item in report["runs"].items():
        summary["runs"][key] = {k: v for k, v in item.items() if k not in {"selection", "refit"}}
        training[key] = {"selection": item["selection"], "refit": item["refit"]}
        with (output / f"{key}.jsonl").open("w", encoding="utf-8", newline="\n") as target:
            for name, evaluation in item["evaluations"].items():
                if not name.replace("-", "").isalnum():
                    raise ValueError("Invalid condition identity")
                result = json.loads((run / key / f"{name}.json").read_text())
                subjects = {}
                for subject in (2, 4, 6, 8):
                    rows = [r for r in result["predictions"] if f"_s{subject}_" in r["trial"]]
                    known = [r for r in rows if r["known"]]
                    unknown = [r for r in rows if not r["known"]]
                    subjects[str(subject)] = {"known_accuracy": float(np.mean([r["actual"] == r["predicted"] for r in known])),
                        "known_macro_f1": float(f1_score([r["actual"] for r in known], [r["predicted"] for r in known], labels=report["labels"], average="macro", zero_division=0)),
                        "known_acceptance": float(np.mean([not r["rejected"] for r in known])),
                        "unknown_rejection_recall": float(np.mean([r["rejected"] for r in unknown])),
                        "unknown_auroc": float(roc_auc_score([not r["known"] for r in rows], [1 - r["confidence"] for r in rows]))}
                summary["runs"][key]["evaluations"][name]["per_test_subject"] = subjects
                for row in result["predictions"]:
                    target.write(json.dumps(dict(row, condition=name), allow_nan=False, separators=(",", ":")) + "\n")
    summary["comparisons"] = {}
    for name in next(iter(report["runs"].values()))["evaluations"]:
        metrics = [item["evaluations"][name]["metrics"] for item in report["runs"].values()]
        aggregates = {}
        for metric in ("known_accuracy", "known_macro_f1", "known_acceptance", "unknown_auroc", "unknown_average_precision", "unknown_rejection_recall"):
            values = [m[metric] for m in metrics]
            aggregates[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        for scheme in ("uncalibrated", "calibrated"):
            for metric in ("nll", "brier", "ece_15_bins"):
                values = [m[scheme][metric] for m in metrics]
                aggregates[f"{scheme}_{metric}"] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        summary["comparisons"][name] = aggregates
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "training.json", training)
    rows = ""
    for name, values in summary["comparisons"].items():
        rows += (f"<tr><td>{escape(name)}</td><td>{values['known_accuracy']['mean']:.1%}</td>"
                 f"<td>{values['known_macro_f1']['mean']:.3f}</td><td>{values['known_acceptance']['mean']:.1%}</td>"
                 f"<td>{values['unknown_auroc']['mean']:.3f}</td><td>{values['unknown_rejection_recall']['mean']:.1%}</td>"
                 f"<td>{values['calibrated_ece_15_bins']['mean']:.3f}</td></tr>")
    links = " · ".join(f"<a href='{escape(key)}.jsonl'>{escape(key)}</a>" for key in summary["runs"])
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Frozen video transfer</title><style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 18px;color:#172c40}}
td,th{{padding:9px;text-align:left;border-bottom:1px solid #cbd5df}}table{{border-collapse:collapse;width:100%}}.scroll{{overflow-x:auto}}a{{color:#175ca4}}</style>
<h1>Frozen video transfer and calibration</h1><p>Means over {len(report['seeds'])} seeds. f25/f50/f75/f100 are observed fractions;
video/imu are available sensors; n10/n30 add normalized IMU noise.</p>
<p>Pooled calibration shares one threshold; conditional calibration fits each clean fraction and sensor configuration separately.
No test or unfamiliar-action examples fit either calibration. Unknown actions are held out from downstream fitting; pretraining can contain related actions.</p>
<div class="scroll"><table><tr><th>Condition and calibration</th><th>Known accuracy</th><th>Known F1</th><th>Known acceptance</th><th>Unknown AUROC</th><th>Unknown rejection</th><th>Calibrated ECE</th></tr>{rows}</table></div>
<p><a href="summary.json">All seeds, sample SD, calibration and runtime</a> · <a href="training.json">Learning curves</a></p><p>Predictions: {links}</p>
<p>This previously observed, segmented-trial benchmark does not establish online detection or reliable rejection of arbitrary unfamiliar activities.</p></html>"""
    (output / "report.html").write_text(html, encoding="utf-8", newline="\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export every transfer seed and calibration condition")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export_report(args.run, args.output)
