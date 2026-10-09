import argparse
from html import escape
import json
from pathlib import Path

import numpy as np

from attention_model import VARIANTS
from deep_experiment import write_json


def export_report(run, output):
    run, output = Path(run), Path(output)
    report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    for key, values in report["runs"].items():
        if values["variant"] not in VARIANTS or type(values["seed"]) is not int or not 0 <= values["seed"] < 2 ** 32:
            raise ValueError("Invalid result identity")
        if key != f"{values['variant']}-{values['seed']}":
            raise ValueError("Invalid result key")
    output.mkdir(parents=True, exist_ok=False)
    summary = {key: value for key, value in report.items() if key not in {"runs", "selections", "refits"}}
    summary["runs"] = {}
    for key, values in report["runs"].items():
        write_json(output / f"{key}.json", values)
        summary["runs"][key] = {"variant": values["variant"], "seed": values["seed"],
                                "accuracy": values["test"]["accuracy"], "macro_f1": values["test"]["macro_f1"],
                                "selected_epoch": report["selections"][key]["supervised"]["best_epoch"],
                                "validation_macro_f1": report["selections"][key]["supervised"]["best_score"],
                                "model_sha256": report["refits"][key]["model_sha256"],
                                "parameters": report["refits"][key]["parameters"], "evaluation": f"{key}.json"}
    summary["paired_macro_f1_differences"] = {}
    for first, second in (("attention", "concatenation"), ("contrastive_attention", "attention")):
        if first in report["summary"] and second in report["summary"]:
            differences = [report["runs"][f"{first}-{seed}"]["test"]["macro_f1"] -
                           report["runs"][f"{second}-{seed}"]["test"]["macro_f1"] for seed in report["seeds"]]
            summary["paired_macro_f1_differences"][f"{first}_minus_{second}"] = {
                "seeds": report["seeds"], "differences": differences, "mean": float(np.mean(differences))}
    write_json(output / "summary.json", summary)
    write_json(output / "training.json", {"selections": report["selections"], "refits": report["refits"]})
    rows = ""
    for variant, values in report["summary"].items():
        accuracy, f1 = values["accuracy"], values["macro_f1"]
        deviation = "n/a" if f1["std"] is None else f"{f1['std']:.3f}"
        rows += f"<tr><td>{escape(variant)}</td><td>{accuracy['mean']:.2%}</td><td>{f1['mean']:.3f}</td><td>{deviation}</td></tr>"
    seed_rows = "".join(f"<tr><td><a href='{escape(key)}.json'>{escape(key)}</a></td><td>{values['selected_epoch']}</td>"
                        f"<td>{values['accuracy']:.2%}</td><td>{values['macro_f1']:.3f}</td></tr>"
                        for key, values in summary["runs"].items())
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Cross-modal model comparison</title><style>body{{font:17px system-ui;max-width:960px;margin:40px auto;padding:0 20px;color:#172c40}}
table{{border-collapse:collapse;width:100%;margin:24px 0}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #cbd5df}}a{{color:#175ca4}}</style>
<h1>Cross-modal model comparison</h1><p>{len(report['seeds'])} seeds per variant; {report['protocol']['counts']['test']} held-out trials per run.</p>
<table><tr><th>Variant</th><th>Mean accuracy</th><th>Mean macro-F1</th><th>F1 sample SD</th></tr>{rows}</table>
<p>All variants share token encoders, data, and selection rules. Attention changes fusion; contrastive pretraining adds a fixed unsupervised training budget.</p>
<h2>Every seed</h2><table><tr><th>Run and predictions</th><th>Selected epoch</th><th>Accuracy</th><th>Macro-F1</th></tr>{seed_rows}</table>
<p>Seed variation is not a confidence interval over people. This benchmark has been evaluated previously. No new blind test or statistical significance is claimed.</p>
<p><a href="summary.json">Protocol and paired differences</a> · <a href="training.json">Learning curves and runtime</a></p></html>"""
    (output / "report.html").write_text(html, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export a compact attention comparison and per-run evidence")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = export_report(args.run, args.output)
    print(json.dumps(summary["summary"], indent=2))


if __name__ == "__main__":
    main()
