import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
import scipy
import sklearn
from scipy.io import loadmat

from activity_data import file_hash, load_trials, validate_signal
from activity_model import features, fit_baseline


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def report_html(report):
    mistakes = [row for row in report["predictions"] if row["actual"] != row["predicted"]]
    rows = "".join(f"<tr><td>{html.escape(row['trial'])}</td><td>{row['actual']}</td>"
                   f"<td>{row['predicted']}</td></tr>" for row in mistakes)
    result = report["svm"]
    video_status = "Video/IMU pairing checked." if report["data"]["paired_video_checked"] else "Video pairing not checked."
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Activity baseline results</title><style>
body{{font:17px system-ui;max-width:900px;margin:40px auto;padding:0 24px;color:#182538}}
h1{{font-size:30px}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #ddd}}
.metrics{{padding:20px;background:#eef4fa;border-radius:8px}}small{{color:#4c5a68}}
</style><h1>Activity recognition: sensor baseline</h1>
<p>Unseen people: subjects 2, 4, 6, and 8. {video_status} Video is not a model input.</p>
<div class="metrics">Accuracy: <b>{result['accuracy']:.1%}</b> · Macro-F1: <b>{result['macro_f1']:.3f}</b>
 · Majority accuracy: {report['majority']['accuracy']:.1%}</div>
<p>RBF SVM, C={report['selected_C']:g}. See metrics.json for the full protocol and results.</p>
<h2>Incorrect predictions ({len(mistakes)})</h2><table><thead><tr><th>Trial</th><th>Actual action</th><th>Predicted action</th></tr></thead>
<tbody>{rows}</tbody></table><p><small>Action numbers follow the UTD-MHAD dataset. These results describe this small benchmark, not live use.</small></p></html>"""


def baseline(args):
    if args.output.exists():
        raise ValueError("Output directory already exists; choose a new run directory")
    started = perf_counter()
    trials, signals, manifest = load_trials(args.inertial, args.rgb)
    model, report = fit_baseline(trials, signals)
    report["data"] = {"dataset": "UTD-MHAD", "trials": len(trials),
                      "inertial_sha256": file_hash(args.inertial),
                      "rgb_sha256": file_hash(args.rgb) if args.rgb else None,
                      "paired_video_checked": args.rgb is not None,
                      "sample_range": [min(map(len, signals)), max(map(len, signals))],
                      "alignment": "Paired trial IDs; no sample timestamps in d_iner"}
    report["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                             "numpy": np.__version__, "scipy": scipy.__version__,
                             "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}
    report["source_sha256"] = {name: file_hash(Path(__file__).parent / name)
                               for name in ("activity.py", "activity_data.py", "activity_model.py")}
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = perf_counter() - started
    args.output.mkdir(parents=True)
    write_json(args.output / "metrics.json", report)
    write_json(args.output / "manifest.json", manifest)
    joblib.dump(model, args.output / "model.joblib")
    (args.output / "report.html").write_text(report_html(report), encoding="utf-8")
    print(json.dumps({"accuracy": report["svm"]["accuracy"], "macro_f1": report["svm"]["macro_f1"],
                      "selected_C": report["selected_C"], "output": str(args.output)}))


def predict(args):
    model = joblib.load(args.model)
    data = loadmat(args.input)
    if "d_iner" not in data:
        raise ValueError("Input must contain d_iner")
    action = int(model.predict(features(validate_signal(data["d_iner"]))[None])[0])
    print(json.dumps({"action": action}))


def main():
    parser = argparse.ArgumentParser(description="Train and inspect an activity-recognition baseline.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("baseline", help="Train with subject-based validation and test splits")
    train.add_argument("--inertial", type=Path, required=True)
    train.add_argument("--rgb", type=Path, help="Optional RGB archive; validate exact trial pairing")
    train.add_argument("--output", type=Path, default=Path("runs/baseline"))
    train.set_defaults(run=baseline)
    inference = commands.add_parser("predict", help="Predict from a local trial and a trusted saved model")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--input", type=Path, required=True)
    inference.set_defaults(run=predict)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
