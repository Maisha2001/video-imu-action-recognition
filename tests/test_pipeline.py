from argparse import Namespace
from io import BytesIO
import json
import subprocess
import sys
from zipfile import ZipFile

import joblib
import numpy as np
import pytest
from scipy.io import savemat

import activity
from activity_data import Trial, archive_index, load_trials, subject_split, validate_signal
from activity_model import features, fit_baseline


@pytest.fixture
def trials_and_signals():
    rng = np.random.default_rng(42)
    trials = [Trial(action, subject, repetition) for action in (1, 2)
              for subject in range(1, 9) for repetition in (1, 2)]
    signals = [rng.normal(trial.action * 3, 0.2, (40 + trial.subject, 6))
               for trial in trials]
    return trials, signals


def make_archives(tmp_path, trials, signals):
    inertial, rgb = tmp_path / "Inertial.zip", tmp_path / "RGB.zip"
    with ZipFile(inertial, "w") as imu, ZipFile(rgb, "w") as video:
        for trial, signal in zip(trials, signals):
            buffer = BytesIO()
            savemat(buffer, {"d_iner": signal})
            imu.writestr(f"Inertial/{trial.key}_inertial.mat", buffer.getvalue())
            video.writestr(f"RGB/{trial.key}_color.avi", b"fixture: pairing only")
    return inertial, rgb


def test_pairing_preserves_trial_order_and_samples(tmp_path, trials_and_signals):
    trials, signals = trials_and_signals
    inertial, rgb = make_archives(tmp_path, trials, signals)
    loaded, values, rows = load_trials(inertial, rgb, require_complete=False)
    assert loaded == trials
    for actual, expected in zip(values, signals):
        np.testing.assert_array_equal(actual, expected)
    assert rows[0]["video_member"] == "RGB/a1_s1_t1_color.avi"
    with pytest.raises(ValueError, match="Expected 861"):
        load_trials(inertial, rgb)


def test_pairing_rejects_missing_video(tmp_path, trials_and_signals):
    trials, signals = trials_and_signals
    inertial, rgb = make_archives(tmp_path, trials, signals)
    with ZipFile(rgb, "w") as bundle:
        bundle.writestr("RGB/a1_s1_t1_color.avi", b"fixture")
    with pytest.raises(ValueError, match="Video/IMU trial mismatch"):
        load_trials(inertial, rgb, require_complete=False)


@pytest.mark.parametrize("name", ["../a1_s1_t1_inertial.mat", "a28_s1_t1_inertial.mat",
                                  "a1_s9_t1_inertial.mat", "unknown.mat"])
def test_bad_archive_names_are_rejected(tmp_path, name):
    path = tmp_path / "invalid.zip"
    with ZipFile(path, "w") as bundle:
        bundle.writestr(name, b"fixture")
    with pytest.raises(ValueError):
        archive_index(path, "inertial")


def test_duplicate_trial_ids_are_rejected(tmp_path):
    path = tmp_path / "duplicate.zip"
    with ZipFile(path, "w") as bundle:
        for folder in ("first", "second"):
            bundle.writestr(f"{folder}/a1_s1_t1_inertial.mat", b"fixture")
    with pytest.raises(ValueError, match="Duplicate trial"):
        archive_index(path, "inertial")


@pytest.mark.parametrize("signal", [np.zeros((7, 6)), np.zeros((20, 5)),
                                    np.zeros(6), np.full((20, 6), np.nan),
                                    np.full((20, 6), np.inf)])
def test_bad_signals_are_rejected(signal):
    with pytest.raises(ValueError):
        validate_signal(signal)


def test_subject_split_rejects_overlap_and_missing_classes(trials_and_signals):
    trials, _ = trials_and_signals
    with pytest.raises(ValueError, match="overlap"):
        subject_split(trials, validation=(5, 7))
    with pytest.raises(ValueError, match="exactly cover"):
        subject_split(trials, test=(2, 4, 6))
    incomplete = [t for t in trials if not (t.subject == 7 and t.action == 2)]
    with pytest.raises(ValueError, match="every action"):
        subject_split(incomplete)


def test_test_subjects_cannot_change_selection_or_fitted_parameters(trials_and_signals):
    trials, signals = trials_and_signals
    model, report = fit_baseline(trials, signals)
    perturbed = [signal * 1000 - 5000 if trial.subject % 2 == 0 else signal
                 for trial, signal in zip(trials, signals)]
    other, other_report = fit_baseline(trials, perturbed)
    assert report["validation_candidates"] == other_report["validation_candidates"]
    assert report["selected_C"] == other_report["selected_C"]
    for field in ("mean_", "scale_"):
        np.testing.assert_array_equal(getattr(model[0], field), getattr(other[0], field))
    np.testing.assert_array_equal(model[1].support_vectors_, other[1].support_vectors_)
    training = np.stack([features(s) for t, s in zip(trials, signals) if t.subject % 2 == 1])
    np.testing.assert_allclose(model[0].mean_, training.mean(axis=0))
    assert report["protocol"]["counts"]["test"] == 16
    assert {row["trial"] for row in report["predictions"]} == {
        t.key for t in trials if t.subject % 2 == 0}


def test_saved_model_and_cli_prediction_agree(tmp_path, trials_and_signals):
    trials, signals = trials_and_signals
    model, _ = fit_baseline(trials, signals)
    model_path, input_path = tmp_path / "model.joblib", tmp_path / "sample.mat"
    joblib.dump(model, model_path)
    savemat(input_path, {"d_iner": signals[0]})
    result = subprocess.run([sys.executable, "src/activity.py", "predict", "--model", str(model_path),
                             "--input", str(input_path)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["action"] == int(model.predict(features(signals[0])[None])[0])


def test_baseline_writes_consistent_outputs(tmp_path, monkeypatch, trials_and_signals, capsys):
    trials, signals = trials_and_signals
    inertial, rgb = make_archives(tmp_path, trials, signals)
    monkeypatch.setattr(activity, "load_trials", lambda imu, video: load_trials(
        imu, video, require_complete=False))
    args = Namespace(inertial=inertial, rgb=rgb, output=tmp_path / "run")
    activity.baseline(args)
    summary = json.loads(capsys.readouterr().out)
    report = json.loads((args.output / "metrics.json").read_text())
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert summary["accuracy"] == report["svm"]["accuracy"]
    assert len(manifest) == 32
    assert report["data"]["paired_video_checked"] is True
    saved = joblib.load(args.output / "model.joblib")
    test_indices = subject_split(trials)[2]
    predicted = saved.predict(np.stack([features(signals[i]) for i in test_indices]))
    assert predicted.tolist() == [row["predicted"] for row in report["predictions"]]
    assert (args.output / "report.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    with pytest.raises(ValueError, match="already exists"):
        activity.baseline(args)


@pytest.mark.parametrize("c_values", [(), (0,), (-1,), (float("nan"),)])
def test_invalid_search_values_are_rejected(trials_and_signals, c_values):
    with pytest.raises(ValueError, match="finite and positive"):
        fit_baseline(*trials_and_signals, c_values=c_values)
