from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from activity_data import Trial
from attention_training import CheckpointPause, configure, state_fingerprint
from deep_model import fit_normalization
from prefix_data import prefix_pair
from robust_experiment import load_model, partitions, run_study
from robust_model import RobustEncoder
from robust_training import PrefixDataset, RobustConfig, conditions, fit_phase
from uncertainty import evaluate_logits, fit_calibration, probabilities


def fixture_data():
    trials = [Trial(a, s, 1) for a in (1, 2, 3) for s in range(1, 9)]
    rng = np.random.default_rng(6)
    video = rng.integers(0, 256, (len(trials), 4, 3, 16, 96, 96), dtype=np.uint8)
    imu = rng.normal(size=(len(trials), 4, 6, 128)).astype(np.float32)
    return trials, video, imu


def test_prefix_never_uses_later_frames_or_samples():
    rng = np.random.default_rng(1)
    decoded = rng.integers(0, 256, (3, 32, 96, 96), dtype=np.uint8)
    signal = rng.normal(size=(64, 6))
    original = prefix_pair(decoded, signal, 0.25)
    decoded[:, 8:] = 255
    signal[16:] = 1e6
    changed = prefix_pair(decoded, signal, 0.25)
    for a, b in zip(original, changed):
        np.testing.assert_array_equal(a, b)
    assert original[0].shape == (3, 16, 96, 96) and original[1].shape == (6, 128)


def test_unknown_actions_and_calibration_people_never_enter_training():
    trials, _, _ = fixture_data()
    split = partitions(trials, [1, 2])
    for name in ("train", "validation", "refit", "calibration"):
        assert all(trials[i].action in (1, 2) for i in split[name])
    assert not set(split["refit"]) & set(split["calibration"])
    assert all(trials[i].subject == 7 for i in split["calibration"])
    assert any(trials[i].action == 3 for i in split["test"])


def test_missing_modality_ignores_arbitrary_values_and_rejects_no_input():
    configure(3)
    model = RobustEncoder(2).eval()
    video, imu = torch.rand(2, 3, 16, 96, 96), torch.randn(2, 6, 128)
    mask = torch.tensor([[0., 1.], [1., 0.]])
    expected = model(video, imu, mask)
    video[0] = float("nan")
    imu[1] = float("nan")
    torch.testing.assert_close(model(video, imu, mask), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="At least one"):
        model(video, imu, torch.zeros_like(mask))


def test_augmentation_is_reproducible_by_epoch_and_example():
    trials, video, imu = fixture_data()
    split = partitions(trials, [1, 2])
    targets = np.array([t.action - 1 for t in trials])
    norm = fit_normalization(imu[:, -1], split["train"])
    dataset = PrefixDataset(video, imu, targets, split["train"], norm, variant="augmented")
    first, repeat = dataset[0], dataset[0]
    for a, b in zip(first[:3], repeat[:3]):
        assert torch.equal(a, b)
    dataset.epoch = 2
    assert any(not torch.equal(a, b) for a, b in zip(first[:3], dataset[0][:3]))


def test_calibration_preserves_rank_and_uses_only_known_labels():
    logits = np.array([[8., 0.], [8., 0.], [0., 8.], [0., 8.]])
    targets = [0, 1, 1, 1]
    calibration = fit_calibration(logits, targets)
    assert calibration["temperature"] > 1
    assert calibration["calibration_nll_after"] < calibration["calibration_nll_before"]
    assert np.array_equal(probabilities(logits).argmax(1), probabilities(logits, calibration["temperature"]).argmax(1))
    with pytest.raises(ValueError, match="known labels"):
        fit_calibration(logits, [0, 1, 1, -1])


def test_unknown_metrics_and_known_rejection_accounting():
    logits = np.array([[8., 0.], [0., 8.], [0., 0.], [0.01, 0.]])
    metrics, rows = evaluate_logits(logits, [1, 2, 3, 3], [1, 2], {"temperature": 1., "threshold": 0.8})
    assert metrics["known_accuracy"] == metrics["known_acceptance"] == 1
    assert metrics["unknown_rejection_recall"] == metrics["unknown_auroc"] == 1
    assert [row["rejected"] for row in rows] == [False, False, True, True]


def test_augmented_checkpoint_resume_is_exact(tmp_path):
    trials, video, imu = fixture_data()
    split = partitions(trials, [1, 2])
    y = np.array([t.action - 1 for t in trials])
    norm = fit_normalization(imu[:, -1], split["train"])
    config = RobustConfig(epochs=2, batch_size=4)
    data = PrefixDataset(video, imu, y, split["train"], norm, variant="augmented")
    configure(9)
    full = RobustEncoder(2)
    original = fit_phase(full, data, [], config, 9, 2, tmp_path / "full.pt", {})
    configure(9)
    with pytest.raises(CheckpointPause):
        fit_phase(RobustEncoder(2), data, [], config, 9, 2, tmp_path / "resume.pt", {}, stop_after_epoch=1)
    configure(9)
    resumed = RobustEncoder(2)
    result = fit_phase(resumed, data, [], config, 9, 2, tmp_path / "resume.pt", {})
    assert result["history"] == original["history"]
    assert state_fingerprint(full.state_dict()) == state_fingerprint(resumed.state_dict())
    with pytest.raises(CheckpointPause):
        fit_phase(RobustEncoder(2), data, [], config, 9, 1, tmp_path / "late.pt", {},
                  stop=datetime.now(timezone.utc) - timedelta(seconds=1))
    assert not (tmp_path / "late.pt").exists()


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    directory = tmp_path_factory.mktemp("robust-study")
    trials, video, imu = fixture_data()
    config = RobustConfig(epochs=1, batch_size=8)
    metadata = {"archive_sha256": {}, "array_sha256": {}}
    result = run_study(trials, video, imu, metadata, directory / "run", config, seeds=(42,), labels=(1, 2))
    return directory, trials, video, imu, config, metadata, result


def test_study_reports_every_condition_and_persists_calibration(study):
    directory, trials, video, imu, config, metadata, result = study
    assert set(result["runs"]) == {"full_only-42", "augmented-42"}
    for key, run in result["runs"].items():
        assert len(run["evaluations"]) == len(conditions()) == 20
        model, artifact = load_model(directory / "run" / key / "model.pt")
        assert artifact["labels"] == [1, 2]
        assert artifact["calibration"]["unique_trials"] == 2
        assert run["evaluations"]["f100-both-n0"]["metrics"]["unknown_trials"] == 4
    assert run_study(trials, video, imu, metadata, directory / "run", config, seeds=(42,), labels=(1, 2), resume=True) == result


def test_unseen_inputs_cannot_change_training_or_calibration(study):
    directory, trials, video, imu, config, metadata, original = study
    video, imu = video.copy(), imu.copy()
    indices = [i for i, t in enumerate(trials) if t.subject in (2, 4, 6, 8) or t.action == 3]
    video[indices], imu[indices] = 0, 9999
    result = run_study(trials, video, imu, metadata, directory / "altered", config, seeds=(42,), labels=(1, 2))
    for key in original["runs"]:
        a, old = load_model(directory / "run" / key / "model.pt")
        b, new = load_model(directory / "altered" / key / "model.pt")
        assert state_fingerprint(a.state_dict()) == state_fingerprint(b.state_dict())
        assert old["calibration"] == new["calibration"]
        for name in old["normalization"]:
            assert torch.equal(old["normalization"][name], new["normalization"][name])


def test_export_includes_all_predictions_and_conditions(study):
    from robust_report import export_report
    directory, _, _, _, _, _, result = study
    output = directory / "export"
    summary = export_report(directory / "run", output)
    assert set(summary["comparisons"]) == {"full_only", "augmented"}
    for key, run in result["runs"].items():
        rows = [json.loads(line) for line in (output / f"{key}.jsonl").read_text().splitlines()]
        assert len(rows) == 20 * 12
        assert len({row["condition"] for row in rows}) == 20
        assert summary["runs"][key]["model_sha256"] == run["model_sha256"]


def test_saved_robustness_verification(study, monkeypatch):
    import verify_robustness
    directory, trials, video, imu, _, metadata, _ = study
    monkeypatch.setattr(verify_robustness, "load_prefix_cache", lambda path: (trials, video, imu, metadata))
    result = verify_robustness.verify("fixture", directory / "run")
    for run in result["checks"].values():
        for condition in run.values():
            assert condition["matching_actions"] == condition["matching_rejections"] == condition["trials"]
            assert condition["max_confidence_difference"] == 0


def test_file_prediction_supports_one_sensor_and_rejects_none(study, tmp_path):
    import cv2
    from scipy.io import savemat
    from robustness import predict
    directory, _, _, _, _, _, _ = study
    model = directory / "run/augmented-42/model.pt"
    imu = tmp_path / "inertial.mat"
    savemat(imu, {"d_iner": np.arange(64 * 6).reshape(64, 6).astype(float)})
    path = tmp_path / "color.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (96, 96))
    assert writer.isOpened()
    for i in range(32):
        writer.write(np.full((96, 96, 3), i * 7, dtype=np.uint8))
    writer.release()
    for video, inertial in ((path, imu), (path, None), (None, imu)):
        result = predict(model, video, inertial, fraction=0.25)
        assert result["available"] == {"video": video is not None, "imu": inertial is not None}
        assert len(result["probabilities"]) == 2 and 0 <= result["confidence"] <= 1
        assert (result["action"] is not None) == result["accepted"]
    with pytest.raises(ValueError, match="at least one"):
        predict(model)
