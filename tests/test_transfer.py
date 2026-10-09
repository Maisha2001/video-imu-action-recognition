from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from activity_data import Trial
from attention_training import CheckpointPause, configure, state_fingerprint
from robust_experiment import partitions
from robust_training import RobustConfig, conditions, fit_phase
from transfer_experiment import calibrate_conditions, calibration_for, load_model, normalization, run_study
from transfer_model import FeatureDataset, TransferFusion
from video_features import prefix_tensor


def fixture_data():
    trials = [Trial(a, s, 1) for a in (1, 2, 3) for s in range(1, 9)]
    rng = np.random.default_rng(123)
    features = rng.normal(size=(len(trials), 4, 512)).astype(np.float32)
    imu = rng.normal(size=(len(trials), 4, 6, 128)).astype(np.float32)
    return trials, features, imu


def test_visual_prefix_excludes_future_frames_and_normalizes_rgb():
    frames = np.zeros((32, 112, 112, 3), dtype=np.uint8)
    frames[:8, :, :, 0] = 255
    prefix = prefix_tensor(frames, (0.25,))
    frames[8:] = 200
    assert torch.equal(prefix, prefix_tensor(frames, (0.25,)))
    assert prefix.shape == (1, 3, 16, 112, 112)
    assert prefix[0, 0, 0, 0, 0] == pytest.approx((1 - 0.43216) / 0.22803)


def test_mask_removes_nonfinite_missing_inputs():
    configure(1)
    model = TransferFusion(2).eval()
    video, imu = torch.rand(2, 512), torch.rand(2, 6, 128)
    mask = torch.tensor([[0., 1.], [1., 0.]])
    expected = model(video, imu, mask)
    video[0], imu[1] = float("nan"), float("nan")
    torch.testing.assert_close(model(video, imu, mask), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="At least one"):
        model(video, imu, torch.zeros_like(mask))


def test_transfer_training_recovers_exactly(tmp_path):
    trials, features, imu = fixture_data()
    split = partitions(trials, [1, 2])
    norm = normalization(features, imu, split["train"])
    data = FeatureDataset(features, imu, np.array([t.action - 1 for t in trials]), split["train"], norm, augment=True)
    config = RobustConfig(epochs=2, batch_size=4)
    configure(8)
    original = TransferFusion(2)
    fitted = fit_phase(original, data, [], config, 8, 2, tmp_path / "full.pt", {})
    configure(8)
    with pytest.raises(CheckpointPause):
        fit_phase(TransferFusion(2), data, [], config, 8, 2, tmp_path / "resume.pt", {}, stop_after_epoch=1)
    configure(8)
    resumed = TransferFusion(2)
    result = fit_phase(resumed, data, [], config, 8, 2, tmp_path / "resume.pt", {})
    assert result["history"] == fitted["history"]
    assert state_fingerprint(original.state_dict()) == state_fingerprint(resumed.state_dict())
    with pytest.raises(CheckpointPause):
        fit_phase(resumed, data, [], config, 8, 1, tmp_path / "late.pt", {},
                  stop=datetime.now(timezone.utc) - timedelta(seconds=1))


def test_condition_calibration_is_separate_and_noise_uses_clean_threshold():
    from robust_training import condition_key
    logits = {condition_key(c): np.array([[2., 0.], [0., 2.], [2., 0.], [0., 2.]])
              for c in conditions() if c["noise"] == 0}
    a = calibrate_conditions(logits, [0, 1, 0, 1])
    logits["f25-video-n0"] = np.array([[0., 2.], [2., 0.], [0., 2.], [2., 0.]])
    b = calibrate_conditions(logits, [0, 1, 0, 1])
    assert a["conditional"]["f100-both-n0"] == b["conditional"]["f100-both-n0"]
    assert a["conditional"]["f25-video-n0"] != b["conditional"]["f25-video-n0"]
    assert calibration_for({"calibration": b}, {"fraction": 1., "modality": "both", "noise": .3}) == b["conditional"]["f100-both-n0"]
    with pytest.raises(ValueError, match="known labels"):
        calibrate_conditions(logits, [0, 1, -1, 1])


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    directory = tmp_path_factory.mktemp("transfer-study")
    trials, features, imu = fixture_data()
    config = RobustConfig(epochs=1, batch_size=8)
    result = run_study(trials, features, imu, {}, directory / "run", config, seeds=(42,), labels=(1, 2))
    return directory, trials, features, imu, config, result


def test_all_conditions_and_calibration_modes_are_saved(study):
    directory, trials, features, imu, config, result = study
    item = result["runs"]["transfer-42"]
    assert len(item["evaluations"]) == 40
    assert item["calibration"]["unique_trials"] == 2
    assert len(item["calibration"]["conditional"]) == 12
    assert all(e["metrics"]["known_trials"] == 8 and e["metrics"]["unknown_trials"] == 4 for e in item["evaluations"].values())
    assert run_study(trials, features, imu, {}, directory / "run", config, seeds=(42,), labels=(1, 2), resume=True) == result


def test_test_people_and_unknown_classes_cannot_change_any_fitting(study):
    directory, trials, features, imu, config, result = study
    features, imu = features.copy(), imu.copy()
    changed = [i for i, trial in enumerate(trials) if trial.subject % 2 == 0 or trial.action == 3]
    features[changed], imu[changed] = 10000, -10000
    run_study(trials, features, imu, {}, directory / "altered", config, seeds=(42,), labels=(1, 2))
    a, original = load_model(directory / "run/transfer-42/model.pt")
    b, altered = load_model(directory / "altered/transfer-42/model.pt")
    assert state_fingerprint(a.state_dict()) == state_fingerprint(b.state_dict())
    assert original["calibration"] == altered["calibration"]
    for stream in original["normalization"]:
        for key in original["normalization"][stream]:
            assert torch.equal(original["normalization"][stream][key], altered["normalization"][stream][key])


def test_saved_model_rejects_bad_calibration(study, tmp_path):
    directory, *_ = study
    _, artifact = load_model(directory / "run/transfer-42/model.pt")
    artifact["calibration"]["conditional"].pop("f25-video-n0")
    path = tmp_path / "bad.pt"
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="Incomplete"):
        load_model(path)


def test_inertial_file_prediction_needs_no_backbone(study, tmp_path):
    from scipy.io import savemat
    from transfer import predict
    directory, *_ = study
    path = tmp_path / "imu.mat"
    savemat(path, {"d_iner": np.arange(384).reshape(64, 6).astype(float)})
    result = predict(directory / "run/transfer-42/model.pt", inertial=path, fraction=.25)
    assert result["modality"] == "imu" and len(result["probabilities"]) == 2
    with pytest.raises(ValueError, match="Supply"):
        predict(directory / "run/transfer-42/model.pt")


def test_feature_extraction_resume_and_checksum(tmp_path, monkeypatch):
    from zipfile import ZipFile
    import video_features
    from attention_training import CheckpointPause
    trials = [Trial(1, 1, 1), Trial(1, 2, 1)]
    rows = [dict(action=t.action, subject=t.subject, repetition=t.repetition, trial=t.key,
                 video_member=f"{t.key}.avi") for t in trials]
    archive = tmp_path / "rgb.zip"
    with ZipFile(archive, "w") as z:
        for row in rows:
            z.writestr(row["video_member"], b"test")
    monkeypatch.setattr(video_features, "load_trials", lambda *args: (trials, [], rows))
    monkeypatch.setattr(video_features, "backbone", lambda *args: None)
    calls = []
    def extract(*args):
        calls.append(1)
        if len(calls) == 2:
            raise CheckpointPause("interrupted")
        return np.ones((4, 512), dtype=np.float32)
    monkeypatch.setattr(video_features, "extract_file", extract)
    with pytest.raises(CheckpointPause):
        video_features.prepare_features(archive, archive, "unused", tmp_path / "cache")
    metadata = video_features.prepare_features(archive, archive, "unused", tmp_path / "cache", resume=True)
    assert len(calls) == 3 and metadata["trials"] == rows
    assert video_features.load_features(tmp_path / "cache")[1].shape == (2, 4, 512)
    with (tmp_path / "cache/features.npy").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        video_features.load_features(tmp_path / "cache")


def test_report_and_saved_predictions_cover_all_modes(study, monkeypatch):
    from transfer_report import export_report
    import verify_transfer
    directory, trials, features, imu, _, result = study
    summary = export_report(directory / "run", directory / "export")
    rows = [json.loads(line) for line in (directory / "export/transfer-42.jsonl").read_text().splitlines()]
    assert len(rows) == 40 * 12 and len(summary["comparisons"]) == 40
    monkeypatch.setattr(verify_transfer, "load_inputs", lambda *args: (trials, features, imu, {}))
    verification = verify_transfer.verify("fixture", "fixture", directory / "run")
    for check in verification["checks"]["transfer-42"].values():
        assert check["trials"] == check["matching_actions"] == check["matching_rejections"]
        assert check["max_confidence_difference"] == 0
