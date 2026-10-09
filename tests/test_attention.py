from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from activity_data import Trial, subject_split
from attention_experiment import load_model, predict_pair, run_study
from attention_model import CrossAttention, PairedEncoder, paired_contrastive_loss
from attention_training import (AttentionConfig, CheckpointPause, PairedDataset, configure,
                                atomic_json, state_fingerprint, train_phase)
from deep_model import fit_normalization


def fixture_data():
    rng = np.random.default_rng(12)
    trials = [Trial(action, subject, 1) for action in (1, 2) for subject in range(1, 9)]
    video = rng.integers(0, 256, size=(16, 3, 16, 96, 96), dtype=np.uint8)
    imu = rng.normal(size=(16, 6, 128)).astype(np.float32)
    labels = np.array([trial.action - 1 for trial in trials])
    return trials, video, imu, labels


def test_cross_attention_uses_context_and_preserves_probability_rows():
    configure(3)
    layer = CrossAttention(dimension=8, heads=2, dropout=0).eval()
    query = torch.randn(2, 3, 8, requires_grad=True)
    context = torch.randn(2, 5, 8, requires_grad=True)
    result, weights = layer(query, context)
    assert result.shape == query.shape and weights.shape == (2, 2, 3, 5)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 2, 3))
    result[..., 0].sum().backward()
    assert context.grad.abs().sum() > 0 and query.grad.abs().sum() > 0
    changed, _ = layer(query, context * 3)
    assert not torch.allclose(result, changed)


def test_contrastive_pairing_and_gradients():
    video = torch.eye(4, requires_grad=True)
    imu = torch.eye(4, requires_grad=True)
    aligned = paired_contrastive_loss(video, imu)
    shuffled = paired_contrastive_loss(video, imu.roll(1, dims=0))
    assert aligned < 0.01 and shuffled > 5
    aligned.backward()
    assert torch.isfinite(video.grad).all() and video.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="at least two"):
        paired_contrastive_loss(video[:1], imu[:1])


def test_paired_pretraining_updates_both_encoders():
    configure(4)
    model = PairedEncoder(2, "contrastive_attention")
    video_before = model.video[0].weight.detach().clone()
    imu_before = model.imu[0].weight.detach().clone()
    head_before = model.head[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    video = torch.rand(4, 3, 16, 96, 96)
    imu = torch.randn(4, 6, 128)
    loss = paired_contrastive_loss(*model.contrastive_embeddings(video, imu))
    loss.backward()
    optimizer.step()
    assert not torch.equal(video_before, model.video[0].weight)
    assert not torch.equal(imu_before, model.imu[0].weight)
    assert torch.equal(head_before, model.head[-1].weight)


@pytest.mark.parametrize("objective", ["classification", "contrastive"])
def test_resume_matches_uninterrupted_parameters_and_losses(tmp_path, objective):
    trials, video, imu, y = fixture_data()
    train, validation, _ = subject_split(trials)
    norm = fit_normalization(imu, train)
    training = PairedDataset(video, imu, y, train, norm)
    validating = PairedDataset(video, imu, y, validation, norm) if objective == "classification" else None
    config = AttentionConfig(epochs=2, pretrain_epochs=2, batch_size=4)
    configure(7)
    uninterrupted = PairedEncoder(2, "attention")
    original = train_phase(uninterrupted, training, validating, config, 7, 2, objective,
                            tmp_path / "full.pt", {"data": "fixture"})
    configure(7)
    interrupted = PairedEncoder(2, "attention")
    with pytest.raises(CheckpointPause):
        train_phase(interrupted, training, validating, config, 7, 2, objective,
                    tmp_path / "resume.pt", {"data": "fixture"}, stop_after_epoch=1)
    configure(7)
    resumed = PairedEncoder(2, "attention")
    result = train_phase(resumed, training, validating, config, 7, 2, objective,
                         tmp_path / "resume.pt", {"data": "fixture"})
    assert original["history"] == result["history"]
    assert original["best_epoch"] == result["best_epoch"]
    assert state_fingerprint(uninterrupted.state_dict()) == state_fingerprint(resumed.state_dict())
    configure(7)
    with pytest.raises(ValueError, match="Checkpoint identity"):
        train_phase(PairedEncoder(2, "attention"), training, validating, config, 7, 2, objective,
                    tmp_path / "resume.pt", {"data": "changed"})


def test_pretraining_does_not_use_action_labels(tmp_path):
    trials, video, imu, y = fixture_data()
    train, _, _ = subject_split(trials)
    norm = fit_normalization(imu, train)
    config = AttentionConfig(epochs=1, pretrain_epochs=1, batch_size=4)
    states = []
    for index, labels in enumerate((y, 1 - y)):
        configure(8)
        model = PairedEncoder(2, "contrastive_attention")
        dataset = PairedDataset(video, imu, labels, train, norm)
        train_phase(model, dataset, None, config, 8, 1, "contrastive", tmp_path / f"labels-{index}.pt", {})
        states.append(state_fingerprint(model.state_dict()))
    assert states[0] == states[1]


def test_stop_deadline_prevents_next_epoch(tmp_path):
    _, video, imu, y = fixture_data()
    norm = fit_normalization(imu, [0, 1])
    dataset = PairedDataset(video, imu, y, [0, 1], norm)
    configure(9)
    path = tmp_path / "not-started.pt"
    with pytest.raises(CheckpointPause):
        train_phase(PairedEncoder(2), dataset, None, AttentionConfig(), 9, 2, "classification", path, {},
                    stop_at_utc=datetime.now(timezone.utc) - timedelta(seconds=1))
    assert not path.exists()


def test_interrupted_json_write_preserves_previous_record(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    atomic_json(path, {"complete": True})
    original = Path.write_text
    def interrupted_write(target, content, **kwargs):
        original(target, content[:5], **kwargs)
        raise OSError("simulated interrupted disk write")
    monkeypatch.setattr(Path, "write_text", interrupted_write)
    with pytest.raises(OSError, match="interrupted"):
        atomic_json(path, {"complete": False, "details": "new record"})
    assert json.loads(path.read_text()) == {"complete": True}
    monkeypatch.setattr(Path, "write_text", original)
    atomic_json(path, {"complete": True, "retry": True})
    assert json.loads(path.read_text())["retry"] is True


@pytest.fixture(scope="module")
def compared(tmp_path_factory):
    directory = tmp_path_factory.mktemp("attention-study")
    trials, video, imu, _ = fixture_data()
    config = AttentionConfig(epochs=1, pretrain_epochs=1, batch_size=4)
    metadata = {"archive_sha256": {}, "array_sha256": {}}
    report = run_study(trials, video, imu, metadata, directory / "study", config, seeds=(6,))
    return directory, trials, video, imu, config, metadata, report


def test_study_ablation_outputs_persistence_and_resume(compared):
    directory, trials, video, imu, config, metadata, report = compared
    assert set(report["summary"]) == {"concatenation", "attention", "contrastive_attention"}
    assert report["protocol"]["counts"] == {"train": 6, "validation": 2, "refit": 8, "test": 8}
    i = trials.index(Trial(1, 2, 1))
    for variant in report["summary"]:
        key = variant + "-6"
        prediction = predict_pair(directory / "study" / key / "model.pt", video[i], imu[i])
        expected = next(row for row in report["runs"][key]["predictions"] if row["trial"] == trials[i].key)
        np.testing.assert_allclose(prediction["probabilities"], expected["probabilities"], atol=1e-6)
        assert bool(prediction["attention"]) == (variant != "concatenation")
    resumed = run_study(trials, video, imu, metadata, directory / "study", config, seeds=(6,), resume=True)
    assert resumed == report
    with pytest.raises(ValueError, match="identity"):
        run_study(trials, video, imu, metadata, directory / "study", config, seeds=(7,), resume=True)


def test_held_out_inputs_do_not_change_fitted_attention(compared):
    directory, trials, video, imu, config, metadata, report = compared
    _, _, test = subject_split(trials)
    video, imu = video.copy(), imu.copy()
    video[test], imu[test] = 0, 100000
    changed = run_study(trials, video, imu, metadata, directory / "changed-test", config, seeds=(6,))
    for key in report["runs"]:
        original_model, _, original_norm = load_model(directory / "study" / key / "model.pt")
        changed_model, _, changed_norm = load_model(directory / "changed-test" / key / "model.pt")
        assert state_fingerprint(original_model.state_dict()) == state_fingerprint(changed_model.state_dict())
        for name in original_norm:
            np.testing.assert_array_equal(original_norm[name], changed_norm[name])
        assert report["selections"][key]["supervised"]["history"] == changed["selections"][key]["supervised"]["history"]


def test_export_preserves_all_seeds_and_predictions(compared):
    from attention_report import export_report
    directory, _, _, _, _, _, report = compared
    output = directory / "exported"
    summary = export_report(directory / "study", output)
    assert summary["summary"] == report["summary"]
    for key, run in report["runs"].items():
        assert json.loads((output / f"{key}.json").read_text()) == run
    assert "attention_minus_concatenation" in summary["paired_macro_f1_differences"]
    with pytest.raises(FileExistsError):
        export_report(directory / "study", output)


def test_attention_verification_rejects_mismatched_data(compared, monkeypatch):
    import verify_attention
    directory, trials, video, imu, _, metadata, _ = compared
    monkeypatch.setattr(verify_attention, "load_cache", lambda path: (trials, video, imu, metadata))
    result = verify_attention.verify("fixture", directory / "study", "cpu")
    for check in result["checks"].values():
        assert check["matching_actions"] == check["trials"] == 8
        assert check["max_probability_difference"] == 0
    invalid = dict(metadata, archive_sha256={"rgb": "wrong"})
    monkeypatch.setattr(verify_attention, "load_cache", lambda path: (trials, video, imu, invalid))
    with pytest.raises(ValueError, match="Cache differs"):
        verify_attention.verify("fixture", directory / "study", "cpu")


def test_completed_resume_rejects_corrupted_model(compared):
    directory, trials, video, imu, config, metadata, _ = compared
    path = directory / "study/attention-6/model.pt"
    original = path.read_bytes()
    try:
        path.write_bytes(b"interrupted or modified model")
        with pytest.raises(ValueError, match="checksum"):
            run_study(trials, video, imu, metadata, directory / "study", config, seeds=(6,), resume=True)
    finally:
        path.write_bytes(original)
