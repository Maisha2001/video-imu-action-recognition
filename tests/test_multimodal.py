from io import BytesIO
import json
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
from scipy.io import savemat

from activity_data import Trial, file_hash, subject_split
from deep_data import decode_video, load_cache, prepare_cache, resample_signal
from deep_experiment import load_model, predict_pair, run_experiment
from deep_model import TrainConfig, fit_normalization, fuse, make_model, seed_training, select_fusion, train_encoder


def write_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (128, 96))
    assert writer.isOpened()
    for i in range(20):
        frame = np.zeros((96, 128, 3), dtype=np.uint8)
        frame[..., 2] = i * 10
        writer.write(frame)
    writer.release()


def test_decode_samples_full_duration_and_rgb(tmp_path):
    path = tmp_path / "clip.avi"
    write_video(path)
    clip, details = decode_video(path)
    assert clip.shape == (3, 16, 96, 96)
    assert clip.dtype == np.uint8
    assert details["decoded_frames"] == 20
    assert details["sampled_frame_indices"][0] == 0
    assert details["sampled_frame_indices"][-1] == 19
    assert clip[0, -1].mean() > 185
    assert clip[2, -1].mean() < 5
    broken = tmp_path / "bad.avi"
    broken.write_bytes(b"invalid video")
    with pytest.raises(ValueError, match="Unreadable"):
        decode_video(broken)


def test_signal_resampling_preserves_channel_order_and_endpoints():
    signal = np.arange(60).reshape(10, 6)
    sampled = resample_signal(signal)
    np.testing.assert_array_equal(sampled[:, 0], signal[0])
    np.testing.assert_array_equal(sampled[:, -1], signal[-1])
    assert sampled.shape == (6, 128)
    with pytest.raises(ValueError, match="finite"):
        resample_signal(np.full((10, 6), np.nan))


def test_cache_pairs_and_integrity(tmp_path):
    path = tmp_path / "clip.avi"
    write_video(path)
    mat = BytesIO()
    savemat(mat, {"d_iner": np.arange(60).reshape(10, 6)})
    inertial, rgb = tmp_path / "imu.zip", tmp_path / "rgb.zip"
    with ZipFile(inertial, "w") as archive:
        archive.writestr("Inertial/a1_s1_t1_inertial.mat", mat.getvalue())
    with ZipFile(rgb, "w") as archive:
        archive.writestr("RGB/a1_s1_t1_color.avi", path.read_bytes())
    cache = tmp_path / "cache"
    prepare_cache(inertial, rgb, cache, require_complete=False)
    trials, video, imu, metadata = load_cache(cache)
    assert trials == [Trial(1, 1, 1)]
    assert metadata["trials"][0]["decoded_frames"] == 20
    assert video.shape[0] == imu.shape[0] == 1
    del video, imu
    metadata["trials"][0]["trial"] = "a2_s1_t1"
    (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="trial IDs"):
        load_cache(cache)
    metadata["trials"][0]["trial"] = "a1_s1_t1"
    (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with (cache / "video.npy").open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xff")
    with pytest.raises(ValueError, match="checksum"):
        load_cache(cache)
    with ZipFile(rgb, "w") as archive:
        archive.writestr("RGB/a1_s2_t1_color.avi", path.read_bytes())
    with pytest.raises(ValueError, match="mismatch"):
        prepare_cache(inertial, rgb, tmp_path / "mismatch", require_complete=False)


def test_normalization_excludes_unselected_people():
    imu = np.arange(4 * 6 * 128, dtype=np.float32).reshape(4, 6, 128)
    expected = fit_normalization(imu, [0, 1])
    imu[2:] = 1e10
    actual = fit_normalization(imu, [0, 1])
    for name in expected:
        np.testing.assert_array_equal(expected[name], actual[name])
    constant = fit_normalization(np.ones((2, 6, 128)), [0, 1])
    assert (constant["std"] > 0).all()


def test_fusion_endpoints_selection_and_invalid_scores():
    video = np.array([[0.9, 0.1], [0.1, 0.9]])
    imu = video[:, ::-1].copy()
    np.testing.assert_array_equal(fuse(video, imu, 0), imu)
    np.testing.assert_array_equal(fuse(video, imu, 1), video)
    weight, candidates = select_fusion(video, imu, np.array([0, 1]))
    assert weight == 0.75
    assert len(candidates) == 5
    assert select_fusion(video, video, np.array([0, 1]))[0] == 0.5
    with pytest.raises(ValueError):
        fuse(video, imu, 2)
    with pytest.raises(ValueError, match="probabilities"):
        fuse(video * 2, imu, 0.5)


def test_encoder_learns_and_backpropagates():
    seed_training(42)
    values = np.stack([np.full((6, 128), (-1) ** i, np.float32) for i in range(16)])
    labels = np.arange(16) % 2
    normalization = fit_normalization(values, np.arange(16))
    config = TrainConfig(epochs=8, batch_size=8)
    model, report = train_encoder(values, labels, np.arange(16), [], "imu", normalization, 2, config, "cpu")
    assert report["history"][-1]["train_loss"] < report["history"][0]["train_loss"] * 0.6
    video_model = make_model("video", 2)
    original = video_model.head.weight.detach().clone()
    optimizer = torch.optim.SGD(video_model.parameters(), lr=0.1)
    loss = torch.nn.functional.cross_entropy(video_model(torch.rand(2, 3, 4, 32, 32)), torch.tensor([0, 1]))
    loss.backward()
    optimizer.step()
    assert not torch.equal(original, video_model.head.weight)


@pytest.fixture(scope="module")
def trained_pair(tmp_path_factory):
    directory = tmp_path_factory.mktemp("deep-run")
    generator = np.random.default_rng(42)
    trials = [Trial(action, subject, 1) for action in (1, 2) for subject in range(1, 9)]
    video = generator.integers(0, 256, size=(16, 3, 16, 96, 96), dtype=np.uint8)
    imu = generator.normal(size=(16, 6, 128)).astype(np.float32)
    metadata = {"archive_sha256": {}, "array_sha256": {}, "preprocess_source_sha256": "fixture", "opencv": cv2.__version__}
    config = TrainConfig(epochs=1, batch_size=4)
    output = directory / "original"
    report = run_experiment(trials, video, imu, metadata, output, config)
    return directory, trials, video, imu, metadata, config, report


def test_saved_inference_matches_evaluation(trained_pair):
    directory, trials, video, imu, _, _, report = trained_pair
    i = trials.index(Trial(1, 2, 1))
    prediction = predict_pair(directory / "original/model.pt", video[i], imu[i])
    row = next(row for row in report["predictions"] if row["trial"] == trials[i].key)
    for name in ("video", "imu", "fusion"):
        np.testing.assert_allclose(prediction[name]["probabilities"], row[name]["probabilities"], atol=1e-6)
    assert prediction["scores_are_calibrated"] is False
    assert report["model_sha256"] == file_hash(directory / "original/model.pt")


def test_test_inputs_cannot_change_selection_or_fitted_model(trained_pair):
    directory, trials, video, imu, metadata, config, report = trained_pair
    _, _, test = subject_split(trials)
    video, imu = video.copy(), imu.copy()
    video[test], imu[test] = 0, 100000
    changed = run_experiment(trials, video, imu, metadata, directory / "changed-test", config)
    original_selection, changed_selection = report["selection"].copy(), changed["selection"].copy()
    original_selection.pop("locked_at_utc")
    changed_selection.pop("locked_at_utc")
    assert original_selection == changed_selection
    first = torch.load(directory / "original/model.pt", weights_only=True)
    second = torch.load(directory / "changed-test/model.pt", weights_only=True)
    for modality in ("video", "imu"):
        for name in first["models"][modality]:
            assert torch.equal(first["models"][modality][name], second["models"][modality][name])
    for name in ("mean", "std"):
        assert torch.equal(first["normalization"][name], second["normalization"][name])


def test_prediction_cli(trained_pair):
    directory, _, _, imu, _, _, _ = trained_pair
    video_path, imu_path = directory / "clip.avi", directory / "signal.mat"
    write_video(video_path)
    savemat(imu_path, {"d_iner": imu[0].T})
    result = subprocess.run([sys.executable, "src/multimodal.py", "predict", "--model",
                             str(directory / "original/model.pt"), "--video", str(video_path),
                             "--inertial", str(imu_path)], capture_output=True, text=True, check=True)
    prediction = json.loads(result.stdout)
    clip, _ = decode_video(video_path)
    direct = predict_pair(directory / "original/model.pt", clip, resample_signal(imu[0].T))
    np.testing.assert_allclose(prediction["fusion"]["probabilities"], direct["fusion"]["probabilities"], atol=1e-6)


def test_reject_incompatible_model_and_bad_training_config(trained_pair):
    directory = trained_pair[0]
    checkpoint = torch.load(directory / "original/model.pt", weights_only=True)
    checkpoint["normalization"]["std"][0] = 0
    torch.save(checkpoint, directory / "invalid.pt")
    with pytest.raises(ValueError, match="normalization"):
        load_model(directory / "invalid.pt")
    for config in (TrainConfig(epochs=0), TrainConfig(batch_size=0), TrainConfig(learning_rate=float("nan"))):
        with pytest.raises(ValueError):
            config.validate()


def test_inference_verifier_checks_identity_and_matches_saved_results(trained_pair, monkeypatch):
    import verify_inference
    directory, trials, video, imu, metadata, _, _ = trained_pair
    monkeypatch.setattr(verify_inference, "load_cache", lambda path: (trials, video, imu, metadata))
    result = verify_inference.verify("fixture", directory / "original", "cpu")
    for check in result["checks"].values():
        assert check["matching_actions"] == check["trials"] == 8
        assert check["max_probability_difference"] == 0
    changed_metadata = dict(metadata, archive_sha256={"inertial": "different"})
    monkeypatch.setattr(verify_inference, "load_cache", lambda path: (trials, video, imu, changed_metadata))
    with pytest.raises(ValueError, match="Cache differs"):
        verify_inference.verify("fixture", directory / "original", "cpu")
