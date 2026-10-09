# Frozen video transfer

This experiment tests a stronger visual representation and condition-specific
confidence calibration. It improves video-only recognition but underperforms
the earlier paired model. It does not establish reliable unknown-action rejection.

## Reproduce

Use Python 3.11 in a separate environment. Install matching PyTorch and Torchvision
builds; these commands match the recorded CUDA 12.8 environment. For CPU, replace
`cu128` with `cpu` and use `--device cpu` below.

```sh
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-transfer.txt
python fetch_data.py --video
python robustness.py prepare --inertial data/Inertial.zip --rgb data/RGB.zip
python -c "from pathlib import Path; from urllib.request import urlretrieve; Path('models').mkdir(exist_ok=True); urlretrieve('https://download.pytorch.org/models/r3d_18-b3b3357e.pth', 'models/r3d_18-b3b3357e.pth')"
python video_features.py --inertial data/Inertial.zip --rgb data/RGB.zip --weights models/r3d_18-b3b3357e.pth --device cuda
python transfer.py train --device cuda
python transfer_report.py --run runs/transfer --output runs/transfer-report
```

The 133,546,016-byte weight file must have SHA256
`b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d`.
The loader checks it before use. Feature preparation saves trial checkpoints;
training saves epoch checkpoints. Use `--resume` with unchanged inputs, sources,
versions, and settings after interruption. Choose new output folders for new
experiments. `--stop-at-utc` accepts an ISO timestamp with an offset.

```sh
python transfer.py predict --model runs/transfer/transfer-42/model.pt --weights models/r3d_18-b3b3357e.pth --video trial_color.avi --inertial trial_inertial.mat --fraction 0.25
python verify_transfer.py --features data/video-features --prefix-cache data/prefix-cache --run runs/transfer --device cpu --output runs/transfer/inference-cpu.json
python -m pytest -q
```

Omit either sensor input to simulate its absence. IMU-only inference does not
need the backbone weights. Predictions return a top action, calibrated scores,
and an accept/reject decision; `action` is null when rejected. The default
calibration is `conditional`; `--mode pooled` selects the shared threshold.
Inputs must describe the same segmented trial. Fractions are 0.25, 0.5, 0.75,
or 1.0 of its known length, not online action onset estimates.

## Model and evaluation

The frozen backbone is Torchvision 0.26.0
[R3D_18_Weights.KINETICS400_V1](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.video.r3d_18.html),
with its classifier removed. Each raw prefix supplies 16 uniformly sampled
frames. OpenCV `INTER_LINEAR` resizes to 128×171; the central 112×112 RGB crop
uses upstream channel means and standard deviations. This explicitly uses
OpenCV resizing, not Torchvision's transform implementation. Extraction uses
evaluation mode, float32, and disabled TF32, producing 512 features per prefix.

A 512→128 visual projection and a 6→32→64→128 temporal IMU encoder feed a
masked concatenation classifier. The trainable model has 139,129 parameters.
Both streams use training-fitted normalization. Training samples the four
fractions uniformly, selects both/video/IMU with probabilities 0.5/0.25/0.25,
and adds normalized IMU noise with SD 0.05. Seeds are 42, 43, and 44; AdamW
uses learning rate 0.001, weight decay 0.0001, batch 32, and dropout 0.1.

The [open-set split](ROBUSTNESS.md) is unchanged: known actions 1–21 train on
people 1 and 3, select on 5, refit on 1/3/5, calibrate on 7, and test on even
people. Actions 22–27 are excluded from every downstream fit. Training has
167 trials, validation 84, refit 251, calibration 84, and test 430 (336 known,
94 unfamiliar). Pretraining can include semantically related actions.

Up to 40 epochs are selected by mean validation macro-F1 across full paired,
quarter paired, full video-only, and full IMU-only conditions; earliest ties
win. Selected epochs are 31, 22, and 30. All selections lock before refitting;
all refits and calibrations finish before testing. This benchmark was already
observed in previous experiments and is not a new blind test.

Pooled calibration fits one temperature and fifth-percentile confidence
threshold to twelve clean fraction/sensor conditions. Conditional calibration
fits them separately for each condition, using the same 84 known calibration
trials. No unknown or test examples choose thresholds. Noisy test conditions
reuse their clean condition's calibration. Each seed reports 20 conditions
under both schemes, with per-person results, all predictions, and seed variation.

Conditional full-paired known acceptance is 95.44%, unknown rejection 0.71%,
and unknown AUROC 0.387; the pooled threshold rejects no unknown paired trials.
Calibration is limited by one participant and does not certify unfamiliar-action
detection. The architectures differ from earlier runs; score changes are
descriptive, not causal evidence for pretraining. No test-based retuning was done.

## Evidence and access

On an RTX 5070, feature extraction took 99.80 seconds with 389.3 MiB peak
allocated GPU memory. Training, selection, refitting, calibration, and evaluation
of the cached-feature models took 22.84 seconds with 69.5 MiB peak allocation.
These exclude downloads, cache validation, and verification, and are not serving
latency measurements. Raw-file inference loads a backbone as well as the head.

[GPU](results/transfer/inference-cuda.json) and [CPU](results/transfer/inference-cpu.json)
checks cover 51,600 cached-feature predictions. GPU confidences match exactly;
CPU changes at most 0.000000761, preserving every action and rejection decision.
The [raw-file evidence](results/transfer/file-inference.json) checks a single
trial at 25% and 100%, each sensor configuration and both calibration schemes,
including three CLI calls. It also re-extracts all four video prefixes on both
devices. This is a smoke test, not an exhaustive raw-video reproduction claim.

Model files and datasets are downloaded or trained locally and are not committed.
UTD-MHAD terms are in [DATA.md](DATA.md). Torchvision source is
[BSD-3-Clause](https://github.com/pytorch/vision/blob/v0.26.0/LICENSE);
[upstream pretrained-weight terms](https://docs.pytorch.org/vision/0.26/models.html)
and the rights of the original training videos remain separate. This repository's
MIT license does not relicense weights or videos. No Kinetics videos are downloaded
or redistributed. Use only model files you created or trust.
