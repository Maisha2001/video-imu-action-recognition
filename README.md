# Video + IMU Activity Recognition

Recognize actions from paired video and wearable motion recordings. The current
release compares a sensor SVM, a 3D video encoder, a 1D sensor encoder, and late
fusion. A controlled three-seed study adds cross-modal attention and paired
contrastive pretraining. Both paths include subject-based evaluation, saved-model
prediction, and local results viewers. A separate open-set study adds early
prefixes, missing-sensor handling, and confidence calibration. A frozen pretrained
video backbone provides a stronger visual comparison with conditional calibration.

## Run

Tested with Python 3.11. Create a virtual environment, then activate it using
`.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS.

```sh
python -m venv .venv
```

After activation:

```sh
python -m pip install -r requirements.txt
python fetch_data.py --video
python activity.py baseline --inertial data/Inertial.zip --rgb data/RGB.zip
```

Open `runs/baseline/report.html` to inspect results and errors. The run also saves
metrics, a trial manifest, and a model. Choose a new `--output` path for each run.
The sensor baseline can run without `--video` and `--rgb`.

```sh
python activity.py predict --model runs/baseline/model.joblib --input trial_inertial.mat
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Load only models you created or trust. See [DATA.md](DATA.md) for data terms,
channel details, and the evaluation split. Code is MIT licensed; data is separate.

## Video and sensor models

Install PyTorch for your machine, then the pinned dependencies. For the recorded
NVIDIA run (CUDA 12.8):

```sh
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-deep.txt
python multimodal.py prepare --inertial data/Inertial.zip --rgb data/RGB.zip
python multimodal.py train --device cuda
```

For CPU use, install PyTorch from `https://download.pytorch.org/whl/cpu` and pass
`--device cpu`. See the [official installation options](https://pytorch.org/get-started/previous-versions/).
The decoded cache uses about 384 MB beyond the downloaded archives. Each run
needs a new output folder; `--output runs/my-run` sets it.

Open `runs/multimodal/report.html` for the comparison. `metrics.json` records
predictions, validation choices, learning curves, input hashes, and runtime.
`model.pt` contains both encoders and the fitted normalization.

```sh
python multimodal.py predict --model runs/multimodal/model.pt --video trial_color.avi --inertial trial_inertial.mat
python -m pytest -q
```

Both inputs must describe the same complete trial. Predictions include each
modality and fusion; their softmax scores are not calibrated confidence.
Models are trained from scratch, so no external model weights are required.

## Evaluation

An RBF SVM uses 180 motion features. Subjects 1, 3, and 5 train the model;
subject 7 selects regularization. The chosen model is refit on odd-numbered
subjects and tested on subjects 2, 4, 6, and 8. Scaling is fitted on training
data only. A majority-class model provides a simple comparison.

On 430 held-out trials, the sensor model achieved **78.14% accuracy** and
**0.775 macro-F1**. The majority baseline reached **3.72% accuracy**.
See [recorded results](results/metrics.json) for predictions, environment,
checksums, and per-subject scores. Training uses 431 trials; video is not an input.

This small, segmented benchmark does not establish performance on live video, new environments, or
unfamiliar actions. Sensor placement depends on the action group.

The neural comparison uses 16 uniformly sampled RGB frames at 96×96 and 128
interpolated IMU samples. Each encoder trains for up to 30 epochs; validation
macro-F1 selects its epoch count and the fusion weight. Both encoders are then
reset and refit on odd subjects before the even subjects are evaluated. See
[the modeling protocol](MODELING.md) for architecture and reproducibility details.

On the same 430 held-out trials:

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Sensor SVM | 78.14% | 0.775 |
| Sensor network | 80.47% | 0.799 |
| Video network | 9.30% | 0.038 |
| Late fusion | 79.30% | 0.782 |

Validation selected 13 video epochs, 12 sensor epochs, and 25% video weight.
**Fusion did not improve over the sensor network.** The video encoder learned
weak features on this small dataset. These results do not establish useful
video modeling or a reliable multimodal improvement.

The neural run took 24.16 seconds on an RTX 5070, excluding download and video
decoding, with 248.4 MiB peak allocated GPU memory. See
[neural results](results/multimodal/metrics.json) and
[saved-model checks](results/multimodal/inference-cpu.json).
Saved GPU inference exactly reproduced all recorded probabilities. CPU inference
preserved every sensor and fusion action; one near-tied video action changed.

## Attention and contrastive learning

With the same environment and paired cache:

```sh
python attention.py train --device cuda
python attention_report.py --run runs/attention --output runs/attention-report
```

The study compares concatenation, bidirectional attention, and contrastive
pretraining followed by attention. It reports every seed and saves recoverable
epoch checkpoints. Use `--resume` after an interruption. See
[the protocol and inference commands](ATTENTION.md).

Each variant was evaluated on the same 430 held-out trials with seeds 42, 43,
and 44. The table reports means and sample standard deviations:

| Paired model | Mean accuracy | Macro-F1, mean ± SD |
| --- | ---: | ---: |
| Concatenation | 81.16% | 0.801 ± 0.021 |
| Cross-modal attention | 79.30% | 0.789 ± 0.032 |
| Contrastive pretraining + attention | 76.82% | 0.764 ± 0.055 |

**Attention and contrastive pretraining did not improve mean performance.**
Attention reduced macro-F1 by 0.012 versus concatenation; pretraining reduced
it by a further 0.025 on average, with substantial seed variation. These are
observed differences, not significance claims. The new encoders also differ
from the earlier baselines, so this does not establish a benefit from video.

See [all nine runs and protocol](results/attention/summary.json),
[learning curves](results/attention/training.json), and the
[local results viewer](results/attention/report.html). The completed comparison
took 31.48 minutes on an RTX 5070 with 242.2 MiB peak allocated GPU memory,
excluding data preparation and verification. All model choices used training
and validation subjects; the previously observed test benchmark was not used
to select settings.

[Saved GPU inference](results/attention/inference-cuda.json) reproduced every
probability exactly. [CPU inference](results/attention/inference-cpu.json)
preserved all 3,870 actions across the nine models; the largest probability
difference was 0.00000191. The [paired-file CLI check](results/attention/paired-file-check.json)
also reproduced its recorded prediction and valid attention matrices.

## Early and missing-sensor recognition

```sh
python robustness.py prepare --inertial data/Inertial.zip --rgb data/RGB.zip
python robustness.py train --device cuda
python robust_report.py --run runs/robustness --output runs/robustness-report
python robustness.py predict --model runs/robustness/augmented-42/model.pt --video trial_color.avi --inertial trial_inertial.mat --fraction 0.25
```

Omit either input for a missing sensor. The model can abstain when confidence
falls below its fitted threshold. This study trains on actions 1–21 and holds
actions 22–27 out of all fitting. A separate person fits calibration; even
subjects remain the test set. Its scores are not directly comparable to the
27-class results above.

Prefixes crop original recordings before resampling, at 25%, 50%, 75%, and
100% of the known trial duration. These are offline segmented-trial results;
they do not establish online detection or exact video/IMU timestamp alignment.
See [the protocol, conditions, and limitations](ROBUSTNESS.md).

Mean known-action accuracy over three seeds and 336 held-out known trials:

| Test input | Full-only training | Prefix/modality augmentation |
| --- | ---: | ---: |
| 25% observed, both sensors | 16.07% | 46.33% |
| 50% observed, both sensors | 56.94% | 71.63% |
| 75% observed, both sensors | 71.92% | 74.01% |
| Complete trial, both sensors | 72.92% | 70.44% |
| Complete trial, IMU only | 57.14% | 67.36% |
| Complete trial, video only | 4.76% | 4.76% |
| Complete trial, both, IMU noise SD 0.3 | 72.02% | 69.44% |

Augmentation helped partial observations and missing video, with a loss on
complete paired trials. **Video-only recognition remains at chance.** For
augmented quarter-trial paired inputs, calibration reduced mean NLL from
2.152 to 1.747 and ECE from 0.261 to 0.114. It did not improve every condition:
full-paired ECE increased from 0.090 to 0.096.

**Unknown-action rejection is ineffective on paired inputs.** On 94 unknown
trials, the fitted threshold rejected none of the complete paired examples
for either training variant. Augmented-model mean unknown AUROC was 0.587.
Pooling calibration across conditions produced a permissive threshold;
confidence is not a reliable safeguard here. The results do not establish
reliable fallback when IMU is missing or reliable unfamiliar-action detection.

The complete six-model study took 6.82 minutes on an RTX 5070, with 242.7 MiB
peak allocated GPU memory, excluding data preparation and verification. See
[every seed and condition](results/robustness/summary.json),
[learning curves](results/robustness/training.json), and the
[local results viewer](results/robustness/report.html).

[Saved GPU inference](results/robustness/inference-cuda.json) reproduced all
51,600 classifications, accept/reject decisions, and confidence scores exactly.
[CPU inference](results/robustness/inference-cpu.json) preserved every decision;
the largest confidence difference was 0.00000110. The
[real-file check](results/robustness/file-inference.json) also reproduced
quarter-trial predictions with both sensors and with either sensor missing.

## Frozen video transfer

The transfer experiment uses pinned Kinetics-400 R3D-18 features and learns a
masked video/IMU classifier. See [setup, protocol, and weight terms](TRANSFER.md).
It retains the 21-known/6-held-out-action split above and all three seeds.

| Known-action accuracy | Earlier augmented model | Frozen video transfer |
| --- | ---: | ---: |
| 25% observed, both sensors | 46.33% | 35.81% |
| Complete trial, both sensors | 70.44% | 52.78% |
| Complete trial, video only | 4.76% | 30.46% |
| Complete trial, IMU only | 67.36% | 63.19% |

Video-only recognition improved, but paired recognition declined. Architecture
and pretraining both changed, so this comparison does not isolate pretraining's
effect. The earlier models and all results remain available.

Separate confidence calibration for each fraction and sensor configuration
rejected only **0.71% of unfamiliar complete paired trials**, while rejecting
4.56% of known trials. Mean unknown-action AUROC was 0.387. These thresholds
are not reliable unfamiliar-action detectors. Held-out actions are excluded
from downstream fitting; related actions may occur in Kinetics pretraining.

Feature extraction took 99.80 seconds and head training/evaluation 22.84 seconds
on an RTX 5070, excluding download and verification. See
[all conditions and seeds](results/transfer/summary.json),
[learning curves](results/transfer/training.json), and the
[local results viewer](results/transfer/report.html). Saved GPU inference
reproduced all 51,600 actions, decisions, and confidence scores exactly; CPU
preserved every action and decision (maximum confidence difference 0.000000761).
The [raw-file check](results/transfer/file-inference.json) covers one trial at
two fractions with each sensor configuration on CPU and GPU.
