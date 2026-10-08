# Video + IMU Activity Recognition

Recognize actions from paired video and wearable motion recordings. The current
release compares a sensor SVM, a 3D video encoder, a 1D sensor encoder, and late
fusion. It includes subject-based evaluation, saved-model prediction, and local
results viewers.

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
