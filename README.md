# Video + IMU Activity Recognition

Recognize actions from paired video and wearable motion recordings. The current
release provides a sensor baseline, subject-based evaluation, video/IMU pairing
checks, and a local results viewer. Video fusion is not implemented yet.

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
