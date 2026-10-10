# Early recognition and uncertainty

Run commands from the repository root.

This experiment compares full paired training with prefix and modality
augmentation. Both use the same compact attention model and seeds 42, 43, and
44. Models train from scratch on **actions 1–21**. Actions 22–27 are unfamiliar
to every fitted model, including its normalization and calibration.

This is a separate 21-class/open-set task. Its accuracy is not directly
comparable to the earlier 27-class studies.

## Data and model

Four prefixes retain 25%, 50%, 75%, or 100% of each original recording.
Original frames and IMU rows are cropped **before** uniform frame sampling and
IMU interpolation. Changing later values cannot change an earlier prefix.
The video path uses 16 RGB frames at 96×96; the sensor path uses 128 samples.
The full prefix exactly matches the earlier paired cache.

Fractions use the known duration of a segmented trial. This is an offline
early-classification benchmark, not online onset detection. The recordings
are paired, but shared frame/sample timestamps are unavailable; equal fractions
do not prove exact temporal synchronization.

An explicit binary mask identifies available modalities. Missing values are
removed before encoding, absent tokens are gated out, and cross-attention falls
back to the available stream. The classifier also receives the mask. Both
modalities absent is an error, not a guess.

The augmented variant samples the four fractions uniformly. It uses both
modalities half the time, video alone a quarter, and IMU alone a quarter.
Available IMU values receive Gaussian noise with standard deviation 0.05 in
training-normalized units. The control sees full, clean, paired recordings.
The architecture is identical; only training exposure changes.

## Selection, calibration, and test

1. Train on people 1 and 3; select epochs on person 5. Selection maximizes
   mean macro-F1 across full paired, quarter paired, full video-only, and full
   IMU-only inputs. Ties use the earlier epoch.
2. Lock all six choices, then reset and refit on people 1, 3, and 5.
3. Fit calibration using only known actions from person 7, across the twelve
   clean fraction/availability conditions. Person 7 never enters model fitting.
4. After every refit and calibration is complete, evaluate people 2, 4, 6, and
   8, including all six unknown actions.

Each selection run allows 40 epochs. AdamW uses learning rate 0.001, weight
decay 0.0001, batch size 16, and dropout 0.1. Deterministic float32 operations
are used with CUDA TF32 disabled. All variants and seeds are reported.

[Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html) fits one
positive scalar per model by known-action negative log likelihood, bounded
between 0.05 and 20. It preserves the top class. The lower fifth percentile
of calibrated maximum probabilities sets a global acceptance threshold.
This targets 95% acceptance on the calibration mixture; it guarantees neither
test coverage nor unknown rejection.

[Maximum softmax probability](https://arxiv.org/abs/1610.02136) is the unknown
detection baseline. Low confidence triggers abstention. Rejection is not proof
of an unfamiliar action, and accepted predictions can still be wrong.

Twenty test conditions cover the twelve clean fraction/mask combinations and
paired inputs with normalized IMU noise of 0.1 or 0.3 at every fraction.
Noise is deterministic by trial and condition. Reports include known accuracy,
macro-F1, NLL, multiclass Brier score, 15-bin ECE, acceptance, accepted-known
accuracy, unknown AUROC/average precision, and unknown rejection recall.
Unknowns are the positive class for detection metrics.

Only 84 unique trials from one person fit calibration; the twelve conditions
reuse those trials. Seeds measure initialization variation, not confidence
across people. This small, previously observed benchmark cannot establish
deployment reliability or performance on arbitrary unknown activities.

## Run

Use the environment and verified archives in the README. The prefix cache
needs about 1.5 GB beyond the archives and existing paired cache.

```sh
python src/robustness.py prepare --inertial data/Inertial.zip --rgb data/RGB.zip
python src/robustness.py train --device cuda
python src/robust_report.py --run runs/robustness --output runs/robustness-report
python src/robustness.py predict --model runs/robustness/augmented-42/model.pt --video trial_color.avi --inertial trial_inertial.mat --fraction 0.25
```

Omit either input for one-sensor prediction. `action` is null when the model
abstains; `top_action` still shows its highest-scoring known class. Open
`runs/robustness-report/report.html` for all conditions and downloadable
predictions. Every saved model includes its fitted normalization, temperature,
threshold, and known-action vocabulary.

Use `--resume` with the same run folder, data, source files, device, and settings
after an interruption. Epoch checkpoints preserve optimizer and random state;
an unfinished epoch restarts. `--stop-at-utc` pauses before further batches or
stages with exit code 75. Completed evaluations may be recomputed on resume
under the unchanged protocol. No model or threshold is selected using test data.

```sh
python src/verify_robustness.py --cache data/prefix-cache --run runs/robustness --device cpu --output runs/robustness-cpu-check.json
```
