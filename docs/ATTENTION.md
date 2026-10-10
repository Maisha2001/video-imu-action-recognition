# Paired representation learning

Run commands from the repository root.

Three controlled variants share compact video/IMU token encoders and the same
training protocol:

1. **Concatenation:** pool each encoder's tokens and classify their concatenation.
2. **Attention:** add bidirectional cross-modal attention before pooling.
3. **Contrastive attention:** pretrain paired representations without action
   labels, then adapt the attention model with supervised training.

These are new encoders, so comparison with the earlier convolutional baselines
does not isolate the effect of attention. The concatenation control does.
Contrasting the two attention variants measures pretraining plus its additional
compute; their total training budgets are not equal.

## Model and objective

Video convolutions produce four temporal positions with a 2×2 spatial grid,
giving 16 tokens. IMU convolutions produce 16 temporal tokens. Each token has
64 features. Group normalization avoids batch running statistics; learned
positions retain token identity. Four-head attention reads the opposite
modality through query/key/value projections, residual connections, layer
normalization, and a feedforward layer. The classifier pools both streams.

Pretraining averages each encoder's tokens, projects them to 32 dimensions,
normalizes the embeddings, and applies symmetric cross-entropy over pairwise
similarities with temperature 0.1. Matching recordings are positives; other
batch members are negatives. Action labels are unused. Different recordings of
the same action can become false negatives. There are no external pretrained
weights or extra recordings.

The attention operation follows [scaled dot-product attention](https://arxiv.org/abs/1706.03762).
The paired objective uses the [symmetric contrastive form](https://arxiv.org/abs/2103.00020)
with video and inertial recordings in place of image/text pairs. This project
does not use CLIP weights or reproduce its experiment.

## Evaluation

Seeds 42, 43, and 44 run every variant. AdamW uses learning rate 0.001, weight
decay 0.0001, batch size 16, and dropout 0.1. Supervised training runs for up to
40 epochs. Contrastive pretraining uses a fixed 20 epochs. All settings were
declared before this comparison's test evaluation.

Subjects 1, 3, and 5 train both pretraining and supervised selection. Subject 7
selects the supervised epoch count by macro-F1, with earlier epochs winning
ties. All nine choices are saved before refitting. Each model is reset and
refit on odd subjects, including pretraining where applicable. All refits
finish before the even-subject test is evaluated. No seed is selected or hidden.

Reports include every seed, mean and sample standard deviation, matched-seed
differences, learning curves, per-class/per-subject scores, and predictions.
Three seeds quantify initialization variability, not participant-level confidence
or statistical significance. Earlier results on this benchmark have already
been observed; this is repeated evaluation, not a new blind test.

## Run and resume

Use the environment and paired cache described in the README:

```sh
python src/attention.py train --device cuda
python src/attention_report.py --run runs/attention --output runs/attention-report
python src/attention.py predict --model runs/attention/attention-42/model.pt --video trial_color.avi --inertial trial_inertial.mat
```

Open `runs/attention-report/report.html`. Prediction also returns average
attention weights; these are internal associations, not causal explanations or
proof of exact temporal alignment. Complete paired trials are required, and
softmax scores are not calibrated confidence.

Attention matrices average the four heads. `video_to_imu` rows are video tokens
ordered by time, spatial row, then spatial column; columns are the 16 IMU
positions. `imu_to_video` reverses those roles. Token positions describe the
resampled trial, not verified shared timestamps.

Every epoch saves model, optimizer, random state, best validation result, and
history through an atomic file replacement. Protocol, selection, and result
records are also replaced atomically. To resume after interruption:

```sh
python src/attention.py train --device cuda --resume
```

Use the same run folder, source files, inputs, configuration, device, and
PyTorch build. Mismatches and changed completed-model checksums are rejected.
An interrupted epoch restarts from the
previous completed epoch. `--stop-at-utc` accepts an ISO timestamp with an offset
and pauses before another epoch; a pause exits with code 75. Normal completion
exits with code 0. Keep the whole run directory to preserve recovery state.

Deterministic operations are enabled and CUDA TF32 is disabled for this
comparison. CPU and GPU resumed training were checked against uninterrupted
training for both objectives. The previous baseline's precision setting is
preserved in its original results.

To check saved-model predictions without retraining:

```sh
python src/verify_attention.py --cache data/paired-cache --run runs/attention --device cpu --output runs/attention-cpu-check.json
```
