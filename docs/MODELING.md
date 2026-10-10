# Video and sensor comparison

Run commands from the repository root.

The video encoder uses three 3D convolutions (16, 32, 64 channels), batch
normalization, ReLU, global mean pooling, and a 27-class head. Strided
convolutions reduce space and time without nondeterministic CUDA pooling.
The IMU encoder uses three 1D convolutions (32, 64, 128 channels), batch
normalization, ReLU, global mean pooling, and a 27-class head.

Video decoding preserves aspect ratio, resizes the shorter edge to 96 pixels,
and takes a central 96×96 crop. Sixteen uniform frame indices span the decoded
clip. RGB values are divided by 255. IMU channels are linearly interpolated to
128 relative positions and standardized using training samples only. There
are no flips, pretrained weights, or claims of exact cross-modal alignment.

## Selection and evaluation

- Train on subjects 1, 3, and 5; validate on subject 7.
- Use AdamW, learning rate 0.001, weight decay 0.0001, batch size 16, seed 42,
  and at most 30 epochs. Select each encoder's best validation macro-F1;
  ties use the earlier epoch.
- Select video weight from 0, 0.25, 0.5, 0.75, and 1 using saved validation
  probabilities. Fusion is a weighted mean of class probabilities. Ties prefer
  the weight closest to 0.5, then the smaller video weight.
- Save selection before evaluating test data. Reset both encoders to the same
  seed, refit on subjects 1, 3, 5, and 7 for their selected epoch counts, and
  refit IMU normalization using only those subjects.
- Evaluate once on subjects 2, 4, 6, and 8. Report video, IMU, and fusion
  accuracy, macro-F1, per-class counts, per-subject scores, and predictions.

The selected fusion weight comes from models trained on three people; refitting
on four can change their relative usefulness. This is a limitation of the fixed
validation protocol. A single seed and four test participants do not establish
statistical significance. No hyperparameters are changed using test scores.

## Reproduce

Use the pinned dependencies and README commands with the default configuration.
Cache files are checksum-checked on loading. Run folders preserve selection,
model weights, metrics, and an HTML viewer; existing folders are not overwritten.
The published result files exclude raw recordings and model binaries. Training
recreates the model locally.

Deterministic PyTorch operations and fixed seeds are enabled. Results may still
differ across PyTorch builds, devices, and operating systems. The report records
the actual device, runtime, peak allocated GPU memory, and source hashes.

The recorded GPU run used PyTorch's default TF32 setting for cuDNN convolutions.
CPU reloading reproduced all 430 sensor and fusion decisions; video agreed on
429 of 430. Maximum probability differences were 0.000145 for IMU, 0.001344
for video, and 0.000336 for fusion. Saved GPU inference matched exactly. Check
your local model without retraining:

```sh
python src/verify_inference.py --cache data/paired-cache --run runs/multimodal --device cpu --output runs/cpu-check.json
```

The verifier checks model and input hashes, then reports differences and scores.
It does not adjust the model or select parameters using those results.

Tests exercise real video decoding, paired-cache integrity, gradients, learning
on a small fixture, saved-model prediction, and CLI inference. A leakage test
changes every test input and verifies that validation selection, fitted
normalization, and final model parameters remain identical. CI runs on generated
fixtures; fixture results are not reported as benchmark performance.
