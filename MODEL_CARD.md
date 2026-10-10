# Results and limitations

This is an experimental activity-recognition application for paired UTD-MHAD
recordings. Its purpose is reproducible comparison and inspection, not safety
monitoring, diagnosis, identity recognition, or deployment on arbitrary videos.

## What was learned

The 27-class studies compare an engineered IMU baseline, independent neural
video/IMU encoders, late fusion, cross-modal attention, and contrastive
pretraining. Three-seed macro-F1 was 0.8013 for concatenation, 0.7890 for attention,
and 0.7640 for contrastive attention. Added complexity did not improve this
benchmark. See [controlled comparisons](ATTENTION.md) and their raw results.

A separate 21-known/6-held-out-action protocol tests partial observations,
missing sensors, normalized IMU noise, and confidence calibration. These
results are not directly comparable with the 27-class studies.

| Known-action accuracy, mean of three seeds | Augmented attention | Frozen video transfer |
| --- | ---: | ---: |
| Full paired recording |70.44%|52.78%|
| Quarter paired recording |46.33%|35.81%|
| Full video only |4.76%|30.46%|

The frozen video representation helps the visual stream but lowers paired
accuracy. Both model families remain available. The demo serves the first
declared seed 42, not a model selected by its test score. All seeds and negative
results are published.

## Failure analysis

[Recomputed error counts](results/release/failure-analysis.json) pool three
seeds over the same 336 known and 94 held-out trials. These counts are repeated
predictions, not 1,290 independent recordings.

For full paired inputs, attention accepted 298 incorrect known predictions
out of 1,008 and accepted all 282 unknown predictions. The transfer model with
conditional calibration accepted 443 incorrect known predictions and 280 of 282
unknown predictions. Confidence thresholds are **not reliable unknown-action
detectors**. The UI shows accepted errors and abstentions explicitly.

Attention's lowest full-paired recall is for catching an object (27.08%). Its
largest confusion is tennis forehand mistaken for bowling (19 predictions).
Transfer misses every throw example across these seeds; its largest confusion
is boxing mistaken for a baseball swing (18 predictions). These observations
describe errors after evaluation and have not been used to retune a model.

## Evaluation boundaries

- People, not individual clips, define splits. Open-set selection uses people 1/3
  for training, 5 for model selection, 1/3/5 for refitting, 7 for calibration, and
  even-numbered people for testing. Held-out actions never enter downstream fitting.
- The benchmark has only eight participants. Seed variation does not measure
  uncertainty across populations. Earlier experiments already exposed test
  results; this is not a new blind external validation.
- Actions 1–21 use a wrist sensor; 22–27 use a thigh sensor. Unknown-action results
  confound activity and sensor placement. Frozen Kinetics pretraining may
  include related activities; downstream-held-out does not mean globally unseen.
- Calibration uses 84 unique known trials from one person, reused across
  conditions. A 95% calibration acceptance target is not a test-time guarantee.
- Early predictions use fractions of a known segmented recording. No shared
  video/sample timestamps support exact synchronization or online onset detection.

## Execution and access

ONNX preserves 25,800 tested calibrated decisions. Representative C++ classifier
and video-encoder outputs match Python ONNX exactly. Native graph execution
does not universally improve latency: the frozen video encoder takes roughly
0.7 seconds on the measured single-thread CPU, before decoding/network overhead.
See [serving measurements](SERVING.md); no real-time guarantee is made.

Code is MIT licensed. Dataset access and research-use statements are documented
in [DATA.md](DATA.md). Data and pretrained weights have separate terms; neither
is included in the code license. Local replay derivatives are not published.
Pin and verify all downloaded archives/weights. Load only trusted models.

Further generalization claims require independent people, settings, activities,
and sensor placements. Reliable unknown detection would require a separately
declared validation protocol and stronger evidence; this release does not
claim that capability simply because it exposes a rejection threshold.
