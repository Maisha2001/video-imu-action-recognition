# Inference service and native runtime

Two explicit model packages are supported: the prefix-augmented attention model
and the frozen-video transfer model. Both use seed 42, the first declared seed,
with their original fitted weights and calibration. They are experimental
comparisons; neither is a reliable unfamiliar-action detector. This work changes
execution, not training or model selection.

## Export and serve

Use the isolated Python 3.11 environment and matching PyTorch/Torchvision builds
from [TRANSFER.md](TRANSFER.md). First reproduce the desired models using
[ROBUSTNESS.md](ROBUSTNESS.md) and [TRANSFER.md](TRANSFER.md). Export to new folders:

```sh
python -m pip install -r requirements-serving.txt
python export_serving.py --family robust --model runs/robustness/augmented-42/model.pt --output runs/serving/robust
python export_serving.py --family transfer --model runs/transfer/transfer-42/model.pt --weights models/r3d_18-b3b3357e.pth --output runs/serving/transfer
python inference_api.py --package robust=runs/serving/robust --package transfer=runs/serving/transfer
```

Only include packages you want to serve. The process binds to `127.0.0.1:8000`;
`--port` changes the port. `--backend torch` uses the original classifier, while
the default uses ONNX Runtime's CPU provider. Transfer video preprocessing uses
the exported ONNX backbone in either mode. No GPU-serving claim is made.

```sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/models
curl -X POST "http://127.0.0.1:8000/predict/robust?fraction=0.25" -F "video=@trial_color.avi" -F "inertial=@trial_inertial.mat"
```

Use `curl.exe` in PowerShell. Omit either sensor file for a missing modality.
Supported fractions are 0.25, 0.5, 0.75, and 1.0 of a known segmented recording.
Transfer accepts `mode=pooled` or `mode=conditional` (default). Robust always
reports its fitted pooled calibration. The response includes top action,
probabilities, confidence threshold, accept/reject decision, source model SHA,
sensor availability, preprocessing/classifier times, and limitations. Rejection
sets `action` to null; it is not proof of an unfamiliar activity.

The server handles one upload/inference request at a time and returns 503 when
busy. It bounds the whole request at 75 MiB, video at 64 MiB, and IMU at 10 MiB;
each upload read has a 30-second idle timeout. Missing, duplicate, malformed,
or unsupported inputs fail explicitly. Filenames do not choose server paths;
temporary uploads are removed after inference. Health and model metadata stay
available during inference. FastAPI telemetry and access logs are disabled.

This is a local service for trusted recordings, without authentication or a
hardened decoder sandbox. Do not expose it publicly. File-size limits do not
guarantee bounded decompression memory for arbitrary malicious media/MAT files.

## Package contract

Export writes `model.onnx`, `source.pt`, and a checksum manifest. Transfer also
includes `backbone.onnx`; it does not require PyTorch weights for video inference.
Normalization belongs to the classifier graph; confidence calibration remains
in the checked manifest. Startup validates every file, shape contract, fitted
label/calibration metadata, and source model identity. Use trusted packages;
checksums detect changes but are not signatures of authenticity.

Graphs use ONNX opset 18, float32, fixed batch size 1, and one CPU thread.
The tested TorchScript exporter emits tracing warnings because shapes and Python
validation branches are fixed. Python and C++ entry points validate input shapes,
finite values, and binary nonempty masks outside the graph. Other batch sizes
are unsupported. Original training sources are unchanged.

## Build the C++ runner

Use the official [ONNX Runtime 1.31.0 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0).
The runtime is MIT licensed; retain its bundled notices. No SDK binaries or
model weights are committed. Verify downloaded archives before extraction:

| Archive | SHA256 |
| --- | --- |
| `onnxruntime-linux-x64-1.31.0.tgz` | `cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57` |
| `onnxruntime-win-x64-1.31.0.zip` | `3958d8a44984160692c1b4ffd0de3cb48f8c39f9d34bb31e433db06e30b82195` |

Linux CI builds with CMake and the runner's C++ compiler:

```sh
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/absolute/path/onnxruntime-linux-x64-1.31.0
cmake --build build --parallel 2
```

The recorded Windows build used portable
[Zig 0.15.2](https://ziglang.org/download/0.15.2/zig-x86_64-windows-0.15.2.zip)
(`3a0ed1e8799a2f8ce2a6e6290a9ff22e6906f8227865911fb7ddedc3cc14cb0c`):

```powershell
zig c++ -std=c++17 -O2 -I C:/tools/onnxruntime-win-x64-1.31.0/include native/infer.cpp C:/tools/onnxruntime-win-x64-1.31.0/lib/onnxruntime.lib -o build/activity_infer.exe
Copy-Item C:/tools/onnxruntime-win-x64-1.31.0/lib/onnxruntime.dll build/
```

Create `build` first and substitute your extracted SDK paths. CMake also accepts
a configured Windows C++ toolchain; the measured Windows run used the command
above. Use `ACTIVITY_NATIVE` to enable native integration tests:

```powershell
$env:ACTIVITY_NATIVE = (Resolve-Path build/activity_infer.exe).Path
python -m pytest -q
```

The native runner executes the same classifier or video-backbone graph:

```sh
build/activity_infer model.onnx inputs.bin logits.bin 10
```

Input is little-endian `VIM1`, a uint32 record count, then each record's float32
tensors in graph input order. Classifier inputs are video, IMU, and availability;
backbone input is video only. Shapes are read from the fixed graph. The file is
limited to 512 MiB and 4,096 records. `verify_serving.write_native_inputs` writes
this format. Output is row-major float32 logits/features, with shape and timing
JSON on stdout. The optional repetition count benchmarks resident inference;
only the first pass is written. Python handles decoding, preprocessing, and
calibration; C++ handles the exported graphs, not raw AVI/MAT parsing.

## Verify and measure

```sh
python verify_serving.py --prefix-cache data/prefix-cache --features data/video-features --packages runs/serving --native build/activity_infer --work runs/native-check --output runs/native-check/verification.json
python verify_api.py --packages runs/serving --rgb data/RGB.zip --inertial data/Inertial.zip --weights models/r3d_18-b3b3357e.pth --native build/activity_infer --work runs/api-check --output runs/api-check/verification.json
python benchmark_serving.py --prefix-cache data/prefix-cache --features data/video-features --packages runs/serving --native build/activity_infer --rgb data/RGB.zip --weights models/r3d_18-b3b3357e.pth --work runs/latency --output runs/latency/latency.json
```

Append `.exe` to the native path on Windows. The HTTP verifier starts a temporary
loopback server and stops it afterward. Run the separate benchmark after other
tests finish. Models and inputs are loaded before timing; decoding, validation,
network transfer, and model loading are excluded from graph benchmarks. The
video backbone is measured separately from the transfer classifier.

[Classifier parity](results/serving/parity.json) covers all 430 held-out trials
and 20 conditions: 8,600 robustness decisions and 17,200 transfer decisions
across both calibration modes. Every action and accept/reject decision matched
PyTorch. Maximum logit differences were 0.00000763 and 0.00000859 respectively.
The C++ runner matched Python ONNX logits exactly on 160 inputs per family,
covering both known/held-out actions, all test people and every condition.

[Real HTTP checks](results/serving/http.json) cover two fixed trials, all four
fractions, three sensor configurations, and both families: 48 requests matched
original file inference. Eight native video-backbone outputs also matched
Python ONNX features exactly. These subsets do not establish exhaustive
raw-video reproduction or production reliability. See the separate
[latency measurements](results/serving/latency.json); small-sample timings are
hardware-specific and do not guarantee real-time operation or a speedup.

Recorded Windows CPU graph latency, median / p95 in milliseconds:

| Graph | PyTorch | Python ONNX | C++ ONNX |
| --- | ---: | ---: | ---: |
| Robustness classifier | 5.058 / 6.807 | 7.220 / 9.663 | 7.264 / 8.793 |
| Transfer classifier | 0.420 / 0.531 | 0.075 / 0.088 | 0.071 / 0.091 |
| Frozen video backbone | 601.985 / 624.122 | 698.048 / 711.954 | 695.360 / 710.111 |

The benchmark uses one thread and batch 1, with 80 classifier samples and 24
backbone samples after five warmups. Native execution helped the small transfer
classifier in this run, but the robustness classifier and video backbone were
slower than PyTorch. The backbone dominates transfer inference cost. Request
latency also includes decoding and HTTP handling; these graph times exclude both.
