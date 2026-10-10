# Local replay

The React/TypeScript app replays paired video frames and six motion channels.
It calls the checked inference service for four observation fractions and
three sensor modes, comparing attention and frozen-video models. It displays
only predictions at or before the playback cursor. Exports contain predictions,
source model identity, and the trial ID; recordings stay local.

## Setup

Use Python 3.11 and Node 24. Create a separate Python environment. Install the
CPU build below, or the matched CUDA 12.8 builds in [TRANSFER.md](TRANSFER.md):

```sh
python -m venv .venv
# Activate .venv using your shell.
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-replay.txt
python fetch_data.py --video
```

Reproduce the trained models using [ROBUSTNESS.md](ROBUSTNESS.md) and
[TRANSFER.md](TRANSFER.md), then export them using [SERVING.md](SERVING.md).
The repository contains measured results and reproducible training code;
trained weights and dataset recordings are not distributed. The frozen
backbone is downloaded from its official host and verified by full checksum.

```sh
python replay_data.py --rgb data/RGB.zip --inertial data/Inertial.zip --output runs/replay
cd web
npm ci
npm run build
cd ..
python replay_server.py --package robust=runs/serving/robust --package transfer=runs/serving/transfer --replay runs/replay
```

Open `http://127.0.0.1:8000`. On Windows use `npm.cmd` if PowerShell blocks its
script wrapper. No global package installation is needed.

Choose a recording, then **Analyze recording**. The service computes 24
conditions sequentially; video processing may take several seconds. Scrub,
play, or choose 25/50/75/100% to compare the three sensor modes. **Export
predictions** saves JSON. A failed analysis retains completed conditions and
shows the error; run analysis again to retry. Changing a recording clears its
displayed results. Before 25%, no prediction is shown.

The four default examples are fixed trial IDs covering two known and two
downstream-held-out actions. They are illustrative, not a representative test
sample. Choose other paired examples with `--trials a3_s2_t1 a24_s4_t1` when
preparing a new folder. The preparation step verifies official archive hashes,
and startup verifies every replay file. JPEG frames avoid browser AVI codec
requirements; inference uses the original AVI and MAT, not the JPEG preview.

The cursor aligns recordings by normalized trial fraction. Shared frame/sample
timestamps are unavailable, so the display does not claim exact synchronization.
Signal plots use the source's numeric units. The released videos decode at 15 FPS;
sensor samples have a nominal 50 Hz source rate. Trial duration and sample counts
differ; this is a segmented offline replay, not streaming onset detection.

## Local experiment tracking

[MLflow](https://mlflow.org/docs/latest/ml/tracking/) imports completed results
into local SQLite and artifact files. Original completion dates and source
hashes are tags; import times are actual. These are result imports, not newly
executed training runs. Repeating an identical successful import reuses its run.
No remote tracking address or telemetry is used.

```sh
python track_results.py --results results/attention/summary.json --store runs/tracking
python track_results.py --results results/robustness/summary.json --store runs/tracking
python track_results.py --results results/transfer/summary.json --store runs/tracking
```

To open MLflow locally, set `MLFLOW_DISABLE_TELEMETRY=true` and
`DO_NOT_TRACK=true` in your shell, then run:

```sh
mlflow server --backend-store-uri sqlite:///runs/tracking/tracking.db --host 127.0.0.1 --port 5000
```

Use a short `--store` path on Windows if long-path support is disabled. Keep
tracking directories local: artifact paths are absolute and are not portable.

## Checks

```sh
python -m pytest -q
cd web
npm test
npm run build
npx playwright install chromium
npm run test:e2e
cd ..
```

CI uses generated inputs for browser error/partial-result checks; it does not
download research recordings or weights. For real-data verification, run the
replay server on port 8017, then:

```sh
python verify_replay.py --replay runs/replay --packages runs/serving --output runs/replay-check.json
```

Optional real browser tests use `ACTIVITY_DEMO_URL=http://127.0.0.1:8017`.
`ACTIVITY_BROWSER_CHANNEL=msedge` uses installed Edge on Windows; otherwise
Playwright uses its installed Chromium. See [release evidence](results/release)
and [model limitations](MODEL_CARD.md). Use trusted local inputs; the service
has no public authentication or hardened media-decoder sandbox.
