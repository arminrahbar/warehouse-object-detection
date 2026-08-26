# Warehouse Object Detection

A reproducible computer-vision pipeline for detecting logistics objects in
video streams. The system accepts local video or UDP input; samples
frames; runs YOLOv4-tiny through OpenCV DNN; applies class-aware non-maximum
suppression; and emits structured detections with optional annotated JPEGs.

The repository also contains an evidence-oriented experiment suite for dataset
characterization, checkpoint comparison, operating-point selection,
controlled input-shift diagnostics, and targeted error review. The integrated
engineering evidence is summarized in the [project report](docs/PROJECT_REPORT.md).

## Capabilities

- Local-file and UDP video ingestion through OpenCV.
- Configurable frame sampling for bounded inference cost.
- Darknet `cfg`/`weights` inference with explicit candidate decoding.
- Combined-confidence filtering and deterministic, class-aware NMS.
- Human-readable frame and detection logs plus optional annotated output.
- One-to-one, same-class matching with experiment-specific 101-point or
  threshold-constrained 11-point AP50.
- Deterministic dataset sampling with density and rare-class coverage controls.
- Controlled NMS and input-shift experiments with reproducible result tables.
- Five specialized error-review queues for focused detector diagnosis.
- A Python 3.12 Docker runtime with external, read-only model/data mounts.

## Runtime architecture

![Three-panel overview of runtime inference, controlled evaluation, and operational diagnosis](docs/figures/01_system_scope.png)

The detector first gates raw candidates by objectness. NMS then ranks the
surviving boxes by `objectness * predicted-class probability` and suppresses
only higher-overlap boxes that predict the same class. Different classes are
never forced to compete for the same spatial region.

## Experiment architecture

| Stage | Controlled decision | Downstream use |
|---|---|---|
| Checkpoint comparison | Select Checkpoint B from paired quality evidence | Fix the detector for later studies |
| Workload design | Preserve measured class, density, and crowding margins in 5,000 indexed paths | Bound repeated analysis |
| NMS sensitivity | Select class-aware IoU 0.30 with a locked quality-first rule | Fix provisional post-processing for diagnostics |
| Input-shift diagnostics | Identify blur as the first field-validation hypothesis | Prioritize camera-derived testing |
| Error review | Build five deterministic, specialized top-250 queues | Direct manual failure analysis |

## Repository layout

```text
detector_service/
├── app.py                         # Runtime orchestration and CLI
├── Dockerfile                     # Reproducible inference image
├── requirements.txt               # Exact runtime dependencies
├── storage/
│   └── README.md                   # External-asset layout and license boundary
└── modules/
    ├── inference/
    │   ├── preprocessing.py       # Video capture and frame sampling
    │   ├── model.py               # OpenCV-DNN model adapter
    │   └── nms.py                 # Combined-confidence, class-aware NMS
    ├── rectification/
    │   ├── augmentation.py        # Bounding-box-aware transforms
    │   └── hard_negative_mining.py # Image-level detector error components
    └── utils/
        └── metrics.py             # IoU, matching, precision/recall, mAP
experiments/scripts/               # Numbered analysis entry points
experiments/figures/               # Accepted figures by experiment
experiments/reports/               # Connected experiment reports
docs/PROJECT_REPORT.md             # Integrated engineering report
docs/figures/                      # Curated project-report figures
tests/                             # Unit, integration, and packaging tests
requirements-analysis.txt          # Runtime plus analysis dependencies
LICENSE                            # MIT terms for original code and documentation
```

Model checkpoints, dataset files, videos, and generated runtime output are
intentionally excluded from version control. The tracked
[experiment output contract](experiments/OUTPUTS.md) defines the numbered
output locations produced when the analyses are reproduced locally.

## Requirements

- Python 3.12
- FFmpeg for UDP smoke tests and stream simulation
- Docker Desktop or Docker Engine for container execution
- External YOLOv4-tiny checkpoints, class names, dataset, and test videos

OpenCV is pinned to `4.13.0.92`. OpenCV 5 removed the legacy Darknet importer
used for the supplied `cfg` and `weights` format, so upgrading OpenCV requires
first converting the checkpoints to a supported format such as ONNX.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-analysis.txt
```

Place the external assets under `detector_service/storage`:

```text
detector_service/storage/
├── logistics/
│   ├── _darknet.labels
│   ├── *.jpg
│   └── *.txt
├── yolo_model_1/
│   ├── logistics.names
│   ├── yolov4-tiny-logistics_size_416_1.cfg
│   └── yolov4-tiny-logistics_size_416_1.weights
├── yolo_model_2/
│   ├── logistics.names
│   ├── yolov4-tiny-logistics_size_416_2.cfg
│   └── yolov4-tiny-logistics_size_416_2.weights
└── test_videos/test_videos/
    └── *.mp4
```

The validated experiment corpus contains 9,525 image-label pairs, 36,721
labeled objects, and 20 ordered classes. A local `detections/` directory may
also exist under `storage/`; it is generated output, not an input asset.

If the assets are maintained outside the repository, link the required child
directories into the tracked storage mount point. For example:

```bash
ln -s /absolute/path/to/storage/logistics detector_service/storage/logistics
ln -s /absolute/path/to/storage/yolo_model_1 detector_service/storage/yolo_model_1
ln -s /absolute/path/to/storage/yolo_model_2 detector_service/storage/yolo_model_2
ln -s /absolute/path/to/storage/test_videos detector_service/storage/test_videos
```

Experiment commands can instead reference an externally managed asset root
through their CLI path options. Run the relevant script with `--help`; the
inventory builder also requires storage-relative `--dataset-dir` and `--classes`
arguments when `--asset-root` is the storage directory itself.

## Run inference

The following bounded run processes five sampled frames and writes annotated
JPEGs to a caller-created output directory:

```bash
python -m detector_service.app \
  --source detector_service/storage/test_videos/test_videos/worker-zone-detection.mp4 \
  --weights detector_service/storage/yolo_model_2/yolov4-tiny-logistics_size_416_2.weights \
  --config detector_service/storage/yolo_model_2/yolov4-tiny-logistics_size_416_2.cfg \
  --classes detector_service/storage/yolo_model_2/logistics.names \
  --frame-interval 60 \
  --candidate-threshold 0.50 \
  --confidence-threshold 0.50 \
  --nms-iou-threshold 0.30 \
  --max-frames 5 \
  --save-dir /tmp/warehouse-object-detections
```

Each sampled frame produces a `[FRAME]` record. Retained detections produce
`[DETECTION]` records containing the sampled-frame index, pixel-space box,
class ID, and combined confidence. Add `--no-save` for log-only execution.

### Receive a UDP stream

Start the detector first so that it is listening before the sender begins:

```bash
python -m detector_service.app \
  --source udp://127.0.0.1:23000 \
  --frame-interval 60 \
  --max-frames 3 \
  --no-save
```

Then publish a video from another terminal:

```bash
ffmpeg -nostdin -re \
  -i detector_service/storage/test_videos/test_videos/worker-zone-detection.mp4 \
  -an -r 30 -c:v mpeg4 -f mpegts \
  "udp://127.0.0.1:23000?pkt_size=1316"
```

## Run with Docker

Build from the repository root so the Dockerfile receives the expected context:

```bash
docker build \
  --file detector_service/Dockerfile \
  --tag warehouse-object-detection:local \
  .
```

The image contains code and runtime dependencies only. Mount assets at runtime
instead of embedding checkpoints or datasets in the image:

```bash
mkdir -p /tmp/warehouse-docker-output

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$(pwd)/detector_service/storage",dst=/app/detector_service/storage,readonly \
  --mount type=bind,src=/tmp/warehouse-docker-output,dst=/app/output \
  warehouse-object-detection:local \
  python -m detector_service.app \
    --source /app/detector_service/storage/test_videos/test_videos/worker-zone-detection.mp4 \
    --max-frames 5 \
    --save-dir /app/output
```

A successful five-sample run writes `sampled_frame_000000.jpg` through
`sampled_frame_000004.jpg`. The sample-based prefix is part of the runtime
output contract and distinguishes these indices from source-video frame
numbers.

The image defaults to its own unprivileged `app` account. The explicit `--user`
mapping above keeps the process non-root while giving it the host user's
identity, so the bind-mounted output directory remains writable on native
Linux and the generated files are owned by the invoking user.

Validate every required asset on the host before launching the container, then
verify the same files through the read-only mount. On Docker Desktop with WSL,
the bind source must also be visible to the active Docker client and daemon. If
`docker` resolves to the Windows executable, a source—or a repository symlink
target—that exists only inside the WSL distro may require a Windows-visible
staging directory or a Linux Docker CLI integrated with that distro. These are
host-path and mount requirements; the image intentionally does not embed model
or media assets.

## Reproduce the analysis

With the standard asset layout in place, run analysis entry points from the
repository root. Stage `00` builds the shared corpus inventory; stages `01`
through `05` correspond to the connected experiment reports.

| Stage | Primary entry point | Engineering question |
|---|---|---|
| 00 · Inventory | `experiments/scripts/02_dataset_analysis/00_build_dataset_inventory.py` | Which image/label pairs and classes form the validated corpus? |
| 01 · Model selection | `experiments/scripts/01_model_selection/01_model_comparison.py` | Which checkpoint provides the stronger quality baseline? |
| 02 · Dataset analysis | `experiments/scripts/02_dataset_analysis/02_dataset_sampling.py` | Which bounded workload preserves class, density, and crowding characteristics? |
| 03 · NMS thresholding | `experiments/scripts/03_nms_thresholding/03_nms_threshold_sweep.py` | Which class-aware IoU setting provides the best measured quality/output trade-off? |
| 04 · Input-shift diagnostics | `experiments/scripts/04_augmentation_robustness/04_augmentation_robustness.py` | Which controlled input changes most strongly affect the selected pipeline? |
| 05 · Error review | `experiments/scripts/05_hard_negative_mining/05_build_error_review_queues.py` | Which images should be prioritized for different kinds of detector review? |

Use each script's `--help` output for its required paths and controls. Start
with a bounded pilot through `--max-images` where supported before launching a
full-corpus evaluation. The [experiment reports](experiments/reports/README.md)
document the fixed populations, metrics, decision rules, and implementation
paths; the [output contract](experiments/OUTPUTS.md) documents generated result
ownership without requiring machine-specific paths.

## Validated results

The validated experiment sequence produced the following evidence.
These are experiment-specific measurements, not claims of general performance
on unrelated warehouse environments.

| Area | Verified result |
|---|---|
| Dataset | 9,525 indexed image paths, 36,721 labeled objects, and 20 classes. |
| Checkpoint selection | Checkpoint B improved 101-point AP50 by `0.032792` over Checkpoint A; the paired source-group bootstrap 95% interval was `[0.027042, 0.039543]`. |
| Selected workload | 5,000 indexed image paths and 19,196 labels with density and protected-class constraints satisfied. |
| NMS operating point | IoU `0.30` retained 7,727 predictions and produced 11-point mAP@0.5 of `0.401573`. |
| NMS tradeoff | IoU `0.20` was a compact near-equivalent; settings above `0.50` added increasingly redundant output without improving measured quality. |
| Brightness shifts | Brighter and darker conditions changed mAP to `0.388012` and `0.379839`. |
| Stronger shifts | Gaussian blur produced `0.276908`; vertical flip produced `0.190519`. |
| Complete misses | 1,677 images had no retained prediction, covering 4,193 labeled objects. |
| Review queues | Five queues of 250 images; maximum off-diagonal Jaccard overlap was `0.259`. |

The nominal NMS choice is intentionally treated as provisional: mAP is nearly
flat from IoU `0.20` through `0.50`, while duplicate-like retained pairs rise at
more permissive thresholds. The selected value balances compact predictions
against near-identical observed AP point estimates under the fixed confidence
policy. The machine-enforced rule first maximizes threshold-constrained 11-point
AP50, then breaks an exact tie by fewer retained predictions and finally by the
lower IoU threshold. The provisional label reflects the absence of an untouched
confirmation set and uncertainty interval, not ambiguity in how the recorded
decision is computed.

## Evaluation semantics

- Predictions are matched to unused ground-truth boxes of the same class.
- A match requires IoU greater than or equal to `0.50`.
- Predictions are ranked by combined confidence.
- Experiment 01 calculates low-floor AP50 at 101 evenly spaced recall levels.
- Experiments 03 and 04 calculate threshold-constrained AP50 at 11 evenly
  spaced recall levels after applying the configured 0.50 confidence filter.
- Aggregate mAP averages all 20 class AP values. Neither experiment-specific
  definition is COCO-style mAP averaged over several IoU thresholds.

The input-shift experiment measures sensitivity to four controlled conditions;
it is not a complete robustness or safety evaluation. Error-review queues are
diagnostic priorities, not automatically corrected training labels. The repository
evaluates existing checkpoints and does not include a model-training pipeline.

## Continuous integration

The GitHub Actions workflow runs on pushes to `main` and pull requests targeting
`main`. Its two independent jobs:

- install the analysis environment under Python 3.12 and run the public test
  suite;
- build the inference container without model, dataset, video, or generated
  artifacts in the Docker context.

The workflow uses read-only repository permissions, receives no secrets, and
does not publish its locally built image. External parity checks self-skip when
their optional comparison sources are unavailable; the asset-independent
runtime, experiment, and packaging contracts still execute.

## Tests

```bash
python -m unittest discover -s tests -v
```

The test suite covers video-resource cleanup, decoding, confidence math,
class-aware suppression, one-to-one matching, augmentation geometry,
deterministic sampling, experiment schemas, atomic artifact writes,
container packaging, and numerical regression behavior. Optional parity checks
self-skip only when their separately managed comparison sources are unavailable.

## License

The repository's original code, documentation, and figures are available under
the [MIT License](LICENSE).

External datasets, model checkpoints, model configuration files, class
vocabularies, and videos are not distributed by this repository and are not
covered by its MIT License. They remain subject to the terms established by
their respective owners. Third-party dependencies also retain their own
licenses.
