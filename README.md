# Warehouse Object Detection

YOLO-based object-detection pipeline for warehouse and logistics video streams. The project supports UDP video ingestion, frame sampling, object detection, non-maximum suppression, annotated frame output, and supporting experiment scripts for model evaluation and detection analysis.

## Overview

The inference service listens for a UDP video stream, processes frames through a YOLO-based detector, applies non-maximum suppression, prints per-frame detections, and saves annotated frames locally.

The repository also includes experiment scripts for model comparison, dataset sampling validation, NMS threshold analysis, augmentation robustness analysis, and hard-negative mining.

## Repository Structure

```text
techtrack/
  app.py
  Dockerfile
  requirements.txt
  modules/
    inference/
    rectification/
    utils/

experiments/
  scripts/
  figures/
```

## Implemented Components

* YOLO model loading and prediction
* Frame extraction with drop-rate sampling
* Prediction filtering and bounding-box conversion
* NumPy-based non-maximum suppression
* mAP evaluation utilities
* YOLO-style loss computation
* Hard-negative mining utilities
* UDP video inference orchestration
* Experiment scripts for model and pipeline analysis

## Required Local Assets

Large runtime assets are not committed to Git. They should exist locally under:

```text
techtrack/storage/
  yolo_model_1/
  yolo_model_2/
  test_videos/
  logistics/
```

Generated detections are written to:

```text
techtrack/storage/detections/
```

## Local Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r techtrack/requirements.txt
```

## Run the Inference Service

Start the service:

```bash
cd techtrack
python app.py
```

The service listens on:

```text
udp://127.0.0.1:23000
```

In a second terminal, stream a test video with FFmpeg:

```bash
cd techtrack
ffmpeg -re -i storage/test_videos/test_videos/worker-zone-detection.mp4 \
  -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
```

Annotated frames are saved to:

```text
techtrack/storage/detections/
```

## Docker Build

Build the image from inside `techtrack/`:

```bash
cd techtrack
docker build -t warehouse-object-detection .
```

## Docker Run

Run the inference container with UDP port `23000` exposed and local storage mounted:

```bash
docker run --rm -it \
  -p 23000:23000/udp \
  -v "$(pwd)/storage:/app/storage" \
  warehouse-object-detection
```

In another terminal, stream a video to the container:

```bash
ffmpeg -re -i storage/test_videos/test_videos/worker-zone-detection.mp4 \
  -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
```

Because `storage/` is mounted into the container, saved detections are available on the host machine.

## Experiment Scripts

Supporting experiment scripts are located under:

```text
experiments/scripts/
```

They cover:

* model comparison
* dataset sampling validation
* NMS threshold analysis
* augmentation robustness analysis
* hard-negative mining analysis

Selected generated figures are tracked under:

```text
experiments/figures/
```

Generated CSV outputs are local-only and ignored by Git.

## Notes

Do not commit model weights, datasets, videos, generated detections, notebook outputs, or experiment CSV outputs. These are intentionally excluded from version control.
