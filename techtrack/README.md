# TechTrack Inference Service

This project implements a YOLO-based object detection inference pipeline for the TechTrack logistics dataset. The service captures a UDP video stream, runs object detection, applies Non-Maximum Suppression, prints per-frame detections, and saves annotated frames to `storage/detections/`.

## Implemented Components

The implementation includes:

* YOLO model loading and prediction in `modules/inference/model.py`
* Prediction filtering and bounding-box conversion
* NumPy-based Non-Maximum Suppression in `modules/inference/nms.py`
* mAP evaluation utilities in `modules/utils/metrics.py`
* YOLO-style loss computation in `modules/utils/loss.py`
* Hard Negative Mining in `modules/rectification/hard_negative_mining.py`
* Frame extraction with drop-rate sampling in `modules/inference/preprocessing.py`
* End-to-end inference orchestration in `app.py`

## Required Local Files

The following files must exist locally under `techtrack/storage/`:

```text
storage/yolo_model_1/
storage/yolo_model_2/
storage/test_videos/
storage/logistics/
```

These files are not committed to GitHub because model weights, videos, and datasets are large runtime assets.

## Quick Start: Local Python Run

From the repository root:

```bash
cd techtrack
python app.py
```

The service listens for a UDP stream on:

```text
udp://127.0.0.1:23000
```

In a second terminal, stream one of the test videos with FFmpeg:

```bash
cd techtrack
ffmpeg -re -i storage/test_videos/test_videos/worker-zone-detection.mp4 -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
```

The service prints detections for each processed frame and saves annotated frames to:

```text
storage/detections/
```

## Quick Start: Docker Build

Build the Docker image from inside the `techtrack/` directory:

```bash
cd techtrack
docker build -t techtrack-inference .
```

## Quick Start: Docker Run

Run the inference container with UDP port `23000` exposed:

```bash
docker run --rm -it -p 23000:23000/udp -v "$(pwd)/storage:/app/storage" techtrack-inference
```

In another terminal, stream the test video to the container:

```bash
ffmpeg -re -i storage/test_videos/test_videos/worker-zone-detection.mp4 -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
```

Annotated frames are written to:

```text
storage/detections/
```

Because `storage/` is mounted into the container, saved detections are available on the host machine.

## Testing

From the repository root, run:

```bash
python -m pytest test/unit_test.py test/unit_test_2.py -v
```

Expected local result:

```text
40 passed, 1 skipped
```


## Endpoint Testing

This implementation is a UDP-based video inference service, not an HTTP API service. Therefore, `curl` is not used for normal inference testing. The service is tested by:

1. Starting `app.py`
2. Streaming video with FFmpeg to UDP port `23000`
3. Checking terminal detection logs
4. Verifying saved annotated frames in `storage/detections/`

## Notes

* Do not commit `storage/`, model weights, datasets, test videos, or generated detection frames.
* Docker uses the headless OpenCV dependency path to avoid GUI requirements inside the container.
* The inference pipeline uses Model 1 by default.
