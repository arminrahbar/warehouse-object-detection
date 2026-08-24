# External runtime assets

This directory is the canonical local mount point for the detector's datasets,
model bundles, and evaluation videos. The repository tracks this contract, but
it intentionally does not version the external assets themselves.

Expected layout:

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

The evidence-producing corpus is expected to contain 9,525 paired JPEG and
YOLO label files, 36,721 labeled objects, and the ordered 20-class vocabulary.
The inventory builder validates those relationships before an experiment uses
them.

An ignored `detections/` directory may also appear here after local inference.
It is generated output rather than a required asset. Run-specific output under
`scratch/` is preferred because it keeps results from separate executions
isolated.

Place the asset tree here, or link its required child directories into this
tracked mount point from an externally managed storage directory. The
application accepts explicit asset file paths, and experiment tools that expose
path options can reference the external storage directory directly. Consult the
relevant script's `--help`; the inventory builder also needs storage-relative
`--dataset-dir` and `--classes` values when its asset root is this directory.

The project MIT license applies to the source code and documentation in this
repository. It does not grant rights to datasets, model weights, or media placed
in this directory; those assets remain subject to their respective terms.
