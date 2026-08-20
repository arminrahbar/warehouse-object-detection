"""Benchmark both detector checkpoints with reproducible end-to-end timing."""

import argparse
import csv
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_INDEX = PROJECT_ROOT / "experiments" / "outputs" / "dataset_index.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "model_selection"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_REPEATS = 1
DEFAULT_WARMUP_IMAGES = 1

# Direct script execution places experiments/scripts, not the repository root,
# on sys.path. Add the package root before the lazy runtime imports below.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CANDIDATE_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.3

MODELS = {
    "model1": {
        "weights": Path(
            "detector_service/storage/yolo_model_1/"
            "yolov4-tiny-logistics_size_416_1.weights"
        ),
        "cfg": Path(
            "detector_service/storage/yolo_model_1/"
            "yolov4-tiny-logistics_size_416_1.cfg"
        ),
        "names": Path("detector_service/storage/yolo_model_1/logistics.names"),
    },
    "model2": {
        "weights": Path(
            "detector_service/storage/yolo_model_2/"
            "yolov4-tiny-logistics_size_416_2.weights"
        ),
        "cfg": Path(
            "detector_service/storage/yolo_model_2/"
            "yolov4-tiny-logistics_size_416_2.cfg"
        ),
        "names": Path("detector_service/storage/yolo_model_2/logistics.names"),
    },
}

OBSERVATION_COLUMNS = [
    "model",
    "repeat_index",
    "sample_position",
    "image_file",
    "image_path",
    "status",
    "detections",
    "read_seconds",
    "predict_seconds",
    "postprocess_seconds",
    "nms_seconds",
    "total_seconds",
]
SUMMARY_COLUMNS = [
    "model",
    "images_in_dataset",
    "sample_requested",
    "unique_images_selected",
    "repeats",
    "warmup_images",
    "successful_observations",
    "unreadable_observations",
    "total_detections",
    "pipeline_setup_seconds",
    "measured_wall_seconds",
    "mean_seconds_per_image",
    "median_seconds_per_image",
    "p95_seconds_per_image",
    "images_per_second",
    "estimated_full_dataset_minutes",
    "candidate_threshold",
    "confidence_threshold",
    "nms_iou_threshold",
    "python_version",
    "opencv_version",
    "platform",
]


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return number


def _absolute_without_dereferencing(path):
    return Path(path).expanduser().absolute()


def load_dataset_sample(path, sample_size):
    """Load a deterministic prefix from the index and report total row count."""

    index_path = _absolute_without_dereferencing(path)
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Dataset index not found: {index_path}. "
            "Run experiments/scripts/02_build_dataset_index.py first."
        )

    with index_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        required = {"image_file", "image_path"}
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(
                "Dataset index is missing required columns: " + ", ".join(missing)
            )
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"Dataset index is empty: {index_path}")

    image_files = [row["image_file"] for row in rows]
    if len(image_files) != len(set(image_files)):
        raise ValueError("Dataset index contains duplicate image_file values.")

    return rows[:sample_size], len(rows)


def resolve_model_assets(paths, asset_root):
    """Resolve and validate one checkpoint's external files."""

    root = _absolute_without_dereferencing(asset_root)
    resolved = {
        name: _absolute_without_dereferencing(root / relative_path)
        for name, relative_path in paths.items()
    }
    missing = [path for path in resolved.values() if not path.is_file()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing model assets:\n{details}")
    return resolved


def linear_percentile(values, percentile):
    """Return a linearly interpolated percentile for a non-empty sequence."""

    if not values:
        raise ValueError("Percentile input must not be empty.")
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _execute_pipeline(detector, nms, frame):
    outputs = detector.predict(frame)
    decoded = detector.post_process(outputs)
    filtered = nms.filter(*decoded)
    return len(filtered[0])


def benchmark_model(
    model_name,
    paths,
    sample_rows,
    dataset_size,
    asset_root,
    repeats=DEFAULT_REPEATS,
    warmup_images=DEFAULT_WARMUP_IMAGES,
    detector_factory=None,
    nms_factory=None,
    image_reader=None,
    clock=time.perf_counter,
    runtime_metadata=None,
):
    """Benchmark one model and return one summary plus observation records."""

    if not sample_rows:
        raise ValueError("At least one sample row is required.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    if warmup_images < 0:
        raise ValueError("warmup_images must be non-negative.")
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive.")

    assets = resolve_model_assets(paths, asset_root)
    if detector_factory is None or nms_factory is None or image_reader is None:
        import cv2

        from detector_service.modules.inference.model import Detector
        from detector_service.modules.inference.nms import NMS

        detector_factory = detector_factory or Detector
        nms_factory = nms_factory or NMS
        image_reader = image_reader or cv2.imread
        if runtime_metadata is None:
            runtime_metadata = {"opencv_version": cv2.__version__}

    metadata = dict(runtime_metadata or {})
    metadata.setdefault("opencv_version", "unknown")
    metadata.setdefault("python_version", platform.python_version())
    metadata.setdefault("platform", platform.platform())

    setup_started = clock()
    detector = detector_factory(
        str(assets["weights"]),
        str(assets["cfg"]),
        str(assets["names"]),
        score_threshold=CANDIDATE_THRESHOLD,
    )
    nms = nms_factory(
        score_threshold=CONFIDENCE_THRESHOLD,
        nms_iou_threshold=NMS_IOU_THRESHOLD,
    )
    pipeline_setup_seconds = clock() - setup_started

    warmed_up = 0
    for row in sample_rows:
        if warmed_up >= warmup_images:
            break
        frame = image_reader(str(Path(asset_root) / row["image_path"]))
        if frame is None:
            continue
        _execute_pipeline(detector, nms, frame)
        warmed_up += 1

    observations = []
    measurement_started = clock()

    for repeat_index in range(1, repeats + 1):
        for sample_position, row in enumerate(sample_rows, start=1):
            observation_started = clock()
            read_started = observation_started
            image_path = Path(asset_root) / row["image_path"]
            frame = image_reader(str(image_path))
            read_finished = clock()

            base = {
                "model": model_name,
                "repeat_index": repeat_index,
                "sample_position": sample_position,
                "image_file": row["image_file"],
                "image_path": row["image_path"],
            }
            if frame is None:
                observations.append(
                    {
                        **base,
                        "status": "unreadable",
                        "detections": 0,
                        "read_seconds": read_finished - read_started,
                        "predict_seconds": "",
                        "postprocess_seconds": "",
                        "nms_seconds": "",
                        "total_seconds": read_finished - observation_started,
                    }
                )
                continue

            predict_started = read_finished
            outputs = detector.predict(frame)
            predict_finished = clock()

            postprocess_started = predict_finished
            decoded = detector.post_process(outputs)
            postprocess_finished = clock()

            nms_started = postprocess_finished
            filtered = nms.filter(*decoded)
            nms_finished = clock()

            observations.append(
                {
                    **base,
                    "status": "processed",
                    "detections": len(filtered[0]),
                    "read_seconds": read_finished - read_started,
                    "predict_seconds": predict_finished - predict_started,
                    "postprocess_seconds": postprocess_finished - postprocess_started,
                    "nms_seconds": nms_finished - nms_started,
                    "total_seconds": nms_finished - observation_started,
                }
            )

    measured_wall_seconds = clock() - measurement_started
    successful = [row for row in observations if row["status"] == "processed"]
    unreadable = [row for row in observations if row["status"] == "unreadable"]
    if not successful:
        raise RuntimeError("No images were processed. Check dataset paths in the index.")

    totals = [float(row["total_seconds"]) for row in successful]
    mean_seconds = statistics.fmean(totals)
    summary = {
        "model": model_name,
        "images_in_dataset": dataset_size,
        "sample_requested": len(sample_rows),
        "unique_images_selected": len(sample_rows),
        "repeats": repeats,
        "warmup_images": warmed_up,
        "successful_observations": len(successful),
        "unreadable_observations": len(unreadable),
        "total_detections": sum(int(row["detections"]) for row in successful),
        "pipeline_setup_seconds": pipeline_setup_seconds,
        "measured_wall_seconds": measured_wall_seconds,
        "mean_seconds_per_image": mean_seconds,
        "median_seconds_per_image": statistics.median(totals),
        "p95_seconds_per_image": linear_percentile(totals, 95),
        "images_per_second": 1.0 / mean_seconds if mean_seconds > 0 else float("inf"),
        "estimated_full_dataset_minutes": mean_seconds * dataset_size / 60.0,
        "candidate_threshold": CANDIDATE_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "python_version": metadata["python_version"],
        "opencv_version": metadata["opencv_version"],
        "platform": metadata["platform"],
    }
    return summary, observations


def _write_csv_atomic(path, fieldnames, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_benchmark_artifacts(output_dir, summaries, observations):
    destination = _absolute_without_dereferencing(output_dir)
    summary_path = destination / "inference_benchmark_summary.csv"
    observations_path = destination / "inference_benchmark_observations.csv"
    _write_csv_atomic(summary_path, SUMMARY_COLUMNS, summaries)
    _write_csv_atomic(observations_path, OBSERVATION_COLUMNS, observations)
    return summary_path, observations_path


def run_benchmark(
    sample_rows,
    dataset_size,
    asset_root,
    output_dir,
    repeats,
    warmup_images,
    models=MODELS,
    **dependencies,
):
    summaries = []
    observations = []
    for model_name, paths in models.items():
        print("=" * 80)
        print(f"Benchmarking {model_name}")
        print("=" * 80)
        summary, model_observations = benchmark_model(
            model_name=model_name,
            paths=paths,
            sample_rows=sample_rows,
            dataset_size=dataset_size,
            asset_root=asset_root,
            repeats=repeats,
            warmup_images=warmup_images,
            **dependencies,
        )
        summaries.append(summary)
        observations.extend(model_observations)
        print(
            f"[RESULT] {model_name}: observations="
            f"{summary['successful_observations']}, "
            f"mean={summary['mean_seconds_per_image']:.6f}s/image, "
            f"throughput={summary['images_per_second']:.3f} images/s, "
            f"estimated full dataset="
            f"{summary['estimated_full_dataset_minutes']:.2f} minutes"
        )

    summary_path, observations_path = write_benchmark_artifacts(
        output_dir,
        summaries,
        observations,
    )
    combined_minutes = sum(
        float(summary["estimated_full_dataset_minutes"]) for summary in summaries
    )
    print(f"[RESULT] Estimated sequential runtime for all models: {combined_minutes:.2f} minutes")
    print(f"[WRITE] Benchmark summary: {summary_path}")
    print(f"[WRITE] Benchmark observations: {observations_path}")
    return summaries, observations, summary_path, observations_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark both detector checkpoints on a deterministic sample."
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root containing detector_service/storage and indexed image paths.",
    )
    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=DEFAULT_DATASET_INDEX,
        help="Dataset index produced by 02_build_dataset_index.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for benchmark summary and observation CSV files.",
    )
    parser.add_argument(
        "--sample-size",
        type=positive_int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of deterministic index rows to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=DEFAULT_REPEATS,
        help="Number of measured passes over the selected rows.",
    )
    parser.add_argument(
        "--warmup-images",
        type=nonnegative_int,
        default=DEFAULT_WARMUP_IMAGES,
        help="Readable images processed before timing each checkpoint.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    asset_root = _absolute_without_dereferencing(args.asset_root)
    if not asset_root.is_dir():
        raise NotADirectoryError(f"Asset root does not exist: {asset_root}")

    dataset_index = args.dataset_index
    if not dataset_index.is_absolute():
        dataset_index = PROJECT_ROOT / dataset_index
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    sample_rows, dataset_size = load_dataset_sample(
        dataset_index,
        args.sample_size,
    )
    print(f"[INFO] Dataset images: {dataset_size}")
    print(f"[INFO] Deterministic sample images: {len(sample_rows)}")
    print(f"[INFO] Repeats: {args.repeats}")
    print(f"[INFO] Warm-up images per model: {args.warmup_images}")
    print(f"[INFO] Asset root: {asset_root}")
    print(
        "[INFO] Pipeline policy: "
        f"objectness>{CANDIDATE_THRESHOLD}, "
        f"combined confidence>={CONFIDENCE_THRESHOLD}, "
        f"class-aware NMS IoU={NMS_IOU_THRESHOLD}"
    )

    return run_benchmark(
        sample_rows=sample_rows,
        dataset_size=dataset_size,
        asset_root=asset_root,
        output_dir=output_dir,
        repeats=args.repeats,
        warmup_images=args.warmup_images,
    )


if __name__ == "__main__":
    main()
