"""Benchmark both detector checkpoints with reproducible end-to-end timing."""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import secrets
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY_DIR = (
    PROJECT_ROOT / "experiments" / "outputs" / "00_dataset_inventory"
)
DEFAULT_DATASET_INDEX = DEFAULT_INVENTORY_DIR / "dataset_index.csv"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "outputs"
    / "01_model_selection"
    / "02_runtime_benchmark"
)
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_REPEATS = 1
DEFAULT_WARMUP_IMAGES = 1
DEFAULT_SEED = 42
DEFAULT_BOOTSTRAP_SAMPLES = 2000
PAIRED_MANIFEST_NAME = "inference_benchmark_manifest.json"
SOURCE_GROUP_MARKER = "_jpg.rf."
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

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
    "benchmark_mode",
    "density_bucket",
    "execution_order",
    "compute_seconds",
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
    "benchmark_mode",
    "seed",
    "sample_selection",
    "mean_compute_seconds",
    "median_compute_seconds",
    "p95_compute_seconds",
]

PAIRED_COMPARISON_COLUMNS = [
    "record_type",
    "repeat_index",
    "sample_position",
    "image_file",
    "image_path",
    "source_group",
    "density_bucket",
    "first_model",
    "model1_compute_ms",
    "model2_compute_ms",
    "delta_model2_minus_model1_ms",
    "faster_model",
    "source_groups",
    "pairs",
    "bootstrap_samples",
    "seed",
    "model1_median_ms",
    "model2_median_ms",
    "relative_median_difference_pct",
    "model1_p95_ms",
    "model2_p95_ms",
    "p95_delta_model2_minus_model1_ms",
    "relative_p95_difference_pct",
    "p95_delta_model2_minus_model1_ci_lower_ms",
    "p95_delta_model2_minus_model1_ci_upper_ms",
    "relative_p95_difference_ci_lower_pct",
    "relative_p95_difference_ci_upper_pct",
    "mean_delta_model2_minus_model1_ms",
    "relative_mean_difference_pct",
    "mean_delta_ci_lower_ms",
    "mean_delta_ci_upper_ms",
    "relative_mean_difference_ci_lower_pct",
    "relative_mean_difference_ci_upper_pct",
]

DENSITY_BUCKETS = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-9", 5, 9),
    ("10-14", 10, 14),
    ("15-19", 15, 19),
    ("20+", 20, None),
)


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


def validate_run_id(value):
    """Reject run identifiers that could escape or obscure the output root."""

    run_id = str(value)
    if not RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
        raise argparse.ArgumentTypeError("invalid run ID")
    return run_id


def _normal_path(path):
    """Return a canonical evidence path without a Windows device prefix."""
    raw = str(path)
    if os.name == "nt":
        if raw.startswith("\\\\?\\UNC\\"):
            raw = "\\\\" + raw[8:]
        elif raw.startswith("\\\\?\\"):
            raw = raw[4:]
    return Path(raw)


def _absolute_without_dereferencing(path):
    return _normal_path(path).expanduser().absolute()


def _filesystem_path(path):
    """Return an extended-length Windows path only for filesystem I/O."""
    normal = _absolute_without_dereferencing(path)
    if os.name != "nt":
        return normal
    raw = str(normal)
    if raw.startswith("\\\\?\\") or len(raw) < 248:
        return normal
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw[2:])
    return Path("\\\\?\\" + raw)


def _resolved_normal_path(path):
    return _normal_path(_filesystem_path(path).resolve())


def _read_image_cv2(path, cv2_module=None):
    """Decode extended Windows paths through Python byte I/O and imdecode."""
    if cv2_module is None:
        import cv2 as cv2_module
    normal = _absolute_without_dereferencing(path)
    filesystem = _filesystem_path(normal)
    if os.name == "nt" and str(filesystem) != str(normal):
        import numpy as np
        try:
            with filesystem.open("rb") as source:
                encoded = source.read()
        except OSError:
            return None
        if not encoded:
            return None
        return cv2_module.imdecode(
            np.frombuffer(encoded, dtype=np.uint8), cv2_module.IMREAD_COLOR
        )
    return cv2_module.imread(str(normal))


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with _filesystem_path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_group_key(image_file):
    """Return the locked source-family key for an indexed image filename."""

    value = str(image_file).strip()
    if not value:
        raise ValueError("image_file must not be empty")
    prefix, marker, _ = value.partition(SOURCE_GROUP_MARKER)
    return prefix if marker and prefix else value


def density_bucket(num_objects):
    """Return the stable object-count density bucket for one indexed image."""

    count = int(num_objects)
    if count < 0:
        raise ValueError("num_objects must be a non-negative integer")
    for name, lower, upper in DENSITY_BUCKETS:
        if count >= lower and (upper is None or count <= upper):
            return name
    raise AssertionError(f"No density bucket configured for {count} objects.")


def _parse_num_objects(row, row_number):
    raw_value = str(row.get("num_objects", "")).strip()
    try:
        count = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"Dataset index row {row_number} has invalid num_objects: {raw_value!r}"
        ) from error
    if count < 0:
        raise ValueError(
            f"Dataset index row {row_number} has invalid num_objects: {raw_value!r}"
        )
    row["num_objects"] = str(count)
    return count


def _allocate_density_sample(bucket_sizes, sample_size):
    """Allocate an exact stratified sample with stable largest remainders."""

    nonempty = [name for name, _, _ in DENSITY_BUCKETS if bucket_sizes.get(name, 0)]
    target = min(sample_size, sum(bucket_sizes.values()))
    allocation = {name: 0 for name in bucket_sizes}
    if target == 0:
        return allocation

    if target >= len(nonempty):
        for name in nonempty:
            allocation[name] = 1

    remaining = target - sum(allocation.values())
    capacities = {
        name: bucket_sizes[name] - allocation[name]
        for name in nonempty
    }
    while remaining > 0:
        total_capacity = sum(capacities.values())
        if total_capacity <= 0:
            break
        quotas = {
            name: remaining * capacities[name] / total_capacity
            for name in nonempty
            if capacities[name] > 0
        }
        floors = {
            name: min(capacities[name], int(quota))
            for name, quota in quotas.items()
        }
        assigned = sum(floors.values())
        for name, count in floors.items():
            allocation[name] += count
            capacities[name] -= count
        remaining -= assigned
        if remaining == 0:
            break

        ranked = sorted(
            (name for name in nonempty if capacities[name] > 0),
            key=lambda name: (
                -(quotas.get(name, 0.0) - int(quotas.get(name, 0.0))),
                nonempty.index(name),
            ),
        )
        if not ranked:
            break
        for name in ranked:
            if remaining == 0:
                break
            allocation[name] += 1
            capacities[name] -= 1
            remaining -= 1

    if sum(allocation.values()) != target:
        raise AssertionError("Density allocation did not produce the requested size.")
    return allocation


def stratified_density_sample(rows, sample_size, seed=DEFAULT_SEED):
    """Choose an exact, seeded sample spanning object-count density buckets."""

    grouped = defaultdict(list)
    for row_number, source_row in enumerate(rows, start=2):
        row = dict(source_row)
        count = _parse_num_objects(row, row_number)
        grouped[density_bucket(count)].append(row)

    bucket_sizes = {name: len(grouped[name]) for name, _, _ in DENSITY_BUCKETS}
    allocation = _allocate_density_sample(bucket_sizes, sample_size)
    rng = random.Random(seed)
    selected = []
    for name, _, _ in DENSITY_BUCKETS:
        candidates = sorted(grouped[name], key=lambda row: row["image_file"])
        rng.shuffle(candidates)
        selected.extend(candidates[: allocation.get(name, 0)])
    rng.shuffle(selected)
    return selected


def load_dataset_sample(path, sample_size, paired=False, seed=DEFAULT_SEED):
    """Load a deterministic prefix or a seeded density-stratified sample."""

    index_path = _absolute_without_dereferencing(path)
    index_filesystem = _filesystem_path(index_path)
    if not index_filesystem.is_file():
        raise FileNotFoundError(
            f"Dataset index not found: {index_path}. "
            "Run experiments/scripts/00_build_dataset_inventory.py first."
        )

    with index_filesystem.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        required = {"image_file", "image_path"}
        if paired:
            required.add("num_objects")
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

    if paired:
        if sample_size > len(rows):
            raise ValueError(
                f"Paired sample requested {sample_size} images, but the index "
                f"contains only {len(rows)}."
            )
        selected = stratified_density_sample(rows, sample_size, seed)
        if len(selected) != sample_size:
            raise RuntimeError(
                f"Paired sampling produced {len(selected)} rows; expected "
                f"exactly {sample_size}."
            )
        return selected, len(rows)
    return rows[:sample_size], len(rows)


def _storage_relative_path(path):
    """Strip a known repository storage prefix from an indexed asset path."""

    relative = Path(path)
    parts = relative.parts
    prefix = ("detector_service", "storage")
    if tuple(parts[: len(prefix)]) == prefix:
        return Path(*parts[len(prefix) :])
    return relative


def resolve_indexed_asset_path(asset_root, indexed_path):
    """Resolve an indexed path against an external storage directory.

    When ``asset_root`` is the storage directory itself, the canonical
    ``detector_service/storage`` prefix is stripped from the indexed path.
    """

    raw = str(indexed_path).strip()
    if not raw:
        raise ValueError("Indexed asset path must not be empty.")
    root = _resolved_normal_path(asset_root)
    raw_path = _normal_path(raw).expanduser()
    if raw_path.is_absolute():
        candidates = [_resolved_normal_path(raw_path)]
    else:
        parts = PurePosixPath(raw.replace("\\", "/")).parts
        if ".." in parts:
            raise ValueError(
                f"Indexed asset path cannot traverse parents: {indexed_path}"
            )
        raw_path = Path(*parts)
        candidates = [root / raw_path]
    storage_relative = _storage_relative_path(raw_path)
    if storage_relative != raw_path:
        candidates.append(root / storage_relative)
    for candidate in candidates:
        resolved = _resolved_normal_path(candidate)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Indexed asset path must remain inside external storage: "
                f"{indexed_path}"
            ) from exc
        if _filesystem_path(resolved).exists():
            return resolved
    return _resolved_normal_path(candidates[-1])


def resolve_model_assets(paths, asset_root):
    """Resolve and validate one checkpoint's external files."""

    resolved = {
        name: resolve_indexed_asset_path(asset_root, relative_path)
        for name, relative_path in paths.items()
    }
    missing = [
        path for path in resolved.values()
        if not _filesystem_path(path).is_file()
    ]
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
        image_reader = image_reader or _read_image_cv2
        if runtime_metadata is None:
            runtime_metadata = {"opencv_version": cv2.__version__}

    metadata = dict(runtime_metadata or {})
    metadata.setdefault("opencv_version", "unknown")
    metadata.setdefault("python_version", platform.python_version())
    metadata.setdefault("platform", platform.platform())

    setup_started = clock()
    detector = detector_factory(
        str(_filesystem_path(assets["weights"])),
        str(_filesystem_path(assets["cfg"])),
        str(_filesystem_path(assets["names"])),
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
        frame = image_reader(
            str(resolve_indexed_asset_path(asset_root, row["image_path"]))
        )
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
            image_path = resolve_indexed_asset_path(asset_root, row["image_path"])
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
                        "benchmark_mode": "independent",
                        "density_bucket": (
                            density_bucket(row["num_objects"])
                            if str(row.get("num_objects", "")).strip()
                            else ""
                        ),
                        "execution_order": "",
                        "compute_seconds": "",
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
                    "benchmark_mode": "independent",
                    "density_bucket": (
                        density_bucket(row["num_objects"])
                        if str(row.get("num_objects", "")).strip()
                        else ""
                    ),
                    "execution_order": "",
                    "compute_seconds": nms_finished - predict_started,
                }
            )

    measured_wall_seconds = clock() - measurement_started
    successful = [row for row in observations if row["status"] == "processed"]
    unreadable = [row for row in observations if row["status"] == "unreadable"]
    if not successful:
        raise RuntimeError("No images were processed. Check dataset paths in the index.")

    totals = [float(row["total_seconds"]) for row in successful]
    compute_totals = [float(row["compute_seconds"]) for row in successful]
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
        "benchmark_mode": "independent",
        "seed": "",
        "sample_selection": "index_prefix",
        "mean_compute_seconds": statistics.fmean(compute_totals),
        "median_compute_seconds": statistics.median(compute_totals),
        "p95_compute_seconds": linear_percentile(compute_totals, 95),
    }
    return summary, observations


def _relative_difference(comparison, baseline):
    baseline = float(baseline)
    if baseline == 0:
        return ""
    return 100.0 * (float(comparison) - baseline) / baseline


def paired_source_group_bootstrap(pair_rows, samples, seed=DEFAULT_SEED):
    """Bootstrap paired mean and p95 contrasts by source family.

    Every resampled source family contributes all of its selected image variants
    and all measured repeats, preserving the dependence structure within a
    source capture.
    """

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    grouped = defaultdict(list)
    for row in pair_rows:
        group = row.get("source_group") or source_group_key(row["image_file"])
        grouped[group].append(row)
    source_groups = sorted(grouped)
    if not source_groups:
        raise ValueError("At least one successful pair is required for bootstrap.")

    rng = random.Random(seed)
    mean_delta_samples = []
    relative_mean_samples = []
    p95_delta_samples = []
    relative_p95_samples = []
    for _ in range(samples):
        sampled_groups = [
            source_groups[rng.randrange(len(source_groups))]
            for _ in source_groups
        ]
        sampled_rows = [
            row
            for group_name in sampled_groups
            for row in grouped[group_name]
        ]
        model1_mean = statistics.fmean(
            float(row["model1_compute_ms"]) for row in sampled_rows
        )
        model2_mean = statistics.fmean(
            float(row["model2_compute_ms"]) for row in sampled_rows
        )
        mean_delta_samples.append(model2_mean - model1_mean)
        relative_mean = _relative_difference(model2_mean, model1_mean)
        if relative_mean != "":
            relative_mean_samples.append(relative_mean)

        model1_p95 = linear_percentile(
            [float(row["model1_compute_ms"]) for row in sampled_rows],
            95,
        )
        model2_p95 = linear_percentile(
            [float(row["model2_compute_ms"]) for row in sampled_rows],
            95,
        )
        p95_delta_samples.append(model2_p95 - model1_p95)
        relative_p95 = _relative_difference(model2_p95, model1_p95)
        if relative_p95 != "":
            relative_p95_samples.append(relative_p95)

    result = {
        "mean_delta_ci_lower_ms": linear_percentile(mean_delta_samples, 2.5),
        "mean_delta_ci_upper_ms": linear_percentile(mean_delta_samples, 97.5),
        "relative_mean_difference_ci_lower_pct": "",
        "relative_mean_difference_ci_upper_pct": "",
        "p95_delta_model2_minus_model1_ci_lower_ms": linear_percentile(
            p95_delta_samples,
            2.5,
        ),
        "p95_delta_model2_minus_model1_ci_upper_ms": linear_percentile(
            p95_delta_samples,
            97.5,
        ),
        "relative_p95_difference_ci_lower_pct": "",
        "relative_p95_difference_ci_upper_pct": "",
    }
    if relative_mean_samples:
        result["relative_mean_difference_ci_lower_pct"] = linear_percentile(
            relative_mean_samples,
            2.5,
        )
        result["relative_mean_difference_ci_upper_pct"] = linear_percentile(
            relative_mean_samples,
            97.5,
        )
    if relative_p95_samples:
        result["relative_p95_difference_ci_lower_pct"] = linear_percentile(
            relative_p95_samples,
            2.5,
        )
        result["relative_p95_difference_ci_upper_pct"] = linear_percentile(
            relative_p95_samples,
            97.5,
        )
    return result


def build_paired_comparison_rows(pair_rows, bootstrap_samples, seed):
    """Append one aggregate record to successful per-pair latency records."""

    if not pair_rows:
        raise RuntimeError("No readable image pairs were processed.")
    model1_values = [float(row["model1_compute_ms"]) for row in pair_rows]
    model2_values = [float(row["model2_compute_ms"]) for row in pair_rows]
    deltas = [float(row["delta_model2_minus_model1_ms"]) for row in pair_rows]
    model1_median = statistics.median(model1_values)
    model2_median = statistics.median(model2_values)
    model1_p95 = linear_percentile(model1_values, 95)
    model2_p95 = linear_percentile(model2_values, 95)
    model1_mean = statistics.fmean(model1_values)
    model2_mean = statistics.fmean(model2_values)
    confidence_intervals = paired_source_group_bootstrap(
        pair_rows,
        bootstrap_samples,
        seed,
    )
    aggregate = {
        "record_type": "aggregate",
        "source_groups": len(
            {
                row.get("source_group") or source_group_key(row["image_file"])
                for row in pair_rows
            }
        ),
        "pairs": len(pair_rows),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "model1_median_ms": model1_median,
        "model2_median_ms": model2_median,
        "relative_median_difference_pct": _relative_difference(
            model2_median,
            model1_median,
        ),
        "model1_p95_ms": model1_p95,
        "model2_p95_ms": model2_p95,
        "p95_delta_model2_minus_model1_ms": model2_p95 - model1_p95,
        "relative_p95_difference_pct": _relative_difference(
            model2_p95,
            model1_p95,
        ),
        "mean_delta_model2_minus_model1_ms": statistics.fmean(deltas),
        "relative_mean_difference_pct": _relative_difference(
            model2_mean,
            model1_mean,
        ),
        **confidence_intervals,
    }
    return [*pair_rows, aggregate]


def benchmark_paired(
    sample_rows,
    dataset_size,
    asset_root,
    repeats=DEFAULT_REPEATS,
    warmup_images=DEFAULT_WARMUP_IMAGES,
    seed=DEFAULT_SEED,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    models=MODELS,
    detector_factory=None,
    nms_factory=None,
    image_reader=None,
    clock=time.perf_counter,
    runtime_metadata=None,
):
    """Benchmark both checkpoints as paired compute measurements per frame."""

    if not sample_rows:
        raise ValueError("At least one sample row is required.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    if warmup_images < 0:
        raise ValueError("warmup_images must be non-negative.")
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive.")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive.")
    if list(models) != ["model1", "model2"]:
        raise ValueError("Paired mode requires model1 followed by model2.")
    if warmup_images > len(sample_rows):
        raise ValueError(
            "warmup_images cannot exceed the number of paired sample rows."
        )
    image_files = [str(row.get("image_file", "")).strip() for row in sample_rows]
    image_paths = [str(row.get("image_path", "")).strip() for row in sample_rows]
    if any(not value for value in image_files):
        raise ValueError("Every paired sample row must have a non-empty image_file.")
    if any(not value for value in image_paths):
        raise ValueError("Every paired sample row must have a non-empty image_path.")
    if len(image_files) != len(set(image_files)):
        raise ValueError("Paired sample rows must have unique image_file values.")
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("Paired sample rows must have unique image_path values.")
    for row_number, row in enumerate(sample_rows, start=1):
        if "num_objects" not in row:
            raise ValueError(
                f"Paired sample row {row_number} is missing num_objects."
            )
        _parse_num_objects(row, row_number)

    if detector_factory is None or nms_factory is None or image_reader is None:
        import cv2

        from detector_service.modules.inference.model import Detector
        from detector_service.modules.inference.nms import NMS

        detector_factory = detector_factory or Detector
        nms_factory = nms_factory or NMS
        image_reader = image_reader or _read_image_cv2
        if runtime_metadata is None:
            runtime_metadata = {"opencv_version": cv2.__version__}

    metadata = dict(runtime_metadata or {})
    metadata.setdefault("opencv_version", "unknown")
    metadata.setdefault("python_version", platform.python_version())
    metadata.setdefault("platform", platform.platform())

    pipelines = {}
    setup_seconds = {}
    for model_name, paths in models.items():
        assets = resolve_model_assets(paths, asset_root)
        setup_started = clock()
        detector = detector_factory(
            str(_filesystem_path(assets["weights"])),
            str(_filesystem_path(assets["cfg"])),
            str(_filesystem_path(assets["names"])),
            score_threshold=CANDIDATE_THRESHOLD,
        )
        nms = nms_factory(
            score_threshold=CONFIDENCE_THRESHOLD,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
        )
        setup_seconds[model_name] = clock() - setup_started
        pipelines[model_name] = (detector, nms)

    warmed_up = 0
    for row in sample_rows:
        if warmed_up >= warmup_images:
            break
        frame = image_reader(
            str(resolve_indexed_asset_path(asset_root, row["image_path"]))
        )
        if frame is None:
            raise RuntimeError(
                "Paired warm-up failed because an image was unreadable: "
                f"{row['image_path']}"
            )
        for model_name in models:
            _execute_pipeline(*pipelines[model_name], frame)
        warmed_up += 1
    if warmed_up != warmup_images:
        raise RuntimeError(
            f"Paired warm-up processed {warmed_up} images; expected "
            f"{warmup_images}."
        )

    observations = []
    successful_pairs = []
    start_offset = random.Random(seed).randrange(2)
    measurement_started = clock()
    pair_number = 0
    for repeat_index in range(1, repeats + 1):
        for sample_position, row in enumerate(sample_rows, start=1):
            image_path = resolve_indexed_asset_path(asset_root, row["image_path"])
            read_started = clock()
            frame = image_reader(str(image_path))
            read_finished = clock()
            read_seconds = read_finished - read_started
            bucket = density_bucket(row["num_objects"])
            model_order = ["model1", "model2"]
            if (pair_number + start_offset) % 2:
                model_order.reverse()
            pair_number += 1

            base = {
                "repeat_index": repeat_index,
                "sample_position": sample_position,
                "image_file": row["image_file"],
                "image_path": row["image_path"],
                "read_seconds": read_seconds,
                "benchmark_mode": "paired",
                "density_bucket": bucket,
            }
            if frame is None:
                raise RuntimeError(
                    "Paired measurement failed because an image was unreadable "
                    f"at repeat {repeat_index}: {row['image_path']}"
                )

            pair_results = {}
            for execution_order, model_name in enumerate(model_order, start=1):
                detector, nms = pipelines[model_name]
                predict_started = clock()
                outputs = detector.predict(frame)
                predict_finished = clock()
                postprocess_started = predict_finished
                decoded = detector.post_process(outputs)
                postprocess_finished = clock()
                nms_started = postprocess_finished
                filtered = nms.filter(*decoded)
                nms_finished = clock()
                compute_seconds = nms_finished - predict_started
                pair_results[model_name] = compute_seconds
                observations.append(
                    {
                        **base,
                        "model": model_name,
                        "status": "processed",
                        "detections": len(filtered[0]),
                        "predict_seconds": predict_finished - predict_started,
                        "postprocess_seconds": (
                            postprocess_finished - postprocess_started
                        ),
                        "nms_seconds": nms_finished - nms_started,
                        "total_seconds": compute_seconds,
                        "execution_order": execution_order,
                        "compute_seconds": compute_seconds,
                    }
                )

            model1_ms = pair_results["model1"] * 1000.0
            model2_ms = pair_results["model2"] * 1000.0
            delta_ms = model2_ms - model1_ms
            successful_pairs.append(
                {
                    "record_type": "pair",
                    "repeat_index": repeat_index,
                    "sample_position": sample_position,
                    "image_file": row["image_file"],
                    "image_path": row["image_path"],
                    "source_group": source_group_key(row["image_file"]),
                    "density_bucket": bucket,
                    "first_model": model_order[0],
                    "model1_compute_ms": model1_ms,
                    "model2_compute_ms": model2_ms,
                    "delta_model2_minus_model1_ms": delta_ms,
                    "faster_model": (
                        "model1" if delta_ms > 0 else "model2" if delta_ms < 0 else "tie"
                    ),
                }
            )

    measured_wall_seconds = clock() - measurement_started
    expected_pairs = len(sample_rows) * repeats
    expected_observations = expected_pairs * len(models)
    unique_processed = {row["image_file"] for row in successful_pairs}
    if len(successful_pairs) != expected_pairs:
        raise RuntimeError(
            f"Incomplete paired benchmark: expected {expected_pairs} pairs, "
            f"recorded {len(successful_pairs)}."
        )
    if len(observations) != expected_observations or any(
        row["status"] != "processed" for row in observations
    ):
        raise RuntimeError(
            "Incomplete paired benchmark observation ledger: expected "
            f"{expected_observations} processed rows."
        )
    if len(unique_processed) != len(sample_rows):
        raise RuntimeError(
            "Incomplete paired benchmark: not every unique sampled frame was "
            "successfully measured."
        )

    summaries = []
    for model_name in models:
        model_rows = [row for row in observations if row["model"] == model_name]
        successful = [row for row in model_rows if row["status"] == "processed"]
        unreadable = [row for row in model_rows if row["status"] == "unreadable"]
        compute_totals = [float(row["compute_seconds"]) for row in successful]
        mean_seconds = statistics.fmean(compute_totals)
        summaries.append(
            {
                "model": model_name,
                "images_in_dataset": dataset_size,
                "sample_requested": len(sample_rows),
                "unique_images_selected": len(sample_rows),
                "repeats": repeats,
                "warmup_images": warmed_up,
                "successful_observations": len(successful),
                "unreadable_observations": len(unreadable),
                "total_detections": sum(int(row["detections"]) for row in successful),
                "pipeline_setup_seconds": setup_seconds[model_name],
                "measured_wall_seconds": measured_wall_seconds,
                "mean_seconds_per_image": mean_seconds,
                "median_seconds_per_image": statistics.median(compute_totals),
                "p95_seconds_per_image": linear_percentile(compute_totals, 95),
                "images_per_second": 1.0 / mean_seconds if mean_seconds > 0 else float("inf"),
                "estimated_full_dataset_minutes": mean_seconds * dataset_size / 60.0,
                "candidate_threshold": CANDIDATE_THRESHOLD,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "nms_iou_threshold": NMS_IOU_THRESHOLD,
                "python_version": metadata["python_version"],
                "opencv_version": metadata["opencv_version"],
                "platform": metadata["platform"],
                "benchmark_mode": "paired",
                "seed": seed,
                "sample_selection": "density_stratified",
                "mean_compute_seconds": mean_seconds,
                "median_compute_seconds": statistics.median(compute_totals),
                "p95_compute_seconds": linear_percentile(compute_totals, 95),
            }
        )

    comparison_rows = build_paired_comparison_rows(
        successful_pairs,
        bootstrap_samples,
        seed,
    )
    return summaries, observations, comparison_rows


def _write_csv_atomic(path, fieldnames, rows):
    destination = _absolute_without_dereferencing(path)
    filesystem_destination = _filesystem_path(destination)
    filesystem_parent = _filesystem_path(destination.parent)
    filesystem_parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.parent / (
        f".{destination.name}.{secrets.token_hex(8)}.tmp"
    )
    try:
        with _filesystem_path(temporary_path).open(
            mode="x",
            encoding="utf-8",
            newline="",
        ) as temporary:
            writer = csv.DictWriter(temporary, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        _filesystem_path(temporary_path).replace(filesystem_destination)
    except Exception:
        _filesystem_path(temporary_path).unlink(missing_ok=True)
        raise


def _write_json_atomic(path, payload):
    destination = _absolute_without_dereferencing(path)
    filesystem_destination = _filesystem_path(destination)
    filesystem_parent = _filesystem_path(destination.parent)
    filesystem_parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.parent / (
        f".{destination.name}.{secrets.token_hex(8)}.tmp"
    )
    try:
        with _filesystem_path(temporary_path).open(
            mode="x",
            encoding="utf-8",
            newline="\n",
        ) as temporary:
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
        _filesystem_path(temporary_path).replace(filesystem_destination)
    except Exception:
        _filesystem_path(temporary_path).unlink(missing_ok=True)
        raise


def _fingerprinted_file(path):
    source = _absolute_without_dereferencing(path)
    filesystem = _filesystem_path(source)
    if not filesystem.is_file():
        raise FileNotFoundError(f"Fingerprint input not found: {source}")
    return {
        "path": str(source),
        "bytes": filesystem.stat().st_size,
        "sha256": _sha256_file(source),
    }


def build_paired_input_provenance(
    dataset_index,
    sample_rows,
    dataset_size,
    asset_root,
    models=MODELS,
):
    """Fingerprint the exact index, ordered sample, and checkpoint bundles."""

    index_path = _absolute_without_dereferencing(dataset_index)
    index_filesystem = _filesystem_path(index_path)
    if not index_filesystem.is_file():
        raise FileNotFoundError(f"Dataset index not found: {index_path}")
    with index_filesystem.open(newline="", encoding="utf-8") as source:
        indexed_rows = sum(1 for _ in csv.DictReader(source))
    if indexed_rows != dataset_size:
        raise ValueError(
            f"Dataset size mismatch: caller reported {dataset_size}, but "
            f"{index_path} contains {indexed_rows} rows."
        )
    normalized_sample = [
        {str(key): str(value) for key, value in sorted(row.items())}
        for row in sample_rows
    ]
    model_records = {}
    for model_name, paths in models.items():
        resolved = resolve_model_assets(paths, asset_root)
        model_records[model_name] = {
            name: _fingerprinted_file(path)
            for name, path in sorted(resolved.items())
        }
    return {
        "dataset": {
            "index": _fingerprinted_file(index_path),
            "images_in_index": int(dataset_size),
            "ordered_sample_images": len(sample_rows),
            "ordered_sample_sha256": _sha256_json(normalized_sample),
            "source_groups": len(
                {source_group_key(row["image_file"]) for row in sample_rows}
            ),
            "source_group_policy": (
                "prefix before '_jpg.rf.'; otherwise complete image_file"
            ),
        },
        "models": model_records,
    }


def build_paired_manifest(
    *,
    input_provenance,
    output_dir,
    summaries,
    observations,
    comparison_rows,
    summary_path,
    observations_path,
    comparison_path,
    repeats,
    warmup_images,
    seed,
    bootstrap_samples,
    artifact_identity_root=None,
):
    """Build a complete evidence manifest after all paired CSVs exist."""

    pair_rows = [row for row in comparison_rows if row["record_type"] == "pair"]
    aggregate_rows = [
        row for row in comparison_rows if row["record_type"] == "aggregate"
    ]
    if len(aggregate_rows) != 1:
        raise RuntimeError("Paired comparison must contain exactly one aggregate row.")
    runtime_records = {
        (
            str(summary["python_version"]),
            str(summary["opencv_version"]),
            str(summary["platform"]),
        )
        for summary in summaries
    }
    if len(runtime_records) != 1:
        raise RuntimeError("Checkpoint summaries disagree on runtime metadata.")
    python_version, opencv_version, platform_name = runtime_records.pop()
    expected_unique = input_provenance["dataset"]["ordered_sample_images"]
    expected_pairs = expected_unique * repeats
    expected_observations = expected_pairs * 2
    completeness = {
        "expected_unique_images": expected_unique,
        "measured_unique_images": len({row["image_file"] for row in pair_rows}),
        "expected_pairs": expected_pairs,
        "measured_pairs": len(pair_rows),
        "expected_observations": expected_observations,
        "processed_observations": sum(
            row["status"] == "processed" for row in observations
        ),
        "unreadable_observations": sum(
            row["status"] == "unreadable" for row in observations
        ),
    }
    if completeness != {
        "expected_unique_images": expected_unique,
        "measured_unique_images": expected_unique,
        "expected_pairs": expected_pairs,
        "measured_pairs": expected_pairs,
        "expected_observations": expected_observations,
        "processed_observations": expected_observations,
        "unreadable_observations": 0,
    }:
        raise RuntimeError(f"Paired manifest completeness check failed: {completeness}")

    artifacts = {
        "summary": {
            **_fingerprinted_file(summary_path),
            "rows": len(summaries),
        },
        "observations": {
            **_fingerprinted_file(observations_path),
            "rows": len(observations),
        },
        "paired_comparison": {
            **_fingerprinted_file(comparison_path),
            "rows": len(comparison_rows),
        },
    }
    if artifact_identity_root is not None:
        identity_root = _absolute_without_dereferencing(artifact_identity_root)
        for record in artifacts.values():
            record["path"] = str(identity_root / Path(record["path"]).name)
    core = {
        "schema_version": 1,
        "status": "complete",
        "benchmark_mode": "paired",
        "output_directory": str(_absolute_without_dereferencing(output_dir)),
        **input_provenance,
        "runtime": {
            "python_version": python_version,
            "opencv_version": opencv_version,
            "platform": platform_name,
        },
        "policy": {
            "candidate_objectness_threshold": CANDIDATE_THRESHOLD,
            "combined_confidence_threshold": CONFIDENCE_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "sample_selection": "seeded_density_stratified",
            "sample_seed": seed,
            "repeats": repeats,
            "warmup_images_per_model": warmup_images,
            "timing_scope": "predict + post_process + class-aware NMS",
            "frame_decode_timed_separately": True,
            "execution_order": "seeded alternating first checkpoint",
            "p95_estimator": "linear interpolation",
            "bootstrap_unit": "source group preserving variants and repeats",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
        },
        "completeness": completeness,
        "artifacts": artifacts,
    }
    return {
        **core,
        "run_fingerprint_sha256": _sha256_json(core),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_benchmark_artifacts(output_dir, summaries, observations):
    destination = _absolute_without_dereferencing(output_dir)
    summary_path = destination / "inference_benchmark_summary.csv"
    observations_path = destination / "inference_benchmark_observations.csv"
    _write_csv_atomic(summary_path, SUMMARY_COLUMNS, summaries)
    _write_csv_atomic(observations_path, OBSERVATION_COLUMNS, observations)
    return summary_path, observations_path


def write_paired_comparison_artifact(output_dir, comparison_rows):
    destination = _absolute_without_dereferencing(output_dir)
    comparison_path = destination / "paired_latency_comparison.csv"
    _write_csv_atomic(
        comparison_path,
        PAIRED_COMPARISON_COLUMNS,
        comparison_rows,
    )
    return comparison_path


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


def run_paired_benchmark(
    sample_rows,
    dataset_size,
    asset_root,
    dataset_index,
    output_dir,
    repeats,
    warmup_images,
    seed=DEFAULT_SEED,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    models=MODELS,
    **dependencies,
):
    destination = _absolute_without_dereferencing(output_dir)
    staging = destination.parent / f".{destination.name}.incomplete"
    if _filesystem_path(destination).exists():
        raise FileExistsError(
            "Paired benchmark output is immutable; choose a new output "
            f"directory. Existing run: {destination}"
        )
    if _filesystem_path(staging).exists():
        raise FileExistsError(
            "An incomplete paired benchmark package already exists; inspect "
            f"or move it before retrying: {staging}"
        )
    input_provenance = build_paired_input_provenance(
        dataset_index=dataset_index,
        sample_rows=sample_rows,
        dataset_size=dataset_size,
        asset_root=asset_root,
        models=models,
    )
    print("=" * 80)
    print("Benchmarking model1 and model2 in paired mode")
    print("=" * 80)
    summaries, observations, comparison_rows = benchmark_paired(
        sample_rows=sample_rows,
        dataset_size=dataset_size,
        asset_root=asset_root,
        repeats=repeats,
        warmup_images=warmup_images,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        models=models,
        **dependencies,
    )
    provenance_after = build_paired_input_provenance(
        dataset_index=dataset_index,
        sample_rows=sample_rows,
        dataset_size=dataset_size,
        asset_root=asset_root,
        models=models,
    )
    if provenance_after != input_provenance:
        raise RuntimeError(
            "Dataset index or checkpoint assets changed while the paired "
            "benchmark was running; no artifacts were written."
        )
    summary_path, observations_path = write_benchmark_artifacts(
        staging,
        summaries,
        observations,
    )
    comparison_path = write_paired_comparison_artifact(
        staging,
        comparison_rows,
    )
    manifest = build_paired_manifest(
        input_provenance=input_provenance,
        output_dir=destination,
        summaries=summaries,
        observations=observations,
        comparison_rows=comparison_rows,
        summary_path=summary_path,
        observations_path=observations_path,
        comparison_path=comparison_path,
        repeats=repeats,
        warmup_images=warmup_images,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        artifact_identity_root=destination,
    )
    manifest_path = staging / PAIRED_MANIFEST_NAME
    _write_json_atomic(manifest_path, manifest)
    expected_names = {
        "inference_benchmark_summary.csv",
        "inference_benchmark_observations.csv",
        "paired_latency_comparison.csv",
        PAIRED_MANIFEST_NAME,
    }
    actual_names = {
        path.name for path in _filesystem_path(staging).iterdir() if path.is_file()
    }
    if actual_names != expected_names:
        raise RuntimeError(
            "Incomplete paired benchmark package; refusing promotion: "
            f"expected={sorted(expected_names)}, actual={sorted(actual_names)}"
        )
    os.replace(_filesystem_path(staging), _filesystem_path(destination))
    summary_path = destination / summary_path.name
    observations_path = destination / observations_path.name
    comparison_path = destination / comparison_path.name
    manifest_path = destination / manifest_path.name
    aggregate = comparison_rows[-1]
    print(
        "[RESULT] Paired compute delta (model2-model1): "
        f"mean={float(aggregate['mean_delta_model2_minus_model1_ms']):.3f} ms, "
        "95% source-group bootstrap CI="
        f"[{float(aggregate['mean_delta_ci_lower_ms']):.3f}, "
        f"{float(aggregate['mean_delta_ci_upper_ms']):.3f}] ms"
    )
    print(f"[WRITE] Benchmark summary: {summary_path}")
    print(f"[WRITE] Benchmark observations: {observations_path}")
    print(f"[WRITE] Paired comparison: {comparison_path}")
    print(f"[WRITE] Paired manifest: {manifest_path}")
    return (
        summaries,
        observations,
        comparison_rows,
        summary_path,
        observations_path,
        comparison_path,
        manifest_path,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark both detector checkpoints on a deterministic sample."
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=PROJECT_ROOT,
        help=(
            "External storage directory. Indexed detector_service/storage "
            "paths are resolved beneath this directory; storage-relative "
            "paths are also supported."
        ),
    )
    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=DEFAULT_DATASET_INDEX,
        help="Dataset index produced by 00_build_dataset_inventory.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Parent directory for immutable benchmark runs. The completed "
            "package is written beneath --run-id."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=validate_run_id,
        required=True,
        help="Safe child-directory name for this benchmark package.",
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
    parser.add_argument(
        "--paired",
        action="store_true",
        help=(
            "Decode each sampled image once, benchmark both checkpoints on the "
            "same frame, and write paired latency evidence."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for paired density sampling, model order, and bootstrap.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=positive_int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Paired source-group bootstrap replicates.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    asset_root = _absolute_without_dereferencing(args.asset_root)
    if not _filesystem_path(asset_root).is_dir():
        raise NotADirectoryError(f"Asset root does not exist: {asset_root}")

    dataset_index = args.dataset_index
    if not dataset_index.is_absolute():
        dataset_index = PROJECT_ROOT / dataset_index
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_dir = output_root / args.run_id
    if _filesystem_path(output_dir).exists():
        raise FileExistsError(
            f"Refusing to overwrite benchmark run directory: {output_dir}"
        )

    sample_rows, dataset_size = load_dataset_sample(
        dataset_index,
        args.sample_size,
        paired=args.paired,
        seed=args.seed,
    )
    print(f"[INFO] Dataset images: {dataset_size}")
    print(f"[INFO] Deterministic sample images: {len(sample_rows)}")
    print(f"[INFO] Repeats: {args.repeats}")
    print(f"[INFO] Warm-up images per model: {args.warmup_images}")
    print(f"[INFO] Mode: {'paired' if args.paired else 'independent'}")
    if args.paired:
        print(f"[INFO] Seed: {args.seed}")
        print(f"[INFO] Bootstrap samples: {args.bootstrap_samples}")
    print(f"[INFO] Asset root: {asset_root}")
    print(
        "[INFO] Pipeline policy: "
        f"objectness>{CANDIDATE_THRESHOLD}, "
        f"combined confidence>={CONFIDENCE_THRESHOLD}, "
        f"class-aware NMS IoU={NMS_IOU_THRESHOLD}"
    )

    runner = run_paired_benchmark if args.paired else run_benchmark
    runner_arguments = {
        "sample_rows": sample_rows,
        "dataset_size": dataset_size,
        "asset_root": asset_root,
        "output_dir": output_dir,
        "repeats": args.repeats,
        "warmup_images": args.warmup_images,
    }
    if args.paired:
        runner_arguments.update(
            dataset_index=dataset_index,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
    return runner(**runner_arguments)


if __name__ == "__main__":
    main()
