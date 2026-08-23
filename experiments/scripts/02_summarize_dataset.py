"""Validate dataset-index artifacts and derive characterization summaries."""

import argparse
import csv
import math
import re
import statistics
import tempfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY_DIR = (
    PROJECT_ROOT / "experiments" / "outputs" / "00_dataset_inventory"
)
DEFAULT_ANALYSIS_ROOT = (
    PROJECT_ROOT / "experiments" / "outputs" / "02_dataset_analysis"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ANALYSIS_ROOT / "01_dataset_summary"
DEFAULT_DATASET_INDEX = DEFAULT_INVENTORY_DIR / "dataset_index.csv"
DEFAULT_CLASS_DISTRIBUTION = DEFAULT_INVENTORY_DIR / "class_distribution.csv"
DEFAULT_OBJECT_DISTRIBUTION = (
    DEFAULT_INVENTORY_DIR / "object_count_distribution.csv"
)

DENSITY_BUCKET_ORDER = ["1", "2-4", "5-9", "10-14", "15-19", "20+"]
DENSE_THRESHOLDS = [5, 10, 15, 20]

SUMMARY_COLUMNS = [
    "dataset",
    "images",
    "total_objects",
    "images_with_zero_objects",
    "mean_objects_per_image",
    "median_objects_per_image",
    "max_objects_per_image",
    "images_ge_5_objects",
    "images_ge_10_objects",
    "images_ge_15_objects",
    "images_ge_20_objects",
]
CLASS_COLUMNS = ["class_id", "class_name", "object_count", "image_count"]
ENRICHED_CLASS_COLUMNS = CLASS_COLUMNS + ["object_share_pct", "image_share_pct"]
DENSITY_COLUMNS = ["density_bucket", "image_count", "image_share_pct"]
DENSE_COUNT_COLUMNS = ["threshold", "image_count", "image_share_pct"]


def clean_column_name(name):
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).lower())
    return normalized.strip("_")


def density_bucket(object_count):
    """Map an object count to the stable downstream density labels."""

    count = int(object_count)
    if count < 0:
        raise ValueError("Object count must be non-negative.")
    if count <= 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 9:
        return "5-9"
    if count <= 14:
        return "10-14"
    if count <= 19:
        return "15-19"
    return "20+"


def _read_csv(path, label):
    source_path = Path(path).expanduser().absolute()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"{label} not found: {source_path}. "
            "Run experiments/scripts/00_build_dataset_inventory.py first."
        )
    with source_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        return reader.fieldnames or [], list(reader)


def _require_columns(actual, required, label):
    missing = [column for column in required if column not in actual]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _nonnegative_integer(value, context):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric.") from error
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite.")
    integer = int(number)
    if number != integer or integer < 0:
        raise ValueError(f"{context} must be a non-negative integer.")
    return integer


def load_and_validate_sources(
    dataset_index_path,
    class_distribution_path,
    object_distribution_path,
):
    """Load all source tables and reconcile their independent aggregates."""

    index_fields, index_rows = _read_csv(dataset_index_path, "Dataset index")
    class_fields, class_rows = _read_csv(
        class_distribution_path,
        "Class distribution",
    )
    distribution_fields, distribution_rows = _read_csv(
        object_distribution_path,
        "Object-count distribution",
    )

    _require_columns(index_fields, ["image_file", "num_objects"], "Dataset index")
    _require_columns(class_fields, CLASS_COLUMNS, "Class distribution")
    _require_columns(
        distribution_fields,
        ["num_objects", "image_count"],
        "Object-count distribution",
    )
    if not index_rows:
        raise RuntimeError("Dataset index is empty.")
    if not class_rows:
        raise RuntimeError("Class distribution is empty.")
    if not distribution_rows:
        raise RuntimeError("Object-count distribution is empty.")

    image_files = [row["image_file"] for row in index_rows]
    if len(image_files) != len(set(image_files)):
        raise ValueError("Dataset index contains duplicate image_file values.")

    parsed_classes = []
    seen_names = set()
    seen_columns = set()
    for expected_id, row in enumerate(class_rows):
        class_id = _nonnegative_integer(row["class_id"], "class_id")
        if class_id != expected_id:
            raise ValueError("Class identifiers must be contiguous and zero-based.")
        class_name = row["class_name"].strip()
        if not class_name:
            raise ValueError(f"Class {class_id} has an empty name.")
        if class_name in seen_names:
            raise ValueError(f"Duplicate class name: {class_name}")
        seen_names.add(class_name)

        column_suffix = clean_column_name(class_name)
        if not column_suffix:
            raise ValueError(f"Class {class_id} cannot form a count column.")
        count_column = f"count_{column_suffix}"
        if count_column in seen_columns:
            raise ValueError(f"Duplicate class count column: {count_column}")
        seen_columns.add(count_column)
        if count_column not in index_fields:
            raise ValueError(f"Dataset index is missing required column: {count_column}")

        parsed_classes.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "count_column": count_column,
                "object_count": _nonnegative_integer(
                    row["object_count"],
                    f"object_count for class {class_id}",
                ),
                "image_count": _nonnegative_integer(
                    row["image_count"],
                    f"image_count for class {class_id}",
                ),
            }
        )

    object_counts = []
    class_object_totals = [0] * len(parsed_classes)
    class_image_totals = [0] * len(parsed_classes)
    observed_distribution = Counter()

    for row_number, row in enumerate(index_rows, start=2):
        count_values = []
        for class_info in parsed_classes:
            count = _nonnegative_integer(
                row[class_info["count_column"]],
                f"{class_info['count_column']} at index row {row_number}",
            )
            count_values.append(count)

        num_objects = _nonnegative_integer(
            row["num_objects"],
            f"num_objects at index row {row_number}",
        )
        if num_objects != sum(count_values):
            raise ValueError(
                f"Index row {row_number} num_objects does not equal its class-count sum."
            )

        object_counts.append(num_objects)
        observed_distribution[num_objects] += 1
        for class_id, count in enumerate(count_values):
            class_object_totals[class_id] += count
            class_image_totals[class_id] += int(count > 0)

    for class_info, objects, images in zip(
        parsed_classes,
        class_object_totals,
        class_image_totals,
    ):
        if class_info["object_count"] != objects:
            raise ValueError(
                f"Object total mismatch for class {class_info['class_id']} "
                f"({class_info['class_name']})."
            )
        if class_info["image_count"] != images:
            raise ValueError(
                f"Image total mismatch for class {class_info['class_id']} "
                f"({class_info['class_name']})."
            )

    supplied_distribution = {}
    for row in distribution_rows:
        num_objects = _nonnegative_integer(row["num_objects"], "distribution num_objects")
        image_count = _nonnegative_integer(row["image_count"], "distribution image_count")
        if num_objects in supplied_distribution:
            raise ValueError(f"Duplicate object-count distribution row: {num_objects}")
        supplied_distribution[num_objects] = image_count
    if supplied_distribution != dict(sorted(observed_distribution.items())):
        raise ValueError("Object-count distribution does not match the dataset index.")

    return index_rows, parsed_classes, object_counts


def build_summary_tables(index_rows, parsed_classes, object_counts):
    """Build the six stable characterization artifacts in memory."""

    image_count = len(index_rows)
    total_objects = sum(object_counts)
    summary = [
        {
            "dataset": "full_dataset",
            "images": image_count,
            "total_objects": total_objects,
            "images_with_zero_objects": sum(count == 0 for count in object_counts),
            "mean_objects_per_image": round(statistics.fmean(object_counts), 3),
            "median_objects_per_image": float(statistics.median(object_counts)),
            "max_objects_per_image": max(object_counts),
            "images_ge_5_objects": sum(count >= 5 for count in object_counts),
            "images_ge_10_objects": sum(count >= 10 for count in object_counts),
            "images_ge_15_objects": sum(count >= 15 for count in object_counts),
            "images_ge_20_objects": sum(count >= 20 for count in object_counts),
        }
    ]

    enriched_classes = []
    for class_info in parsed_classes:
        enriched_classes.append(
            {
                "class_id": class_info["class_id"],
                "class_name": class_info["class_name"],
                "object_count": class_info["object_count"],
                "image_count": class_info["image_count"],
                "object_share_pct": round(
                    100.0 * class_info["object_count"] / total_objects,
                    4,
                )
                if total_objects
                else 0.0,
                "image_share_pct": round(
                    100.0 * class_info["image_count"] / image_count,
                    4,
                ),
            }
        )

    top_classes = sorted(
        enriched_classes,
        key=lambda row: (-row["object_count"], row["class_id"]),
    )[:10]
    bottom_classes = sorted(
        enriched_classes,
        key=lambda row: (row["object_count"], row["class_id"]),
    )[:10]

    bucket_counts = Counter(density_bucket(count) for count in object_counts)
    density_summary = [
        {
            "density_bucket": bucket,
            "image_count": bucket_counts[bucket],
            "image_share_pct": round(100.0 * bucket_counts[bucket] / image_count, 4),
        }
        for bucket in DENSITY_BUCKET_ORDER
    ]

    dense_counts = []
    for threshold in DENSE_THRESHOLDS:
        count = sum(value >= threshold for value in object_counts)
        dense_counts.append(
            {
                "threshold": f">={threshold} objects",
                "image_count": count,
                "image_share_pct": round(100.0 * count / image_count, 4),
            }
        )

    return {
        "full_dataset_summary.csv": (SUMMARY_COLUMNS, summary),
        "class_distribution_enriched.csv": (
            ENRICHED_CLASS_COLUMNS,
            enriched_classes,
        ),
        "top10_classes_by_object_count.csv": (
            ENRICHED_CLASS_COLUMNS,
            top_classes,
        ),
        "bottom10_classes_by_object_count.csv": (
            ENRICHED_CLASS_COLUMNS,
            bottom_classes,
        ),
        "density_bucket_distribution.csv": (DENSITY_COLUMNS, density_summary),
        "dense_image_counts.csv": (DENSE_COUNT_COLUMNS, dense_counts),
    }


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


def write_summary_tables(output_dir, tables):
    destination = Path(output_dir).expanduser().absolute()
    paths = []
    for filename, (fieldnames, rows) in tables.items():
        path = destination / filename
        _write_csv_atomic(path, fieldnames, rows)
        paths.append(path)
    return paths


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate and summarize dataset-index artifacts."
    )
    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=DEFAULT_DATASET_INDEX,
    )
    parser.add_argument(
        "--class-distribution",
        type=Path,
        default=DEFAULT_CLASS_DISTRIBUTION,
    )
    parser.add_argument(
        "--object-count-distribution",
        type=Path,
        default=DEFAULT_OBJECT_DISTRIBUTION,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser


def _project_relative(path):
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main(argv=None):
    args = build_parser().parse_args(argv)
    index_rows, parsed_classes, object_counts = load_and_validate_sources(
        _project_relative(args.dataset_index),
        _project_relative(args.class_distribution),
        _project_relative(args.object_count_distribution),
    )
    tables = build_summary_tables(index_rows, parsed_classes, object_counts)
    output_paths = write_summary_tables(
        _project_relative(args.output_dir),
        tables,
    )

    summary = tables["full_dataset_summary.csv"][1][0]
    print("DATASET SUMMARY")
    print("---------------")
    for key in SUMMARY_COLUMNS:
        print(f"{key}: {summary[key]}")

    print("\nTOP 10 CLASSES BY OBJECT COUNT")
    for row in tables["top10_classes_by_object_count.csv"][1]:
        print(
            f"{row['class_id']:>2} {row['class_name']}: "
            f"objects={row['object_count']} images={row['image_count']}"
        )

    print("\nDENSITY BUCKET DISTRIBUTION")
    for row in tables["density_bucket_distribution.csv"][1]:
        print(
            f"{row['density_bucket']:>5}: images={row['image_count']} "
            f"share={row['image_share_pct']:.4f}%"
        )

    for path in output_paths:
        print(f"[WRITE] {path}")
    return tables, output_paths


if __name__ == "__main__":
    main()
