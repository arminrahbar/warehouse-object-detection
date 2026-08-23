"""Build the stage-00 dataset inventory from paired YOLO images and labels."""

import argparse
import csv
import json
import math
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_RELATIVE = Path("detector_service/storage/logistics")
DEFAULT_CLASSES_RELATIVE = Path(
    "detector_service/storage/yolo_model_1/logistics.names"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "outputs" / "00_dataset_inventory"
)

BASE_INDEX_COLUMNS = [
    "image_path",
    "label_path",
    "image_file",
    "label_file",
    "num_objects",
    "class_ids_present",
    "class_names_present",
]
CLASS_SUMMARY_COLUMNS = [
    "class_id",
    "class_name",
    "object_count",
    "image_count",
]
OBJECT_DISTRIBUTION_COLUMNS = ["num_objects", "image_count"]


@dataclass(frozen=True)
class IndexBuildResult:
    dataset_index_path: Path
    class_distribution_path: Path
    object_count_distribution_path: Path
    images_discovered: int
    images_indexed: int
    missing_labels: int
    invalid_annotations: int
    total_objects: int


def clean_column_name(name):
    """Convert a display class name into its stable count-column suffix."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).lower())
    return normalized.strip("_")


def load_classes(path):
    """Read an ordered class vocabulary without allowing ID renumbering."""

    class_path = Path(path)
    if not class_path.is_file():
        raise FileNotFoundError(f"Class names file not found: {class_path}")

    lines = class_path.read_text(encoding="utf-8").splitlines()
    if not lines or all(not line.strip() for line in lines):
        raise ValueError(f"Class names file is empty: {class_path}")
    blank_lines = [index for index, line in enumerate(lines, start=1) if not line.strip()]
    if blank_lines:
        rendered = ", ".join(str(index) for index in blank_lines)
        raise ValueError(
            "Class names file contains blank entries that would renumber "
            f"class IDs (lines: {rendered}): {class_path}"
        )
    classes = [line.strip() for line in lines]

    suffixes = [clean_column_name(name) for name in classes]
    if any(not suffix for suffix in suffixes):
        raise ValueError("Every class name must contain at least one letter or digit.")

    duplicate_suffixes = sorted(
        suffix for suffix, count in Counter(suffixes).items() if count > 1
    )
    if duplicate_suffixes:
        details = ", ".join(duplicate_suffixes)
        raise ValueError(f"Class names produce duplicate count columns: {details}")

    return classes


def _annotation_error(label_path, line_number, reason, strict):
    message = f"{label_path}:{line_number}: {reason}"
    if strict:
        raise ValueError(message)
    return message


def parse_label_file(path, class_count, strict=True):
    """Count valid YOLO rows and report rows excluded by validation."""

    label_path = Path(path)
    counts = Counter()
    errors = []

    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) != 5:
            errors.append(
                _annotation_error(
                    label_path,
                    line_number,
                    "expected exactly five YOLO fields",
                    strict,
                )
            )
            continue

        try:
            values = [float(value) for value in parts]
        except ValueError:
            errors.append(
                _annotation_error(
                    label_path,
                    line_number,
                    "all five YOLO fields must be numeric",
                    strict,
                )
            )
            continue

        if not all(math.isfinite(value) for value in values):
            errors.append(
                _annotation_error(
                    label_path,
                    line_number,
                    "YOLO fields must be finite",
                    strict,
                )
            )
            continue

        raw_class_id, center_x, center_y, width, height = values
        class_id = int(raw_class_id)
        if raw_class_id != class_id:
            errors.append(
                _annotation_error(
                    label_path,
                    line_number,
                    "class identifier must be integer-valued",
                    strict,
                )
            )
            continue
        if not 0 <= class_id < class_count:
            errors.append(
                _annotation_error(
                    label_path,
                    line_number,
                    f"class identifier {class_id} is outside [0, {class_count})",
                    strict,
                )
            )
            continue
        if not (
            0.0 <= center_x <= 1.0
            and 0.0 <= center_y <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            errors.append(
                _annotation_error(
                    label_path,
                    line_number,
                    "box coordinates must be normalized YOLO values",
                    strict,
                )
            )
            continue

        counts[class_id] += 1

    return counts, errors


def _portable_path(path, asset_root):
    absolute_path = Path(path).expanduser().absolute()
    absolute_root = Path(asset_root).expanduser().absolute()
    try:
        return absolute_path.relative_to(absolute_root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Asset path is outside the declared asset root {absolute_root}: "
            f"{absolute_path}"
        ) from error


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


def build_dataset_index(
    dataset_dir,
    classes_path,
    output_dir,
    asset_root,
    strict=True,
):
    """Build all index artifacts and return their paths and validation counts."""

    # Keep logical paths intact: resolving here would dereference an external
    # storage symlink and leak host-specific paths into the portable index.
    dataset_path = Path(dataset_dir).expanduser().absolute()
    class_names_path = Path(classes_path).expanduser().absolute()
    destination = Path(output_dir).expanduser().absolute()
    root = Path(asset_root).expanduser().absolute()

    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {dataset_path}")

    classes = load_classes(class_names_path)
    count_columns = [f"count_{clean_column_name(name)}" for name in classes]
    image_paths = sorted(dataset_path.glob("*.jpg"), key=lambda path: path.name)

    rows = []
    object_counter = Counter()
    image_counter = Counter()
    object_count_distribution = Counter()
    missing_labels = []
    annotation_errors = []

    for image_path in image_paths:
        label_path = image_path.with_suffix(".txt")
        if not label_path.is_file():
            missing_labels.append(image_path)
            if strict:
                raise FileNotFoundError(
                    f"Label file not found for image {image_path}: {label_path}"
                )
            continue

        class_counts, errors = parse_label_file(
            label_path,
            len(classes),
            strict=strict,
        )
        annotation_errors.extend(errors)
        total_objects = sum(class_counts.values())

        object_counter.update(class_counts)
        for class_id in class_counts:
            image_counter[class_id] += 1
        object_count_distribution[total_objects] += 1

        class_ids = sorted(class_counts)
        row = {
            "image_path": _portable_path(image_path, root),
            "label_path": _portable_path(label_path, root),
            "image_file": image_path.name,
            "label_file": label_path.name,
            "num_objects": total_objects,
            "class_ids_present": json.dumps(class_ids),
            "class_names_present": json.dumps(
                [classes[class_id] for class_id in class_ids]
            ),
        }
        row.update(
            {
                column: class_counts.get(class_id, 0)
                for class_id, column in enumerate(count_columns)
            }
        )
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No image-label pairs were indexed from {dataset_path}")

    class_rows = [
        {
            "class_id": class_id,
            "class_name": class_name,
            "object_count": object_counter[class_id],
            "image_count": image_counter[class_id],
        }
        for class_id, class_name in enumerate(classes)
    ]
    distribution_rows = [
        {"num_objects": num_objects, "image_count": image_count}
        for num_objects, image_count in sorted(object_count_distribution.items())
    ]

    dataset_index_path = destination / "dataset_index.csv"
    class_distribution_path = destination / "class_distribution.csv"
    object_count_distribution_path = destination / "object_count_distribution.csv"

    _write_csv_atomic(
        dataset_index_path,
        BASE_INDEX_COLUMNS + count_columns,
        rows,
    )
    _write_csv_atomic(
        class_distribution_path,
        CLASS_SUMMARY_COLUMNS,
        class_rows,
    )
    _write_csv_atomic(
        object_count_distribution_path,
        OBJECT_DISTRIBUTION_COLUMNS,
        distribution_rows,
    )

    return IndexBuildResult(
        dataset_index_path=dataset_index_path,
        class_distribution_path=class_distribution_path,
        object_count_distribution_path=object_count_distribution_path,
        images_discovered=len(image_paths),
        images_indexed=len(rows),
        missing_labels=len(missing_labels),
        invalid_annotations=len(annotation_errors),
        total_objects=sum(object_counter.values()),
    )


def _path_from_root(value, asset_root, default_relative):
    path = Path(value).expanduser() if value is not None else Path(default_relative)
    return path if path.is_absolute() else asset_root / path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Index paired JPEG images and YOLO label files."
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root used to resolve assets and serialize portable relative paths.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Dataset directory, absolute or relative to --asset-root. "
            "Defaults to detector_service/storage/logistics."
        ),
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=None,
        help=(
            "Class-name file, absolute or relative to --asset-root. Defaults to "
            "detector_service/storage/yolo_model_1/logistics.names."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the three CSV artifacts.",
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help=(
            "Diagnostic mode: skip missing labels and invalid annotation rows. "
            "Strict validation is the default for evidence-producing runs."
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    asset_root = args.asset_root.expanduser().resolve()
    if not asset_root.is_dir():
        raise NotADirectoryError(f"Asset root does not exist: {asset_root}")

    dataset_dir = _path_from_root(
        args.dataset_dir,
        asset_root,
        DEFAULT_DATASET_RELATIVE,
    )
    classes_path = _path_from_root(
        args.classes,
        asset_root,
        DEFAULT_CLASSES_RELATIVE,
    )
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    result = build_dataset_index(
        dataset_dir=dataset_dir,
        classes_path=classes_path,
        output_dir=output_dir,
        asset_root=asset_root,
        strict=not args.allow_invalid,
    )

    print(f"[WRITE] Dataset index: {result.dataset_index_path}")
    print(f"[WRITE] Class distribution: {result.class_distribution_path}")
    print(
        "[WRITE] Object-count distribution: "
        f"{result.object_count_distribution_path}"
    )
    print(f"[INFO] Images discovered: {result.images_discovered}")
    print(f"[INFO] Images indexed: {result.images_indexed}")
    print(f"[INFO] Missing labels: {result.missing_labels}")
    print(f"[INFO] Invalid annotations skipped: {result.invalid_annotations}")
    print(f"[INFO] Total labeled objects: {result.total_objects}")
    return result


if __name__ == "__main__":
    main()
