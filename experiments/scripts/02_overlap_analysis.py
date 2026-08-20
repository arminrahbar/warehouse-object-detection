"""Measure ground-truth box overlap in the full dataset and selected sample."""

import argparse
import math
import sys
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector_service.modules.utils.metrics import calculate_iou


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "outputs"
DEFAULT_SAMPLING_DIR = DEFAULT_OUTPUT_ROOT / "dataset_sampling"
DEFAULT_DATASET_INDEX = DEFAULT_OUTPUT_ROOT / "dataset_index.csv"
DEFAULT_SELECTED_INDEX = DEFAULT_SAMPLING_DIR / "selected_sample_index.csv"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "experiments" / "figures" / "02_dataset_sampling"

IOU_THRESHOLDS = (0.1, 0.3, 0.5)
CROWDING_BUCKET_ORDER = ("0", "1-4", "5-19", "20+")
PROFILE_COLUMNS = [
    "image_file",
    "image_path",
    "label_path",
    "num_objects",
    "pair_count",
    "max_pairwise_iou",
    "mean_pairwise_iou",
    "pairs_iou_gt_0_1",
    "pairs_iou_gt_0_3",
    "pairs_iou_gt_0_5",
    "crowding_bucket",
]
SUMMARY_COLUMNS = [
    "dataset",
    "images",
    "mean_pair_count",
    "mean_max_pairwise_iou",
    "mean_pairs_iou_gt_0_1",
    "images_with_any_iou_gt_0_1",
    "images_with_any_iou_gt_0_3",
    "images_with_any_iou_gt_0_5",
    "images_with_20plus_iou_gt_0_1_pairs",
]
CROWDING_COMPARISON_COLUMNS = [
    "dataset_full",
    "crowding_bucket",
    "image_count_full",
    "image_share_pct_full",
    "dataset_sample",
    "image_count_sample",
    "image_share_pct_sample",
    "image_share_diff_pp",
]


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def yolo_to_xywh(center_x, center_y, width, height):
    """Convert normalized YOLO center geometry to top-left ``xywh``."""

    cx, cy, box_width, box_height = map(
        float,
        (center_x, center_y, width, height),
    )
    return [cx - box_width / 2.0, cy - box_height / 2.0, box_width, box_height]


def crowding_bucket(pairs_above_point_one):
    """Map a non-negative overlapping-pair count to its reporting bucket."""

    count = int(pairs_above_point_one)
    if count != pairs_above_point_one or count < 0:
        raise ValueError("Overlapping-pair count must be a non-negative integer.")
    if count == 0:
        return "0"
    if count <= 4:
        return "1-4"
    if count <= 19:
        return "5-19"
    return "20+"


def parse_yolo_boxes(label_path):
    """Read validated normalized YOLO rows and return top-left ``xywh`` boxes."""

    path = Path(label_path)
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")

    boxes = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"{path}:{line_number}: expected at least five YOLO fields")
        try:
            values = [float(value) for value in parts[:5]]
        except ValueError as error:
            raise ValueError(
                f"{path}:{line_number}: the first five YOLO fields must be numeric"
            ) from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: YOLO fields must be finite")

        raw_class_id, center_x, center_y, width, height = values
        if raw_class_id != int(raw_class_id) or raw_class_id < 0:
            raise ValueError(
                f"{path}:{line_number}: class identifier must be a non-negative integer"
            )
        if not (
            0.0 <= center_x <= 1.0
            and 0.0 <= center_y <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise ValueError(
                f"{path}:{line_number}: box coordinates must be normalized YOLO values"
            )
        boxes.append(yolo_to_xywh(center_x, center_y, width, height))
    return boxes


def compute_overlap_for_boxes(boxes):
    """Return pairwise-overlap statistics for one image's ground-truth boxes."""

    normalized_boxes = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError("Every bounding box must contain four xywh values.")
        values = np.asarray(box, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Bounding-box values must be finite.")
        if values[2] < 0.0 or values[3] < 0.0:
            raise ValueError("Bounding-box width and height must be non-negative.")
        normalized_boxes.append(values)

    pair_count = len(normalized_boxes) * (len(normalized_boxes) - 1) // 2
    if pair_count == 0:
        return {
            "pair_count": 0,
            "max_pairwise_iou": 0.0,
            "mean_pairwise_iou": 0.0,
            "pairs_iou_gt_0_1": 0,
            "pairs_iou_gt_0_3": 0,
            "pairs_iou_gt_0_5": 0,
        }

    overlaps = np.empty(pair_count, dtype=float)
    position = 0
    for first_index, first_box in enumerate(normalized_boxes[:-1]):
        for second_box in normalized_boxes[first_index + 1 :]:
            overlaps[position] = np.clip(
                calculate_iou(first_box, second_box),
                0.0,
                1.0,
            )
            position += 1

    return {
        "pair_count": pair_count,
        "max_pairwise_iou": float(overlaps.max()),
        "mean_pairwise_iou": float(overlaps.mean()),
        "pairs_iou_gt_0_1": int(np.count_nonzero(overlaps > IOU_THRESHOLDS[0])),
        "pairs_iou_gt_0_3": int(np.count_nonzero(overlaps > IOU_THRESHOLDS[1])),
        "pairs_iou_gt_0_5": int(np.count_nonzero(overlaps > IOU_THRESHOLDS[2])),
    }


def compute_overlap_for_label(label_path):
    """Read one label file and compute its pairwise-overlap statistics."""

    return compute_overlap_for_boxes(parse_yolo_boxes(label_path))


def _required_columns(table, columns, label):
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _validated_counts(series, label):
    numeric = pd.to_numeric(series, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain finite values.")
    if (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label} must contain non-negative integers.")
    return numeric.astype(np.int64)


def load_and_validate_indexes(dataset_index_path, selected_index_path):
    """Load the full and selected indexes and prove subset consistency."""

    full_path = Path(dataset_index_path).expanduser().absolute()
    selected_path = Path(selected_index_path).expanduser().absolute()
    if not full_path.is_file():
        raise FileNotFoundError(
            f"Dataset index not found: {full_path}. "
            "Run experiments/scripts/02_build_dataset_index.py first."
        )
    if not selected_path.is_file():
        raise FileNotFoundError(
            f"Selected sample index not found: {selected_path}. "
            "Run experiments/scripts/02_dataset_sampling.py first."
        )

    full = pd.read_csv(full_path)
    selected = pd.read_csv(selected_path)
    required = ["image_file", "image_path", "label_path", "num_objects"]
    _required_columns(full, required, "Dataset index")
    _required_columns(selected, required, "Selected sample index")
    if full.empty:
        raise RuntimeError("Dataset index is empty.")
    if selected.empty:
        raise RuntimeError("Selected sample index is empty.")

    for table, label in ((full, "Dataset index"), (selected, "Selected sample index")):
        if table["image_file"].isna().any() or table["image_path"].isna().any():
            raise ValueError(f"{label} contains missing image identifiers.")
        if table["image_file"].duplicated().any():
            raise ValueError(f"{label} contains duplicate image_file values.")
        if table["image_path"].duplicated().any():
            raise ValueError(f"{label} contains duplicate image_path values.")
        table["num_objects"] = _validated_counts(table["num_objects"], "num_objects")

    full_by_file = full.set_index("image_file", drop=False)
    missing = selected.loc[
        ~selected["image_file"].isin(full_by_file.index),
        "image_file",
    ].tolist()
    if missing:
        raise ValueError(
            "Selected sample contains images absent from the full index: "
            + ", ".join(map(str, missing[:3]))
        )

    aligned_full = full_by_file.loc[selected["image_file"]]
    for column in ("image_path", "label_path", "num_objects"):
        expected = aligned_full[column].reset_index(drop=True)
        observed = selected[column].reset_index(drop=True)
        if not observed.equals(expected):
            raise ValueError(f"Selected sample {column} values do not match the full index.")
    return full, selected


def resolve_label_path(indexed_path, asset_root=None, project_root=PROJECT_ROOT):
    """Resolve a serialized label path without changing its stored namespace."""

    raw_value = str(indexed_path).strip()
    if not raw_value:
        raise ValueError("Label path cannot be empty.")
    direct = Path(raw_value).expanduser()
    if direct.is_absolute():
        return direct

    logical = PurePosixPath(raw_value.replace("\\", "/"))
    parts = logical.parts
    if asset_root is not None:
        root = Path(asset_root).expanduser().absolute()
        supported_prefixes = {
            ("detector_service", "storage"),
            ("techtrack", "storage"),
        }
        if len(parts) >= 2 and tuple(parts[:2]) in supported_prefixes:
            return root.joinpath(*parts[2:])
    return Path(project_root).joinpath(*parts)


def build_overlap_profile(index, asset_root=None, progress_interval=1000):
    """Build one auditable overlap record for every indexed image."""

    rows = []
    total = len(index)
    for position, row in enumerate(index.itertuples(index=False), start=1):
        label_path = resolve_label_path(row.label_path, asset_root=asset_root)
        boxes = parse_yolo_boxes(label_path)
        expected_objects = int(row.num_objects)
        if len(boxes) != expected_objects:
            raise ValueError(
                f"Label count mismatch for {row.image_file}: "
                f"index={expected_objects}, parsed={len(boxes)}"
            )
        metrics = compute_overlap_for_boxes(boxes)
        rows.append(
            {
                "image_file": row.image_file,
                "image_path": row.image_path,
                "label_path": row.label_path,
                "num_objects": expected_objects,
                **metrics,
                "crowding_bucket": crowding_bucket(metrics["pairs_iou_gt_0_1"]),
            }
        )
        if progress_interval and position % progress_interval == 0:
            print(f"[INFO] Processed {position}/{total} images")
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


def summarize_overlap(table, label):
    """Summarize per-image overlap evidence for one dataset view."""

    if table.empty:
        raise ValueError("Cannot summarize an empty overlap table.")
    _required_columns(table, PROFILE_COLUMNS, "Overlap profile")
    return {
        "dataset": str(label),
        "images": int(len(table)),
        "mean_pair_count": round(float(table["pair_count"].mean()), 3),
        "mean_max_pairwise_iou": round(float(table["max_pairwise_iou"].mean()), 4),
        "mean_pairs_iou_gt_0_1": round(float(table["pairs_iou_gt_0_1"].mean()), 3),
        "images_with_any_iou_gt_0_1": int((table["pairs_iou_gt_0_1"] > 0).sum()),
        "images_with_any_iou_gt_0_3": int((table["pairs_iou_gt_0_3"] > 0).sum()),
        "images_with_any_iou_gt_0_5": int((table["pairs_iou_gt_0_5"] > 0).sum()),
        "images_with_20plus_iou_gt_0_1_pairs": int(
            (table["pairs_iou_gt_0_1"] >= 20).sum()
        ),
    }


def crowding_distribution(table, label):
    """Return the stable four-bucket crowding distribution for a dataset view."""

    if table.empty:
        raise ValueError("Cannot build a crowding distribution from an empty table.")
    counts = (
        table["crowding_bucket"]
        .value_counts()
        .reindex(CROWDING_BUCKET_ORDER, fill_value=0)
    )
    return pd.DataFrame(
        {
            "dataset": str(label),
            "crowding_bucket": CROWDING_BUCKET_ORDER,
            "image_count": counts.to_numpy(dtype=np.int64),
            "image_share_pct": 100.0 * counts.to_numpy(dtype=float) / len(table),
        }
    )


def build_overlap_evidence(profile, selected_index, selected_name):
    """Construct summary and crowding artifacts from a validated profile."""

    _required_columns(profile, PROFILE_COLUMNS, "Overlap profile")
    selected_files = set(selected_index["image_file"])
    selected_profile = profile[profile["image_file"].isin(selected_files)].copy()
    if len(selected_profile) != len(selected_index):
        raise RuntimeError("Overlap profile does not cover every selected image.")

    summary = pd.DataFrame(
        [
            summarize_overlap(profile, "full_dataset"),
            summarize_overlap(selected_profile, selected_name),
        ],
        columns=SUMMARY_COLUMNS,
    )
    full_distribution = crowding_distribution(profile, "full_dataset")
    selected_distribution = crowding_distribution(selected_profile, selected_name)
    comparison = full_distribution.merge(
        selected_distribution,
        on="crowding_bucket",
        suffixes=("_full", "_sample"),
        validate="one_to_one",
    )
    comparison["image_share_diff_pp"] = (
        comparison["image_share_pct_sample"] - comparison["image_share_pct_full"]
    )
    comparison = comparison[CROWDING_COMPARISON_COLUMNS]
    return {
        "overlap_profile.csv": profile,
        "overlap_summary.csv": summary,
        "crowding_distribution_comparison.csv": comparison,
    }


def _write_dataframe_atomic(path, table):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv",
            prefix=f".{path.stem}-",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            table.to_csv(handle, index=False)
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_overlap_artifacts(output_dir, artifacts):
    """Atomically write the three tabular overlap artifacts."""

    directory = Path(output_dir).expanduser().absolute()
    output_paths = {}
    for name, table in artifacts.items():
        output_path = directory / name
        _write_dataframe_atomic(output_path, table)
        output_paths[name] = output_path
    return output_paths


def build_crowding_figure(comparison, figure_dir):
    """Render the full-versus-selected crowding distribution."""

    import matplotlib.pyplot as plt

    _required_columns(
        comparison,
        CROWDING_COMPARISON_COLUMNS,
        "Crowding comparison",
    )
    directory = Path(figure_dir).expanduser().absolute()
    directory.mkdir(parents=True, exist_ok=True)
    positions = np.arange(len(comparison))
    width = 0.38

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.bar(
        positions - width / 2,
        comparison["image_share_pct_full"],
        width=width,
        label="Full dataset",
    )
    axis.bar(
        positions + width / 2,
        comparison["image_share_pct_sample"],
        width=width,
        label="Selected sample",
    )
    axis.set_xticks(positions, comparison["crowding_bucket"])
    axis.set_xlabel("Box-pair overlap bucket: number of pairs with IoU > 0.1")
    axis.set_ylabel("Image share (%)")
    axis.set_title("Full dataset vs selected sample crowding distribution")
    axis.legend()
    figure.tight_layout()

    figure_path = directory / "04_crowding_distribution.png"
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare ground-truth overlap in the full dataset and selected sample."
    )
    parser.add_argument("--dataset-index", type=Path, default=DEFAULT_DATASET_INDEX)
    parser.add_argument("--selected-index", type=Path, default=DEFAULT_SELECTED_INDEX)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SAMPLING_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--selected-name")
    parser.add_argument("--progress-interval", type=positive_int, default=1000)
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Write validated CSV evidence without importing Matplotlib.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    full_index, selected_index = load_and_validate_indexes(
        args.dataset_index,
        args.selected_index,
    )
    selected_name = args.selected_name or (
        f"rare_aware_density_stratified_{len(selected_index)}"
    )
    profile = build_overlap_profile(
        full_index,
        asset_root=args.asset_root,
        progress_interval=args.progress_interval,
    )
    artifacts = build_overlap_evidence(profile, selected_index, selected_name)
    output_paths = write_overlap_artifacts(args.output_dir, artifacts)

    figure_path = None
    if not args.skip_figure:
        figure_path = build_crowding_figure(
            artifacts["crowding_distribution_comparison.csv"],
            args.figure_dir,
        )

    print("\nOVERLAP SUMMARY")
    print(artifacts["overlap_summary.csv"].to_string(index=False))
    print("\nCROWDING DISTRIBUTION")
    print(artifacts["crowding_distribution_comparison.csv"].to_string(index=False))
    for path in output_paths.values():
        print(f"[WRITE] {path}")
    if figure_path is not None:
        print(f"[WRITE] {figure_path}")
    return artifacts, output_paths, figure_path


if __name__ == "__main__":
    main()
