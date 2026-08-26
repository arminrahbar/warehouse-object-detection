"""Build one interpretable detector-error record per selected image."""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector_service.modules.rectification.hard_negative_mining import (
    ERROR_COMPONENT_COLUMNS,
    compute_image_error_components,
)
from experiments.scripts.experiment_contracts import (
    load_verified_checkpoint_selection,
    load_verified_operating_point,
    threshold_tag,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "outputs"
DEFAULT_SAMPLE_PATH = (
    DEFAULT_OUTPUT_ROOT
    / "02_dataset_analysis"
    / "02_sample_selection"
    / "selected_sample_index.csv"
)
DEFAULT_NMS_OUTPUT_DIR = (
    DEFAULT_OUTPUT_ROOT
    / "03_nms_thresholding"
    / "01_threshold_sweep"
)
DEFAULT_SELECTION_RUN = (
    DEFAULT_OUTPUT_ROOT
    / "01_model_selection"
    / "03_checkpoint_decision"
    / "selection-20260821-v1"
)
DEFAULT_OPERATING_POINT = DEFAULT_NMS_OUTPUT_DIR / "operating_point.json"
DEFAULT_GROUND_TRUTH_PATH = (
    DEFAULT_NMS_OUTPUT_DIR / "ground_truth_sample5000.csv"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_OUTPUT_ROOT
    / "05_hard_negative_mining"
    / "01_error_components"
)

MATCH_IOU_THRESHOLD = 0.5
CONFIDENCE_FLOOR = 0.5
DATASET_NAME = "rare_aware_density_stratified_5000"
ORIGINAL_CONDITION = "original"
PREDICTION_RUN_LABEL = "sample5000"
DENSITY_BUCKETS = ("1", "2-4", "5-9", "10-14", "15-19", "20+")

SAMPLE_COLUMNS = [
    "image_file",
    "image_path",
    "num_objects",
    "density_bucket",
    "class_names_present",
]
PREDICTION_COLUMNS = [
    "image_file",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "class_id",
    "combined_confidence",
]
PREDICTION_PROVENANCE_COLUMNS = ["model", "nms_threshold"]
PREDICTION_INPUT_COLUMNS = [*PREDICTION_PROVENANCE_COLUMNS, *PREDICTION_COLUMNS]
GROUND_TRUTH_COLUMNS = [
    "image_file",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "class_id",
]
COUNT_COLUMNS = [
    "prediction_count",
    "ground_truth_count",
    "matched_prediction_count",
    "false_positive_prediction_count",
    "matched_gt_count",
    "missed_gt_count",
]
MEAN_COLUMNS = ["mean_matched_iou", "mean_matched_confidence"]
COMPONENT_COLUMNS = [
    *SAMPLE_COLUMNS,
    *ERROR_COMPONENT_COLUMNS,
    *COUNT_COLUMNS,
    *MEAN_COLUMNS,
]


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def resolve_prediction_input(predictions, selection_run, operating_point):
    """Resolve predictions from verified checkpoint and NMS decisions."""

    selection = load_verified_checkpoint_selection(selection_run)
    operating = load_verified_operating_point(operating_point, selection_run)
    model_name = selection["selected_model"]
    nms_threshold = float(operating["selected_nms_iou_threshold"])
    default_path = (
        Path(operating["path"]).parent
        / (
            f"{model_name}_predictions_nms_{threshold_tag(nms_threshold)}_"
            f"{PREDICTION_RUN_LABEL}.csv"
        )
    )
    source = default_path if predictions is None else Path(predictions)
    return {
        "prediction_path": source.expanduser().absolute(),
        "selected_model": model_name,
        "nms_threshold": nms_threshold,
        "selection": selection,
        "operating_point": operating,
    }


# Retained as a pure canonical-layout constant for repository-wide path checks.
# Runtime resolution remains dynamic: ``main`` verifies the selected checkpoint
# and operating point before deriving the prediction artifact. Keeping this
# constant side-effect free also makes imports and ``--help`` work in a clean
# checkout where ignored experiment evidence is intentionally absent.
DEFAULT_PREDICTION_PATH = (
    DEFAULT_NMS_OUTPUT_DIR / "model2_predictions_nms_0_3_sample5000.csv"
)


def _required_columns(table, columns, label):
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _integer_series(series, label):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError(f"{label} must contain finite integers.")
    if (numeric < 0).any() or (numeric % 1 != 0).any():
        raise ValueError(f"{label} must contain non-negative integers.")
    return numeric.astype("int64")


def _numeric_columns(table, columns, label):
    normalized = table.copy()
    for column in columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if not np.isfinite(normalized[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains non-finite numeric values.")
    return normalized


def _validate_exact_string_column(table, column, expected, label):
    if table[column].isna().any():
        raise ValueError(f"{label} {column} provenance contains missing values.")
    observed = set(table[column].astype(str))
    if observed != {expected}:
        raise ValueError(
            f"{label} {column} provenance must be {expected!r}; "
            f"found {sorted(observed)!r}."
        )


def _validate_optional_dataset_condition(table, label):
    if "dataset" in table.columns:
        _validate_exact_string_column(table, "dataset", DATASET_NAME, label)
    for column in ("augmentation_condition", "condition"):
        if column in table.columns:
            _validate_exact_string_column(
                table,
                column,
                ORIGINAL_CONDITION,
                label,
            )


def validate_prediction_provenance(
    table,
    *,
    expected_model,
    expected_nms_threshold,
):
    """Validate upstream identity fields before analysis columns are reduced."""

    _required_columns(
        table,
        PREDICTION_PROVENANCE_COLUMNS,
        "Prediction cache",
    )
    if table.empty:
        raise ValueError("Prediction cache is empty; provenance cannot be verified.")
    _validate_exact_string_column(
        table,
        "model",
        str(expected_model),
        "Prediction cache",
    )
    thresholds = pd.to_numeric(table["nms_threshold"], errors="coerce")
    if thresholds.isna().any() or not np.isfinite(thresholds).all():
        raise ValueError(
            "Prediction cache nms_threshold provenance must contain finite values."
        )
    if not np.isclose(
        thresholds.to_numpy(dtype=float),
        float(expected_nms_threshold),
        rtol=0.0,
        atol=1e-12,
    ).all():
        observed = sorted(set(thresholds.astype(float)))
        raise ValueError(
            "Prediction cache nms_threshold provenance must equal "
            f"{float(expected_nms_threshold):.12g}; found {observed!r}."
        )
    _validate_optional_dataset_condition(table, "Prediction cache")
    return True


def _write_csv(path, table):
    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv",
            prefix=f".{destination.stem}-",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            table.to_csv(handle, index=False)
        temporary.replace(destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def load_sample(path, max_images=None):
    """Load a unique selected-image manifest with queue metadata."""

    source = Path(path).expanduser().absolute()
    if not source.is_file():
        raise FileNotFoundError(f"Selected sample index not found: {source}")
    sample = pd.read_csv(source)
    _required_columns(sample, SAMPLE_COLUMNS, "Selected sample index")
    if sample.empty:
        raise ValueError("Selected sample index is empty.")
    if sample[["image_file", "image_path"]].isna().any().any():
        raise ValueError("Selected sample index contains missing image identifiers.")
    for column in ("image_file", "image_path"):
        if sample[column].duplicated().any():
            raise ValueError(f"Selected sample index contains duplicate {column} values.")
    sample = sample[SAMPLE_COLUMNS].copy()
    sample["num_objects"] = _integer_series(sample["num_objects"], "num_objects")
    if not set(sample["density_bucket"].astype(str)).issubset(DENSITY_BUCKETS):
        raise ValueError("Selected sample index contains an unsupported density bucket.")
    if sample["class_names_present"].isna().any():
        raise ValueError("Selected sample index contains missing class-presence metadata.")
    if max_images is not None:
        if max_images > len(sample):
            raise ValueError(
                f"max_images {max_images} exceeds selected sample size {len(sample)}."
            )
        sample = sample.head(max_images).copy()
    return sample


def load_predictions(
    path,
    sample,
    *,
    expected_model,
    expected_nms_threshold,
):
    """Load fixed post-NMS predictions and validate the confidence boundary."""

    source = Path(path).expanduser().absolute()
    if not source.is_file():
        raise FileNotFoundError(f"Post-NMS prediction cache not found: {source}")
    predictions = pd.read_csv(source)
    _required_columns(
        predictions,
        PREDICTION_INPUT_COLUMNS,
        "Prediction cache",
    )
    validate_prediction_provenance(
        predictions,
        expected_model=expected_model,
        expected_nms_threshold=expected_nms_threshold,
    )
    predictions = predictions[PREDICTION_COLUMNS].copy()
    predictions = _numeric_columns(
        predictions,
        ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "class_id", "combined_confidence"],
        "Prediction cache",
    )
    predictions["class_id"] = _integer_series(predictions["class_id"], "class_id")
    if (predictions[["bbox_w", "bbox_h"]] < 0).any().any():
        raise ValueError("Prediction widths and heights cannot be negative.")
    if not predictions["combined_confidence"].between(CONFIDENCE_FLOOR, 1.0).all():
        raise ValueError(
            "Post-NMS predictions must satisfy the configured confidence floor."
        )
    selected = set(sample["image_file"])
    predictions = predictions[predictions["image_file"].isin(selected)].copy()
    return predictions.reset_index(drop=True)


def load_ground_truth(path, sample):
    """Load ground truth and reconcile every selected image count."""

    source = Path(path).expanduser().absolute()
    if not source.is_file():
        raise FileNotFoundError(f"Ground-truth cache not found: {source}")
    ground_truth = pd.read_csv(source)
    _required_columns(ground_truth, GROUND_TRUTH_COLUMNS, "Ground-truth cache")
    _validate_optional_dataset_condition(ground_truth, "Ground-truth cache")
    ground_truth = ground_truth[GROUND_TRUTH_COLUMNS].copy()
    ground_truth = _numeric_columns(
        ground_truth,
        ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "class_id"],
        "Ground-truth cache",
    )
    ground_truth["class_id"] = _integer_series(ground_truth["class_id"], "class_id")
    if (ground_truth[["bbox_w", "bbox_h"]] < 0).any().any():
        raise ValueError("Ground-truth widths and heights cannot be negative.")
    selected = set(sample["image_file"])
    ground_truth = ground_truth[ground_truth["image_file"].isin(selected)].copy()
    observed = ground_truth.groupby("image_file").size()
    expected = sample.set_index("image_file")["num_objects"]
    observed = observed.reindex(expected.index, fill_value=0).astype("int64")
    if not np.array_equal(observed.to_numpy(), expected.to_numpy()):
        raise ValueError("Ground-truth counts disagree with the selected sample index.")
    return ground_truth.reset_index(drop=True)


def _prediction_arrays(group):
    if group is None or group.empty:
        return [], [], []
    return (
        group[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(dtype=float),
        group["class_id"].to_numpy(dtype=int),
        group["combined_confidence"].to_numpy(dtype=float),
    )


def _ground_truth_arrays(group):
    if group is None or group.empty:
        return [], []
    return (
        group[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(dtype=float),
        group["class_id"].to_numpy(dtype=int),
    )


def build_component_table(sample, predictions, ground_truth):
    """Calculate bounded error components in selected-manifest order."""

    prediction_groups = (
        {name: group for name, group in predictions.groupby("image_file", sort=False)}
        if not predictions.empty
        else {}
    )
    ground_truth_groups = (
        {name: group for name, group in ground_truth.groupby("image_file", sort=False)}
        if not ground_truth.empty
        else {}
    )
    rows = []
    for position, image_row in enumerate(sample.itertuples(index=False), start=1):
        prediction_values = _prediction_arrays(prediction_groups.get(image_row.image_file))
        truth_values = _ground_truth_arrays(ground_truth_groups.get(image_row.image_file))
        components = compute_image_error_components(
            *prediction_values,
            *truth_values,
            iou_threshold=MATCH_IOU_THRESHOLD,
            confidence_floor=CONFIDENCE_FLOOR,
        )
        rows.append(
            {
                "image_file": image_row.image_file,
                "image_path": image_row.image_path,
                "num_objects": int(image_row.num_objects),
                "density_bucket": image_row.density_bucket,
                "class_names_present": image_row.class_names_present,
                **components,
            }
        )
        if position % 1000 == 0:
            print(f"[COMPONENTS] Processed {position}/{len(sample)} images")
    return pd.DataFrame(rows, columns=COMPONENT_COLUMNS)


def validate_component_table(table):
    """Assert count identities and bounded values on the produced evidence."""

    _required_columns(table, COMPONENT_COLUMNS, "Component table")
    normalized = table[COMPONENT_COLUMNS].copy()
    normalized = _numeric_columns(
        normalized,
        [*ERROR_COMPONENT_COLUMNS, *COUNT_COLUMNS, *MEAN_COLUMNS, "num_objects"],
        "Component table",
    )
    for column in [*COUNT_COLUMNS, "num_objects"]:
        normalized[column] = _integer_series(normalized[column], column)
    for column in ERROR_COMPONENT_COLUMNS:
        if not normalized[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} values must be within [0, 1].")
    if not normalized["mean_matched_iou"].between(0.0, 1.0).all():
        raise ValueError("mean_matched_iou values must be within [0, 1].")
    if not normalized["mean_matched_confidence"].between(0.0, 1.0).all():
        raise ValueError("mean_matched_confidence values must be within [0, 1].")
    identities = (
        normalized["matched_prediction_count"].equals(normalized["matched_gt_count"])
        and normalized["false_positive_prediction_count"].equals(
            normalized["prediction_count"] - normalized["matched_prediction_count"]
        )
        and normalized["missed_gt_count"].equals(
            normalized["ground_truth_count"] - normalized["matched_gt_count"]
        )
        and normalized["ground_truth_count"].equals(normalized["num_objects"])
    )
    if not identities:
        raise ValueError("Component count identities are inconsistent.")
    return normalized


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute image-level detector error components."
    )
    parser.add_argument("--sample-index", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--selection-run", type=Path, default=DEFAULT_SELECTION_RUN)
    parser.add_argument(
        "--operating-point",
        type=Path,
        default=DEFAULT_OPERATING_POINT,
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help=(
            "Optional post-NMS cache override. Its model and NMS provenance "
            "must match the verified upstream decisions."
        ),
    )
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-images", type=positive_int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    prediction_input = resolve_prediction_input(
        args.predictions,
        args.selection_run,
        args.operating_point,
    )
    sample = load_sample(args.sample_index, args.max_images)
    predictions = load_predictions(
        prediction_input["prediction_path"],
        sample,
        expected_model=prediction_input["selected_model"],
        expected_nms_threshold=prediction_input["nms_threshold"],
    )
    ground_truth = load_ground_truth(args.ground_truth, sample)
    components = validate_component_table(
        build_component_table(sample, predictions, ground_truth)
    )
    run_label = f"first_{args.max_images}" if args.max_images else "sample5000"
    output_path = _write_csv(
        Path(args.output_dir) / f"image_error_components_{run_label}.csv",
        components,
    )
    print(f"[WRITE] {output_path}")
    print(f"[INFO] Images evaluated: {len(components)}")
    print(
        "[INFO] Prediction provenance: "
        f"model={prediction_input['selected_model']} "
        f"nms_iou={prediction_input['nms_threshold']:.12g}"
    )
    print(f"[INFO] Predictions evaluated: {int(components['prediction_count'].sum())}")
    print(f"[INFO] Ground truths evaluated: {int(components['ground_truth_count'].sum())}")
    print("\nERROR COMPONENT DISTRIBUTION")
    print(
        components[list(ERROR_COMPONENT_COLUMNS)]
        .describe(percentiles=[0.5, 0.75, 0.95])
        .T.to_string()
    )
    zero_prediction = components[components["prediction_count"] == 0]
    print("\nZERO-PREDICTION IMAGES")
    print(f"count: {len(zero_prediction)}")
    print(
        "ground truths in zero-prediction images: "
        f"{int(zero_prediction['ground_truth_count'].sum())}"
    )
    return components, output_path


if __name__ == "__main__":
    main()
