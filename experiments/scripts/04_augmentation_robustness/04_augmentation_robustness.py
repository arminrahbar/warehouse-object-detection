"""Measure detector sensitivity to controlled image perturbations."""

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector_service.modules.inference.nms import NMS
from detector_service.modules.utils.metrics import (
    calculate_map_x_point_interpolated,
    calculate_precision_recall_curve,
    match_detections,
)
from experiments.scripts.experiment_contracts import (
    EvidenceContractError,
    load_verified_checkpoint_selection,
    load_verified_operating_point,
    resolve_indexed_path as resolve_contract_path,
    resolve_selected_model_assets,
    sha256_file,
    threshold_tag,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "outputs"
DEFAULT_SAMPLE_INDEX = (
    DEFAULT_OUTPUT_ROOT
    / "02_dataset_analysis"
    / "02_sample_selection"
    / "selected_sample_index.csv"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_OUTPUT_ROOT
    / "04_augmentation_robustness"
    / "01_condition_evaluation"
)
DEFAULT_SELECTION_RUN = (
    DEFAULT_OUTPUT_ROOT
    / "01_model_selection"
    / "03_checkpoint_decision"
    / "selection-20260821-v1"
)
DEFAULT_OPERATING_POINT = (
    DEFAULT_OUTPUT_ROOT
    / "03_nms_thresholding"
    / "01_threshold_sweep"
    / "operating_point.json"
)
DEFAULT_FIGURE_DIR = (
    PROJECT_ROOT / "scratch" / "diagnostic-figures" / "04_augmentation_robustness"
)
DEFAULT_ASSET_ROOT = PROJECT_ROOT / "detector_service" / "storage"

MODEL_NAME = "model2"
DATASET_NAME = "rare_aware_density_stratified_5000"
OBJECTNESS_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.3
MAP_IOU_THRESHOLD = 0.5
EVAL_TYPE = "combined"
CACHE_MANIFEST_SCHEMA_VERSION = 1

MODEL_ASSETS = {
    "weights": "yolo_model_2/yolov4-tiny-logistics_size_416_2.weights",
    "config": "yolo_model_2/yolov4-tiny-logistics_size_416_2.cfg",
    "classes": "yolo_model_2/logistics.names",
}

CONDITIONS = (
    {"tag": "original", "display": "Original", "type": "none"},
    {
        "tag": "gaussian_blur_k9",
        "display": "Gaussian blur",
        "type": "gaussian_blur",
        "kernel_size": 9,
        "sigma": 0,
    },
    {
        "tag": "vertical_flip",
        "display": "Vertical flip",
        "type": "vertical_flip",
    },
    {
        "tag": "brightness_increase",
        "display": "Brighter / higher contrast",
        "type": "brightness",
        "alpha": 1.15,
        "beta": 35,
    },
    {
        "tag": "brightness_decrease",
        "display": "Darker / lower contrast",
        "type": "brightness",
        "alpha": 0.85,
        "beta": -35,
    },
)
LEGACY_DISPLAY_ALIASES = {
    "brightness_increase": {"Brightness increase"},
    "brightness_decrease": {"Brightness decrease"},
}

RAW_COLUMNS = [
    "model",
    "dataset",
    "augmentation_condition",
    "augmentation_display",
    "image_file",
    "image_path",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "class_id",
    "class_name",
    "object_score",
    "predicted_class_score",
    "combined_confidence",
    "class_scores_json",
]
LEDGER_COLUMNS = [
    "model",
    "dataset",
    "augmentation_condition",
    "image_file",
    "image_path",
    "status",
    "candidate_count",
]
PREDICTION_COLUMNS = RAW_COLUMNS + ["nms_threshold"]
GROUND_TRUTH_COLUMNS = [
    "dataset",
    "augmentation_condition",
    "augmentation_display",
    "image_file",
    "image_path",
    "class_id",
    "class_name",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
]
SUMMARY_COLUMNS = [
    "model",
    "dataset",
    "augmentation_condition",
    "augmentation_display",
    "mAP@0.5_11_point",
    "total_ground_truth",
    "total_predictions_after_nms",
    "evaluation_rows",
    "candidate_objectness_threshold",
    "nms_confidence_threshold",
    "nms_iou_threshold",
    "map_iou_threshold",
    "eval_type",
    "mAP_change_vs_original",
    "mAP_percent_change_vs_original",
    "prediction_change_vs_original",
]
PER_CLASS_COLUMNS = [
    "model",
    "dataset",
    "augmentation_condition",
    "augmentation_display",
    "class_id",
    "class_name",
    "ground_truth_count",
    "prediction_count",
    "ap_11_point",
    "original_ap_11_point",
    "ap_change_vs_original",
]


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


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


def _float_columns(table, columns, label):
    normalized = table.copy()
    for column in columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    values = normalized[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite numeric values.")
    return normalized


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


def _run_label(max_images):
    return f"first_{max_images}" if max_images is not None else "sample5000"


def _condition(tag):
    matches = [condition for condition in CONDITIONS if condition["tag"] == tag]
    if not matches:
        raise ValueError(f"Unknown augmentation condition: {tag}")
    return matches[0]


def load_sample_index(path, max_images=None):
    source = Path(path).expanduser().absolute()
    if not source.is_file():
        raise FileNotFoundError(
            f"Selected sample index not found: {source}. "
            "Run experiments/scripts/02_dataset_analysis/02_dataset_sampling.py first."
        )
    index = pd.read_csv(source)
    _required_columns(
        index,
        ["image_file", "image_path", "label_path", "num_objects"],
        "Selected sample index",
    )
    if index.empty:
        raise ValueError("Selected sample index is empty.")
    if index[["image_file", "image_path", "label_path"]].isna().any().any():
        raise ValueError("Selected sample index contains missing paths.")
    for column in ("image_file", "image_path"):
        if index[column].duplicated().any():
            raise ValueError(f"Selected sample index contains duplicate {column} values.")
    index = index.copy()
    index["num_objects"] = _integer_series(index["num_objects"], "num_objects")
    if max_images is not None:
        if max_images > len(index):
            raise ValueError(
                f"max_images {max_images} exceeds selected sample size {len(index)}."
            )
        index = index.head(max_images).copy()
    return index


def resolve_indexed_path(value, asset_root=None, project_root=PROJECT_ROOT):
    return resolve_contract_path(
        value,
        asset_root=asset_root,
        project_root=project_root,
    )


def _cache_paths(
    directory,
    condition,
    run_label,
    model_name=MODEL_NAME,
    nms_iou_threshold=NMS_IOU_THRESHOLD,
):
    root = Path(directory).expanduser().absolute()
    tag = condition["tag"]
    return {
        "ground_truth": root / f"ground_truth_{tag}_{run_label}.csv",
        "raw": root / f"{model_name}_raw_predictions_{tag}_{run_label}.csv",
        "ledger": root / f"{model_name}_inference_ledger_{tag}_{run_label}.csv",
        "predictions": root
        / (
            f"{model_name}_predictions_{tag}_class_aware_nms_"
            f"{threshold_tag(nms_iou_threshold)}_"
            f"{run_label}.csv"
        ),
    }


def _manifest_path(directory, run_label):
    return (
        Path(directory).expanduser().absolute()
        / f"condition_cache_manifest_{run_label}.json"
    )


def _write_json(path, payload):
    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".json",
            prefix=f".{destination.stem}-",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def _ordered_vocabulary_sha256(classes):
    encoded = json.dumps(
        list(classes),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selected_records_sha256(index):
    columns = ["image_file", "image_path", "label_path", "num_objects"]
    records = index[columns].to_dict(orient="records")
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _condition_contract(
    index,
    sample_index_path,
    selection,
    operating_point,
    classes,
    run_label,
    model_name,
    nms_iou_threshold,
):
    return {
        "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "run_label": run_label,
        "dataset": DATASET_NAME,
        "sample_index": {
            "sha256": sha256_file(sample_index_path),
            "selected_rows": int(len(index)),
            "selected_records_sha256": _selected_records_sha256(index),
        },
        "checkpoint_selection": {
            "run_id": selection["selection_run_id"],
            "manifest_sha256": selection["selection_manifest_sha256"],
            "decision_sha256": selection["decision_sha256"],
            "selected_checkpoint": selection["selected_checkpoint"],
            "selected_model": model_name,
            "asset_paths": dict(selection["asset_paths"]),
            "model_identity": dict(selection["model_identity"]),
        },
        "vocabulary": {
            "class_count": len(classes),
            "names_asset_sha256": selection["model_identity"]["names"],
            "ordered_classes_sha256": _ordered_vocabulary_sha256(classes),
        },
        "operating_point": {
            "sha256": operating_point["sha256"],
            "selected_model": model_name,
            "selected_nms_iou_threshold": nms_iou_threshold,
        },
        "conditions": [dict(condition) for condition in CONDITIONS],
        "policy": {
            "candidate_objectness_threshold": OBJECTNESS_THRESHOLD,
            "nms_confidence_threshold": SCORE_THRESHOLD,
            "nms_iou_threshold": nms_iou_threshold,
            "map_iou_threshold": MAP_IOU_THRESHOLD,
            "eval_type": EVAL_TYPE,
        },
    }


def _primary_cache_paths(
    directory,
    run_label,
    model_name,
    nms_iou_threshold,
):
    return {
        condition["tag"]: {
            kind: path
            for kind, path in _cache_paths(
                directory,
                condition,
                run_label,
                model_name=model_name,
                nms_iou_threshold=nms_iou_threshold,
            ).items()
            if kind in {"ground_truth", "raw", "ledger"}
        }
        for condition in CONDITIONS
    }


def _artifact_identity(path):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Condition-cache artifact not found: {source}")
    try:
        rows = len(pd.read_csv(source))
    except (OSError, pd.errors.ParserError) as error:
        raise EvidenceContractError(
            f"Condition-cache artifact is not readable CSV: {source}"
        ) from error
    return {
        "file": source.name,
        "rows": rows,
        "sha256": sha256_file(source),
    }


def write_condition_cache_manifest(
    directory,
    index,
    sample_index_path,
    selection,
    operating_point,
    classes,
    run_label,
    model_name=MODEL_NAME,
    nms_iou_threshold=NMS_IOU_THRESHOLD,
):
    """Atomically record all primary evidence identities for one full run."""

    payload = _condition_contract(
        index,
        sample_index_path,
        selection,
        operating_point,
        classes,
        run_label,
        model_name,
        nms_iou_threshold,
    )
    paths = _primary_cache_paths(
        directory,
        run_label,
        model_name,
        nms_iou_threshold,
    )
    payload["artifacts"] = {
        tag: {kind: _artifact_identity(path) for kind, path in kind_paths.items()}
        for tag, kind_paths in paths.items()
    }
    return _write_json(_manifest_path(directory, run_label), payload)


def validate_condition_cache_manifest(
    directory,
    index,
    sample_index_path,
    selection,
    operating_point,
    classes,
    run_label,
    model_name=MODEL_NAME,
    nms_iou_threshold=NMS_IOU_THRESHOLD,
):
    """Validate provenance and bytes before replaying primary cache evidence."""

    path = _manifest_path(directory, run_label)
    if not path.is_file():
        raise FileNotFoundError(
            "Full cache replay requires a complete condition-cache manifest: "
            f"{path}"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceContractError(
            f"Condition-cache manifest is not valid JSON: {path}"
        ) from error
    expected = _condition_contract(
        index,
        sample_index_path,
        selection,
        operating_point,
        classes,
        run_label,
        model_name,
        nms_iou_threshold,
    )
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise EvidenceContractError(
                f"Condition-cache manifest {key!r} does not match this run."
            )
    expected_paths = _primary_cache_paths(
        directory,
        run_label,
        model_name,
        nms_iou_threshold,
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_paths):
        raise EvidenceContractError(
            "Condition-cache manifest does not cover the exact condition set."
        )
    for tag, kind_paths in expected_paths.items():
        recorded = artifacts.get(tag)
        if not isinstance(recorded, dict) or set(recorded) != set(kind_paths):
            raise EvidenceContractError(
                f"Condition-cache manifest is incomplete for {tag!r}."
            )
        for kind, artifact_path in kind_paths.items():
            if recorded[kind] != _artifact_identity(artifact_path):
                raise EvidenceContractError(
                    f"Condition-cache {kind} identity mismatch for {tag!r}."
                )
    return manifest


def _assert_disjoint_directories(cache_input, output_dir):
    if cache_input is None:
        return
    cache = Path(cache_input).expanduser().resolve(strict=False)
    output = Path(output_dir).expanduser().resolve(strict=False)

    def contained(candidate, root):
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            return False

    if contained(cache, output) or contained(output, cache):
        raise ValueError(
            "--cache-input-dir and --output-dir must be disjoint; neither may "
            "contain the other."
        )


def load_classes(class_file, expected_sha256):
    """Load the complete ordered vocabulary bound to checkpoint selection."""

    source = Path(class_file).expanduser().absolute() if class_file else None
    if source is None or not source.is_file():
        raise FileNotFoundError(
            "The selected checkpoint's class-name file is required for a full "
            f"robustness run: {source}"
        )
    observed_sha256 = sha256_file(source)
    if observed_sha256 != expected_sha256:
        raise EvidenceContractError(
            "Class-name file does not match the verified selected-checkpoint "
            f"identity: {source}"
        )
    classes = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Class-name file must contain unique, non-empty names.")
    return classes


def _canonical_image_path(value):
    raw = str(value).strip().replace("\\", "/")
    if not raw:
        raise EvidenceContractError("Logical image_path cannot be empty.")
    parts = PurePosixPath(raw).parts
    if ".." in parts:
        raise EvidenceContractError(
            f"Logical image_path cannot contain parent traversal: {value}"
        )
    if tuple(parts[:2]) == ("detector_service", "storage"):
        return ("storage", *parts[2:])
    return parts


def _reconcile_image_paths(table, index, label):
    """Bind every cached logical path to its selected-index image identity."""

    normalized = table.copy()
    expected = index.set_index("image_file")["image_path"]
    selected_files = set(expected.index.astype(str))
    observed_files = normalized["image_file"].astype(str)
    if not set(observed_files).issubset(selected_files):
        raise ValueError(f"{label} contains images outside the selected index.")
    for position, (image_file, observed_path) in enumerate(
        zip(observed_files, normalized["image_path"], strict=True),
        start=1,
    ):
        expected_path = expected.loc[image_file]
        if _canonical_image_path(observed_path) != _canonical_image_path(expected_path):
            raise EvidenceContractError(
                f"{label} row {position} image_path does not match selected-index "
                f"identity for {image_file!r}."
            )
    normalized["image_path"] = observed_files.map(expected).to_numpy()
    return normalized


def apply_condition(image, condition, augmenter=None):
    """Apply exactly one declared perturbation to a decoded BGR image."""

    if augmenter is None:
        from detector_service.modules.rectification.augmentation import Augmenter

        augmenter = Augmenter
    condition_type = condition["type"]
    if condition_type == "none":
        return image.copy()
    if condition_type == "gaussian_blur":
        return augmenter.gaussian_blur(
            image=image,
            kernel_size=condition["kernel_size"],
            sigma=condition["sigma"],
        )
    if condition_type == "vertical_flip":
        return augmenter.vertical_flip(image=image)
    if condition_type == "brightness":
        return augmenter.change_brightness(
            image=image,
            alpha=condition["alpha"],
            beta=condition["beta"],
        )
    raise ValueError(f"Unsupported augmentation type: {condition_type}")


def parse_yolo_ground_truth(label_path, image_width, image_height, classes, condition):
    """Convert normalized YOLO labels to condition-aligned pixel xywh boxes."""

    source = Path(label_path)
    if not source.is_file():
        raise FileNotFoundError(f"Label file not found: {source}")
    rows = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) != 5:
            raise ValueError(f"{source}:{line_number} must contain five values.")
        try:
            values = np.asarray(fields, dtype=float)
        except ValueError as error:
            raise ValueError(f"{source}:{line_number} contains non-numeric values.") from error
        if not np.isfinite(values).all():
            raise ValueError(f"{source}:{line_number} contains non-finite values.")
        class_value = values[0]
        if class_value % 1 != 0 or not 0 <= class_value < len(classes):
            raise ValueError(f"{source}:{line_number} has an invalid class ID.")
        if ((values[1:] < 0.0) | (values[1:] > 1.0)).any():
            raise ValueError(f"{source}:{line_number} has coordinates outside [0, 1].")
        class_id = int(class_value)
        center_x, center_y, width, height = values[1:]
        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"{source}:{line_number} must have positive box width and height."
            )
        if condition["type"] == "vertical_flip":
            center_y = 1.0 - center_y
        pixel_width = width * image_width
        pixel_height = height * image_height
        left = float(np.clip(center_x * image_width - pixel_width / 2, 0, image_width))
        top = float(np.clip(center_y * image_height - pixel_height / 2, 0, image_height))
        right = float(np.clip(center_x * image_width + pixel_width / 2, 0, image_width))
        bottom = float(
            np.clip(center_y * image_height + pixel_height / 2, 0, image_height)
        )
        clipped_width = right - left
        clipped_height = bottom - top
        if clipped_width <= 0.0 or clipped_height <= 0.0:
            raise ValueError(f"{source}:{line_number} clips to an empty box.")
        rows.append(
            {
                "class_id": class_id,
                "class_name": classes[class_id],
                "bbox_x": left,
                "bbox_y": top,
                "bbox_w": clipped_width,
                "bbox_h": clipped_height,
            }
        )
    return rows


def build_ground_truth(index, classes, condition, asset_root=None):
    import cv2

    rows = []
    for position, image_row in enumerate(index.itertuples(index=False), start=1):
        image_path = resolve_indexed_path(image_row.image_path, asset_root)
        label_path = resolve_indexed_path(image_row.label_path, asset_root)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not decode image for ground truth: {image_path}")
        height, width = image.shape[:2]
        labels = parse_yolo_ground_truth(
            label_path,
            width,
            height,
            classes,
            condition,
        )
        for label in labels:
            rows.append(
                {
                    "dataset": DATASET_NAME,
                    "augmentation_condition": condition["tag"],
                    "augmentation_display": condition["display"],
                    "image_file": image_row.image_file,
                    "image_path": image_row.image_path,
                    **label,
                }
            )
        if position % 1000 == 0:
            print(f"[GT {condition['tag']}] Processed {position}/{len(index)} images")
    return pd.DataFrame(rows, columns=GROUND_TRUTH_COLUMNS)


def _validate_identity(table, classes, label):
    normalized = table.copy()
    normalized["class_id"] = _integer_series(normalized["class_id"], f"{label} class_id")
    if not normalized.empty and (normalized["class_id"] >= len(classes)).any():
        raise ValueError(f"{label} contains a class ID outside the vocabulary.")
    expected = normalized["class_id"].map(dict(enumerate(classes)))
    if not normalized["class_name"].astype(str).equals(expected.astype(str)):
        raise ValueError(f"{label} class names do not match the vocabulary.")
    return normalized


def _normalize_display_label(table, condition, label):
    observed = set(table["augmentation_display"].astype(str))
    accepted = {
        condition["display"],
        *LEGACY_DISPLAY_ALIASES.get(condition["tag"], set()),
    }
    if not observed or not observed.issubset(accepted):
        raise ValueError(f"{label} display label does not match the condition.")
    normalized = table.copy()
    normalized["augmentation_display"] = condition["display"]
    return normalized


def validate_ground_truth(table, index, classes, condition):
    label = "Ground-truth cache"
    _required_columns(table, GROUND_TRUTH_COLUMNS, label)
    normalized = _validate_identity(table[GROUND_TRUTH_COLUMNS], classes, label)
    normalized = _float_columns(
        normalized,
        ["bbox_x", "bbox_y", "bbox_w", "bbox_h"],
        label,
    )
    if (normalized[["bbox_w", "bbox_h"]] <= 0).any().any():
        raise ValueError("Ground-truth widths and heights must be positive.")
    if set(normalized["dataset"].astype(str)) != {DATASET_NAME}:
        raise ValueError(f"Ground-truth cache must use dataset {DATASET_NAME}.")
    if set(normalized["augmentation_condition"].astype(str)) != {condition["tag"]}:
        raise ValueError("Ground-truth cache condition does not match its filename.")
    normalized = _normalize_display_label(normalized, condition, label)
    normalized = _reconcile_image_paths(normalized, index, label)
    observed = normalized.groupby("image_file").size()
    expected = index.set_index("image_file")["num_objects"]
    observed = observed.reindex(expected.index, fill_value=0).astype("int64")
    if not np.array_equal(observed.to_numpy(), expected.to_numpy()):
        raise ValueError("Ground-truth counts do not match the selected index.")
    return normalized.reset_index(drop=True)


def _parse_score_vector(value, row_number, class_count):
    try:
        vector = np.asarray(json.loads(value), dtype=float).reshape(-1)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Raw prediction row {row_number} has invalid class scores.") from error
    if len(vector) != class_count or not np.isfinite(vector).all():
        raise ValueError(
            f"Raw prediction row {row_number} must contain {class_count} finite scores."
        )
    if ((vector < 0.0) | (vector > 1.0)).any():
        raise ValueError("Class probabilities must be between zero and one.")
    return vector


def validate_raw_predictions(
    table,
    index,
    classes,
    condition,
    model_name=MODEL_NAME,
):
    label = "Raw-prediction cache"
    _required_columns(table, RAW_COLUMNS, label)
    normalized = _validate_identity(table[RAW_COLUMNS], classes, label)
    if normalized.empty:
        return normalized
    normalized = _float_columns(
        normalized,
        [
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "object_score",
            "predicted_class_score",
            "combined_confidence",
        ],
        label,
    )
    expected_constants = {
        "model": model_name,
        "dataset": DATASET_NAME,
        "augmentation_condition": condition["tag"],
    }
    for column, expected in expected_constants.items():
        if set(normalized[column].astype(str)) != {expected}:
            raise ValueError(f"Raw-prediction {column} must equal {expected!r}.")
    normalized = _normalize_display_label(normalized, condition, label)
    if (normalized[["bbox_w", "bbox_h"]] < 0).any().any():
        raise ValueError("Raw-prediction widths and heights cannot be negative.")
    for column in ("object_score", "predicted_class_score", "combined_confidence"):
        if not normalized[column].between(0.0, 1.0).all():
            raise ValueError(f"Raw-prediction {column} values must be probabilities.")
    normalized = _reconcile_image_paths(normalized, index, label)
    recomputed_scores = []
    for position, row in enumerate(normalized.itertuples(index=False), start=1):
        vector = _parse_score_vector(row.class_scores_json, position, len(classes))
        predicted_class = int(np.argmax(vector))
        if predicted_class != int(row.class_id):
            raise ValueError("Predicted class does not match the maximum class score.")
        recomputed_scores.append(float(vector[predicted_class]))
    if not np.allclose(
        normalized["predicted_class_score"],
        recomputed_scores,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError("Predicted-class scores do not match class-score vectors.")
    combined = normalized["object_score"] * normalized["predicted_class_score"]
    if not np.allclose(
        normalized["combined_confidence"],
        combined,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError("Combined confidence does not match its components.")
    return normalized.reset_index(drop=True)


def validate_inference_ledger(
    table,
    index,
    raw_predictions,
    condition,
    model_name=MODEL_NAME,
):
    """Prove that a condition was inferred once for every selected image."""

    _required_columns(table, LEDGER_COLUMNS, "Inference ledger")
    normalized = table[LEDGER_COLUMNS].copy()
    if len(normalized) != len(index):
        raise ValueError(
            "Inference ledger row count does not match the selected-image count."
        )
    if normalized["image_file"].duplicated().any():
        raise ValueError("Inference ledger contains duplicate image_file values.")
    expected_constants = {
        "model": model_name,
        "dataset": DATASET_NAME,
        "augmentation_condition": condition["tag"],
        "status": "processed",
    }
    for column, expected in expected_constants.items():
        if set(normalized[column].astype(str)) != {expected}:
            raise ValueError(f"Inference-ledger {column} must equal {expected!r}.")
    normalized = _reconcile_image_paths(normalized, index, "Inference ledger")
    expected_images = index[["image_file", "image_path"]].reset_index(drop=True)
    observed_images = normalized[["image_file", "image_path"]].reset_index(drop=True)
    if not observed_images.equals(expected_images):
        raise ValueError(
            "Inference ledger image order and paths do not match the selected index."
        )
    normalized["candidate_count"] = _integer_series(
        normalized["candidate_count"],
        "Inference-ledger candidate_count",
    )
    actual_counts = (
        raw_predictions.groupby("image_file").size()
        if not raw_predictions.empty
        else pd.Series(dtype="int64")
    )
    expected_counts = normalized["image_file"].map(actual_counts).fillna(0).astype("int64")
    if not np.array_equal(
        normalized["candidate_count"].to_numpy(),
        expected_counts.to_numpy(),
    ):
        raise ValueError(
            "Inference-ledger candidate counts do not match the raw-prediction cache."
        )
    return normalized.reset_index(drop=True)


def _serialize_detection(
    image_row,
    condition,
    box,
    class_id,
    score,
    vector,
    classes,
    model_name=MODEL_NAME,
):
    class_id = int(class_id)
    object_score = float(score)
    probabilities = np.asarray(vector, dtype=float).reshape(-1)
    predicted_score = float(probabilities[class_id])
    return {
        "model": model_name,
        "dataset": DATASET_NAME,
        "augmentation_condition": condition["tag"],
        "augmentation_display": condition["display"],
        "image_file": image_row.image_file,
        "image_path": image_row.image_path,
        "bbox_x": float(box[0]),
        "bbox_y": float(box[1]),
        "bbox_w": float(box[2]),
        "bbox_h": float(box[3]),
        "class_id": class_id,
        "class_name": classes[class_id],
        "object_score": object_score,
        "predicted_class_score": predicted_score,
        "combined_confidence": object_score * predicted_score,
        "class_scores_json": json.dumps([float(value) for value in probabilities]),
    }


def run_raw_inference(
    index,
    classes,
    condition,
    asset_root,
    model_name=MODEL_NAME,
    model_assets=None,
    ledger_rows=None,
):
    import cv2

    from detector_service.modules.inference.model import Detector

    root = Path(asset_root).expanduser().absolute()
    paths = (
        {name: Path(path) for name, path in model_assets.items()}
        if model_assets is not None
        else {name: root / relative for name, relative in MODEL_ASSETS.items()}
    )
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing model assets:\n{details}")
    detector = Detector(
        str(paths["weights"]),
        str(paths["config"]),
        str(paths["classes"]),
        score_threshold=OBJECTNESS_THRESHOLD,
    )
    rows = []
    start = time.perf_counter()
    for position, image_row in enumerate(index.itertuples(index=False), start=1):
        image_path = resolve_indexed_path(image_row.image_path, root)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not decode inference image: {image_path}")
        transformed = apply_condition(image, condition)
        outputs = detector.predict(transformed)
        decoded = detector.post_process(outputs)
        for values in zip(*decoded):
            rows.append(
                _serialize_detection(
                    image_row,
                    condition,
                    *values,
                    classes,
                    model_name=model_name,
                )
            )
        if ledger_rows is not None:
            ledger_rows.append(
                {
                    "model": model_name,
                    "dataset": DATASET_NAME,
                    "augmentation_condition": condition["tag"],
                    "image_file": image_row.image_file,
                    "image_path": image_row.image_path,
                    "status": "processed",
                    "candidate_count": len(decoded[0]),
                }
            )
        if position % 250 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[RAW {condition['tag']}] Processed {position}/{len(index)} "
                f"images | candidates={len(rows)} | elapsed={elapsed:.1f}s"
            )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def apply_fixed_nms(
    raw_predictions,
    index,
    classes,
    condition,
    model_name=MODEL_NAME,
    nms_iou_threshold=NMS_IOU_THRESHOLD,
):
    nms = NMS(
        score_threshold=SCORE_THRESHOLD,
        nms_iou_threshold=nms_iou_threshold,
    )
    groups = (
        {
            image_file: group
            for image_file, group in raw_predictions.groupby("image_file", sort=False)
        }
        if not raw_predictions.empty
        else {}
    )
    rows = []
    for image_row in index.itertuples(index=False):
        group = groups.get(image_row.image_file)
        if group is None or group.empty:
            continue
        boxes = group[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(dtype=float).tolist()
        class_ids = group["class_id"].astype(int).tolist()
        scores = group["object_score"].astype(float).tolist()
        vectors = [json.loads(value) for value in group["class_scores_json"]]
        retained = nms.filter(boxes, class_ids, scores, vectors)
        for values in zip(*retained):
            row = _serialize_detection(
                image_row,
                condition,
                *values,
                classes,
                model_name=model_name,
            )
            row["nms_threshold"] = nms_iou_threshold
            rows.append(row)
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def validate_predictions(
    table,
    index,
    classes,
    condition,
    raw_predictions=None,
    model_name=MODEL_NAME,
    nms_iou_threshold=NMS_IOU_THRESHOLD,
):
    _required_columns(table, PREDICTION_COLUMNS, "Prediction cache")
    normalized = validate_raw_predictions(
        table[RAW_COLUMNS],
        index,
        classes,
        condition,
        model_name=model_name,
    )
    thresholds = pd.to_numeric(table["nms_threshold"], errors="coerce")
    if thresholds.isna().any() or not np.allclose(thresholds, nms_iou_threshold):
        raise ValueError("Prediction cache uses the wrong NMS threshold.")
    normalized["nms_threshold"] = thresholds.to_numpy(dtype=float)
    normalized = normalized[PREDICTION_COLUMNS].reset_index(drop=True)
    if raw_predictions is not None:
        expected = apply_fixed_nms(
            raw_predictions,
            index,
            classes,
            condition,
            model_name=model_name,
            nms_iou_threshold=nms_iou_threshold,
        ).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                normalized,
                expected,
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as error:
            raise EvidenceContractError(
                "Prediction cache does not match recomputation from its "
                "validated raw-prediction cache."
            ) from error
    return normalized


def build_metric_lists(index, predictions, ground_truth):
    prediction_groups = (
        {name: group for name, group in predictions.groupby("image_file", sort=False)}
        if not predictions.empty
        else {}
    )
    truth_groups = (
        {name: group for name, group in ground_truth.groupby("image_file", sort=False)}
        if not ground_truth.empty
        else {}
    )
    boxes, classes, scores, vectors, truth_boxes, truth_classes = [], [], [], [], [], []
    for image_row in index.itertuples(index=False):
        predicted = prediction_groups.get(image_row.image_file)
        if predicted is None:
            boxes.append([])
            classes.append([])
            scores.append([])
            vectors.append([])
        else:
            boxes.append(predicted[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(dtype=float).tolist())
            classes.append(predicted["class_id"].astype(int).tolist())
            scores.append(predicted["object_score"].astype(float).tolist())
            vectors.append([json.loads(value) for value in predicted["class_scores_json"]])
        truth = truth_groups.get(image_row.image_file)
        if truth is None:
            truth_boxes.append([])
            truth_classes.append([])
        else:
            truth_boxes.append(truth[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(dtype=float).tolist())
            truth_classes.append(truth["class_id"].astype(int).tolist())
    return boxes, classes, scores, vectors, truth_boxes, truth_classes


def evaluate_condition(
    index,
    predictions,
    ground_truth,
    classes,
    condition,
    model_name=MODEL_NAME,
    nms_iou_threshold=NMS_IOU_THRESHOLD,
):
    metric_inputs = build_metric_lists(index, predictions, ground_truth)
    matches, truth_counts = match_detections(
        boxes=metric_inputs[0],
        classes=metric_inputs[1],
        scores=metric_inputs[2],
        cls_scores=metric_inputs[3],
        gt_boxes=metric_inputs[4],
        gt_classes=metric_inputs[5],
        map_iou_threshold=MAP_IOU_THRESHOLD,
        eval_type=EVAL_TYPE,
    )
    precision, recall, _ = calculate_precision_recall_curve(
        matches,
        truth_counts,
        num_classes=len(classes),
    )
    points = {
        class_id: list(zip(recall[class_id], precision[class_id]))
        for class_id in range(len(classes))
    }
    map_score = calculate_map_x_point_interpolated(
        points,
        num_classes=len(classes),
        num_interpolated_points=11,
    )
    per_class_rows = []
    for class_id, class_name in enumerate(classes):
        class_ap = calculate_map_x_point_interpolated(
            {0: points[class_id]},
            num_classes=1,
            num_interpolated_points=11,
        )
        per_class_rows.append(
            {
                "model": model_name,
                "dataset": DATASET_NAME,
                "augmentation_condition": condition["tag"],
                "augmentation_display": condition["display"],
                "class_id": class_id,
                "class_name": class_name,
                "ground_truth_count": int((ground_truth["class_id"] == class_id).sum()),
                "prediction_count": int((predictions["class_id"] == class_id).sum()),
                "ap_11_point": class_ap,
            }
        )
    summary = {
        "model": model_name,
        "dataset": DATASET_NAME,
        "augmentation_condition": condition["tag"],
        "augmentation_display": condition["display"],
        "mAP@0.5_11_point": map_score,
        "total_ground_truth": int(len(ground_truth)),
        "total_predictions_after_nms": int(len(predictions)),
        "evaluation_rows": int(sum(len(rows) for rows in matches.values())),
        "candidate_objectness_threshold": OBJECTNESS_THRESHOLD,
        "nms_confidence_threshold": SCORE_THRESHOLD,
        "nms_iou_threshold": nms_iou_threshold,
        "map_iou_threshold": MAP_IOU_THRESHOLD,
        "eval_type": EVAL_TYPE,
    }
    return summary, pd.DataFrame(per_class_rows)


def add_baseline_changes(summary, per_class):
    baseline_rows = summary[summary["augmentation_condition"] == "original"]
    if len(baseline_rows) != 1:
        raise ValueError("Exactly one original-condition summary row is required.")
    baseline_map = float(baseline_rows.iloc[0]["mAP@0.5_11_point"])
    baseline_predictions = int(baseline_rows.iloc[0]["total_predictions_after_nms"])
    summary = summary.copy()
    summary["mAP_change_vs_original"] = summary["mAP@0.5_11_point"] - baseline_map
    summary["mAP_percent_change_vs_original"] = np.where(
        baseline_map == 0.0,
        0.0,
        summary["mAP_change_vs_original"] / baseline_map * 100.0,
    )
    summary["prediction_change_vs_original"] = (
        summary["total_predictions_after_nms"] - baseline_predictions
    )
    baseline_ap = (
        per_class[per_class["augmentation_condition"] == "original"]
        [["class_id", "ap_11_point"]]
        .rename(columns={"ap_11_point": "original_ap_11_point"})
    )
    if baseline_ap["class_id"].duplicated().any():
        raise ValueError("Original per-class evidence contains duplicate class IDs.")
    per_class = per_class.merge(baseline_ap, on="class_id", how="left", validate="many_to_one")
    if per_class["original_ap_11_point"].isna().any():
        raise ValueError("Original per-class evidence does not cover every class.")
    per_class["ap_change_vs_original"] = (
        per_class["ap_11_point"] - per_class["original_ap_11_point"]
    )
    return summary[SUMMARY_COLUMNS], per_class[PER_CLASS_COLUMNS]


def build_figures(summary, per_class, figure_dir):
    import matplotlib.pyplot as plt

    directory = Path(figure_dir).expanduser().absolute()
    directory.mkdir(parents=True, exist_ok=True)
    order = {condition["tag"]: position for position, condition in enumerate(CONDITIONS)}
    ordered = summary.copy()
    ordered["condition_order"] = ordered["augmentation_condition"].map(order)
    if ordered["condition_order"].isna().any():
        raise ValueError("Summary contains an unknown augmentation condition.")
    ordered = ordered.sort_values("condition_order")
    definitions = [
        (
            "02_map_by_condition.png",
            ordered["mAP@0.5_11_point"],
            "11-point mean AP at IoU 0.5",
            "Threshold-constrained detection score by image condition",
        ),
        (
            "04_prediction_count_by_condition.png",
            ordered["total_predictions_after_nms"],
            "Predictions retained after NMS",
            "Post-NMS prediction count by image condition",
        ),
    ]
    paths = []
    for name, values, y_label, title in definitions:
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.bar(ordered["augmentation_display"], values)
        axis.set_xlabel("Image condition")
        axis.set_ylabel(y_label)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylim(bottom=0.0)
        figure.tight_layout()
        path = directory / name
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

    changed = ordered[ordered["augmentation_condition"] != "original"].copy()
    changed["mAP_drop_vs_original"] = -changed["mAP_change_vs_original"]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(changed["augmentation_display"], changed["mAP_drop_vs_original"])
    axis.set_xlabel("Image condition")
    axis.set_ylabel("Mean AP drop from original images")
    axis.set_title("Detection degradation under controlled perturbations")
    axis.tick_params(axis="x", rotation=25)
    axis.set_ylim(bottom=0.0)
    figure.tight_layout()
    path = directory / "03_map_drop_vs_baseline.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    changed_classes = per_class[per_class["augmentation_condition"] != "original"].copy()
    changed_classes["ap_drop_vs_original"] = -changed_classes["ap_change_vs_original"]
    largest = changed_classes.sort_values(
        ["ap_drop_vs_original", "augmentation_condition", "class_id"],
        ascending=[False, True, True],
    ).head(12)
    largest = largest.copy()
    largest["label"] = largest["augmentation_display"] + " / " + largest["class_name"]
    largest = largest.sort_values("ap_drop_vs_original")
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(largest["label"], largest["ap_drop_vs_original"])
    axis.set_xlabel("11-point AP drop from original images")
    axis.set_ylabel("Image condition / class")
    axis.set_title("Largest per-class score drops under perturbation")
    figure.tight_layout()
    path = directory / "05_largest_per_class_ap_drops.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)
    return sorted(paths)


def _validate_derived_artifacts(summary, per_class):
    """Reject partial or policy-mixed tables before rendering retained evidence."""

    if summary.empty or per_class.empty:
        raise EvidenceContractError(
            "Derived robustness artifacts must be non-empty and complete."
        )
    expected_tags = {condition["tag"] for condition in CONDITIONS}
    if set(summary["augmentation_condition"].astype(str)) != expected_tags:
        raise EvidenceContractError(
            "Robustness summary does not contain the exact condition set."
        )
    if summary["augmentation_condition"].duplicated().any():
        raise EvidenceContractError(
            "Robustness summary must contain one row per condition."
        )
    if set(per_class["augmentation_condition"].astype(str)) != expected_tags:
        raise EvidenceContractError(
            "Per-class robustness evidence does not contain the exact condition set."
        )
    expected_constants = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "eval_type": EVAL_TYPE,
    }
    for label, table in (("summary", summary), ("per-class evidence", per_class)):
        for column in ("model", "dataset"):
            observed = set(table[column].astype(str))
            if observed != {expected_constants[column]}:
                raise EvidenceContractError(
                    f"Robustness {label} {column} identity is incomplete or mixed."
                )
    if set(summary["eval_type"].astype(str)) != {EVAL_TYPE}:
        raise EvidenceContractError("Robustness summary evaluation type is invalid.")
    numeric_policy = {
        "candidate_objectness_threshold": OBJECTNESS_THRESHOLD,
        "nms_confidence_threshold": SCORE_THRESHOLD,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "map_iou_threshold": MAP_IOU_THRESHOLD,
    }
    for column, expected in numeric_policy.items():
        values = pd.to_numeric(summary[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.allclose(values, expected):
            raise EvidenceContractError(
                f"Robustness summary {column} does not match the fixed policy."
            )

    normalized_summary = summary.copy()
    normalized_per_class = per_class.copy()
    for condition in CONDITIONS:
        summary_mask = normalized_summary["augmentation_condition"] == condition["tag"]
        normalized_summary.loc[summary_mask] = _normalize_display_label(
            normalized_summary.loc[summary_mask],
            condition,
            "Robustness summary",
        )
        class_mask = normalized_per_class["augmentation_condition"] == condition["tag"]
        normalized_per_class.loc[class_mask] = _normalize_display_label(
            normalized_per_class.loc[class_mask],
            condition,
            "Per-class robustness evidence",
        )

    normalized_per_class["class_id"] = _integer_series(
        normalized_per_class["class_id"],
        "Per-class robustness class_id",
    )
    if normalized_per_class[
        ["augmentation_condition", "class_id"]
    ].duplicated().any():
        raise EvidenceContractError(
            "Per-class robustness evidence contains duplicate condition/class rows."
        )
    baseline = normalized_per_class[
        normalized_per_class["augmentation_condition"] == "original"
    ][["class_id", "class_name"]].sort_values("class_id")
    expected_ids = list(range(len(baseline)))
    if baseline["class_id"].tolist() != expected_ids:
        raise EvidenceContractError(
            "Per-class robustness vocabulary must use contiguous IDs from zero."
        )
    expected_names = baseline["class_name"].astype(str).tolist()
    if any(not name.strip() for name in expected_names) or len(expected_names) != len(
        set(expected_names)
    ):
        raise EvidenceContractError(
            "Per-class robustness vocabulary must contain unique, non-empty names."
        )
    for condition in CONDITIONS:
        condition_rows = normalized_per_class[
            normalized_per_class["augmentation_condition"] == condition["tag"]
        ].sort_values("class_id")
        if condition_rows["class_id"].tolist() != expected_ids or condition_rows[
            "class_name"
        ].astype(str).tolist() != expected_names:
            raise EvidenceContractError(
                "Per-class robustness vocabulary is incomplete or changes by condition."
            )

    summary_numeric = [
        column
        for column in SUMMARY_COLUMNS
        if column
        not in {"model", "dataset", "augmentation_condition", "augmentation_display", "eval_type"}
    ]
    class_numeric = [
        column
        for column in PER_CLASS_COLUMNS
        if column
        not in {"model", "dataset", "augmentation_condition", "augmentation_display", "class_name"}
    ]
    for label, table, columns in (
        ("summary", normalized_summary, summary_numeric),
        ("per-class evidence", normalized_per_class, class_numeric),
    ):
        values = table[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise EvidenceContractError(
                f"Robustness {label} contains non-finite numeric values."
            )
    return (
        normalized_summary[SUMMARY_COLUMNS].reset_index(drop=True),
        normalized_per_class[PER_CLASS_COLUMNS].reset_index(drop=True),
    )


def load_derived_artifacts(output_dir, run_label):
    directory = Path(output_dir).expanduser().absolute()
    paths = {
        "summary": directory / f"summary_by_condition_{run_label}.csv",
        "per_class": directory / f"per_class_ap_by_condition_{run_label}.csv",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing derived robustness artifacts:\n{details}")
    summary = pd.read_csv(paths["summary"])
    per_class = pd.read_csv(paths["per_class"])
    _required_columns(summary, SUMMARY_COLUMNS, "Robustness summary")
    _required_columns(per_class, PER_CLASS_COLUMNS, "Per-class robustness summary")
    return _validate_derived_artifacts(
        summary[SUMMARY_COLUMNS],
        per_class[PER_CLASS_COLUMNS],
    )


def run_experiment(
    index,
    classes,
    output_dir,
    cache_input_dir,
    asset_root,
    run_label,
    force=False,
    refresh_postprocessing=False,
    model_name=MODEL_NAME,
    nms_iou_threshold=NMS_IOU_THRESHOLD,
    model_assets=None,
):
    summary_rows = []
    per_class_tables = []
    prediction_paths = []
    for condition in CONDITIONS:
        managed = _cache_paths(
            output_dir,
            condition,
            run_label,
            model_name=model_name,
            nms_iou_threshold=nms_iou_threshold,
        )
        inputs = (
            _cache_paths(
                cache_input_dir,
                condition,
                run_label,
                model_name=model_name,
                nms_iou_threshold=nms_iou_threshold,
            )
            if cache_input_dir
            else managed
        )
        if force or not inputs["ground_truth"].is_file():
            if cache_input_dir:
                raise FileNotFoundError(f"Ground-truth cache not found: {inputs['ground_truth']}")
            ground_truth = build_ground_truth(index, classes, condition, asset_root)
            _write_csv(managed["ground_truth"], ground_truth)
            print(f"[WRITE] {managed['ground_truth']} rows={len(ground_truth)}")
        else:
            ground_truth = pd.read_csv(inputs["ground_truth"])
        ground_truth = validate_ground_truth(ground_truth, index, classes, condition)

        if force or not inputs["raw"].is_file():
            if cache_input_dir:
                raise FileNotFoundError(f"Raw-prediction cache not found: {inputs['raw']}")
            ledger_rows = []
            raw = run_raw_inference(
                index,
                classes,
                condition,
                asset_root,
                model_name=model_name,
                model_assets=model_assets,
                ledger_rows=ledger_rows,
            )
            ledger = pd.DataFrame(ledger_rows, columns=LEDGER_COLUMNS)
            _write_csv(managed["raw"], raw)
            _write_csv(managed["ledger"], ledger)
            print(f"[WRITE] {managed['raw']} rows={len(raw)}")
            print(f"[WRITE] {managed['ledger']} rows={len(ledger)}")
        else:
            raw = pd.read_csv(inputs["raw"])
            if not inputs["ledger"].is_file():
                raise FileNotFoundError(
                    "Raw-prediction cache has no completeness ledger. Rebuild the "
                    f"condition or provide its ledger: {inputs['ledger']}"
                )
            ledger = pd.read_csv(inputs["ledger"])
        raw = validate_raw_predictions(
            raw,
            index,
            classes,
            condition,
            model_name=model_name,
        )
        validate_inference_ledger(
            ledger,
            index,
            raw,
            condition,
            model_name=model_name,
        )

        if managed["predictions"].is_file() and not refresh_postprocessing and not force:
            predictions = pd.read_csv(managed["predictions"])
        else:
            predictions = apply_fixed_nms(
                raw,
                index,
                classes,
                condition,
                model_name=model_name,
                nms_iou_threshold=nms_iou_threshold,
            )
            _write_csv(managed["predictions"], predictions)
            print(f"[WRITE] {managed['predictions']} rows={len(predictions)}")
        predictions = validate_predictions(
            predictions,
            index,
            classes,
            condition,
            raw_predictions=raw,
            model_name=model_name,
            nms_iou_threshold=nms_iou_threshold,
        )
        prediction_paths.append(managed["predictions"])

        summary, per_class = evaluate_condition(
            index,
            predictions,
            ground_truth,
            classes,
            condition,
            model_name=model_name,
            nms_iou_threshold=nms_iou_threshold,
        )
        summary_rows.append(summary)
        per_class_tables.append(per_class)
    summary = pd.DataFrame(summary_rows)
    per_class = pd.concat(per_class_tables, ignore_index=True)
    summary, per_class = add_baseline_changes(summary, per_class)
    return summary, per_class, prediction_paths


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the verified selected checkpoint across fixed image "
            "perturbation conditions."
        )
    )
    parser.add_argument("--sample-index", type=Path, default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument("--selection-run", type=Path, default=DEFAULT_SELECTION_RUN)
    parser.add_argument(
        "--operating-point",
        type=Path,
        default=DEFAULT_OPERATING_POINT,
    )
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument(
        "--class-file",
        type=Path,
        help=(
            "Optional alternate location of the selected checkpoint's exact "
            "class-name file; its SHA-256 identity is verified."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--cache-input-dir",
        type=Path,
        help="Read ground-truth and raw-prediction caches from this directory without modifying it.",
    )
    parser.add_argument("--max-images", type=positive_int)
    parser.add_argument("--refresh-postprocessing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.force and args.cache_input_dir:
        raise ValueError(
            "--force cannot be combined with --cache-input-dir; external caches are read-only."
        )
    run_label = _run_label(args.max_images)
    if args.figures_only:
        summary, per_class = load_derived_artifacts(args.output_dir, run_label)
        figures = build_figures(summary, per_class, args.figure_dir)
        for path in figures:
            print(f"[WRITE] {path}")
        return summary, per_class, {}, figures

    index = load_sample_index(args.sample_index, args.max_images)
    selection = load_verified_checkpoint_selection(args.selection_run)
    operating_point = load_verified_operating_point(
        args.operating_point,
        args.selection_run,
    )
    model_name = selection["selected_model"]
    nms_iou_threshold = float(operating_point["selected_nms_iou_threshold"])
    output_dir = Path(args.output_dir).expanduser().absolute()
    asset_root = Path(args.asset_root).expanduser().absolute()
    cache_input = (
        Path(args.cache_input_dir).expanduser().absolute()
        if args.cache_input_dir
        else None
    )
    _assert_disjoint_directories(cache_input, output_dir)
    cache_root = cache_input or output_dir
    primary_paths = _primary_cache_paths(
        cache_root,
        run_label,
        model_name=model_name,
        nms_iou_threshold=nms_iou_threshold,
    )
    needs_inference = cache_input is None and (
        args.force
        or any(
            not _cache_paths(
                cache_root,
                condition,
                run_label,
                model_name=model_name,
                nms_iou_threshold=nms_iou_threshold,
            )["raw"].is_file()
            for condition in CONDITIONS
        )
    )
    selected_assets = (
        resolve_selected_model_assets(asset_root, args.selection_run)
        if needs_inference
        else None
    )
    selected_relative_assets = selection["asset_paths"]
    class_file = args.class_file or (
        selected_assets["resolved_paths"]["classes"]
        if selected_assets is not None
        else asset_root / selected_relative_assets["classes"]
    )
    classes = load_classes(
        class_file,
        selection["model_identity"]["names"],
    )
    any_primary_evidence = any(
        path.is_file()
        for condition_paths in primary_paths.values()
        for path in condition_paths.values()
    )
    if cache_input is not None or (any_primary_evidence and not args.force):
        validate_condition_cache_manifest(
            cache_root,
            index,
            args.sample_index,
            selection,
            operating_point,
            classes,
            run_label,
            model_name=model_name,
            nms_iou_threshold=nms_iou_threshold,
        )
    print(f"[INFO] Images selected: {len(index)}")
    print(f"[INFO] Classes: {len(classes)}")
    print(
        f"[INFO] Model held constant: {model_name} "
        f"(checkpoint {selection['selected_checkpoint']}, "
        f"selection run {selection['selection_run_id']})"
    )
    print(f"[INFO] Conditions: {[condition['tag'] for condition in CONDITIONS]}")
    print(
        f"[INFO] Policy: objectness>{OBJECTNESS_THRESHOLD}, "
        f"combined confidence>={SCORE_THRESHOLD}, NMS IoU={nms_iou_threshold}"
    )
    summary, per_class, prediction_paths = run_experiment(
        index,
        classes,
        output_dir,
        cache_input,
        asset_root,
        run_label,
        force=args.force,
        refresh_postprocessing=args.refresh_postprocessing,
        model_name=model_name,
        nms_iou_threshold=nms_iou_threshold,
        model_assets=(
            selected_assets["resolved_paths"]
            if selected_assets is not None
            else None
        ),
    )
    artifact_paths = {
        "summary": _write_csv(
            output_dir / f"summary_by_condition_{run_label}.csv",
            summary,
        ),
        "per_class": _write_csv(
            output_dir / f"per_class_ap_by_condition_{run_label}.csv",
            per_class,
        ),
    }
    if cache_input is None:
        artifact_paths["cache_manifest"] = write_condition_cache_manifest(
            output_dir,
            index,
            args.sample_index,
            selection,
            operating_point,
            classes,
            run_label,
            model_name=model_name,
            nms_iou_threshold=nms_iou_threshold,
        )
    figures = []
    if not args.skip_figures:
        figures = build_figures(summary, per_class, args.figure_dir)
    print("\nAUGMENTATION ROBUSTNESS SUMMARY")
    print(summary.to_string(index=False))
    for path in [*artifact_paths.values(), *figures]:
        print(f"[WRITE] {path}")
    return summary, per_class, {"derived": artifact_paths, "predictions": prediction_paths}, figures


if __name__ == "__main__":
    main()
