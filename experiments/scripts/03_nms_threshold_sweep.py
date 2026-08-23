"""Evaluate one cached detector output across controlled NMS IoU thresholds."""

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector_service.modules.inference.nms import NMS
from detector_service.modules.utils.metrics import (
    calculate_iou,
    calculate_map_x_point_interpolated,
    calculate_precision_recall_curve,
    match_detections,
)
from experiments.scripts.experiment_contracts import (
    EvidenceContractError,
    load_verified_checkpoint_selection,
    resolve_indexed_path as resolve_contract_path,
    resolve_selected_model_assets,
    sha256_file,
    threshold_tag as contract_threshold_tag,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "outputs"
DEFAULT_SAMPLE_SELECTION_DIR = (
    DEFAULT_OUTPUT_ROOT / "02_dataset_analysis" / "02_sample_selection"
)
DEFAULT_OVERLAP_DIR = (
    DEFAULT_OUTPUT_ROOT / "02_dataset_analysis" / "03_overlap_analysis"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_OUTPUT_ROOT / "03_nms_thresholding" / "01_threshold_sweep"
)
DEFAULT_SELECTION_RUN = (
    DEFAULT_OUTPUT_ROOT
    / "01_model_selection"
    / "03_checkpoint_decision"
    / "selection-20260821-v1"
)
DEFAULT_FIGURE_DIR = (
    PROJECT_ROOT / "scratch" / "diagnostic-figures" / "03_nms_thresholding"
)
DEFAULT_SAMPLE_INDEX = DEFAULT_SAMPLE_SELECTION_DIR / "selected_sample_index.csv"
DEFAULT_OVERLAP_PROFILE = DEFAULT_OVERLAP_DIR / "overlap_profile.csv"
DEFAULT_CLASS_FILE = (
    PROJECT_ROOT / "detector_service" / "storage" / "yolo_model_2" / "logistics.names"
)

MODEL_NAME = "model2"
DATASET_NAME = "rare_aware_density_stratified_5000"
SCORE_THRESHOLD = 0.5
MAP_IOU_THRESHOLD = 0.5
EVAL_TYPE = "combined"
NMS_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7)
DUPLICATE_IOU_THRESHOLD = 0.5
INFERENCE_CACHE_SCHEMA_VERSION = 1

RAW_COLUMNS = [
    "model",
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
    "image_file",
    "image_path",
    "status",
    "candidate_count",
]
PREDICTION_COLUMNS = RAW_COLUMNS + ["nms_threshold"]
GROUND_TRUTH_COLUMNS = [
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
    "nms_threshold",
    "mAP@0.5_11_point",
    "total_ground_truth",
    "total_predictions_after_nms",
    "evaluation_rows",
    "score_threshold",
    "map_iou_threshold",
    "eval_type",
]
PER_CLASS_COLUMNS = [
    "model",
    "dataset",
    "nms_threshold",
    "class_id",
    "class_name",
    "ground_truth_count",
    "prediction_count",
    "ap_11_point",
]
DUPLICATE_COLUMNS = [
    "duplicate_like_pairs_iou_gt_0_5",
    "images_with_duplicate_like_pairs",
    "mean_duplicate_like_pairs_per_image",
    "nms_threshold",
    "total_predictions_after_nms",
]
SUBSET_COLUMNS = [
    "subset_name",
    "image_count",
    "ground_truth_count",
    "nms_threshold",
    "mAP@0.5_11_point",
    "total_predictions_after_nms",
    "evaluation_rows",
    "duplicate_like_pairs_iou_gt_0_5",
    "images_with_duplicate_like_pairs",
    "mean_duplicate_like_pairs_per_image",
]
INFERENCE_CACHE_ARTIFACTS = {
    "ground_truth": GROUND_TRUTH_COLUMNS,
    "raw_predictions": RAW_COLUMNS,
    "inference_ledger": LEDGER_COLUMNS,
}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _required_columns(table, required, label):
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _validated_integer_series(series, label):
    numeric = pd.to_numeric(series, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain finite values.")
    if (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label} must contain non-negative integers.")
    return numeric.astype(np.int64)


def _validated_float_columns(table, columns, label):
    normalized = table.copy()
    for column in columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        values = normalized[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{label} {column} must contain finite values.")
    return normalized


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


def _write_json_atomic(path, payload):
    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
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
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def _ordered_vocabulary_sha256(classes):
    payload = json.dumps(
        list(classes),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selected_index_sha256(index):
    payload = index.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_exact_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise EvidenceContractError(
            f"{label} does not match the inference-cache contract: "
            f"expected={sorted(expected)}, observed={observed}"
        )


def build_inference_cache_manifest(
    *,
    sample_index_path,
    index,
    selection,
    classes,
    class_file,
    model_name,
    run_label,
    artifact_paths,
    artifact_tables,
):
    """Build a content-addressed manifest for validated pre-NMS evidence."""

    sample_source = Path(sample_index_path).expanduser().absolute()
    vocabulary_source = Path(class_file).expanduser().absolute()
    artifacts = {}
    for role in INFERENCE_CACHE_ARTIFACTS:
        path = Path(artifact_paths[role]).expanduser().absolute()
        table = artifact_tables[role]
        if not path.is_file():
            raise FileNotFoundError(f"Inference-cache artifact not found: {path}")
        artifacts[role] = {
            "file_name": path.name,
            "rows": int(len(table)),
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": INFERENCE_CACHE_SCHEMA_VERSION,
        "status": "complete",
        "run_label": run_label,
        "sample_index": {
            "file_name": sample_source.name,
            "file_sha256": sha256_file(sample_source),
            "selected_rows": int(len(index)),
            "selected_rows_sha256": _selected_index_sha256(index),
        },
        "checkpoint_selection": {
            "run_id": selection["selection_run_id"],
            "manifest_sha256": selection["selection_manifest_sha256"],
            "decision_sha256": selection["decision_sha256"],
        },
        "model": {
            "dataset": DATASET_NAME,
            "selected_model": model_name,
            "asset_identities": dict(selection["model_identity"]),
        },
        "vocabulary": {
            "file_name": vocabulary_source.name,
            "file_sha256": sha256_file(vocabulary_source),
            "class_count": len(classes),
            "ordered_classes_sha256": _ordered_vocabulary_sha256(classes),
        },
        "candidate_policy": {
            "objectness_operator": ">",
            "objectness_threshold": SCORE_THRESHOLD,
            "predicted_class_rule": "argmax(class_scores_json)",
            "combined_confidence_formula": (
                "object_score * predicted_class_score"
            ),
            "postprocessing_confidence_operator": ">=",
            "postprocessing_confidence_threshold": SCORE_THRESHOLD,
        },
        "artifacts": artifacts,
    }


def validate_inference_cache_manifest(
    manifest_path,
    *,
    sample_index_path,
    index,
    selection,
    classes,
    class_file,
    model_name,
    run_label,
    artifact_paths,
    artifact_tables,
):
    """Verify that cached pre-NMS evidence belongs to this exact experiment run."""

    source = Path(manifest_path).expanduser().absolute()
    if not source.is_file():
        raise FileNotFoundError(
            "Inference-cache manifest not found. Historical raw caches are not "
            f"trusted; regenerate them with --force: {source}"
        )
    try:
        observed = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceContractError(
            f"Inference-cache manifest is not valid JSON: {source}"
        ) from error
    expected = build_inference_cache_manifest(
        sample_index_path=sample_index_path,
        index=index,
        selection=selection,
        classes=classes,
        class_file=class_file,
        model_name=model_name,
        run_label=run_label,
        artifact_paths=artifact_paths,
        artifact_tables=artifact_tables,
    )
    _require_exact_keys(observed, expected, "Inference-cache manifest")
    if observed != expected:
        raise EvidenceContractError(
            "Inference-cache manifest is stale or does not match the selected "
            "index, checkpoint, policy, vocabulary, or cache artifacts."
        )
    return observed


def resolve_indexed_path(value, asset_root=None, project_root=PROJECT_ROOT):
    """Resolve a logical index path while enforcing root containment."""

    return resolve_contract_path(
        value,
        asset_root=asset_root,
        project_root=project_root,
    )


def load_sample_index(path, max_images=None):
    """Load a unique selected-image manifest and apply an optional prefix bound."""

    source = Path(path).expanduser().absolute()
    if not source.is_file():
        raise FileNotFoundError(
            f"Selected sample index not found: {source}. "
            "Run experiments/scripts/02_dataset_sampling.py first."
        )
    index = pd.read_csv(source)
    required = ["image_file", "image_path", "label_path", "num_objects"]
    _required_columns(index, required, "Selected sample index")
    if index.empty:
        raise RuntimeError("Selected sample index is empty.")
    if index["image_file"].isna().any() or index["image_path"].isna().any():
        raise ValueError("Selected sample index contains missing image identifiers.")
    if index["image_file"].duplicated().any():
        raise ValueError("Selected sample index contains duplicate image_file values.")
    if index["image_path"].duplicated().any():
        raise ValueError("Selected sample index contains duplicate image_path values.")
    index["num_objects"] = _validated_integer_series(index["num_objects"], "num_objects")

    if max_images is not None:
        if max_images > len(index):
            raise ValueError(
                f"max_images {max_images} exceeds selected sample size {len(index)}."
            )
        index = index.head(max_images).copy()
    return index


def load_classes(class_file, expected_sha256=None):
    """Load a complete, ordered ``.names`` vocabulary with optional identity proof."""

    source = Path(class_file).expanduser().absolute()
    if not source.is_file():
        raise FileNotFoundError(
            "Verified full class vocabulary not found. Cached detections cannot prove "
            f"that trailing unseen classes are absent: {source}"
        )
    if expected_sha256 is not None and sha256_file(source) != expected_sha256:
        raise EvidenceContractError(
            "Class-name file hash does not match the selected checkpoint identity: "
            f"{source}"
        )
    classes = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not classes:
        raise ValueError(f"Class-name file is empty: {source}")
    if len(classes) != len(set(classes)):
        raise ValueError("Class names must be unique.")
    return classes


def _canonical_logical_image_path(value):
    """Normalize an indexed image path without weakening its logical identity."""

    raw = str(value).strip().replace("\\", "/")
    if not raw or raw.lower() == "nan":
        raise ValueError("image_path values must be non-empty.")
    parts = tuple(
        part for part in PurePosixPath(raw).parts if part not in ("", ".", "/")
    )
    if ".." in parts:
        raise ValueError("image_path values cannot contain parent traversal.")
    if parts[:2] in {
        ("detector_service", "storage"),
        ("techtrack", "storage"),
    }:
        return ("storage", *parts[2:])
    return ("logical", *parts)


def _validate_image_paths_against_index(table, index, label):
    """Bind every evidence row to the selected index's path for its image file."""

    expected = {
        str(row.image_file): _canonical_logical_image_path(row.image_path)
        for row in index.itertuples(index=False)
    }
    for row_number, row in enumerate(
        table[["image_file", "image_path"]].itertuples(index=False),
        start=1,
    ):
        image_file = str(row.image_file)
        if image_file not in expected:
            raise ValueError(f"{label} contains images outside the selected index.")
        observed = _canonical_logical_image_path(row.image_path)
        if observed != expected[image_file]:
            raise ValueError(
                f"{label} row {row_number} image_path does not match the selected "
                f"index for {image_file!r}."
            )
    return table


def _validate_class_identity(table, classes, label):
    if table.empty:
        return table
    normalized = table.copy()
    normalized["class_id"] = _validated_integer_series(
        normalized["class_id"],
        f"{label} class_id",
    )
    if (normalized["class_id"] >= len(classes)).any():
        raise ValueError(f"{label} contains a class ID outside the class vocabulary.")
    expected_names = normalized["class_id"].map(dict(enumerate(classes)))
    if not normalized["class_name"].astype(str).equals(expected_names.astype(str)):
        raise ValueError(f"{label} class names do not match the class vocabulary.")
    return normalized


def validate_ground_truth(table, index, classes):
    """Validate a ground-truth cache against the selected manifest."""

    _required_columns(table, GROUND_TRUTH_COLUMNS, "Ground-truth cache")
    normalized = _validate_class_identity(table[GROUND_TRUTH_COLUMNS], classes, "Ground-truth cache")
    normalized = _validated_float_columns(
        normalized,
        ["bbox_x", "bbox_y", "bbox_w", "bbox_h"],
        "Ground-truth cache",
    )
    if (normalized[["bbox_w", "bbox_h"]] < 0).any().any():
        raise ValueError("Ground-truth widths and heights must be non-negative.")
    _validate_image_paths_against_index(normalized, index, "Ground-truth cache")

    observed = normalized.groupby("image_file").size()
    expected = index.set_index("image_file")["num_objects"]
    observed = observed.reindex(expected.index, fill_value=0).astype(np.int64)
    if not np.array_equal(observed.to_numpy(), expected.to_numpy()):
        raise ValueError("Ground-truth row counts do not match selected-index object counts.")
    return normalized.reset_index(drop=True)


def _parse_score_vector(serialized, row_number, classes):
    try:
        vector = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Raw prediction row {row_number} has invalid class_scores_json.") from error
    values = np.asarray(vector, dtype=float).reshape(-1)
    if len(values) != len(classes) or not np.isfinite(values).all():
        raise ValueError(
            f"Raw prediction row {row_number} must contain {len(classes)} finite class scores."
        )
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Class probabilities must be between zero and one.")
    return values


def validate_raw_predictions(table, index, classes, model_name=MODEL_NAME):
    """Validate raw cached detections and their derived score fields."""

    _required_columns(table, RAW_COLUMNS, "Raw-prediction cache")
    normalized = table[RAW_COLUMNS].copy()
    if normalized.empty:
        return normalized
    normalized = _validate_class_identity(normalized, classes, "Raw-prediction cache")
    normalized = _validated_float_columns(
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
        "Raw-prediction cache",
    )
    if set(normalized["model"].astype(str)) != {model_name}:
        raise ValueError(
            f"Raw-prediction cache must contain only {model_name} rows."
        )
    if (normalized[["bbox_w", "bbox_h"]] < 0).any().any():
        raise ValueError("Raw-prediction widths and heights must be non-negative.")
    for column in ("object_score", "predicted_class_score", "combined_confidence"):
        if not normalized[column].between(0.0, 1.0).all():
            raise ValueError(f"Raw-prediction {column} values must be probabilities.")
    _validate_image_paths_against_index(normalized, index, "Raw-prediction cache")

    predicted_scores = []
    predicted_class_ids = []
    for position, row in enumerate(normalized.itertuples(index=False), start=1):
        vector = _parse_score_vector(row.class_scores_json, position, classes)
        predicted_scores.append(float(vector[int(row.class_id)]))
        predicted_class_ids.append(int(np.argmax(vector)))
    if not np.array_equal(
        normalized["class_id"].to_numpy(dtype=np.int64),
        np.asarray(predicted_class_ids, dtype=np.int64),
    ):
        raise ValueError("Raw-prediction class_id must equal argmax(class_scores_json).")
    if not np.allclose(
        normalized["predicted_class_score"],
        predicted_scores,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError("Predicted-class scores do not match class-score vectors.")
    expected_combined = normalized["object_score"] * normalized["predicted_class_score"]
    if not np.allclose(
        normalized["combined_confidence"],
        expected_combined,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError("Combined confidence does not equal objectness times class probability.")
    return normalized.reset_index(drop=True)


def validate_inference_ledger(table, index, raw_predictions, model_name=MODEL_NAME):
    """Prove that inference completed once for every selected image."""

    _required_columns(table, LEDGER_COLUMNS, "Inference ledger")
    normalized = table[LEDGER_COLUMNS].copy()
    if len(normalized) != len(index):
        raise ValueError(
            "Inference ledger row count does not match the selected-image count."
        )
    if normalized["image_file"].duplicated().any():
        raise ValueError("Inference ledger contains duplicate image_file values.")
    if set(normalized["model"].astype(str)) != {model_name}:
        raise ValueError(f"Inference ledger must contain only {model_name} rows.")
    if set(normalized["status"].astype(str)) != {"processed"}:
        raise ValueError("Every inference-ledger row must have status='processed'.")

    expected_files = index["image_file"].astype(str).tolist()
    observed_files = normalized["image_file"].astype(str).tolist()
    if observed_files != expected_files:
        raise ValueError(
            "Inference ledger image order does not match the selected index."
        )
    _validate_image_paths_against_index(normalized, index, "Inference ledger")
    normalized["candidate_count"] = _validated_integer_series(
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


def load_overlap_profile(path, index):
    """Load overlap evidence and return complete and crowded selected views."""

    source = Path(path).expanduser().absolute()
    if not source.is_file():
        raise FileNotFoundError(
            f"Overlap profile not found: {source}. "
            "Run experiments/scripts/02_overlap_analysis.py first."
        )
    overlap = pd.read_csv(source)
    _required_columns(overlap, ["image_file", "pairs_iou_gt_0_1"], "Overlap profile")
    if overlap["image_file"].duplicated().any():
        raise ValueError("Overlap profile contains duplicate image_file values.")
    overlap["pairs_iou_gt_0_1"] = _validated_integer_series(
        overlap["pairs_iou_gt_0_1"],
        "pairs_iou_gt_0_1",
    )
    indexed = set(index["image_file"])
    available = set(overlap["image_file"])
    missing = indexed - available
    if missing:
        raise ValueError("Overlap profile does not cover every selected image.")
    crowded_files = set(
        overlap.loc[
            overlap["image_file"].isin(indexed)
            & (overlap["pairs_iou_gt_0_1"] > 0),
            "image_file",
        ]
    )
    return overlap, index[index["image_file"].isin(crowded_files)].copy()


def yolo_label_to_xywh(label_path, image_width, image_height, classes):
    """Convert a strict normalized YOLO label file into pixel-space rows."""

    path = Path(label_path)
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")
    width = int(image_width)
    height = int(image_height)
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")

    rows = []
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
            raise ValueError(f"{path}:{line_number}: YOLO fields must be numeric") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: YOLO fields must be finite")
        raw_class_id, center_x, center_y, box_width, box_height = values
        class_id = int(raw_class_id)
        if raw_class_id != class_id or not 0 <= class_id < len(classes):
            raise ValueError(f"{path}:{line_number}: class identifier is outside the vocabulary")
        if not (
            0.0 <= center_x <= 1.0
            and 0.0 <= center_y <= 1.0
            and 0.0 < box_width <= 1.0
            and 0.0 < box_height <= 1.0
        ):
            raise ValueError(f"{path}:{line_number}: invalid normalized YOLO geometry")
        pixel_width = box_width * width
        pixel_height = box_height * height
        rows.append(
            {
                "class_id": class_id,
                "class_name": classes[class_id],
                "bbox_x": center_x * width - pixel_width / 2.0,
                "bbox_y": center_y * height - pixel_height / 2.0,
                "bbox_w": pixel_width,
                "bbox_h": pixel_height,
            }
        )
    return rows


def build_ground_truth(index, classes, asset_root=None, image_reader=None, progress_interval=1000):
    """Decode selected images and build pixel-space ground truth."""

    if image_reader is None:
        import cv2

        image_reader = cv2.imread
    rows = []
    for position, row in enumerate(index.itertuples(index=False), start=1):
        image_path = resolve_indexed_path(row.image_path, asset_root=asset_root)
        label_path = resolve_indexed_path(row.label_path, asset_root=asset_root)
        image = image_reader(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read selected image: {image_path}")
        image_height, image_width = image.shape[:2]
        labels = yolo_label_to_xywh(label_path, image_width, image_height, classes)
        if len(labels) != int(row.num_objects):
            raise ValueError(
                f"Label count mismatch for {row.image_file}: "
                f"index={row.num_objects}, parsed={len(labels)}"
            )
        for label in labels:
            rows.append(
                {
                    "image_file": row.image_file,
                    "image_path": row.image_path,
                    **label,
                }
            )
        if progress_interval and position % progress_interval == 0:
            print(f"[GT] Processed {position}/{len(index)} images")
    return pd.DataFrame(rows, columns=GROUND_TRUTH_COLUMNS)


def _resolve_model_assets(asset_root, selection_run):
    root = asset_root or PROJECT_ROOT / "detector_service" / "storage"
    return resolve_selected_model_assets(root, selection_run)


def _serialize_raw_detection(
    image_row,
    bbox,
    class_id,
    object_score,
    score_vector,
    classes,
    model_name=MODEL_NAME,
):
    probabilities = np.asarray(score_vector, dtype=float).reshape(-1)
    if len(probabilities) != len(classes):
        raise ValueError("Detector class-score vector length does not match the vocabulary.")
    class_id = int(class_id)
    if not 0 <= class_id < len(classes):
        raise ValueError("Detector returned a class ID outside the vocabulary.")
    predicted_score = float(probabilities[class_id])
    objectness = float(object_score)
    return {
        "model": model_name,
        "image_file": image_row.image_file,
        "image_path": image_row.image_path,
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_w": float(bbox[2]),
        "bbox_h": float(bbox[3]),
        "class_id": class_id,
        "class_name": classes[class_id],
        "object_score": objectness,
        "predicted_class_score": predicted_score,
        "combined_confidence": objectness * predicted_score,
        "class_scores_json": json.dumps(probabilities.tolist()),
    }


def run_raw_inference(
    index,
    classes,
    asset_root=None,
    image_reader=None,
    detector_factory=None,
    progress_interval=250,
    model_name=MODEL_NAME,
    model_assets=None,
    selection_run=None,
    expected_names_sha256=None,
    ledger_rows=None,
):
    """Run the selected checkpoint once and retain pre-NMS evidence."""

    if image_reader is None:
        import cv2

        image_reader = cv2.imread
    if model_assets is None:
        if selection_run is None:
            raise ValueError(
                "selection_run is required when model assets are not pre-resolved."
            )
        resolved = _resolve_model_assets(asset_root, selection_run)
        model_name = resolved["selected_model"]
        paths = resolved["resolved_paths"]
        expected_names_sha256 = resolved["model_identity"]["names"]
    else:
        paths = {name: Path(path) for name, path in model_assets.items()}
        if expected_names_sha256 is None:
            raise ValueError(
                "expected_names_sha256 is required with pre-resolved model assets."
            )
    model_classes = load_classes(
        paths["classes"],
        expected_sha256=expected_names_sha256,
    )
    if list(classes) != model_classes:
        raise EvidenceContractError(
            "Inference class vocabulary does not match the selected checkpoint's "
            "ordered .names vocabulary."
        )
    if detector_factory is None:
        from detector_service.modules.inference.model import Detector

        detector_factory = Detector
    detector = detector_factory(
        str(paths["weights"]),
        str(paths["config"]),
        str(paths["classes"]),
        score_threshold=SCORE_THRESHOLD,
    )
    rows = []
    start = time.perf_counter()
    for position, image_row in enumerate(index.itertuples(index=False), start=1):
        image_path = resolve_indexed_path(image_row.image_path, asset_root=asset_root)
        image = image_reader(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read selected image: {image_path}")
        outputs = detector.predict(image)
        boxes, class_ids, object_scores, class_scores = detector.post_process(outputs)
        lengths = {len(boxes), len(class_ids), len(object_scores), len(class_scores)}
        if len(lengths) != 1:
            raise ValueError("Detector output collections must have equal lengths.")
        for values in zip(boxes, class_ids, object_scores, class_scores):
            rows.append(
                _serialize_raw_detection(
                    image_row,
                    *values,
                    classes,
                    model_name=model_name,
                )
            )
        if ledger_rows is not None:
            ledger_rows.append(
                {
                    "model": model_name,
                    "image_file": image_row.image_file,
                    "image_path": image_row.image_path,
                    "status": "processed",
                    "candidate_count": len(boxes),
                }
            )
        if progress_interval and position % progress_interval == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[RAW] Processed {position}/{len(index)} images | "
                f"candidates={len(rows)} | elapsed={elapsed:.1f}s"
            )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def threshold_tag(threshold):
    return contract_threshold_tag(threshold)


def apply_nms_threshold(
    raw_predictions,
    index,
    classes,
    nms_threshold,
    model_name=MODEL_NAME,
):
    """Apply the project class-aware NMS implementation at one IoU threshold."""

    threshold = float(nms_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("NMS IoU threshold must be between zero and one.")
    nms = NMS(score_threshold=SCORE_THRESHOLD, nms_iou_threshold=threshold)
    groups = (
        {name: group for name, group in raw_predictions.groupby("image_file", sort=False)}
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
        object_scores = group["object_score"].astype(float).tolist()
        class_scores = [json.loads(value) for value in group["class_scores_json"]]
        retained = nms.filter(boxes, class_ids, object_scores, class_scores)
        for values in zip(*retained):
            row = _serialize_raw_detection(
                image_row,
                *values,
                classes,
                model_name=model_name,
            )
            row["nms_threshold"] = threshold
            rows.append(row)
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def build_metric_lists(index, predictions, ground_truth):
    """Assemble image-aligned metric inputs, including empty images."""

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
    boxes, classes, scores, class_scores = [], [], [], []
    truth_boxes, truth_classes = [], []
    for row in index.itertuples(index=False):
        predicted = prediction_groups.get(row.image_file)
        if predicted is None:
            boxes.append([])
            classes.append([])
            scores.append([])
            class_scores.append([])
        else:
            boxes.append(predicted[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(dtype=float).tolist())
            classes.append(predicted["class_id"].astype(int).tolist())
            scores.append(predicted["object_score"].astype(float).tolist())
            class_scores.append([json.loads(value) for value in predicted["class_scores_json"]])
        truth = truth_groups.get(row.image_file)
        if truth is None:
            truth_boxes.append([])
            truth_classes.append([])
        else:
            truth_boxes.append(truth[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(dtype=float).tolist())
            truth_classes.append(truth["class_id"].astype(int).tolist())
    return boxes, classes, scores, class_scores, truth_boxes, truth_classes


def evaluate_predictions(
    index,
    predictions,
    ground_truth,
    classes,
    nms_threshold,
    model_name=MODEL_NAME,
):
    """Calculate per-class 11-point AP and aggregate mAP at IoU 0.50."""

    metric_lists = build_metric_lists(index, predictions, ground_truth)
    matches, truth_counts = match_detections(
        boxes=metric_lists[0],
        classes=metric_lists[1],
        scores=metric_lists[2],
        cls_scores=metric_lists[3],
        gt_boxes=metric_lists[4],
        gt_classes=metric_lists[5],
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
    per_class = []
    for class_id, class_name in enumerate(classes):
        ap = calculate_map_x_point_interpolated(
            {0: points[class_id]},
            num_classes=1,
            num_interpolated_points=11,
        )
        per_class.append(
            {
                "model": model_name,
                "dataset": DATASET_NAME,
                "nms_threshold": float(nms_threshold),
                "class_id": class_id,
                "class_name": class_name,
                "ground_truth_count": int((ground_truth["class_id"] == class_id).sum()),
                "prediction_count": int((predictions["class_id"] == class_id).sum()),
                "ap_11_point": ap,
            }
        )
    summary = {
        "model": model_name,
        "dataset": DATASET_NAME,
        "nms_threshold": float(nms_threshold),
        "mAP@0.5_11_point": map_score,
        "total_ground_truth": int(len(ground_truth)),
        "total_predictions_after_nms": int(len(predictions)),
        "evaluation_rows": int(sum(len(records) for records in matches.values())),
        "score_threshold": SCORE_THRESHOLD,
        "map_iou_threshold": MAP_IOU_THRESHOLD,
        "eval_type": EVAL_TYPE,
    }
    return summary, pd.DataFrame(per_class, columns=PER_CLASS_COLUMNS)


def count_duplicate_like_prediction_pairs(predictions, iou_threshold=DUPLICATE_IOU_THRESHOLD):
    """Count surviving same-class pairs above a diagnostic IoU boundary."""

    threshold = float(iou_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Duplicate IoU threshold must be between zero and one.")
    if predictions.empty:
        return {
            "duplicate_like_pairs_iou_gt_0_5": 0,
            "images_with_duplicate_like_pairs": 0,
            "mean_duplicate_like_pairs_per_image": 0.0,
        }
    total_pairs = 0
    images_with_pairs = 0
    for _, image_group in predictions.groupby("image_file", sort=False):
        image_pairs = 0
        for _, class_group in image_group.groupby("class_id", sort=False):
            boxes = class_group[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(dtype=float)
            for first in range(len(boxes)):
                for second in range(first + 1, len(boxes)):
                    iou = float(np.clip(calculate_iou(boxes[first], boxes[second]), 0.0, 1.0))
                    if iou > threshold:
                        image_pairs += 1
        total_pairs += image_pairs
        images_with_pairs += int(image_pairs > 0)
    detected_images = int(predictions["image_file"].nunique())
    return {
        "duplicate_like_pairs_iou_gt_0_5": int(total_pairs),
        "images_with_duplicate_like_pairs": int(images_with_pairs),
        "mean_duplicate_like_pairs_per_image": (
            float(total_pairs) / detected_images if detected_images else 0.0
        ),
    }


def evaluate_subset(
    subset_name,
    subset_index,
    predictions,
    ground_truth,
    classes,
    threshold,
    model_name=MODEL_NAME,
):
    files = set(subset_index["image_file"])
    subset_predictions = predictions[predictions["image_file"].isin(files)].copy()
    subset_truth = ground_truth[ground_truth["image_file"].isin(files)].copy()
    summary, _ = evaluate_predictions(
        subset_index,
        subset_predictions,
        subset_truth,
        classes,
        threshold,
        model_name=model_name,
    )
    duplicates = count_duplicate_like_prediction_pairs(subset_predictions)
    return {
        "subset_name": subset_name,
        "image_count": int(len(subset_index)),
        "ground_truth_count": int(len(subset_truth)),
        "nms_threshold": float(threshold),
        "mAP@0.5_11_point": summary["mAP@0.5_11_point"],
        "total_predictions_after_nms": int(len(subset_predictions)),
        "evaluation_rows": summary["evaluation_rows"],
        **duplicates,
    }


def validate_prediction_cache(
    table,
    raw_predictions,
    index,
    classes,
    threshold,
    model_name=MODEL_NAME,
):
    """Validate a post-NMS cache and prove that it derives from the raw cache."""

    _required_columns(table, PREDICTION_COLUMNS, "NMS prediction cache")
    normalized = validate_raw_predictions(
        table[RAW_COLUMNS],
        index,
        classes,
        model_name=model_name,
    )
    thresholds = pd.to_numeric(table["nms_threshold"], errors="raise")
    if not np.isfinite(thresholds.to_numpy(dtype=float)).all() or not np.allclose(
        thresholds,
        float(threshold),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("NMS prediction cache uses the wrong threshold.")
    normalized["nms_threshold"] = thresholds.to_numpy(dtype=float)
    normalized = normalized[PREDICTION_COLUMNS].reset_index(drop=True)
    expected = apply_nms_threshold(
        raw_predictions,
        index,
        classes,
        threshold,
        model_name=model_name,
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
            "NMS prediction cache does not match recomputation from the "
            "validated raw-prediction cache."
        ) from error
    return normalized


def _prediction_path(output_dir, threshold, run_label, model_name=MODEL_NAME):
    return Path(output_dir) / (
        f"{model_name}_predictions_nms_{threshold_tag(threshold)}_{run_label}.csv"
    )


def run_threshold_sweep(
    index,
    raw_predictions,
    ground_truth,
    classes,
    crowded_index,
    output_dir,
    run_label,
    refresh_postprocessing=False,
    model_name=MODEL_NAME,
):
    """Apply all thresholds and assemble four auditable result tables."""

    output_dir = Path(output_dir).expanduser().absolute()
    summary_rows, class_tables, duplicate_rows, subset_rows = [], [], [], []
    subsets = {
        "all_selected": index,
        "crowded_any_overlap": crowded_index,
    }
    for threshold in NMS_THRESHOLDS:
        path = _prediction_path(output_dir, threshold, run_label, model_name)
        if path.is_file() and not refresh_postprocessing:
            predictions = pd.read_csv(path)
            predictions = validate_prediction_cache(
                predictions,
                raw_predictions,
                index,
                classes,
                threshold,
                model_name=model_name,
            )
        else:
            predictions = apply_nms_threshold(
                raw_predictions,
                index,
                classes,
                threshold,
                model_name=model_name,
            )
            _write_dataframe_atomic(path, predictions)
            print(f"[WRITE] {path} rows={len(predictions)}")

        summary, per_class = evaluate_predictions(
            index,
            predictions,
            ground_truth,
            classes,
            threshold,
            model_name=model_name,
        )
        duplicates = count_duplicate_like_prediction_pairs(predictions)
        duplicates["nms_threshold"] = float(threshold)
        duplicates["total_predictions_after_nms"] = int(len(predictions))
        summary_rows.append(summary)
        class_tables.append(per_class)
        duplicate_rows.append(duplicates)
        for subset_name, subset_index in subsets.items():
            subset_rows.append(
                evaluate_subset(
                    subset_name,
                    subset_index,
                    predictions,
                    ground_truth,
                    classes,
                    threshold,
                    model_name=model_name,
                )
            )

    return {
        f"nms_threshold_summary_{run_label}.csv": pd.DataFrame(
            summary_rows,
            columns=SUMMARY_COLUMNS,
        ),
        f"per_class_ap_by_threshold_{run_label}.csv": pd.concat(
            class_tables,
            ignore_index=True,
        )[PER_CLASS_COLUMNS],
        f"duplicate_summary_by_threshold_{run_label}.csv": pd.DataFrame(
            duplicate_rows,
            columns=DUPLICATE_COLUMNS,
        ),
        f"subset_summary_by_threshold_{run_label}.csv": pd.DataFrame(
            subset_rows,
            columns=SUBSET_COLUMNS,
        ),
    }


def write_sweep_artifacts(output_dir, artifacts):
    output_paths = {}
    for name, table in artifacts.items():
        path = Path(output_dir).expanduser().absolute() / name
        _write_dataframe_atomic(path, table)
        output_paths[name] = path
    return output_paths


def build_operating_point_record(
    summary,
    duplicates,
    summary_path,
    duplicate_path,
    selection,
):
    """Select and record the quality-first NMS operating point."""

    _required_columns(summary, SUMMARY_COLUMNS, "NMS summary")
    _required_columns(duplicates, DUPLICATE_COLUMNS, "NMS duplicate summary")
    if len(summary) != len(NMS_THRESHOLDS) or len(duplicates) != len(NMS_THRESHOLDS):
        raise EvidenceContractError(
            "NMS operating-point selection requires one row per declared threshold."
        )
    observed = sorted(float(value) for value in summary["nms_threshold"])
    expected = sorted(float(value) for value in NMS_THRESHOLDS)
    if not np.allclose(observed, expected, rtol=0.0, atol=1e-12):
        raise EvidenceContractError(
            "NMS summary thresholds do not match the declared sweep."
        )
    if set(summary["model"].astype(str)) != {selection["selected_model"]}:
        raise EvidenceContractError(
            "NMS summary model does not match the checkpoint-selection decision."
        )

    ranked = summary.sort_values(
        ["mAP@0.5_11_point", "total_predictions_after_nms", "nms_threshold"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    selected = ranked.iloc[0]
    threshold = float(selected["nms_threshold"])
    duplicate_row = duplicates.loc[
        np.isclose(
            pd.to_numeric(duplicates["nms_threshold"], errors="raise"),
            threshold,
            rtol=0.0,
            atol=1e-12,
        )
    ]
    if len(duplicate_row) != 1:
        raise EvidenceContractError(
            "Duplicate diagnostics do not identify the selected NMS threshold uniquely."
        )
    duplicate_row = duplicate_row.iloc[0]
    return {
        "schema_version": 1,
        "status": "complete",
        "selected_model": selection["selected_model"],
        "selected_nms_iou_threshold": threshold,
        "selection_rule": (
            "maximize threshold-constrained 11-point AP50; then minimize retained "
            "predictions; then choose the lower IoU threshold"
        ),
        "selected_metrics": {
            "mAP@0.5_11_point": float(selected["mAP@0.5_11_point"]),
            "total_predictions_after_nms": int(
                selected["total_predictions_after_nms"]
            ),
            "duplicate_like_pairs_iou_gt_0_5": int(
                duplicate_row["duplicate_like_pairs_iou_gt_0_5"]
            ),
        },
        "checkpoint_selection": {
            "run_id": selection["selection_run_id"],
            "manifest_sha256": selection["selection_manifest_sha256"],
            "decision_sha256": selection["decision_sha256"],
        },
        "evidence": {
            Path(summary_path).name: {
                "rows": int(len(summary)),
                "sha256": sha256_file(summary_path),
            },
            Path(duplicate_path).name: {
                "rows": int(len(duplicates)),
                "sha256": sha256_file(duplicate_path),
            },
        },
    }


def write_operating_point(output_dir, artifacts, output_paths, selection, run_label):
    summary_name = f"nms_threshold_summary_{run_label}.csv"
    duplicate_name = f"duplicate_summary_by_threshold_{run_label}.csv"
    record = build_operating_point_record(
        artifacts[summary_name],
        artifacts[duplicate_name],
        output_paths[summary_name],
        output_paths[duplicate_name],
        selection,
    )
    return _write_json_atomic(Path(output_dir) / "operating_point.json", record)


def _normalize_derived_thresholds(table, label, expected_rows_per_threshold):
    values = pd.to_numeric(table["nms_threshold"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise EvidenceContractError(f"{label} contains non-finite NMS thresholds.")
    declared = np.asarray(NMS_THRESHOLDS, dtype=float)
    normalized = []
    for value in values:
        matches = np.flatnonzero(
            np.isclose(declared, value, rtol=0.0, atol=1e-12)
        )
        if len(matches) != 1:
            raise EvidenceContractError(
                f"{label} contains an undeclared NMS threshold: {value}"
            )
        normalized.append(float(declared[int(matches[0])]))
    counts = pd.Series(normalized).value_counts().to_dict()
    expected = {
        float(threshold): int(expected_rows_per_threshold)
        for threshold in NMS_THRESHOLDS
    }
    if counts != expected:
        raise EvidenceContractError(
            f"{label} does not contain the required rows for every NMS threshold."
        )
    result = table.copy()
    result["nms_threshold"] = normalized
    return result


def _validate_probability_column(table, column, label):
    values = pd.to_numeric(table[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise EvidenceContractError(
            f"{label} {column} must contain finite values between zero and one."
        )
    return values


def validate_derived_artifacts(artifacts, run_label):
    """Validate the complete four-table result package used by figures-only mode."""

    schemas = {
        f"nms_threshold_summary_{run_label}.csv": SUMMARY_COLUMNS,
        f"per_class_ap_by_threshold_{run_label}.csv": PER_CLASS_COLUMNS,
        f"duplicate_summary_by_threshold_{run_label}.csv": DUPLICATE_COLUMNS,
        f"subset_summary_by_threshold_{run_label}.csv": SUBSET_COLUMNS,
    }
    _require_exact_keys(artifacts, schemas, "Derived NMS artifact package")
    normalized = {}
    for name, columns in schemas.items():
        table = artifacts[name]
        if list(table.columns) != columns:
            raise EvidenceContractError(
                f"{name} schema mismatch: expected={columns}, "
                f"observed={list(table.columns)}"
            )
        normalized[name] = table.copy()

    summary_name = f"nms_threshold_summary_{run_label}.csv"
    class_name = f"per_class_ap_by_threshold_{run_label}.csv"
    duplicate_name = f"duplicate_summary_by_threshold_{run_label}.csv"
    subset_name = f"subset_summary_by_threshold_{run_label}.csv"
    summary = _normalize_derived_thresholds(
        normalized[summary_name],
        "NMS threshold summary",
        1,
    )
    if summary.empty:
        raise EvidenceContractError("NMS threshold summary cannot be empty.")
    models = set(summary["model"].astype(str))
    if len(models) != 1 or not models.issubset({"model1", "model2"}):
        raise EvidenceContractError("NMS summary must identify exactly one supported model.")
    model_name = next(iter(models))
    if set(summary["dataset"].astype(str)) != {DATASET_NAME}:
        raise EvidenceContractError("NMS summary dataset policy does not match this experiment.")
    if set(summary["eval_type"].astype(str)) != {EVAL_TYPE}:
        raise EvidenceContractError("NMS summary evaluation policy does not match this experiment.")
    for column, expected_value in (
        ("score_threshold", SCORE_THRESHOLD),
        ("map_iou_threshold", MAP_IOU_THRESHOLD),
    ):
        values = pd.to_numeric(summary[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.allclose(
            values,
            expected_value,
            rtol=0.0,
            atol=1e-12,
        ):
            raise EvidenceContractError(
                f"NMS summary {column} policy does not match this experiment."
            )
    summary["mAP@0.5_11_point"] = _validate_probability_column(
        summary,
        "mAP@0.5_11_point",
        "NMS summary",
    )
    for column in (
        "total_ground_truth",
        "total_predictions_after_nms",
        "evaluation_rows",
    ):
        summary[column] = _validated_integer_series(
            summary[column],
            f"NMS summary {column}",
        )
    if summary["total_ground_truth"].nunique() != 1:
        raise EvidenceContractError(
            "NMS summary ground-truth count must be constant across thresholds."
        )
    if not summary["evaluation_rows"].equals(
        summary["total_predictions_after_nms"]
    ):
        raise EvidenceContractError(
            "NMS summary evaluation rows must equal retained prediction rows."
        )

    per_class_source = normalized[class_name]
    if per_class_source.empty or len(per_class_source) % len(NMS_THRESHOLDS):
        raise EvidenceContractError(
            "Per-class NMS results must contain the same non-empty vocabulary "
            "for every threshold."
        )
    class_count = len(per_class_source) // len(NMS_THRESHOLDS)
    per_class = _normalize_derived_thresholds(
        per_class_source,
        "Per-class NMS results",
        class_count,
    )
    if set(per_class["model"].astype(str)) != {model_name}:
        raise EvidenceContractError("Per-class results use a different model.")
    if set(per_class["dataset"].astype(str)) != {DATASET_NAME}:
        raise EvidenceContractError("Per-class results use a different dataset.")
    per_class["class_id"] = _validated_integer_series(
        per_class["class_id"],
        "Per-class class_id",
    )
    for column in ("ground_truth_count", "prediction_count"):
        per_class[column] = _validated_integer_series(
            per_class[column],
            f"Per-class {column}",
        )
    per_class["ap_11_point"] = _validate_probability_column(
        per_class,
        "ap_11_point",
        "Per-class results",
    )
    expected_class_ids = list(range(class_count))
    canonical_class_names = None
    summary_by_threshold = {
        float(row["nms_threshold"]): row for _, row in summary.iterrows()
    }
    for threshold, group in per_class.groupby("nms_threshold", sort=False):
        ordered = group.sort_values("class_id")
        if ordered["class_id"].tolist() != expected_class_ids:
            raise EvidenceContractError(
                "Per-class results must contain each contiguous class ID exactly once "
                "per threshold."
            )
        names = ordered["class_name"].astype(str).tolist()
        if any(not name.strip() for name in names):
            raise EvidenceContractError("Per-class names must be non-empty.")
        if canonical_class_names is None:
            canonical_class_names = names
        elif names != canonical_class_names:
            raise EvidenceContractError(
                "Per-class vocabulary changes between NMS thresholds."
            )
        summary_row = summary_by_threshold[float(threshold)]
        if int(ordered["ground_truth_count"].sum()) != int(
            summary_row["total_ground_truth"]
        ):
            raise EvidenceContractError(
                "Per-class ground-truth counts do not reconcile with the summary."
            )
        if int(ordered["prediction_count"].sum()) != int(
            summary_row["total_predictions_after_nms"]
        ):
            raise EvidenceContractError(
                "Per-class prediction counts do not reconcile with the summary."
            )
        if not np.isclose(
            float(ordered["ap_11_point"].mean()),
            float(summary_row["mAP@0.5_11_point"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise EvidenceContractError(
                "Mean per-class AP does not reconcile with summary mAP."
            )

    duplicates = _normalize_derived_thresholds(
        normalized[duplicate_name],
        "Duplicate diagnostics",
        1,
    )
    for column in (
        "duplicate_like_pairs_iou_gt_0_5",
        "images_with_duplicate_like_pairs",
        "total_predictions_after_nms",
    ):
        duplicates[column] = _validated_integer_series(
            duplicates[column],
            f"Duplicate diagnostics {column}",
        )
    duplicate_means = pd.to_numeric(
        duplicates["mean_duplicate_like_pairs_per_image"],
        errors="raise",
    ).to_numpy(dtype=float)
    if not np.isfinite(duplicate_means).all() or (duplicate_means < 0.0).any():
        raise EvidenceContractError(
            "Duplicate diagnostic means must be finite and non-negative."
        )
    duplicates["mean_duplicate_like_pairs_per_image"] = duplicate_means
    duplicates_by_threshold = {
        float(row["nms_threshold"]): row for _, row in duplicates.iterrows()
    }
    for threshold, summary_row in summary_by_threshold.items():
        duplicate_row = duplicates_by_threshold[threshold]
        if int(duplicate_row["total_predictions_after_nms"]) != int(
            summary_row["total_predictions_after_nms"]
        ):
            raise EvidenceContractError(
                "Duplicate diagnostics do not reconcile with summary prediction counts."
            )

    subsets = _normalize_derived_thresholds(
        normalized[subset_name],
        "NMS subset summary",
        2,
    )
    for column in (
        "image_count",
        "ground_truth_count",
        "total_predictions_after_nms",
        "evaluation_rows",
        "duplicate_like_pairs_iou_gt_0_5",
        "images_with_duplicate_like_pairs",
    ):
        subsets[column] = _validated_integer_series(
            subsets[column],
            f"NMS subset summary {column}",
        )
    subsets["mAP@0.5_11_point"] = _validate_probability_column(
        subsets,
        "mAP@0.5_11_point",
        "NMS subset summary",
    )
    subset_means = pd.to_numeric(
        subsets["mean_duplicate_like_pairs_per_image"],
        errors="raise",
    ).to_numpy(dtype=float)
    if not np.isfinite(subset_means).all() or (subset_means < 0.0).any():
        raise EvidenceContractError(
            "NMS subset duplicate means must be finite and non-negative."
        )
    subsets["mean_duplicate_like_pairs_per_image"] = subset_means
    if not subsets["evaluation_rows"].equals(
        subsets["total_predictions_after_nms"]
    ):
        raise EvidenceContractError(
            "NMS subset evaluation rows must equal retained prediction rows."
        )
    expected_subsets = {"all_selected", "crowded_any_overlap"}
    if set(subsets["subset_name"].astype(str)) != expected_subsets:
        raise EvidenceContractError("NMS subset identities do not match the experiment.")
    for name, group in subsets.groupby("subset_name"):
        if group["image_count"].nunique() != 1:
            raise EvidenceContractError(
                f"NMS subset image count changes across thresholds: {name}"
            )
    for threshold, group in subsets.groupby("nms_threshold", sort=False):
        if set(group["subset_name"].astype(str)) != expected_subsets:
            raise EvidenceContractError(
                "Every NMS threshold must contain both declared subsets."
            )
        indexed = group.set_index("subset_name")
        all_row = indexed.loc["all_selected"]
        crowded_row = indexed.loc["crowded_any_overlap"]
        summary_row = summary_by_threshold[float(threshold)]
        duplicate_row = duplicates_by_threshold[float(threshold)]
        exact_pairs = (
            ("ground_truth_count", "total_ground_truth"),
            ("total_predictions_after_nms", "total_predictions_after_nms"),
            ("evaluation_rows", "evaluation_rows"),
        )
        for subset_column, summary_column in exact_pairs:
            if int(all_row[subset_column]) != int(
                summary_row[summary_column]
            ):
                raise EvidenceContractError(
                    "All-selected subset counts do not reconcile with the NMS summary."
                )
        if not np.isclose(
            float(all_row["mAP@0.5_11_point"]),
            float(summary_row["mAP@0.5_11_point"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise EvidenceContractError(
                "All-selected subset mAP does not reconcile with the NMS summary."
            )
        for column in (
            "duplicate_like_pairs_iou_gt_0_5",
            "images_with_duplicate_like_pairs",
        ):
            if int(all_row[column]) != int(duplicate_row[column]):
                raise EvidenceContractError(
                    "All-selected subset duplicate counts do not reconcile with "
                    "duplicate diagnostics."
                )
        if not np.isclose(
            float(all_row["mean_duplicate_like_pairs_per_image"]),
            float(duplicate_row["mean_duplicate_like_pairs_per_image"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise EvidenceContractError(
                "All-selected subset duplicate mean does not reconcile with diagnostics."
            )
        for column in (
            "image_count",
            "ground_truth_count",
            "total_predictions_after_nms",
            "evaluation_rows",
        ):
            if int(crowded_row[column]) > int(all_row[column]):
                raise EvidenceContractError(
                    "Crowded-subset counts cannot exceed all-selected counts."
                )

    normalized[summary_name] = summary
    normalized[class_name] = per_class
    normalized[duplicate_name] = duplicates
    normalized[subset_name] = subsets
    return normalized


def load_derived_artifacts(output_dir, run_label):
    names = [
        f"nms_threshold_summary_{run_label}.csv",
        f"per_class_ap_by_threshold_{run_label}.csv",
        f"duplicate_summary_by_threshold_{run_label}.csv",
        f"subset_summary_by_threshold_{run_label}.csv",
    ]
    paths = [Path(output_dir).expanduser().absolute() / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing derived NMS artifacts: " + "; ".join(missing))
    artifacts = {path.name: pd.read_csv(path) for path in paths}
    return validate_derived_artifacts(artifacts, run_label)


def build_figures(summary, duplicates, subset_summary, figure_dir):
    """Render the four canonical threshold-analysis figures."""

    import matplotlib.pyplot as plt

    directory = Path(figure_dir).expanduser().absolute()
    directory.mkdir(parents=True, exist_ok=True)
    figure_paths = []

    definitions = [
        (
            "01_map_by_threshold.png",
            summary["nms_threshold"],
            summary["mAP@0.5_11_point"],
            "Mean 11-point AP at IoU 0.5",
            "Detection score by NMS threshold",
            (0.0, 1.0),
        ),
        (
            "02_prediction_count_by_threshold.png",
            summary["nms_threshold"],
            summary["total_predictions_after_nms"],
            "Predictions after NMS",
            "Prediction count by NMS threshold",
            None,
        ),
        (
            "03_duplicate_pairs_by_threshold.png",
            duplicates["nms_threshold"],
            duplicates["duplicate_like_pairs_iou_gt_0_5"],
            "Same-class prediction pairs with IoU > 0.5",
            "Duplicate-like prediction pairs by NMS threshold",
            None,
        ),
    ]
    for name, x_values, y_values, y_label, title, limits in definitions:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(x_values, y_values, marker="o")
        axis.set_xlabel("NMS IoU threshold")
        axis.set_ylabel(y_label)
        axis.set_title(title)
        if limits is None:
            axis.set_ylim(bottom=0.0)
        else:
            axis.set_ylim(*limits)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = directory / name
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        figure_paths.append(path)

    figure, axis = plt.subplots(figsize=(8, 5))
    labels = {
        "all_selected": "All selected images",
        "crowded_any_overlap": "Crowded subset",
    }
    for subset_name, group in subset_summary.groupby("subset_name", sort=False):
        group = group.sort_values("nms_threshold")
        axis.plot(
            group["nms_threshold"],
            group["mAP@0.5_11_point"],
            marker="o",
            label=labels.get(subset_name, subset_name),
        )
    axis.set_xlabel("NMS IoU threshold")
    axis.set_ylabel("Mean 11-point AP at IoU 0.5")
    axis.set_title("Detection score by NMS threshold and subset")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = directory / "04_map_by_threshold_and_subset.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    figure_paths.append(path)
    return figure_paths


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the verified selected checkpoint across NMS IoU thresholds."
        )
    )
    parser.add_argument("--sample-index", type=Path, default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument("--overlap-profile", type=Path, default=DEFAULT_OVERLAP_PROFILE)
    parser.add_argument("--selection-run", type=Path, default=DEFAULT_SELECTION_RUN)
    parser.add_argument("--class-file", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--ground-truth-cache", type=Path)
    parser.add_argument("--raw-predictions-cache", type=Path)
    parser.add_argument(
        "--inference-ledger",
        type=Path,
        help=(
            "One-row-per-image ledger paired with --raw-predictions-cache. "
            "Managed runs create it automatically."
        ),
    )
    parser.add_argument(
        "--inference-cache-manifest",
        type=Path,
        help=(
            "Content-addressed manifest covering the ground-truth, raw-prediction, "
            "and inference-ledger cache package."
        ),
    )
    parser.add_argument("--max-images", type=positive_int)
    parser.add_argument("--refresh-postprocessing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    return parser


def _run_label(max_images):
    return f"first_{max_images}" if max_images is not None else "sample5000"


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_label = _run_label(args.max_images)
    cache_overrides = {
        "ground truth": args.ground_truth_cache,
        "raw predictions": args.raw_predictions_cache,
        "inference ledger": args.inference_ledger,
    }
    supplied_cache_overrides = {
        label for label, value in cache_overrides.items() if value is not None
    }
    if args.force and (
        supplied_cache_overrides or args.inference_cache_manifest is not None
    ):
        raise ValueError(
            "--force cannot be combined with external cache paths; "
            "use the output directory for rebuilt caches."
        )
    if (supplied_cache_overrides or args.inference_cache_manifest is not None) and (
        len(supplied_cache_overrides) != len(cache_overrides)
        or args.inference_cache_manifest is None
    ):
        raise ValueError(
            "External cache replay requires --ground-truth-cache, "
            "--raw-predictions-cache, --inference-ledger, and "
            "--inference-cache-manifest together."
        )
    if args.figures_only:
        artifacts = load_derived_artifacts(args.output_dir, run_label)
        figure_paths = build_figures(
            artifacts[f"nms_threshold_summary_{run_label}.csv"],
            artifacts[f"duplicate_summary_by_threshold_{run_label}.csv"],
            artifacts[f"subset_summary_by_threshold_{run_label}.csv"],
            args.figure_dir,
        )
        for path in figure_paths:
            print(f"[WRITE] {path}")
        return artifacts, {}, figure_paths

    index = load_sample_index(args.sample_index, args.max_images)
    selection = load_verified_checkpoint_selection(args.selection_run)
    model_name = selection["selected_model"]
    output_dir = Path(args.output_dir).expanduser().absolute()
    ground_truth_path = (
        Path(args.ground_truth_cache).expanduser().absolute()
        if args.ground_truth_cache
        else output_dir / f"ground_truth_{run_label}.csv"
    )
    raw_path = (
        Path(args.raw_predictions_cache).expanduser().absolute()
        if args.raw_predictions_cache
        else output_dir / f"{model_name}_raw_predictions_{run_label}.csv"
    )
    ledger_path = (
        Path(args.inference_ledger).expanduser().absolute()
        if args.inference_ledger
        else output_dir / f"{model_name}_inference_ledger_{run_label}.csv"
    )
    cache_manifest_path = (
        Path(args.inference_cache_manifest).expanduser().absolute()
        if args.inference_cache_manifest
        else output_dir / f"inference_cache_manifest_{run_label}.json"
    )
    cache_paths = {
        "ground_truth": ground_truth_path,
        "raw_predictions": raw_path,
        "inference_ledger": ledger_path,
    }
    package_paths = [*cache_paths.values(), cache_manifest_path]
    package_exists = [path.is_file() for path in package_paths]
    if supplied_cache_overrides and not all(package_exists):
        missing = [
            str(path)
            for path, exists in zip(package_paths, package_exists)
            if not exists
        ]
        raise FileNotFoundError(
            "Every externally supplied cache package file must already exist: "
            + ", ".join(missing)
        )
    if args.force:
        generate_cache = True
    elif all(package_exists):
        generate_cache = False
    elif not any(package_exists):
        generate_cache = True
    else:
        missing = [
            path.name
            for path, exists in zip(package_paths, package_exists)
            if not exists
        ]
        raise FileNotFoundError(
            "Inference-cache package is incomplete and cannot be trusted. "
            f"Missing {missing}; regenerate the managed package with --force."
        )

    default_asset_root = (
        Path(args.asset_root).expanduser().absolute()
        if args.asset_root is not None
        else PROJECT_ROOT / "detector_service" / "storage"
    )
    selected_assets = (
        resolve_selected_model_assets(default_asset_root, args.selection_run)
        if generate_cache
        else None
    )
    class_file = args.class_file or (
        selected_assets["resolved_paths"]["classes"]
        if selected_assets is not None
        else default_asset_root / selection["asset_paths"]["classes"]
    )
    classes = load_classes(
        class_file,
        expected_sha256=selection["model_identity"]["names"],
    )
    _, crowded_index = load_overlap_profile(args.overlap_profile, index)

    if generate_cache:
        ground_truth = build_ground_truth(index, classes, asset_root=args.asset_root)
        ledger_rows = []
        raw_predictions = run_raw_inference(
            index,
            classes,
            asset_root=args.asset_root,
            model_name=model_name,
            model_assets=(
                selected_assets["resolved_paths"]
                if selected_assets is not None
                else None
            ),
            selection_run=args.selection_run,
            expected_names_sha256=selection["model_identity"]["names"],
            ledger_rows=ledger_rows,
        )
        ledger = pd.DataFrame(ledger_rows, columns=LEDGER_COLUMNS)
        ground_truth = validate_ground_truth(ground_truth, index, classes)
        raw_predictions = validate_raw_predictions(
            raw_predictions,
            index,
            classes,
            model_name=model_name,
        )
        ledger = validate_inference_ledger(
            ledger,
            index,
            raw_predictions,
            model_name=model_name,
        )
        _write_dataframe_atomic(ground_truth_path, ground_truth)
        _write_dataframe_atomic(raw_path, raw_predictions)
        _write_dataframe_atomic(ledger_path, ledger)
        ground_truth = validate_ground_truth(
            pd.read_csv(ground_truth_path),
            index,
            classes,
        )
        raw_predictions = validate_raw_predictions(
            pd.read_csv(raw_path),
            index,
            classes,
            model_name=model_name,
        )
        ledger = validate_inference_ledger(
            pd.read_csv(ledger_path),
            index,
            raw_predictions,
            model_name=model_name,
        )
        artifact_tables = {
            "ground_truth": ground_truth,
            "raw_predictions": raw_predictions,
            "inference_ledger": ledger,
        }
        cache_manifest = build_inference_cache_manifest(
            sample_index_path=args.sample_index,
            index=index,
            selection=selection,
            classes=classes,
            class_file=class_file,
            model_name=model_name,
            run_label=run_label,
            artifact_paths=cache_paths,
            artifact_tables=artifact_tables,
        )
        _write_json_atomic(cache_manifest_path, cache_manifest)
        validate_inference_cache_manifest(
            cache_manifest_path,
            sample_index_path=args.sample_index,
            index=index,
            selection=selection,
            classes=classes,
            class_file=class_file,
            model_name=model_name,
            run_label=run_label,
            artifact_paths=cache_paths,
            artifact_tables=artifact_tables,
        )
        print(f"[WRITE] {ground_truth_path} rows={len(ground_truth)}")
        print(f"[WRITE] {raw_path} rows={len(raw_predictions)}")
        print(f"[WRITE] {ledger_path} rows={len(ledger)}")
        print(f"[WRITE] {cache_manifest_path}")
    else:
        ground_truth = pd.read_csv(ground_truth_path)
        raw_predictions = pd.read_csv(raw_path)
        ledger = pd.read_csv(ledger_path)
        artifact_tables = {
            "ground_truth": ground_truth,
            "raw_predictions": raw_predictions,
            "inference_ledger": ledger,
        }
        validate_inference_cache_manifest(
            cache_manifest_path,
            sample_index_path=args.sample_index,
            index=index,
            selection=selection,
            classes=classes,
            class_file=class_file,
            model_name=model_name,
            run_label=run_label,
            artifact_paths=cache_paths,
            artifact_tables=artifact_tables,
        )
        ground_truth = validate_ground_truth(ground_truth, index, classes)
        raw_predictions = validate_raw_predictions(
            raw_predictions,
            index,
            classes,
            model_name=model_name,
        )
        ledger = validate_inference_ledger(
            ledger,
            index,
            raw_predictions,
            model_name=model_name,
        )

    print(f"[INFO] Images selected: {len(index)}")
    print(f"[INFO] Crowded subset images: {len(crowded_index)}")
    print(f"[INFO] Classes: {len(classes)}")
    print(
        f"[INFO] Selected checkpoint: {selection['selected_checkpoint']} "
        f"({model_name}) from {selection['selection_run_id']}"
    )
    print(f"[INFO] Raw candidates reused: {len(raw_predictions)}")
    print(f"[INFO] NMS thresholds: {list(NMS_THRESHOLDS)}")

    artifacts = run_threshold_sweep(
        index,
        raw_predictions,
        ground_truth,
        classes,
        crowded_index,
        output_dir,
        run_label,
        refresh_postprocessing=(
            args.refresh_postprocessing or args.force or generate_cache
        ),
        model_name=model_name,
    )
    artifacts = validate_derived_artifacts(artifacts, run_label)
    output_paths = write_sweep_artifacts(output_dir, artifacts)
    output_paths[cache_manifest_path.name] = cache_manifest_path
    operating_point_path = write_operating_point(
        output_dir,
        artifacts,
        output_paths,
        selection,
        run_label,
    )
    output_paths[operating_point_path.name] = operating_point_path
    figure_paths = []
    if not args.skip_figures:
        figure_paths = build_figures(
            artifacts[f"nms_threshold_summary_{run_label}.csv"],
            artifacts[f"duplicate_summary_by_threshold_{run_label}.csv"],
            artifacts[f"subset_summary_by_threshold_{run_label}.csv"],
            args.figure_dir,
        )

    print("\nNMS THRESHOLD SUMMARY")
    print(artifacts[f"nms_threshold_summary_{run_label}.csv"].to_string(index=False))
    print("\nDUPLICATE-LIKE PREDICTION SUMMARY")
    print(artifacts[f"duplicate_summary_by_threshold_{run_label}.csv"].to_string(index=False))
    print("\nSUBSET SUMMARY")
    print(artifacts[f"subset_summary_by_threshold_{run_label}.csv"].to_string(index=False))
    for path in output_paths.values():
        print(f"[WRITE] {path}")
    for path in figure_paths:
        print(f"[WRITE] {path}")
    return artifacts, output_paths, figure_paths


if __name__ == "__main__":
    main()
