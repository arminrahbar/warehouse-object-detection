"""Evaluate one cached detector output across controlled NMS IoU thresholds."""

import argparse
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


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "outputs"
DEFAULT_SAMPLING_DIR = DEFAULT_OUTPUT_ROOT / "dataset_sampling"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "nms_thresholding"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "experiments" / "figures" / "03_nms_thresholding"
DEFAULT_SAMPLE_INDEX = DEFAULT_SAMPLING_DIR / "selected_sample_index.csv"
DEFAULT_OVERLAP_PROFILE = DEFAULT_SAMPLING_DIR / "overlap_profile.csv"
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

MODEL_ASSET_PATHS = {
    "weights": "yolo_model_2/yolov4-tiny-logistics_size_416_2.weights",
    "config": "yolo_model_2/yolov4-tiny-logistics_size_416_2.cfg",
    "classes": "yolo_model_2/logistics.names",
}

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


def _logical_parts(value):
    raw = str(value).strip()
    if not raw:
        raise ValueError("Asset path cannot be empty.")
    return PurePosixPath(raw.replace("\\", "/")).parts


def resolve_indexed_path(value, asset_root=None, project_root=PROJECT_ROOT):
    """Resolve a logical index path through an optional external storage root."""

    direct = Path(str(value)).expanduser()
    if direct.is_absolute():
        return direct
    parts = _logical_parts(value)
    if asset_root is not None and len(parts) >= 2:
        supported = {
            ("detector_service", "storage"),
            ("techtrack", "storage"),
        }
        if tuple(parts[:2]) in supported:
            return Path(asset_root).expanduser().absolute().joinpath(*parts[2:])
    return Path(project_root).joinpath(*parts)


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


def _class_mapping_from_evidence(paths):
    rows = []
    for path in paths:
        if path is None:
            continue
        source = Path(path).expanduser().absolute()
        if source.is_file():
            table = pd.read_csv(source, usecols=["class_id", "class_name"])
            rows.append(table)
    if not rows:
        return None

    combined = pd.concat(rows, ignore_index=True).drop_duplicates()
    combined["class_id"] = _validated_integer_series(combined["class_id"], "class_id")
    mapping = {}
    for row in combined.itertuples(index=False):
        class_id = int(row.class_id)
        class_name = str(row.class_name).strip()
        if not class_name:
            raise ValueError("Cached class names must be non-empty.")
        if class_id in mapping and mapping[class_id] != class_name:
            raise ValueError(
                f"Conflicting names for class {class_id}: "
                f"{mapping[class_id]!r} and {class_name!r}"
            )
        mapping[class_id] = class_name
    if not mapping:
        raise ValueError("Cached experiment evidence contains no class mapping.")
    expected = list(range(max(mapping) + 1))
    if sorted(mapping) != expected:
        raise ValueError(
            "Cached class mapping must contain contiguous IDs from zero."
        )
    return [mapping[class_id] for class_id in expected]


def load_classes(class_file=None, evidence_paths=()):
    """Load the class vocabulary from a names file or consistent cache evidence."""

    source = Path(class_file).expanduser().absolute() if class_file else None
    if source is not None and source.is_file():
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

    cached = _class_mapping_from_evidence(evidence_paths)
    if cached is not None:
        print("[INFO] Loaded class names from cached experiment evidence.")
        return cached
    location = source or DEFAULT_CLASS_FILE
    raise FileNotFoundError(
        f"Class-name file not found: {location}. No complete cached mapping is available."
    )


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
    selected_files = set(index["image_file"])
    if not set(normalized["image_file"]).issubset(selected_files):
        raise ValueError("Ground-truth cache contains images outside the selected index.")

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


def validate_raw_predictions(table, index, classes):
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
    if set(normalized["model"].astype(str)) != {MODEL_NAME}:
        raise ValueError(f"Raw-prediction cache must contain only {MODEL_NAME} rows.")
    if (normalized[["bbox_w", "bbox_h"]] < 0).any().any():
        raise ValueError("Raw-prediction widths and heights must be non-negative.")
    for column in ("object_score", "predicted_class_score", "combined_confidence"):
        if not normalized[column].between(0.0, 1.0).all():
            raise ValueError(f"Raw-prediction {column} values must be probabilities.")
    if not set(normalized["image_file"]).issubset(set(index["image_file"])):
        raise ValueError("Raw-prediction cache contains images outside the selected index.")

    predicted_scores = []
    for position, row in enumerate(normalized.itertuples(index=False), start=1):
        vector = _parse_score_vector(row.class_scores_json, position, classes)
        predicted_scores.append(float(vector[int(row.class_id)]))
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


def _resolve_model_assets(asset_root):
    if asset_root is None:
        root = PROJECT_ROOT / "detector_service" / "storage"
    else:
        root = Path(asset_root).expanduser().absolute()
    paths = {name: root / relative for name, relative in MODEL_ASSET_PATHS.items()}
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model assets: " + "; ".join(missing))
    return paths


def _serialize_raw_detection(image_row, bbox, class_id, object_score, score_vector, classes):
    probabilities = np.asarray(score_vector, dtype=float).reshape(-1)
    if len(probabilities) != len(classes):
        raise ValueError("Detector class-score vector length does not match the vocabulary.")
    class_id = int(class_id)
    if not 0 <= class_id < len(classes):
        raise ValueError("Detector returned a class ID outside the vocabulary.")
    predicted_score = float(probabilities[class_id])
    objectness = float(object_score)
    return {
        "model": MODEL_NAME,
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
):
    """Run Model 2 once and retain pre-NMS candidate evidence."""

    if image_reader is None:
        import cv2

        image_reader = cv2.imread
    paths = _resolve_model_assets(asset_root)
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
            rows.append(_serialize_raw_detection(image_row, *values, classes))
        if progress_interval and position % progress_interval == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[RAW] Processed {position}/{len(index)} images | "
                f"candidates={len(rows)} | elapsed={elapsed:.1f}s"
            )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def threshold_tag(threshold):
    return format(float(threshold), "g").replace(".", "_")


def apply_nms_threshold(raw_predictions, index, classes, nms_threshold):
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
            row = _serialize_raw_detection(image_row, *values, classes)
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


def evaluate_predictions(index, predictions, ground_truth, classes, nms_threshold):
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
                "model": MODEL_NAME,
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
        "model": MODEL_NAME,
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


def evaluate_subset(subset_name, subset_index, predictions, ground_truth, classes, threshold):
    files = set(subset_index["image_file"])
    subset_predictions = predictions[predictions["image_file"].isin(files)].copy()
    subset_truth = ground_truth[ground_truth["image_file"].isin(files)].copy()
    summary, _ = evaluate_predictions(
        subset_index,
        subset_predictions,
        subset_truth,
        classes,
        threshold,
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


def _prediction_path(output_dir, threshold, run_label):
    return Path(output_dir) / (
        f"{MODEL_NAME}_predictions_nms_{threshold_tag(threshold)}_{run_label}.csv"
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
):
    """Apply all thresholds and assemble four auditable result tables."""

    output_dir = Path(output_dir).expanduser().absolute()
    summary_rows, class_tables, duplicate_rows, subset_rows = [], [], [], []
    subsets = {
        "all_selected": index,
        "crowded_any_overlap": crowded_index,
    }
    for threshold in NMS_THRESHOLDS:
        path = _prediction_path(output_dir, threshold, run_label)
        if path.is_file() and not refresh_postprocessing:
            predictions = pd.read_csv(path)
            _required_columns(predictions, PREDICTION_COLUMNS, "NMS prediction cache")
            predictions = predictions[PREDICTION_COLUMNS]
        else:
            predictions = apply_nms_threshold(raw_predictions, index, classes, threshold)
            _write_dataframe_atomic(path, predictions)
            print(f"[WRITE] {path} rows={len(predictions)}")

        summary, per_class = evaluate_predictions(
            index,
            predictions,
            ground_truth,
            classes,
            threshold,
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
    return {path.name: pd.read_csv(path) for path in paths}


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
        description="Evaluate cached Model 2 detections across NMS IoU thresholds."
    )
    parser.add_argument("--sample-index", type=Path, default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument("--overlap-profile", type=Path, default=DEFAULT_OVERLAP_PROFILE)
    parser.add_argument("--class-file", type=Path, default=DEFAULT_CLASS_FILE)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--ground-truth-cache", type=Path)
    parser.add_argument("--raw-predictions-cache", type=Path)
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
    if args.force and (args.ground_truth_cache or args.raw_predictions_cache):
        raise ValueError(
            "--force cannot be combined with external cache paths; "
            "use the output directory for rebuilt caches."
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
    output_dir = Path(args.output_dir).expanduser().absolute()
    ground_truth_path = (
        Path(args.ground_truth_cache).expanduser().absolute()
        if args.ground_truth_cache
        else output_dir / f"ground_truth_{run_label}.csv"
    )
    raw_path = (
        Path(args.raw_predictions_cache).expanduser().absolute()
        if args.raw_predictions_cache
        else output_dir / f"{MODEL_NAME}_raw_predictions_{run_label}.csv"
    )
    classes = load_classes(args.class_file, (ground_truth_path, raw_path))
    _, crowded_index = load_overlap_profile(args.overlap_profile, index)

    if args.ground_truth_cache and not ground_truth_path.is_file():
        raise FileNotFoundError(f"Ground-truth cache not found: {ground_truth_path}")
    if args.force or not ground_truth_path.is_file():
        ground_truth = build_ground_truth(index, classes, asset_root=args.asset_root)
        _write_dataframe_atomic(ground_truth_path, ground_truth)
        print(f"[WRITE] {ground_truth_path} rows={len(ground_truth)}")
    else:
        ground_truth = pd.read_csv(ground_truth_path)
        ground_truth = ground_truth[
            ground_truth["image_file"].isin(set(index["image_file"]))
        ].copy()
    ground_truth = validate_ground_truth(ground_truth, index, classes)

    if args.raw_predictions_cache and not raw_path.is_file():
        raise FileNotFoundError(f"Raw-prediction cache not found: {raw_path}")
    if args.force or not raw_path.is_file():
        raw_predictions = run_raw_inference(index, classes, asset_root=args.asset_root)
        _write_dataframe_atomic(raw_path, raw_predictions)
        print(f"[WRITE] {raw_path} rows={len(raw_predictions)}")
    else:
        raw_predictions = pd.read_csv(raw_path)
        raw_predictions = raw_predictions[
            raw_predictions["image_file"].isin(set(index["image_file"]))
        ].copy()
    raw_predictions = validate_raw_predictions(raw_predictions, index, classes)

    print(f"[INFO] Images selected: {len(index)}")
    print(f"[INFO] Crowded subset images: {len(crowded_index)}")
    print(f"[INFO] Classes: {len(classes)}")
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
        refresh_postprocessing=args.refresh_postprocessing or args.force,
    )
    output_paths = write_sweep_artifacts(output_dir, artifacts)
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
