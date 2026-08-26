"""Strict, staged, reproducible comparison of two detector checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import secrets
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INVENTORY_DIR = (
    PROJECT_ROOT / "experiments" / "outputs" / "00_dataset_inventory"
)
DEFAULT_DATASET_INDEX = DEFAULT_INVENTORY_DIR / "dataset_index.csv"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "outputs"
    / "01_model_selection"
    / "01_quality_comparison"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector_service.modules.inference.nms import NMS
from detector_service.modules.utils.metrics import (
    calculate_map_x_point_interpolated,
    calculate_precision_recall_curve,
    match_detections,
)

DEFAULT_CANDIDATE_FLOOR = 0.001
DEFAULT_DEPLOYMENT_CONFIDENCE = 0.50
DEFAULT_NMS_IOU = 0.30
# Backward-compatible names describe the deployed operating point consumed by
# the separate runtime benchmark. The strict quality runner uses the low floor
# above and derives the deployment view from the same post-NMS evidence.
DETECTOR_OBJECTNESS_THRESHOLD = DEFAULT_DEPLOYMENT_CONFIDENCE
NMS_CONFIDENCE_THRESHOLD = DEFAULT_DEPLOYMENT_CONFIDENCE
NMS_THRESHOLD = DEFAULT_NMS_IOU
MAP_IOU_THRESHOLD = 0.50
PRIMARY_AP_POINTS = 101
LEGACY_AP_POINTS = 11
EVAL_TYPE = "combined"
RUN_SCHEMA_VERSION = 2

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

INDEX_REQUIRED_COLUMNS = ["image_file", "image_path", "label_path", "num_objects"]
GT_COLUMNS = [
    "image_file", "image_path", "class_id", "class_name",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
]
PREDICTION_COLUMNS = [
    "model", "image_index", "image_file", "image_path",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "class_id", "class_name", "object_score", "predicted_class_score",
    "combined_confidence", "nms_iou_threshold",
]
LEDGER_COLUMNS = [
    "model", "image_index", "image_file", "image_path", "status",
    "raw_candidate_count", "post_nms_prediction_count", "read_seconds",
    "predict_seconds", "postprocess_seconds", "nms_seconds", "total_seconds",
]
AGGREGATE_COLUMNS = [
    "model", "images_evaluated", "total_ground_truth", "low_floor_predictions",
    "deployment_predictions", "mAP50_101pt",
    "threshold_constrained_mAP50_11pt", "deployment_true_positives",
    "deployment_false_positives", "deployment_false_negatives",
    "deployment_micro_precision", "deployment_micro_recall", "deployment_micro_f1",
    "deployment_macro_precision", "deployment_macro_recall", "deployment_macro_f1",
    "candidate_floor", "deployment_confidence", "nms_iou_threshold",
    "map_iou_threshold",
]
PER_CLASS_COLUMNS = [
    "model", "class_id", "class_name", "ground_truth_count",
    "low_floor_prediction_count", "ap50_101pt", "deployment_prediction_count",
    "deployment_true_positives", "deployment_false_positives",
    "deployment_false_negatives", "deployment_precision", "deployment_recall",
    "deployment_f1", "threshold_constrained_ap50_11pt",
]
SOURCE_FILES = (
    Path("experiments/scripts/01_model_selection/01_model_comparison.py"),
    Path("detector_service/modules/inference/model.py"),
    Path("detector_service/modules/inference/nms.py"),
    Path("detector_service/modules/utils/metrics.py"),
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _normal_path(path):
    """Return a canonical display path without a Windows device prefix."""
    raw = str(path)
    if os.name == "nt":
        if raw.startswith("\\\\?\\UNC\\"):
            raw = "\\\\" + raw[8:]
        elif raw.startswith("\\\\?\\"):
            raw = raw[4:]
    return Path(raw)


def _absolute(path):
    return _normal_path(path).expanduser().absolute()


def _filesystem_path(path):
    """Return an extended-length Windows path only at the filesystem boundary."""
    normal = _absolute(path)
    if os.name != "nt":
        return normal
    raw = str(normal)
    if raw.startswith("\\\\?\\") or len(raw) < 248:
        return normal
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw[2:])
    return Path("\\\\?\\" + raw)


def _resolved_normal_path(path):
    """Resolve links with long-path I/O, then remove the device prefix."""
    return _normal_path(_filesystem_path(path).resolve())


def _read_image_cv2(path, cv2_module=None):
    """Decode long Windows filenames without exposing device paths as evidence."""
    if cv2_module is None:
        import cv2 as cv2_module
    normal = _absolute(path)
    filesystem = _filesystem_path(normal)
    if os.name == "nt" and str(filesystem) != str(normal):
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


def _sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with _filesystem_path(path).open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_identity(path):
    source = _absolute(path)
    filesystem = _filesystem_path(source)
    if not filesystem.is_file():
        raise FileNotFoundError(f"Required file not found: {source}")
    return {
        "path": str(source),
        "size_bytes": filesystem.stat().st_size,
        "sha256": _sha256_file(source),
    }


def probability(value):
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return number


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
    run_id = str(value)
    if not RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
        raise argparse.ArgumentTypeError("invalid run ID")
    return run_id


def resolve_indexed_path(value, asset_root):
    """Resolve a canonical or storage-relative path inside an asset root."""
    raw = str(value).strip()
    if not raw:
        raise ValueError("Indexed asset path cannot be empty.")
    root = _resolved_normal_path(asset_root)
    direct = _normal_path(raw).expanduser()
    if direct.is_absolute():
        candidate = _resolved_normal_path(direct)
    else:
        parts = PurePosixPath(raw.replace("\\", "/")).parts
        if ".." in parts:
            raise ValueError(f"Indexed path cannot traverse parents: {value}")
        if len(parts) >= 2 and tuple(parts[:2]) == (
            "detector_service", "storage"
        ):
            parts = parts[2:]
        candidate = _resolved_normal_path(root.joinpath(*parts))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Indexed asset path must remain inside external storage: {value}"
        ) from exc
    return candidate


def validate_run_scope(args):
    """Require explicit, count-gated full runs and explicit bounded pilots."""
    pilot = bool(getattr(args, "pilot", False))
    max_images = getattr(args, "max_images", None)
    expected_images = getattr(args, "expected_images", None)
    expected_labels = getattr(args, "expected_labels", None)
    if pilot:
        if max_images is None:
            raise ValueError("pilot mode requires --max-images")
        return "pilot"
    if max_images is not None:
        raise ValueError("--max-images is allowed only with --pilot")
    if expected_images is None or expected_labels is None:
        raise ValueError(
            "full mode requires --expected-images and --expected-labels"
        )
    return "full"


def load_and_validate_index(
    path, max_images=None, expected_images=None, expected_labels=None
):
    source = _absolute(path)
    filesystem = _filesystem_path(source)
    if not filesystem.is_file():
        raise FileNotFoundError(f"Dataset index not found: {source}")
    index = pd.read_csv(filesystem)
    missing = [column for column in INDEX_REQUIRED_COLUMNS if column not in index]
    if missing:
        raise ValueError("Dataset index is missing columns: " + ", ".join(missing))
    if index.empty:
        raise RuntimeError("Dataset index is empty.")
    for column in ("image_file", "image_path", "label_path"):
        if index[column].isna().any() or index[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Dataset index {column} values must be non-empty.")
    if index["image_file"].duplicated().any():
        raise ValueError("Dataset index contains duplicate image_file values.")
    if index["image_path"].duplicated().any():
        raise ValueError("Dataset index contains duplicate image_path values.")
    numeric = pd.to_numeric(index["num_objects"], errors="raise")
    values = numeric.to_numpy(float)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError("Dataset index num_objects must contain non-negative integers.")
    index = index.copy()
    index["num_objects"] = numeric.astype(np.int64)
    if max_images is not None:
        if max_images > len(index):
            raise ValueError(f"max_images {max_images} exceeds dataset size {len(index)}.")
        index = index.head(max_images).copy()
    image_count = len(index)
    label_count = int(index["num_objects"].sum())
    if expected_images is not None and image_count != expected_images:
        raise ValueError(f"Expected {expected_images} images, found {image_count}.")
    if expected_labels is not None and label_count != expected_labels:
        raise ValueError(f"Expected {expected_labels} labels, found {label_count}.")
    return source, index.reset_index(drop=True)


def _load_vocabulary(path, model_name):
    source = _absolute(path)
    filesystem = _filesystem_path(source)
    if not filesystem.is_file():
        raise FileNotFoundError(f"Missing {model_name} class vocabulary: {source}")
    raw_lines = filesystem.read_text(encoding="utf-8").splitlines()
    classes = [line.strip() for line in raw_lines]
    if not classes or any(not name for name in classes):
        raise ValueError(f"{model_name} class vocabulary is empty or contains blanks.")
    if len(classes) != len(set(classes)):
        raise ValueError(f"{model_name} class vocabulary contains duplicates.")
    return classes


def _resolve_model_bundle_paths(asset_root, models=MODELS):
    root = _resolved_normal_path(asset_root)
    if not _filesystem_path(root).is_dir():
        raise NotADirectoryError(f"External storage directory not found: {root}")
    resolved = {}
    for model_name, bundle in models.items():
        paths = {
            name: resolve_indexed_path(relative.as_posix(), root)
            for name, relative in bundle.items()
        }
        missing = [
            f"{name}: {path}" for name, path in paths.items()
            if not _filesystem_path(path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {model_name} bundle assets: " + "; ".join(missing)
            )
        resolved[model_name] = paths
    return root, resolved


def _validate_cfg_class_count(path, model_name, expected_count):
    declarations = []
    for raw_line in _filesystem_path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].split(";", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key.lower() == "classes":
            try:
                declarations.append(int(value))
            except ValueError as exc:
                raise ValueError(
                    f"{model_name} cfg contains a non-integer classes value."
                ) from exc
    if not declarations:
        raise ValueError(f"{model_name} cfg contains no classes declaration.")
    if any(count != expected_count for count in declarations):
        raise ValueError(
            f"{model_name} cfg class count does not match its vocabulary."
        )


def resolve_and_validate_model_bundles(asset_root, models=MODELS):
    root, resolved = _resolve_model_bundle_paths(asset_root, models=models)
    vocabularies = {}
    for model_name, paths in resolved.items():
        vocabularies[model_name] = _load_vocabulary(paths["names"], model_name)
    first_name = next(iter(vocabularies))
    classes = vocabularies[first_name]
    for model_name, vocabulary in vocabularies.items():
        if vocabulary != classes:
            raise ValueError(
                f"Checkpoint class vocabularies differ: {first_name} != {model_name}"
            )
        _validate_cfg_class_count(
            resolved[model_name]["cfg"], model_name, len(vocabulary)
        )
    return root, resolved, classes


def parse_yolo_labels_strict(label_path, image_width, image_height, classes):
    source = _absolute(label_path)
    filesystem = _filesystem_path(source)
    if not filesystem.is_file():
        raise FileNotFoundError(f"Label file not found: {source}")
    labels = []
    for line_number, raw_line in enumerate(
        filesystem.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 5:
            raise ValueError(f"{source}:{line_number}: expected exactly five YOLO fields")
        try:
            values = [float(field) for field in fields]
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number}: YOLO fields must be numeric") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{source}:{line_number}: YOLO fields must be finite")
        raw_class, center_x, center_y, width, height = values
        class_id = int(raw_class)
        if raw_class != class_id or not 0 <= class_id < len(classes):
            raise ValueError(f"{source}:{line_number}: invalid class identifier")
        if not (
            0 <= center_x <= 1 and 0 <= center_y <= 1
            and 0 < width <= 1 and 0 < height <= 1
        ):
            raise ValueError(f"{source}:{line_number}: invalid normalized YOLO box")
        pixel_w, pixel_h = width * image_width, height * image_height
        labels.append({
            "class_id": class_id,
            "class_name": classes[class_id],
            "bbox_x": center_x * image_width - pixel_w / 2,
            "bbox_y": center_y * image_height - pixel_h / 2,
            "bbox_w": pixel_w,
            "bbox_h": pixel_h,
        })
    return labels


def yolo_label_to_xywh(label_path, image_w, image_h, classes):
    return parse_yolo_labels_strict(label_path, image_w, image_h, classes)


@contextmanager
def _atomic_csv_stream(path, fieldnames):
    destination = _absolute(path)
    filesystem_destination = _filesystem_path(destination)
    filesystem_parent = _filesystem_path(destination.parent)
    filesystem_parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.parent / (
        f".{destination.name}.{secrets.token_hex(8)}.tmp"
    )
    handle = None
    try:
        handle = _filesystem_path(temporary_path).open(
            mode="x", encoding="utf-8", newline="",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        yield writer
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        _filesystem_path(temporary_path).replace(filesystem_destination)
    except Exception:
        if handle is not None:
            handle.close()
        _filesystem_path(temporary_path).unlink(missing_ok=True)
        raise


def _write_dataframe_atomic(path, table, columns):
    with _atomic_csv_stream(path, columns) as writer:
        for record in table.reindex(columns=columns).to_dict(orient="records"):
            writer.writerow(record)


def _write_json_atomic(path, value):
    destination = _absolute(path)
    filesystem_destination = _filesystem_path(destination)
    filesystem_parent = _filesystem_path(destination.parent)
    filesystem_parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.parent / (
        f".{destination.name}.{secrets.token_hex(8)}.tmp"
    )
    try:
        with _filesystem_path(temporary_path).open(
            mode="x", encoding="utf-8",
        ) as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _filesystem_path(temporary_path).replace(filesystem_destination)
    except Exception:
        _filesystem_path(temporary_path).unlink(missing_ok=True)
        raise


def build_ground_truth(index, classes, asset_root, output_path=None, image_reader=None):
    """Decode every image and require parsed labels to match the index."""
    if image_reader is None:
        image_reader = _read_image_cv2
    records = []
    for position, row in enumerate(index.itertuples(index=False), start=1):
        image_path = resolve_indexed_path(row.image_path, asset_root)
        label_path = resolve_indexed_path(row.label_path, asset_root)
        frame = image_reader(str(image_path))
        if frame is None:
            raise ValueError(f"Unable to read indexed image: {image_path}")
        image_height, image_width = frame.shape[:2]
        labels = parse_yolo_labels_strict(
            label_path, image_width, image_height, classes
        )
        if len(labels) != int(row.num_objects):
            raise ValueError(
                f"Label count mismatch for {row.image_file}: "
                f"index={row.num_objects}, parsed={len(labels)}"
            )
        records.extend({
            "image_file": row.image_file,
            "image_path": row.image_path,
            **label,
        } for label in labels)
        if position % 1000 == 0:
            print(f"[GT] Processed {position}/{len(index)} images")
    ground_truth = pd.DataFrame(records, columns=GT_COLUMNS)
    if len(ground_truth) != int(index["num_objects"].sum()):
        raise RuntimeError("Ground-truth output does not match indexed label total.")
    if output_path is not None:
        _write_dataframe_atomic(output_path, ground_truth, GT_COLUMNS)
    return ground_truth


def _serialize_prediction(
    model_name, image_index, image_row, bbox, class_id, object_score,
    score_vector, classes, nms_iou_threshold,
):
    class_id = int(class_id)
    probabilities = np.asarray(score_vector, dtype=float).reshape(-1)
    if len(probabilities) != len(classes):
        raise ValueError(f"{model_name} returned the wrong class-score vector length.")
    if not 0 <= class_id < len(classes):
        raise ValueError(f"{model_name} returned a class ID outside the vocabulary.")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError(f"{model_name} returned invalid class probabilities.")
    objectness = float(object_score)
    if not math.isfinite(objectness) or not 0 <= objectness <= 1:
        raise ValueError(f"{model_name} returned invalid objectness.")
    predicted_score = float(probabilities[class_id])
    return {
        "model": model_name,
        "image_index": image_index,
        "image_file": image_row.image_file,
        "image_path": image_row.image_path,
        "bbox_x": float(bbox[0]), "bbox_y": float(bbox[1]),
        "bbox_w": float(bbox[2]), "bbox_h": float(bbox[3]),
        "class_id": class_id, "class_name": classes[class_id],
        "object_score": objectness,
        "predicted_class_score": predicted_score,
        "combined_confidence": objectness * predicted_score,
        "nms_iou_threshold": float(nms_iou_threshold),
    }


def run_inference_for_model(
    model_name, bundle, index, classes, asset_root, candidate_floor,
    nms_iou_threshold, prediction_path, ledger_writer, image_reader=None,
    detector_factory=None, clock=time.perf_counter,
):
    """Stream compact post-NMS rows and one complete ledger row per image."""
    if image_reader is None or detector_factory is None:
        from detector_service.modules.inference.model import Detector
        image_reader = image_reader or _read_image_cv2
        detector_factory = detector_factory or Detector
    detector = detector_factory(
        str(_filesystem_path(bundle["weights"])),
        str(_filesystem_path(bundle["cfg"])),
        str(_filesystem_path(bundle["names"])),
        score_threshold=candidate_floor,
    )
    nms = NMS(score_threshold=candidate_floor, nms_iou_threshold=nms_iou_threshold)
    total_raw = total_retained = 0
    with _atomic_csv_stream(prediction_path, PREDICTION_COLUMNS) as prediction_writer:
        for position, row in enumerate(index.itertuples(index=False), start=1):
            total_started = read_started = clock()
            image_path = resolve_indexed_path(row.image_path, asset_root)
            frame = image_reader(str(image_path))
            read_finished = clock()
            if frame is None:
                raise ValueError(f"Unable to read indexed image: {image_path}")
            predict_started = read_finished
            outputs = detector.predict(frame)
            predict_finished = clock()
            postprocess_started = predict_finished
            decoded = detector.post_process(outputs)
            postprocess_finished = clock()
            if len(decoded) != 4 or len({len(values) for values in decoded}) != 1:
                raise ValueError(
                    f"{model_name} returned inconsistent detections for {row.image_file}."
                )
            for score_vector in decoded[3]:
                probabilities = np.asarray(score_vector, dtype=float).reshape(-1)
                if len(probabilities) != len(classes):
                    raise ValueError(
                        f"{model_name} returned the wrong class-score vector length."
                    )
            image_raw = len(decoded[0])
            nms_started = postprocess_finished
            retained = nms.filter(*decoded)
            nms_finished = clock()
            image_retained = len(retained[0])
            total_raw += image_raw
            total_retained += image_retained
            for values in zip(*retained):
                prediction_writer.writerow(_serialize_prediction(
                    model_name, position, row, *values, classes, nms_iou_threshold
                ))
            ledger_writer.writerow({
                "model": model_name, "image_index": position,
                "image_file": row.image_file, "image_path": row.image_path,
                "status": "processed", "raw_candidate_count": image_raw,
                "post_nms_prediction_count": image_retained,
                "read_seconds": read_finished - read_started,
                "predict_seconds": predict_finished - predict_started,
                "postprocess_seconds": postprocess_finished - postprocess_started,
                "nms_seconds": nms_finished - nms_started,
                "total_seconds": nms_finished - total_started,
            })
            if position % 500 == 0:
                print(
                    f"[{model_name}] Processed {position}/{len(index)} images | "
                    f"raw={total_raw} | retained={total_retained}"
                )
    return {
        "images_processed": len(index), "raw_candidates": total_raw,
        "post_nms_predictions": total_retained,
    }


def _groups_by_image(table):
    if table.empty:
        return {}
    return {
        image_file: group
        for image_file, group in table.groupby("image_file", sort=False)
    }


def validate_prediction_table(table, model_name, index, classes, policy):
    missing = [column for column in PREDICTION_COLUMNS if column not in table]
    if missing:
        raise ValueError(f"{model_name} predictions are missing: {', '.join(missing)}")
    normalized = table[PREDICTION_COLUMNS].copy()
    if normalized.empty:
        return normalized
    if set(normalized["model"].astype(str)) != {model_name}:
        raise ValueError(f"Prediction table must contain only {model_name} rows.")
    image_numeric = pd.to_numeric(normalized["image_index"], errors="raise")
    image_values = image_numeric.to_numpy(float)
    if (
        not np.isfinite(image_values).all()
        or not np.equal(image_values, np.floor(image_values)).all()
    ):
        raise ValueError(f"{model_name} predictions contain invalid image indexes.")
    normalized["image_index"] = image_numeric.astype(np.int64)
    expected_keys = {
        (position, str(row.image_file), str(row.image_path))
        for position, row in enumerate(index.itertuples(index=False), start=1)
    }
    observed_keys = set(zip(
        normalized["image_index"].astype(int),
        normalized["image_file"].astype(str),
        normalized["image_path"].astype(str),
    ))
    if not observed_keys.issubset(expected_keys):
        raise ValueError(
            f"{model_name} prediction image keys do not match the dataset index."
        )
    class_numeric = pd.to_numeric(normalized["class_id"], errors="raise")
    class_values = class_numeric.to_numpy(float)
    if not np.equal(class_values, np.floor(class_values)).all():
        raise ValueError(f"{model_name} predictions contain non-integer class IDs.")
    normalized["class_id"] = class_numeric.astype(np.int64)
    if (normalized["class_id"] < 0).any() or (
        normalized["class_id"] >= len(classes)
    ).any():
        raise ValueError(f"{model_name} predictions contain invalid class IDs.")
    expected_names = normalized["class_id"].map(dict(enumerate(classes)))
    if not normalized["class_name"].astype(str).equals(expected_names.astype(str)):
        raise ValueError(f"{model_name} prediction class names do not match IDs.")
    numeric_columns = [
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "object_score",
        "predicted_class_score", "combined_confidence", "nms_iou_threshold",
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        if not np.isfinite(normalized[column].to_numpy(float)).all():
            raise ValueError(f"{model_name} {column} must be finite.")
    if (normalized[["bbox_w", "bbox_h"]] < 0).any().any():
        raise ValueError(f"{model_name} predictions contain negative dimensions.")
    for column in ("object_score", "predicted_class_score", "combined_confidence"):
        if not normalized[column].between(0, 1).all():
            raise ValueError(f"{model_name} {column} values must be probabilities.")
    expected_combined = normalized["object_score"] * normalized["predicted_class_score"]
    if not np.allclose(
        normalized["combined_confidence"], expected_combined,
        rtol=1e-12, atol=1e-15,
    ):
        raise ValueError(f"{model_name} combined confidence is inconsistent.")
    if not np.allclose(
        normalized["nms_iou_threshold"], policy["nms_iou_threshold"],
        rtol=0, atol=1e-15,
    ):
        raise ValueError(f"{model_name} predictions use the wrong NMS threshold.")
    if (normalized["combined_confidence"] < policy["candidate_floor"] - 1e-15).any():
        raise ValueError(f"{model_name} predictions fall below the low floor.")
    return normalized.reset_index(drop=True)


def validate_ledger(ledger, index, model_names):
    missing = [column for column in LEDGER_COLUMNS if column not in ledger]
    if missing:
        raise ValueError("Inference ledger is missing: " + ", ".join(missing))
    normalized = ledger[LEDGER_COLUMNS].copy()
    expected_files = index["image_file"].astype(str).tolist()
    expected_paths = index["image_path"].astype(str).tolist()
    model_names = list(model_names)
    observed_models = set(normalized["model"].astype(str))
    if observed_models - set(model_names):
        raise ValueError("Inference ledger contains unexpected model rows.")
    for model_name in model_names:
        rows = normalized[normalized["model"].astype(str) == model_name].copy()
        if len(rows) != len(index):
            raise ValueError(f"Inference ledger is incomplete for {model_name}.")
        positions = pd.to_numeric(rows["image_index"], errors="raise")
        position_values = positions.to_numpy(float)
        if (
            not np.isfinite(position_values).all()
            or not np.equal(position_values, np.floor(position_values)).all()
        ):
            raise ValueError(f"Inference ledger positions are invalid for {model_name}.")
        rows["image_index"] = positions.astype(np.int64)
        rows = rows.sort_values("image_index")
        if rows["image_file"].astype(str).tolist() != expected_files:
            raise ValueError(f"Inference ledger order is invalid for {model_name}.")
        if rows["image_path"].astype(str).tolist() != expected_paths:
            raise ValueError(f"Inference ledger paths are invalid for {model_name}.")
        if set(rows["status"].astype(str)) != {"processed"}:
            raise ValueError(f"Inference ledger contains failed rows for {model_name}.")
        if rows["image_index"].tolist() != list(range(1, len(index) + 1)):
            raise ValueError(f"Inference ledger positions are invalid for {model_name}.")
        for column in ("raw_candidate_count", "post_nms_prediction_count"):
            values = pd.to_numeric(rows[column], errors="raise").to_numpy(float)
            if (
                not np.isfinite(values).all() or (values < 0).any()
                or not np.equal(values, np.floor(values)).all()
            ):
                raise ValueError(f"Inference ledger {column} is invalid for {model_name}.")
            rows[column] = values.astype(np.int64)
        if (
            rows["post_nms_prediction_count"] > rows["raw_candidate_count"]
        ).any():
            raise ValueError(
                f"Inference ledger retained count exceeds raw count for {model_name}."
            )
        timing_columns = [
            "read_seconds", "predict_seconds", "postprocess_seconds",
            "nms_seconds", "total_seconds",
        ]
        for column in timing_columns:
            rows[column] = pd.to_numeric(rows[column], errors="raise")
            values = rows[column].to_numpy(float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(
                    f"Inference ledger {column} is invalid for {model_name}."
                )
        component_total = rows[[
            "read_seconds", "predict_seconds", "postprocess_seconds", "nms_seconds"
        ]].sum(axis=1)
        if not np.allclose(
            rows["total_seconds"], component_total, rtol=1e-7, atol=1e-9
        ):
            raise ValueError(
                f"Inference ledger timing totals are inconsistent for {model_name}."
            )
        normalized.loc[rows.index, LEDGER_COLUMNS] = rows[LEDGER_COLUMNS]
    if len(normalized) != len(index) * len(model_names):
        raise ValueError("Inference ledger contains unexpected model rows.")
    return normalized.reset_index(drop=True)


def validate_prediction_ledger_alignment(predictions, ledger, model_name, index):
    rows = ledger[ledger["model"].astype(str) == model_name].copy()
    rows["image_index"] = pd.to_numeric(rows["image_index"], errors="raise").astype(int)
    expected_counts = rows.set_index("image_index")["post_nms_prediction_count"].astype(int)
    observed_counts = predictions.groupby("image_index").size().reindex(
        range(1, len(index) + 1), fill_value=0
    )
    expected_counts = expected_counts.reindex(range(1, len(index) + 1))
    if expected_counts.isna().any() or not np.array_equal(
        observed_counts.to_numpy(int), expected_counts.to_numpy(int)
    ):
        raise ValueError(
            f"Predictions and ledger disagree by image for {model_name}."
        )


def validate_inference_summary(summary, ledger, model_name, image_count):
    rows = ledger[ledger["model"].astype(str) == model_name]
    expected = {
        "images_processed": int(image_count),
        "raw_candidates": int(rows["raw_candidate_count"].sum()),
        "post_nms_predictions": int(rows["post_nms_prediction_count"].sum()),
    }
    if summary != expected:
        raise ValueError(f"Inference summary disagrees with ledger for {model_name}.")


def build_metric_lists(index, predictions, ground_truth):
    prediction_groups = _groups_by_image(predictions)
    truth_groups = _groups_by_image(ground_truth)
    boxes, classes, scores, class_scores, truth_boxes, truth_classes = (
        [], [], [], [], [], []
    )
    for row in index.itertuples(index=False):
        predicted = prediction_groups.get(row.image_file)
        if predicted is None:
            boxes.append([])
            classes.append([])
            scores.append([])
            class_scores.append([])
        else:
            boxes.append(predicted[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(float).tolist())
            classes.append(predicted["class_id"].astype(int).tolist())
            scores.append(predicted["object_score"].astype(float).tolist())
            # A one-element vector is explicitly supported by metrics.py.
            class_scores.append([[float(value)] for value in predicted["predicted_class_score"]])
        truth = truth_groups.get(row.image_file)
        if truth is None:
            truth_boxes.append([])
            truth_classes.append([])
        else:
            truth_boxes.append(truth[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(float).tolist())
            truth_classes.append(truth["class_id"].astype(int).tolist())
    return boxes, classes, scores, class_scores, truth_boxes, truth_classes


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision, recall):
    return (
        float(2 * precision * recall / (precision + recall))
        if precision + recall else 0.0
    )


def _ap_by_class(index, predictions, ground_truth, classes, point_count):
    inputs = build_metric_lists(index, predictions, ground_truth)
    matches, truth_counts = match_detections(
        boxes=inputs[0], classes=inputs[1], scores=inputs[2], cls_scores=inputs[3],
        gt_boxes=inputs[4], gt_classes=inputs[5],
        map_iou_threshold=MAP_IOU_THRESHOLD, eval_type=EVAL_TYPE,
    )
    precision, recall, _ = calculate_precision_recall_curve(
        matches, truth_counts, num_classes=len(classes)
    )
    ap = {}
    for class_id in range(len(classes)):
        curve = list(zip(recall[class_id], precision[class_id]))
        ap[class_id] = calculate_map_x_point_interpolated(
            {0: curve}, num_classes=1, num_interpolated_points=point_count
        )
    return float(np.mean(list(ap.values()))), ap, matches, truth_counts


def evaluate_model(model_name, index, predictions, ground_truth, classes, policy):
    primary_map, primary_ap, _, _ = _ap_by_class(
        index, predictions, ground_truth, classes, PRIMARY_AP_POINTS
    )
    deployment = predictions[
        predictions["combined_confidence"] >= policy["deployment_confidence"]
    ].copy()
    legacy_map, legacy_ap, matches, truth_counts = _ap_by_class(
        index, deployment, ground_truth, classes, LEGACY_AP_POINTS
    )
    rows = []
    for class_id, class_name in enumerate(classes):
        tp = sum(bool(matched) for _, matched in matches.get(class_id, []))
        prediction_count = len(matches.get(class_id, []))
        fp = prediction_count - tp
        gt = int(truth_counts.get(class_id, 0))
        fn = gt - tp
        precision = _safe_ratio(tp, prediction_count)
        recall = _safe_ratio(tp, gt)
        rows.append({
            "model": model_name, "class_id": class_id, "class_name": class_name,
            "ground_truth_count": gt,
            "low_floor_prediction_count": int((predictions["class_id"] == class_id).sum()),
            "ap50_101pt": primary_ap[class_id],
            "deployment_prediction_count": prediction_count,
            "deployment_true_positives": tp, "deployment_false_positives": fp,
            "deployment_false_negatives": fn, "deployment_precision": precision,
            "deployment_recall": recall, "deployment_f1": _f1(precision, recall),
            "threshold_constrained_ap50_11pt": legacy_ap[class_id],
        })
    per_class = pd.DataFrame(rows, columns=PER_CLASS_COLUMNS)
    tp = int(per_class["deployment_true_positives"].sum())
    fp = int(per_class["deployment_false_positives"].sum())
    fn = int(per_class["deployment_false_negatives"].sum())
    micro_precision, micro_recall = _safe_ratio(tp, tp + fp), _safe_ratio(tp, tp + fn)
    aggregate = {
        "model": model_name, "images_evaluated": len(index),
        "total_ground_truth": len(ground_truth),
        "low_floor_predictions": len(predictions),
        "deployment_predictions": len(deployment), "mAP50_101pt": primary_map,
        "threshold_constrained_mAP50_11pt": legacy_map,
        "deployment_true_positives": tp, "deployment_false_positives": fp,
        "deployment_false_negatives": fn,
        "deployment_micro_precision": micro_precision,
        "deployment_micro_recall": micro_recall,
        "deployment_micro_f1": _f1(micro_precision, micro_recall),
        "deployment_macro_precision": float(per_class["deployment_precision"].mean()),
        "deployment_macro_recall": float(per_class["deployment_recall"].mean()),
        "deployment_macro_f1": float(per_class["deployment_f1"].mean()),
        "candidate_floor": policy["candidate_floor"],
        "deployment_confidence": policy["deployment_confidence"],
        "nms_iou_threshold": policy["nms_iou_threshold"],
        "map_iou_threshold": policy["map_iou_threshold"],
    }
    return aggregate, per_class


def _csv_row_count(path):
    with _filesystem_path(path).open("r", encoding="utf-8", newline="") as source:
        return max(0, sum(1 for _ in source) - 1)


def _artifact_identity(path, columns, rows):
    source = _absolute(path)
    filesystem = _filesystem_path(source)
    with filesystem.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if header != list(columns):
        raise ValueError(f"Artifact schema mismatch: {source.name}")
    observed_rows = _csv_row_count(source)
    if observed_rows != int(rows):
        raise ValueError(
            f"Artifact row mismatch for {source.name}: {observed_rows} != {rows}"
        )
    return {
        "sha256": _sha256_file(source), "size_bytes": filesystem.stat().st_size,
        "rows": observed_rows, "columns": list(columns),
    }


def _source_identities():
    identities = {}
    for relative in SOURCE_FILES:
        identity = _file_identity(PROJECT_ROOT / relative)
        identity["path"] = relative.as_posix()
        identities[relative.as_posix()] = identity
    return identities


def _model_input_identities(bundles):
    return {
        model_name: {key: _file_identity(path) for key, path in bundle.items()}
        for model_name, bundle in bundles.items()
    }


def _dataset_asset_identity(index, asset_root):
    """Hash every selected image and label into one deterministic identity."""
    digest = hashlib.sha256()
    total_size = 0
    record_count = 0
    for image_index, row in enumerate(index.itertuples(index=False), start=1):
        for kind, logical_path in (
            ("image", row.image_path), ("label", row.label_path)
        ):
            identity = _file_identity(resolve_indexed_path(logical_path, asset_root))
            record = {
                "image_index": image_index,
                "kind": kind,
                "indexed_path": str(logical_path),
                **identity,
            }
            digest.update(
                json.dumps(
                    record, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            )
            digest.update(b"\n")
            total_size += int(identity["size_bytes"])
            record_count += 1
    return {
        "images": len(index), "labels": len(index), "files": record_count,
        "size_bytes": total_size, "sha256": digest.hexdigest(),
    }


def _capture_input_identities(dataset_path, index, asset_root, bundles):
    return {
        "dataset_index": _file_identity(dataset_path),
        "dataset_assets": _dataset_asset_identity(index, asset_root),
        "models": _model_input_identities(bundles),
        "source_files": _source_identities(),
    }


def _assert_inputs_unchanged(expected, observed):
    if observed == expected:
        return
    changed = [
        name for name in expected
        if expected.get(name) != observed.get(name)
    ]
    raise RuntimeError(
        "Run inputs changed after their initial snapshot: " + ", ".join(changed)
    )


def _environment_metadata():
    try:
        import cv2
        opencv_version = cv2.__version__
        opencv_threads = int(cv2.getNumThreads())
    except Exception:
        opencv_version, opencv_threads = "unavailable", None
    return {
        "python_version": platform.python_version(),
        "opencv_version": opencv_version, "opencv_threads": opencv_threads,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
    }


def _manifest_input_identities(manifest):
    dataset = manifest["dataset"]
    return {
        "dataset_index": {
            key: dataset[key] for key in ("path", "size_bytes", "sha256")
        },
        "dataset_assets": dataset["asset_identity"],
        "models": manifest["models"],
        "source_files": manifest["source_files"],
    }


def _validate_manifest_structure(manifest):
    required = {
        "schema_version", "run_id", "run_scope", "status", "dataset",
        "external_storage_root", "models", "class_vocabulary", "policy",
        "policy_sha256", "source_files", "source_policy_sha256",
        "input_identities_sha256", "artifacts",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError("Run manifest is missing: " + ", ".join(missing))
    if manifest["schema_version"] != RUN_SCHEMA_VERSION:
        raise ValueError("Run manifest schema version is unsupported.")
    if manifest["status"] != "complete":
        raise ValueError("Run manifest is not marked complete.")
    if manifest["run_scope"] not in {"full", "pilot"}:
        raise ValueError("Run manifest scope is invalid.")
    if _sha256_json(manifest["policy"]) != manifest["policy_sha256"]:
        raise ValueError("Run manifest policy identity is invalid.")
    if _sha256_json({
        "policy": manifest["policy"], "source_files": manifest["source_files"]
    }) != manifest["source_policy_sha256"]:
        raise ValueError("Run manifest source-policy identity is invalid.")
    inputs = _manifest_input_identities(manifest)
    if _sha256_json(inputs) != manifest["input_identities_sha256"]:
        raise ValueError("Run manifest input identity is invalid.")
    dataset = manifest["dataset"]
    for key in (
        "path", "size_bytes", "sha256", "selected_images", "selected_labels",
        "asset_identity", "selection_max_images",
    ):
        if key not in dataset:
            raise ValueError(f"Run manifest dataset is missing: {key}")
    assets = dataset["asset_identity"]
    selected_images = int(dataset["selected_images"])
    if (
        selected_images <= 0
        or int(assets.get("images", -1)) != selected_images
        or int(assets.get("labels", -1)) != selected_images
        or int(assets.get("files", -1)) != 2 * selected_images
    ):
        raise ValueError("Run manifest dataset asset counts are invalid.")
    if manifest["run_scope"] == "full" and dataset["selection_max_images"] is not None:
        raise ValueError("Full run manifest cannot contain a maximum image selection.")
    if manifest["run_scope"] == "pilot" and dataset["selection_max_images"] is None:
        raise ValueError("Pilot run manifest must contain a maximum image selection.")
    if set(manifest["models"]) != set(MODELS):
        raise ValueError("Run manifest model set is invalid.")
    expected_artifacts = {
        "ground_truth.csv", "inference_ledger.csv", "aggregate_metrics.csv",
        "per_class_metrics.csv",
        *(f"{model_name}_predictions.csv" for model_name in MODELS),
    }
    if set(manifest["artifacts"]) != expected_artifacts:
        raise ValueError("Run manifest artifact set is invalid.")


def _recapture_manifest_inputs(manifest):
    dataset = manifest["dataset"]
    max_images = dataset["selection_max_images"]
    dataset_path, index = load_and_validate_index(
        dataset["path"], max_images=max_images,
        expected_images=int(dataset["selected_images"]),
        expected_labels=int(dataset["selected_labels"]),
    )
    asset_root = Path(manifest["external_storage_root"])
    bundles = {}
    for model_name, bundle in manifest["models"].items():
        bundles[model_name] = {}
        for key, identity in bundle.items():
            path = resolve_indexed_path(identity["path"], asset_root)
            bundles[model_name][key] = path
    return _capture_input_identities(
        dataset_path, index, asset_root, bundles
    )


def verify_run_directory(run_directory, manifest=None, current_inputs=None):
    """Verify the on-disk manifest, every input identity, and every artifact."""
    directory = _absolute(run_directory)
    manifest_path = directory / "run_manifest.json"
    manifest_filesystem = _filesystem_path(manifest_path)
    if not manifest_filesystem.is_file():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    disk_manifest = json.loads(manifest_filesystem.read_text(encoding="utf-8"))
    if manifest is not None and disk_manifest != manifest:
        raise ValueError("On-disk run manifest does not match the expected manifest.")
    manifest = disk_manifest
    _validate_manifest_structure(manifest)
    expected_inputs = _manifest_input_identities(manifest)
    observed_inputs = (
        _recapture_manifest_inputs(manifest)
        if current_inputs is None else current_inputs
    )
    _assert_inputs_unchanged(expected_inputs, observed_inputs)
    for name, expected in manifest["artifacts"].items():
        relative = PurePosixPath(str(name).replace("\\", "/"))
        if relative.is_absolute() or len(relative.parts) != 1 or ".." in relative.parts:
            raise ValueError(f"Run artifact path is invalid: {name}")
        path = directory / name
        filesystem = _filesystem_path(path)
        if not filesystem.is_file():
            raise FileNotFoundError(f"Run artifact not found: {path}")
        if _sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Run artifact hash mismatch: {name}")
        if filesystem.stat().st_size != expected["size_bytes"]:
            raise ValueError(f"Run artifact size mismatch: {name}")
        if _csv_row_count(path) != expected["rows"]:
            raise ValueError(f"Run artifact row-count mismatch: {name}")
        with filesystem.open("r", encoding="utf-8", newline="") as handle:
            if next(csv.reader(handle), []) != expected["columns"]:
                raise ValueError(f"Run artifact schema mismatch: {name}")
    return True


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a strict full-corpus checkpoint comparison."
    )
    parser.add_argument("--dataset-index", type=Path, default=DEFAULT_DATASET_INDEX)
    parser.add_argument(
        "--asset-root", type=Path, required=True,
        help="External storage directory containing logistics and yolo_model_*.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=validate_run_id, required=True)
    parser.add_argument(
        "--pilot", action="store_true",
        help="Run an explicitly bounded pilot; requires --max-images.",
    )
    parser.add_argument(
        "--expected-images", type=positive_int,
        help="Required exact image count in full mode.",
    )
    parser.add_argument(
        "--expected-labels", type=nonnegative_int,
        help="Required exact label count in full mode.",
    )
    parser.add_argument(
        "--max-images", type=positive_int,
        help="Head-of-index image limit; allowed only with --pilot.",
    )
    parser.add_argument(
        "--candidate-floor", type=probability, default=DEFAULT_CANDIDATE_FLOOR
    )
    parser.add_argument(
        "--deployment-confidence", type=probability,
        default=DEFAULT_DEPLOYMENT_CONFIDENCE,
    )
    parser.add_argument("--nms-iou", type=probability, default=DEFAULT_NMS_IOU)
    return parser


def run_experiment(
    args, detector_factory=None, image_reader=None, clock=time.perf_counter
):
    """Generate and promote a complete immutable run, or leave staging evidence."""
    started_utc = _utc_now()
    run_scope = validate_run_scope(args)
    if args.deployment_confidence < args.candidate_floor:
        raise ValueError("deployment confidence must be at least candidate floor")

    dataset_candidate = _absolute(args.dataset_index)
    dataset_identity = _file_identity(dataset_candidate)
    source_identities = _source_identities()
    prevalidated_root, prevalidated_bundles = _resolve_model_bundle_paths(
        args.asset_root
    )
    model_identities = _model_input_identities(prevalidated_bundles)
    dataset_path, index = load_and_validate_index(
        args.dataset_index, max_images=args.max_images,
        expected_images=args.expected_images, expected_labels=args.expected_labels,
    )
    if _file_identity(dataset_path) != dataset_identity:
        raise RuntimeError("Dataset index changed while it was being loaded.")
    asset_root, bundles, classes = resolve_and_validate_model_bundles(args.asset_root)
    if asset_root != prevalidated_root or bundles != prevalidated_bundles:
        raise RuntimeError("Resolved model bundle paths changed during validation.")
    if _model_input_identities(bundles) != model_identities:
        raise RuntimeError("Model inputs changed while they were being validated.")
    input_snapshot = {
        "dataset_index": dataset_identity,
        "dataset_assets": _dataset_asset_identity(index, asset_root),
        "models": model_identities,
        "source_files": source_identities,
    }

    output_root = _absolute(args.output_root)
    _filesystem_path(output_root).mkdir(parents=True, exist_ok=True)
    final_directory = output_root / args.run_id
    staging_directory = output_root / f".{args.run_id}.incomplete"
    if _filesystem_path(final_directory).exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {final_directory}")
    if _filesystem_path(staging_directory).exists():
        raise FileExistsError(
            f"Incomplete run already exists; inspect it or use a new ID: {staging_directory}"
        )
    _filesystem_path(staging_directory).mkdir()

    policy = {
        "candidate_objectness_comparator": ">",
        "candidate_floor": float(args.candidate_floor),
        "low_floor_combined_confidence_comparator": ">=",
        "deployment_confidence": float(args.deployment_confidence),
        "deployment_confidence_comparator": ">=",
        "nms_iou_threshold": float(args.nms_iou),
        "map_iou_threshold": MAP_IOU_THRESHOLD,
        "primary_ap_interpolation_points": PRIMARY_AP_POINTS,
        "legacy_ap_interpolation_points": LEGACY_AP_POINTS,
        "eval_type": EVAL_TYPE,
    }

    ground_truth_path = staging_directory / "ground_truth.csv"
    ledger_path = staging_directory / "inference_ledger.csv"
    ground_truth = build_ground_truth(
        index, classes, asset_root, output_path=ground_truth_path,
        image_reader=image_reader,
    )
    inference_summaries = {}
    with _atomic_csv_stream(ledger_path, LEDGER_COLUMNS) as ledger_writer:
        for model_name, bundle in bundles.items():
            inference_summaries[model_name] = run_inference_for_model(
                model_name, bundle, index, classes, asset_root,
                args.candidate_floor, args.nms_iou,
                staging_directory / f"{model_name}_predictions.csv",
                ledger_writer, image_reader=image_reader,
                detector_factory=detector_factory, clock=clock,
            )

    ledger = validate_ledger(pd.read_csv(ledger_path), index, bundles.keys())
    aggregate_rows, per_class_tables, prediction_tables = [], [], {}
    for model_name in bundles:
        predictions = pd.read_csv(staging_directory / f"{model_name}_predictions.csv")
        predictions = validate_prediction_table(
            predictions, model_name, index, classes, policy
        )
        validate_prediction_ledger_alignment(
            predictions, ledger, model_name, index
        )
        validate_inference_summary(
            inference_summaries[model_name], ledger, model_name, len(index)
        )
        aggregate, per_class = evaluate_model(
            model_name, index, predictions, ground_truth, classes, policy
        )
        aggregate_rows.append(aggregate)
        per_class_tables.append(per_class)
        prediction_tables[model_name] = predictions

    aggregate_table = pd.DataFrame(aggregate_rows, columns=AGGREGATE_COLUMNS)
    per_class_table = pd.concat(per_class_tables, ignore_index=True).reindex(
        columns=PER_CLASS_COLUMNS
    )
    _write_dataframe_atomic(
        staging_directory / "aggregate_metrics.csv", aggregate_table, AGGREGATE_COLUMNS
    )
    _write_dataframe_atomic(
        staging_directory / "per_class_metrics.csv", per_class_table, PER_CLASS_COLUMNS
    )

    artifact_specs = {
        "ground_truth.csv": (GT_COLUMNS, len(ground_truth)),
        "inference_ledger.csv": (LEDGER_COLUMNS, len(ledger)),
        "aggregate_metrics.csv": (AGGREGATE_COLUMNS, len(aggregate_table)),
        "per_class_metrics.csv": (PER_CLASS_COLUMNS, len(per_class_table)),
    }
    for model_name, predictions in prediction_tables.items():
        artifact_specs[f"{model_name}_predictions.csv"] = (
            PREDICTION_COLUMNS, len(predictions)
        )
    artifacts = {
        name: _artifact_identity(staging_directory / name, columns, rows)
        for name, (columns, rows) in artifact_specs.items()
    }
    observed_inputs = _capture_input_identities(
        dataset_path, index, asset_root, bundles
    )
    _assert_inputs_unchanged(input_snapshot, observed_inputs)
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION, "run_id": args.run_id,
        "run_scope": run_scope, "status": "complete", "started_utc": started_utc,
        "completed_utc": _utc_now(),
        "dataset": {
            **input_snapshot["dataset_index"], "selected_images": len(index),
            "selected_labels": int(index["num_objects"].sum()),
            "selection_max_images": args.max_images,
            "asset_identity": input_snapshot["dataset_assets"],
        },
        "external_storage_root": str(asset_root),
        "models": input_snapshot["models"],
        "class_vocabulary": classes, "policy": policy,
        "policy_sha256": _sha256_json(policy),
        "source_files": input_snapshot["source_files"],
        "source_policy_sha256": _sha256_json(
            {"policy": policy, "source_files": input_snapshot["source_files"]}
        ),
        "input_identities_sha256": _sha256_json(input_snapshot),
        "environment": _environment_metadata(),
        "command": [str(value) for value in sys.argv],
        "inference_summaries": inference_summaries, "artifacts": artifacts,
    }
    manifest_path = staging_directory / "run_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    loaded = json.loads(
        _filesystem_path(manifest_path).read_text(encoding="utf-8")
    )
    if loaded != manifest:
        raise ValueError("Run manifest changed during serialization.")
    verify_run_directory(
        staging_directory, loaded, current_inputs=observed_inputs
    )
    _filesystem_path(staging_directory).replace(_filesystem_path(final_directory))
    print(f"[COMPLETE] Promoted verified run: {final_directory}")
    print(aggregate_table.to_string(index=False))
    return final_directory, manifest, aggregate_table, per_class_table


def main(argv=None):
    return run_experiment(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
