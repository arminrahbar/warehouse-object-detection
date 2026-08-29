"""Select a detector checkpoint from verified quality and runtime evidence.

This stage performs no model inference. It revalidates immutable evidence,
reconstructs image-level matching, estimates paired source-family uncertainty,
and applies the locked lexicographic decision rule.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "outputs"
    / "01_model_selection"
    / "03_checkpoint_decision"
)
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_SEED = 20260821
DEFAULT_EXPECTED_IMAGES = 9525
DEFAULT_EXPECTED_LABELS = 36721
SOURCE_GROUP_MARKER = "_jpg.rf."
SELECTION_SCHEMA_VERSION = 1
BOOTSTRAP_EVENT_CELL_BUDGET = 2_000_000
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
QUALITY_SOURCE_FILES = {
    "experiments/scripts/01_model_selection/01_compare_model_quality.py",
    "detector_service/modules/inference/model.py",
    "detector_service/modules/inference/nms.py",
    "detector_service/modules/utils/metrics.py",
}
LEGACY_QUALITY_SOURCE_FILES = {
    "experiments/scripts/01_model_selection/01_model_comparison.py",
    "detector_service/modules/inference/model.py",
    "detector_service/modules/inference/nms.py",
    "detector_service/modules/utils/metrics.py",
}
EARLY_LEGACY_QUALITY_SOURCE_FILES = {
    "experiments/scripts/01_model_comparison.py",
    "detector_service/modules/inference/model.py",
    "detector_service/modules/inference/nms.py",
    "detector_service/modules/utils/metrics.py",
}

LOCKED_MODEL_HASHES = {
    "model1": {
        "weights": "5f3d6e98255618c8d0d2a6275cd24c7f6fb7d50ff120498c6d94fb437a83f14e",
        "cfg": "75c54cb0b72cef2680282904e80d82eae2f6ed55590a327a7b3867a77898ad1c",
        "names": "83398377d1db963fcca23f2db159401928ffaf8f5e06bc7ff244ec00868d8d69",
    },
    "model2": {
        "weights": "b58fbc33e9fcbaed09972b2dc8767737b8c27da13c4c711d99174ec5ccd1b7c8",
        "cfg": "75c54cb0b72cef2680282904e80d82eae2f6ed55590a327a7b3867a77898ad1c",
        "names": "83398377d1db963fcca23f2db159401928ffaf8f5e06bc7ff244ec00868d8d69",
    },
}

GROUND_TRUTH_COLUMNS = [
    "image_file", "image_path", "class_id", "class_name",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
]
PREDICTION_COLUMNS = [
    "model", "image_index", "image_file", "image_path",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h", "class_id", "class_name",
    "object_score", "predicted_class_score", "combined_confidence",
    "nms_iou_threshold",
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
    "deployment_micro_precision", "deployment_micro_recall",
    "deployment_micro_f1", "deployment_macro_precision",
    "deployment_macro_recall", "deployment_macro_f1", "candidate_floor",
    "deployment_confidence", "nms_iou_threshold", "map_iou_threshold",
]
PER_CLASS_COLUMNS = [
    "model", "class_id", "class_name", "ground_truth_count",
    "low_floor_prediction_count", "ap50_101pt",
    "deployment_prediction_count", "deployment_true_positives",
    "deployment_false_positives", "deployment_false_negatives",
    "deployment_precision", "deployment_recall", "deployment_f1",
    "threshold_constrained_ap50_11pt",
]
QUALITY_SCHEMAS = {
    "ground_truth.csv": GROUND_TRUTH_COLUMNS,
    "model1_predictions.csv": PREDICTION_COLUMNS,
    "model2_predictions.csv": PREDICTION_COLUMNS,
    "inference_ledger.csv": LEDGER_COLUMNS,
    "aggregate_metrics.csv": AGGREGATE_COLUMNS,
    "per_class_metrics.csv": PER_CLASS_COLUMNS,
}

RUNTIME_SUMMARY_COLUMNS = [
    "model", "images_in_dataset", "sample_requested", "unique_images_selected",
    "repeats", "warmup_images", "successful_observations",
    "unreadable_observations", "total_detections", "pipeline_setup_seconds",
    "measured_wall_seconds", "mean_seconds_per_image",
    "median_seconds_per_image", "p95_seconds_per_image", "images_per_second",
    "estimated_full_dataset_minutes", "candidate_threshold",
    "confidence_threshold", "nms_iou_threshold", "python_version",
    "opencv_version", "platform", "benchmark_mode", "seed",
    "sample_selection", "mean_compute_seconds", "median_compute_seconds",
    "p95_compute_seconds",
]
RUNTIME_OBSERVATION_COLUMNS = [
    "model", "repeat_index", "sample_position", "image_file", "image_path",
    "status", "detections", "read_seconds", "predict_seconds",
    "postprocess_seconds", "nms_seconds", "total_seconds", "benchmark_mode",
    "density_bucket", "execution_order", "compute_seconds",
]
RUNTIME_COMPARISON_COLUMNS = [
    "record_type", "repeat_index", "sample_position", "image_file",
    "image_path", "source_group", "density_bucket", "first_model",
    "model1_compute_ms", "model2_compute_ms",
    "delta_model2_minus_model1_ms", "faster_model", "source_groups", "pairs",
    "bootstrap_samples", "seed", "model1_median_ms", "model2_median_ms",
    "relative_median_difference_pct", "model1_p95_ms", "model2_p95_ms",
    "p95_delta_model2_minus_model1_ms",
    "relative_p95_difference_pct",
    "p95_delta_model2_minus_model1_ci_lower_ms",
    "p95_delta_model2_minus_model1_ci_upper_ms",
    "relative_p95_difference_ci_lower_pct",
    "relative_p95_difference_ci_upper_pct",
    "mean_delta_model2_minus_model1_ms", "relative_mean_difference_pct",
    "mean_delta_ci_lower_ms", "mean_delta_ci_upper_ms",
    "relative_mean_difference_ci_lower_pct",
    "relative_mean_difference_ci_upper_pct",
]
RUNTIME_ARTIFACTS = {
    "summary": ("inference_benchmark_summary.csv", RUNTIME_SUMMARY_COLUMNS),
    "observations": (
        "inference_benchmark_observations.csv", RUNTIME_OBSERVATION_COLUMNS
    ),
    "paired_comparison": (
        "paired_latency_comparison.csv", RUNTIME_COMPARISON_COLUMNS
    ),
}

SUMMARY_COLUMNS = [
    "metric", "model1_value", "model2_value", "delta_model2_minus_model1",
    "ci_lower", "ci_upper", "practical_threshold", "relative_effect_pct",
    "selection_role",
]
BOOTSTRAP_COLUMNS = [
    "replicate", "sampled_source_group_draw_sha256",
    "delta_mAP50_101pt", "delta_deployment_macro_f1",
]

LOCKED_QUALITY_POLICY = {
    "candidate_objectness_comparator": ">",
    "candidate_floor": 0.001,
    "low_floor_combined_confidence_comparator": ">=",
    "deployment_confidence": 0.50,
    "deployment_confidence_comparator": ">=",
    "nms_iou_threshold": 0.30,
    "map_iou_threshold": 0.50,
    "primary_ap_interpolation_points": 101,
    "legacy_ap_interpolation_points": 11,
    "eval_type": "combined",
}


class IntegrityError(ValueError):
    """Raised when evidence fails a locked integrity requirement."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _absolute(path):
    return Path(path).expanduser().absolute()


def _sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require(condition, message):
    if not condition:
        raise IntegrityError(message)


def _require_sha256(value, context):
    text = str(value)
    _require(
        len(text) == 64 and all(character in "0123456789abcdef" for character in text),
        f"{context} is not a lowercase SHA-256 digest.",
    )
    return text


def source_group_key(image_file):
    """Return the locked source-family key, with a safe complete-name fallback."""

    value = str(image_file).strip()
    if not value:
        raise IntegrityError("image_file must not be empty.")
    prefix, marker, _ = value.partition(SOURCE_GROUP_MARKER)
    return prefix if marker and prefix else value


def _csv_row_count(path):
    with Path(path).open("r", encoding="utf-8", newline="") as source:
        return max(0, sum(1 for _ in source) - 1)


def _read_json(path, context):
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(f"Unable to read {context}: {error}") from error
    _require(isinstance(value, dict), f"{context} must contain a JSON object.")
    return value


def _read_csv_exact(path, columns, context):
    source = Path(path)
    _require(source.is_file(), f"Missing {context}: {source}")
    try:
        table = pd.read_csv(source)
    except Exception as error:
        raise IntegrityError(f"Unable to read {context}: {error}") from error
    _require(
        table.columns.tolist() == list(columns),
        f"{context} schema does not match the locked ordered columns.",
    )
    return table


def _verify_csv_identity(path, record, columns, context, size_key="size_bytes"):
    _require(isinstance(record, dict), f"{context} identity must be an object.")
    source = Path(path)
    _require(source.is_file(), f"Missing {context}: {source}")
    _require(
        int(record.get(size_key, -1)) == source.stat().st_size,
        f"{context} byte size does not match its manifest.",
    )
    expected_hash = _require_sha256(record.get("sha256", ""), f"{context} hash")
    _require(_sha256_file(source) == expected_hash, f"{context} hash mismatch.")
    _require(
        int(record.get("rows", -1)) == _csv_row_count(source),
        f"{context} row count does not match its manifest.",
    )
    if "columns" in record:
        _require(
            record["columns"] == list(columns),
            f"{context} manifest schema is incorrect.",
        )


def _numeric(table, columns, context, finite=True):
    for column in columns:
        try:
            table[column] = pd.to_numeric(table[column], errors="raise")
        except Exception as error:
            raise IntegrityError(f"{context}.{column} must be numeric.") from error
        if finite:
            _require(
                np.isfinite(table[column].to_numpy(dtype=float)).all(),
                f"{context}.{column} must be finite.",
            )


def _integer(table, columns, context, nonnegative=False):
    _numeric(table, columns, context)
    for column in columns:
        values = table[column].to_numpy(dtype=float)
        _require(
            np.equal(values, np.floor(values)).all(),
            f"{context}.{column} must contain integers.",
        )
        if nonnegative:
            _require((values >= 0).all(), f"{context}.{column} must be non-negative.")
        table[column] = table[column].astype(np.int64)


def _assert_close(actual, expected, context, atol=1e-10):
    _require(
        math.isfinite(float(actual)) and math.isfinite(float(expected)),
        f"{context} must be finite.",
    )
    _require(
        math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=atol),
        f"{context} mismatch: {actual} != {expected}.",
    )


def _validate_file_identity_shape(record, context, size_key="size_bytes"):
    _require(isinstance(record, dict), f"{context} must be an identity object.")
    _require(str(record.get("path", "")).strip(), f"{context} path is missing.")
    _require(int(record.get(size_key, -1)) >= 0, f"{context} byte size is invalid.")
    _require_sha256(record.get("sha256", ""), f"{context} hash")


def _validate_quality_manifest(directory, expected_images, expected_labels, locked_hashes):
    manifest_path = directory / "run_manifest.json"
    manifest = _read_json(manifest_path, "quality manifest")
    _require(manifest.get("schema_version") == 2, "Unsupported quality schema.")
    _require(manifest.get("status") == "complete", "Quality run is not complete.")
    _require(manifest.get("run_scope") == "full",
             "Checkpoint selection requires a strict full-corpus quality run.")

    expected_files = set(QUALITY_SCHEMAS) | {"run_manifest.json"}
    observed_files = {path.name for path in directory.iterdir() if path.is_file()}
    _require(observed_files == expected_files, "Quality run must contain exactly seven files.")

    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, dict), "Quality artifact manifest is missing.")
    _require(set(artifacts) == set(QUALITY_SCHEMAS), "Quality artifact set is invalid.")
    for name, columns in QUALITY_SCHEMAS.items():
        _verify_csv_identity(directory / name, artifacts[name], columns, name)

    dataset = manifest.get("dataset", {})
    _validate_file_identity_shape(dataset, "quality dataset index")
    _require(
        int(dataset.get("selected_images", -1)) == expected_images,
        f"Quality run must contain exactly {expected_images} selected images.",
    )
    _require(
        int(dataset.get("selected_labels", -1)) == expected_labels,
        f"Quality run must contain exactly {expected_labels} selected labels.",
    )
    _require(dataset.get("selection_max_images") is None,
             "Full quality evidence cannot contain an image limit.")
    asset_identity = dataset.get("asset_identity", {})
    _require(
        int(asset_identity.get("images", -1)) == expected_images
        and int(asset_identity.get("labels", -1)) == expected_images
        and int(asset_identity.get("files", -1)) == expected_images * 2
        and int(asset_identity.get("size_bytes", -1)) >= 0,
        "Quality dataset-asset identity has invalid counts.",
    )
    _require_sha256(asset_identity.get("sha256", ""), "quality dataset-asset hash")

    policy = manifest.get("policy")
    _require(policy == LOCKED_QUALITY_POLICY, "Quality policy is not the locked policy.")
    _require(
        manifest.get("policy_sha256") == _sha256_json(policy),
        "Quality policy digest mismatch.",
    )
    source_files = manifest.get("source_files")
    _require(isinstance(source_files, dict)
             and set(source_files) in (
                 QUALITY_SOURCE_FILES,
                 LEGACY_QUALITY_SOURCE_FILES,
                 EARLY_LEGACY_QUALITY_SOURCE_FILES,
             ),
             "Quality source identity set is invalid.")
    for name, identity in source_files.items():
        _validate_file_identity_shape(identity, f"quality source {name}")
    _require(
        manifest.get("source_policy_sha256")
        == _sha256_json({"policy": policy, "source_files": source_files}),
        "Quality source/policy digest mismatch.",
    )

    classes = manifest.get("class_vocabulary")
    _require(
        isinstance(classes, list) and len(classes) == 20,
        "Quality run must use an ordered 20-class vocabulary.",
    )
    _require(
        all(str(value).strip() for value in classes) and len(set(classes)) == 20,
        "Quality vocabulary must contain 20 unique non-empty names.",
    )
    models = manifest.get("models")
    _require(isinstance(models, dict) and set(models) == {"model1", "model2"},
             "Quality manifest must identify model1 and model2.")
    for model_name in ("model1", "model2"):
        _require(set(models[model_name]) == {"weights", "cfg", "names"},
                 f"{model_name} bundle is incomplete.")
        for asset_name, identity in models[model_name].items():
            _validate_file_identity_shape(identity, f"{model_name}.{asset_name}")
            if locked_hashes is not None:
                _require(
                    identity["sha256"] == locked_hashes[model_name][asset_name],
                    f"{model_name}.{asset_name} does not match the locked asset.",
                )
    _require(
        models["model1"]["cfg"]["sha256"] == models["model2"]["cfg"]["sha256"]
        and models["model1"]["names"]["sha256"]
        == models["model2"]["names"]["sha256"],
        "Checkpoint configuration or vocabulary identities differ.",
    )
    _require(
        models["model1"]["weights"]["sha256"]
        != models["model2"]["weights"]["sha256"],
        "Checkpoint weight identities must differ.",
    )
    input_identities = {
        "dataset_index": {
            key: dataset[key] for key in ("path", "size_bytes", "sha256")
        },
        "dataset_assets": asset_identity,
        "models": models,
        "source_files": source_files,
    }
    _require(
        manifest.get("input_identities_sha256") == _sha256_json(input_identities),
        "Quality input-identity digest mismatch.",
    )
    _require(isinstance(manifest.get("environment"), dict),
             "Quality environment metadata is missing.")
    _require(isinstance(manifest.get("command"), list),
             "Quality command metadata is missing.")
    return manifest


def _validate_quality_tables(directory, manifest, expected_images, expected_labels):
    tables = {
        name: _read_csv_exact(directory / name, columns, name)
        for name, columns in QUALITY_SCHEMAS.items()
    }
    classes = [str(value) for value in manifest["class_vocabulary"]]
    class_map = dict(enumerate(classes))

    ledger = tables["inference_ledger.csv"]
    _integer(
        ledger,
        ["image_index", "raw_candidate_count", "post_nms_prediction_count"],
        "inference_ledger",
        nonnegative=True,
    )
    _numeric(
        ledger,
        ["read_seconds", "predict_seconds", "postprocess_seconds", "nms_seconds",
         "total_seconds"],
        "inference_ledger",
    )
    _require((ledger[["read_seconds", "predict_seconds", "postprocess_seconds",
                      "nms_seconds", "total_seconds"]] >= 0).all().all(),
             "Inference timings must be non-negative.")
    _require(set(ledger["model"].astype(str)) == {"model1", "model2"},
             "Inference ledger model set is invalid.")
    ordered = None
    ordered_paths = None
    for model_name in ("model1", "model2"):
        rows = ledger.loc[ledger["model"].astype(str) == model_name].copy()
        rows = rows.sort_values("image_index", kind="stable")
        _require(len(rows) == expected_images, f"{model_name} ledger is incomplete.")
        _require(rows["image_index"].tolist() == list(range(1, expected_images + 1)),
                 f"{model_name} ledger positions are invalid.")
        _require(set(rows["status"].astype(str)) == {"processed"},
                 f"{model_name} ledger contains unsuccessful rows.")
        files = rows["image_file"].astype(str).tolist()
        paths = rows["image_path"].astype(str).tolist()
        _require(all(value.strip() for value in files + paths),
                 "Image identifiers and paths must be non-empty.")
        _require(len(set(files)) == expected_images and len(set(paths)) == expected_images,
                 "Quality image identifiers and paths must be unique.")
        if ordered is None:
            ordered, ordered_paths = files, paths
        else:
            _require(files == ordered and paths == ordered_paths,
                     "Checkpoint ledgers do not use identical ordered images.")
    _require(len(ledger) == expected_images * 2, "Inference ledger has extra rows.")
    image_to_index = {name: index for index, name in enumerate(ordered)}
    image_to_path = dict(zip(ordered, ordered_paths))

    ground_truth = tables["ground_truth.csv"]
    _require(len(ground_truth) == expected_labels, "Ground-truth row count is invalid.")
    _integer(ground_truth, ["class_id"], "ground_truth")
    _numeric(ground_truth, ["bbox_x", "bbox_y", "bbox_w", "bbox_h"], "ground_truth")
    _require(ground_truth["class_id"].between(0, len(classes) - 1).all(),
             "Ground truth contains invalid class IDs.")
    expected_names = ground_truth["class_id"].map(class_map).astype(str)
    _require(ground_truth["class_name"].astype(str).equals(expected_names),
             "Ground-truth class names do not match IDs.")
    _require(set(ground_truth["image_file"].astype(str)).issubset(image_to_index),
             "Ground truth contains unknown images.")
    _require(
        ground_truth.apply(
            lambda row: str(row["image_path"]) == image_to_path[str(row["image_file"])],
            axis=1,
        ).all(),
        "Ground-truth image paths do not match the ledger.",
    )
    _require((ground_truth[["bbox_w", "bbox_h"]] > 0).all().all(),
             "Ground-truth dimensions must be positive.")
    ground_truth["image_index"] = ground_truth["image_file"].map(image_to_index)

    predictions = {}
    for model_name in ("model1", "model2"):
        name = f"{model_name}_predictions.csv"
        table = tables[name]
        if not table.empty:
            _integer(table, ["image_index", "class_id"], name)
            _numeric(
                table,
                ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "object_score",
                 "predicted_class_score", "combined_confidence",
                 "nms_iou_threshold"],
                name,
            )
            _require(set(table["model"].astype(str)) == {model_name},
                     f"{name} contains an invalid model label.")
            _require(table["class_id"].between(0, len(classes) - 1).all(),
                     f"{name} contains invalid class IDs.")
            expected_names = table["class_id"].map(class_map).astype(str)
            _require(table["class_name"].astype(str).equals(expected_names),
                     f"{name} class names do not match IDs.")
            _require(set(table["image_file"].astype(str)).issubset(image_to_index),
                     f"{name} contains unknown images.")
            expected_indices = table["image_file"].map(image_to_index) + 1
            _require(table["image_index"].equals(expected_indices.astype(np.int64)),
                     f"{name} image indices do not match the ledger.")
            expected_paths = table["image_file"].map(image_to_path).astype(str)
            _require(table["image_path"].astype(str).equals(expected_paths),
                     f"{name} image paths do not match the ledger.")
            _require((table[["bbox_w", "bbox_h"]] >= 0).all().all(),
                     f"{name} contains negative box dimensions.")
            for column in ("object_score", "predicted_class_score",
                           "combined_confidence"):
                _require(table[column].between(0, 1).all(),
                         f"{name}.{column} is outside [0, 1].")
            _require(
                np.allclose(
                    table["combined_confidence"],
                    table["object_score"] * table["predicted_class_score"],
                    rtol=1e-12, atol=1e-15,
                ),
                f"{name} contains inconsistent combined confidence.",
            )
            _require(
                np.allclose(table["nms_iou_threshold"], 0.30, rtol=0, atol=1e-15),
                f"{name} uses the wrong NMS threshold.",
            )
            _require((table["combined_confidence"] >= 0.001 - 1e-15).all(),
                     f"{name} falls below the low-floor policy.")
            table["image_index_zero"] = table["image_index"] - 1
        else:
            table["image_index_zero"] = pd.Series(dtype=np.int64)
        ledger_count = int(ledger.loc[ledger["model"] == model_name,
                                      "post_nms_prediction_count"].sum())
        _require(ledger_count == len(table), f"{model_name} prediction count disagrees with ledger.")
        predictions[model_name] = table

    aggregate = tables["aggregate_metrics.csv"]
    per_class = tables["per_class_metrics.csv"]
    _require(aggregate["model"].astype(str).tolist() == ["model1", "model2"],
             "Aggregate metrics must contain model1 then model2.")
    _require(len(per_class) == len(classes) * 2, "Per-class metric row count is invalid.")
    for model_name in ("model1", "model2"):
        rows = per_class.loc[per_class["model"].astype(str) == model_name]
        _require(rows["class_id"].astype(int).tolist() == list(range(len(classes))),
                 f"{model_name} per-class ordering is invalid.")
        _require(rows["class_name"].astype(str).tolist() == classes,
                 f"{model_name} per-class names are invalid.")

    return {
        "tables": tables,
        "classes": classes,
        "ordered_images": ordered,
        "ordered_paths": ordered_paths,
        "image_to_index": image_to_index,
        "ground_truth": ground_truth,
        "predictions": predictions,
        "ledger": ledger,
        "aggregate": aggregate,
        "per_class": per_class,
    }


def validate_quality_run(
    quality_run, expected_images=DEFAULT_EXPECTED_IMAGES,
    expected_labels=DEFAULT_EXPECTED_LABELS, locked_hashes=LOCKED_MODEL_HASHES,
):
    directory = _absolute(quality_run)
    _require(directory.is_dir(), f"Quality run directory not found: {directory}")
    manifest = _validate_quality_manifest(
        directory, expected_images, expected_labels, locked_hashes
    )
    evidence = _validate_quality_tables(
        directory, manifest, expected_images, expected_labels
    )
    evidence.update({"directory": directory, "manifest": manifest})
    return evidence


def _linear_percentile(values, percentile):
    ordered = sorted(float(value) for value in values)
    _require(ordered, "Cannot calculate a percentile from empty evidence.")
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _runtime_bootstrap(pair_rows, samples, seed):
    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[str(row["source_group"])].append(row)
    groups = sorted(grouped)
    _require(groups, "Runtime comparison contains no pairs.")
    import random
    rng = random.Random(seed)
    mean_deltas, p95_deltas = [], []
    for _ in range(samples):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        rows = [row for group in sampled for row in grouped[group]]
        model1 = [float(row["model1_compute_ms"]) for row in rows]
        model2 = [float(row["model2_compute_ms"]) for row in rows]
        mean_deltas.append(float(np.mean(model2)) - float(np.mean(model1)))
        p95_deltas.append(
            _linear_percentile(model2, 95) - _linear_percentile(model1, 95)
        )
    return {
        "mean_lower": _linear_percentile(mean_deltas, 2.5),
        "mean_upper": _linear_percentile(mean_deltas, 97.5),
        "p95_lower": _linear_percentile(p95_deltas, 2.5),
        "p95_upper": _linear_percentile(p95_deltas, 97.5),
    }


def _verify_runtime_manifest(runtime_dir, quality, bootstrap_samples, seed):
    manifest_path = runtime_dir / "inference_benchmark_manifest.json"
    manifest = _read_json(manifest_path, "runtime manifest")
    _require(manifest.get("schema_version") == 1, "Unsupported runtime schema.")
    _require(manifest.get("status") == "complete", "Runtime run is not complete.")
    _require(manifest.get("benchmark_mode") == "paired", "Runtime run is not paired.")

    expected_files = {
        "inference_benchmark_summary.csv", "inference_benchmark_observations.csv",
        "paired_latency_comparison.csv", "inference_benchmark_manifest.json",
    }
    observed = {path.name for path in runtime_dir.iterdir() if path.is_file()}
    _require(observed == expected_files, "Runtime run must contain exactly four files.")

    core = {
        key: value for key, value in manifest.items()
        if key not in {"run_fingerprint_sha256", "created_at_utc"}
    }
    _require(
        manifest.get("run_fingerprint_sha256") == _sha256_json(core),
        "Runtime manifest fingerprint mismatch.",
    )

    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, dict) and set(artifacts) == set(RUNTIME_ARTIFACTS),
             "Runtime artifact set is invalid.")
    tables = {}
    for key, (filename, columns) in RUNTIME_ARTIFACTS.items():
        record = artifacts[key]
        _require(Path(str(record.get("path", ""))).name == filename,
                 f"Runtime {key} manifest path is invalid.")
        _verify_csv_identity(runtime_dir / filename, record, columns, filename,
                             size_key="bytes")
        tables[key] = _read_csv_exact(runtime_dir / filename, columns, filename)

    dataset = manifest.get("dataset", {})
    _validate_file_identity_shape(dataset.get("index", {}), "runtime dataset index",
                                  size_key="bytes")
    _require(
        dataset["index"]["sha256"] == quality["manifest"]["dataset"]["sha256"],
        "Quality and runtime dataset-index identities differ.",
    )
    _require(int(dataset.get("images_in_index", -1)) == len(quality["ordered_images"]),
             "Runtime dataset size differs from quality evidence.")
    sample_images = int(dataset.get("ordered_sample_images", -1))
    _require(0 < sample_images <= len(quality["ordered_images"]),
             "Runtime sample size is invalid.")
    _require_sha256(dataset.get("ordered_sample_sha256", ""),
                    "runtime ordered-sample digest")
    _require(dataset.get("source_group_policy")
             == "prefix before '_jpg.rf.'; otherwise complete image_file",
             "Runtime source-group policy differs from the locked rule.")

    runtime_models = manifest.get("models")
    quality_models = quality["manifest"]["models"]
    _require(isinstance(runtime_models, dict) and set(runtime_models) == {"model1", "model2"},
             "Runtime model identities are incomplete.")
    for model_name in ("model1", "model2"):
        _require(set(runtime_models[model_name]) == {"weights", "cfg", "names"},
                 f"Runtime {model_name} bundle is incomplete.")
        for asset_name in ("weights", "cfg", "names"):
            identity = runtime_models[model_name][asset_name]
            _validate_file_identity_shape(identity, f"runtime {model_name}.{asset_name}",
                                          size_key="bytes")
            _require(identity["sha256"] == quality_models[model_name][asset_name]["sha256"],
                     f"Runtime and quality {model_name}.{asset_name} identities differ.")
            _require(int(identity["bytes"])
                     == int(quality_models[model_name][asset_name]["size_bytes"]),
                     f"Runtime and quality {model_name}.{asset_name} sizes differ.")

    policy = manifest.get("policy", {})
    _require(float(policy.get("candidate_objectness_threshold", -1)) == 0.50,
             "Runtime candidate threshold is invalid.")
    _require(float(policy.get("combined_confidence_threshold", -1)) == 0.50,
             "Runtime confidence threshold is invalid.")
    _require(float(policy.get("nms_iou_threshold", -1)) == 0.30,
             "Runtime NMS threshold is invalid.")
    _require(policy.get("sample_selection") == "seeded_density_stratified",
             "Runtime sample-selection policy is invalid.")
    _require(int(policy.get("sample_seed", -1)) == seed,
             "Runtime sample seed differs from the selection protocol.")
    _require(policy.get("timing_scope") == "predict + post_process + class-aware NMS",
             "Runtime timing scope is invalid.")
    _require(policy.get("frame_decode_timed_separately") is True,
             "Runtime decode timing must be separate.")
    _require(policy.get("execution_order") == "seeded alternating first checkpoint",
             "Runtime order policy is invalid.")
    _require(policy.get("p95_estimator") == "linear interpolation",
             "Runtime p95 estimator is invalid.")
    _require(policy.get("bootstrap_unit") == "source group preserving variants and repeats",
             "Runtime bootstrap unit is invalid.")
    _require(int(policy.get("bootstrap_samples", -1)) == bootstrap_samples,
             "Runtime bootstrap count differs from the selection protocol.")
    _require(int(policy.get("bootstrap_seed", -1)) == seed,
             "Runtime bootstrap seed differs from the selection protocol.")

    completeness = manifest.get("completeness", {})
    repeats = int(policy.get("repeats", -1))
    _require(repeats > 0 and int(policy.get("warmup_images_per_model", -1)) >= 0,
             "Runtime repeat or warm-up policy is invalid.")
    expected_pairs = sample_images * repeats
    expected_observations = expected_pairs * 2
    expected_completeness = {
        "expected_unique_images": sample_images,
        "measured_unique_images": sample_images,
        "expected_pairs": expected_pairs,
        "measured_pairs": expected_pairs,
        "expected_observations": expected_observations,
        "processed_observations": expected_observations,
        "unreadable_observations": 0,
    }
    _require(completeness == expected_completeness, "Runtime completeness gate failed.")
    return manifest, tables


def _validate_runtime_tables(manifest, tables, quality, bootstrap_samples, seed):
    summary = tables["summary"]
    observations = tables["observations"]
    comparison = tables["paired_comparison"]
    _require(summary["model"].astype(str).tolist() == ["model1", "model2"],
             "Runtime summary must contain model1 then model2.")
    summary_numeric = [
        "images_in_dataset", "sample_requested", "unique_images_selected", "repeats",
        "warmup_images", "successful_observations", "unreadable_observations",
        "total_detections", "pipeline_setup_seconds", "measured_wall_seconds",
        "mean_seconds_per_image", "median_seconds_per_image", "p95_seconds_per_image",
        "images_per_second", "estimated_full_dataset_minutes", "candidate_threshold",
        "confidence_threshold", "nms_iou_threshold", "seed", "mean_compute_seconds",
        "median_compute_seconds", "p95_compute_seconds",
    ]
    _numeric(summary, summary_numeric, "runtime_summary")
    _require(set(summary["benchmark_mode"].astype(str)) == {"paired"},
             "Runtime summary mode is invalid.")
    _require((summary["unreadable_observations"] == 0).all(),
             "Runtime summary contains unreadable observations.")
    _require(np.allclose(summary["candidate_threshold"], 0.5),
             "Runtime summary candidate threshold is invalid.")
    _require(np.allclose(summary["confidence_threshold"], 0.5),
             "Runtime summary confidence threshold is invalid.")
    _require(np.allclose(summary["nms_iou_threshold"], 0.3),
             "Runtime summary NMS threshold is invalid.")

    _integer(observations, ["repeat_index", "sample_position", "detections",
                            "execution_order"], "runtime_observations", nonnegative=True)
    _numeric(observations, ["read_seconds", "predict_seconds", "postprocess_seconds",
                            "nms_seconds", "total_seconds", "compute_seconds"],
             "runtime_observations")
    _require(set(observations["model"].astype(str)) == {"model1", "model2"},
             "Runtime observations contain invalid model labels.")
    _require(set(observations["status"].astype(str)) == {"processed"},
             "Runtime observations contain unsuccessful rows.")
    _require(set(observations["benchmark_mode"].astype(str)) == {"paired"},
             "Runtime observations are not paired.")
    _require((observations[["read_seconds", "predict_seconds", "postprocess_seconds",
                            "nms_seconds", "total_seconds", "compute_seconds"]] >= 0).all().all(),
             "Runtime observations contain negative timings.")
    _require(np.allclose(observations["compute_seconds"], observations["total_seconds"],
                         rtol=1e-10, atol=1e-12),
             "Runtime compute and total timings disagree.")
    _require(np.allclose(
        observations["compute_seconds"],
        observations["predict_seconds"] + observations["postprocess_seconds"]
        + observations["nms_seconds"], rtol=1e-10, atol=1e-12),
        "Runtime stage timings do not sum to compute time.",
    )

    pair_rows = comparison.loc[comparison["record_type"].astype(str) == "pair"].copy()
    aggregate_rows = comparison.loc[
        comparison["record_type"].astype(str) == "aggregate"
    ].copy()
    _require(len(aggregate_rows) == 1 and len(pair_rows) + 1 == len(comparison),
             "Runtime comparison must contain pairs followed by one aggregate row.")
    _require(comparison.iloc[-1]["record_type"] == "aggregate",
             "Runtime aggregate row must be final.")
    for column in ("repeat_index", "sample_position", "model1_compute_ms",
                   "model2_compute_ms", "delta_model2_minus_model1_ms"):
        pair_rows[column] = pd.to_numeric(pair_rows[column], errors="raise")
    quality_images = set(quality["ordered_images"])
    _require(set(pair_rows["image_file"].astype(str)).issubset(quality_images),
             "Runtime pairs contain images absent from quality evidence.")
    _require(pair_rows.apply(
        lambda row: str(row["source_group"]) == source_group_key(row["image_file"]), axis=1
    ).all(), "Runtime source-group values are invalid.")
    _require(
        len(set(pair_rows["source_group"].astype(str)))
        == int(manifest["dataset"]["source_groups"]),
        "Runtime source-group count differs from its manifest.",
    )
    pair_key_columns = ["repeat_index", "sample_position"]
    _require(not pair_rows.duplicated(pair_key_columns).any(),
             "Runtime comparison contains duplicate repeat/sample positions.")
    expected_repeats = int(manifest["policy"]["repeats"])
    expected_sample_size = int(manifest["dataset"]["ordered_sample_images"])
    sample_identity = None
    for repeat_index in range(1, expected_repeats + 1):
        repeat_rows = pair_rows.loc[pair_rows["repeat_index"] == repeat_index].sort_values(
            "sample_position"
        )
        _require(
            repeat_rows["sample_position"].astype(int).tolist()
            == list(range(1, expected_sample_size + 1)),
            "Runtime comparison has incomplete sample positions.",
        )
        current_identity = list(zip(
            repeat_rows["image_file"].astype(str),
            repeat_rows["image_path"].astype(str),
        ))
        if sample_identity is None:
            sample_identity = current_identity
            _require(
                len({name for name, _ in sample_identity}) == expected_sample_size
                and len({path for _, path in sample_identity}) == expected_sample_size,
                "Runtime sample image identifiers and paths are not unique.",
            )
        else:
            _require(current_identity == sample_identity,
                     "Runtime repeats do not use the same ordered sample.")
    _require(np.allclose(
        pair_rows["delta_model2_minus_model1_ms"],
        pair_rows["model2_compute_ms"] - pair_rows["model1_compute_ms"],
        rtol=1e-10, atol=1e-10), "Runtime pair deltas are inconsistent.")
    expected_faster = np.where(
        pair_rows["delta_model2_minus_model1_ms"] > 0, "model1",
        np.where(pair_rows["delta_model2_minus_model1_ms"] < 0, "model2", "tie"),
    )
    _require(pair_rows["faster_model"].astype(str).tolist() == expected_faster.tolist(),
             "Runtime faster-model labels are inconsistent.")

    observation_keys = observations.groupby(
        ["repeat_index", "sample_position", "image_file", "image_path"], sort=False
    )
    _require(len(observation_keys) == len(pair_rows),
             "Runtime observation pairing is incomplete.")
    pair_lookup = {
        (int(row.repeat_index), int(row.sample_position), str(row.image_file), str(row.image_path)): row
        for row in pair_rows.itertuples(index=False)
    }
    for key, rows in observation_keys:
        _require(key in pair_lookup and len(rows) == 2,
                 "Runtime observation does not map to exactly one pair.")
        _require(set(rows["model"].astype(str)) == {"model1", "model2"}
                 and set(rows["execution_order"].astype(int)) == {1, 2},
                 "Runtime pair model/order evidence is invalid.")
        pair = pair_lookup[key]
        first = rows.sort_values("execution_order").iloc[0]["model"]
        _require(str(pair.first_model) == str(first), "Runtime first-model evidence differs.")
        for model_name in ("model1", "model2"):
            seconds = float(rows.loc[rows["model"] == model_name, "compute_seconds"].iloc[0])
            _assert_close(seconds * 1000.0, getattr(pair, f"{model_name}_compute_ms"),
                          f"runtime {model_name} pair timing", atol=1e-8)

    aggregate = aggregate_rows.iloc[0]
    for column in RUNTIME_COMPARISON_COLUMNS[12:]:
        if column in aggregate and str(aggregate[column]).strip() not in {"", "nan"}:
            try:
                float(aggregate[column])
            except ValueError as error:
                raise IntegrityError(f"Runtime aggregate {column} must be numeric.") from error
    _require(int(float(aggregate["pairs"])) == len(pair_rows),
             "Runtime aggregate pair count is invalid.")
    _require(int(float(aggregate["source_groups"]))
             == len(set(pair_rows["source_group"].astype(str))),
             "Runtime aggregate source-group count is invalid.")
    _require(int(float(aggregate["bootstrap_samples"])) == bootstrap_samples
             and int(float(aggregate["seed"])) == seed,
             "Runtime aggregate bootstrap parameters are invalid.")
    model1_values = pair_rows["model1_compute_ms"].to_numpy(dtype=float)
    model2_values = pair_rows["model2_compute_ms"].to_numpy(dtype=float)
    model1_p95 = _linear_percentile(model1_values, 95)
    model2_p95 = _linear_percentile(model2_values, 95)
    model1_mean = float(np.mean(model1_values))
    model2_mean = float(np.mean(model2_values))
    _assert_close(aggregate["model1_p95_ms"], model1_p95, "runtime model1 p95")
    _assert_close(aggregate["model2_p95_ms"], model2_p95, "runtime model2 p95")
    _assert_close(aggregate["p95_delta_model2_minus_model1_ms"],
                  model2_p95 - model1_p95, "runtime p95 delta")
    _assert_close(aggregate["mean_delta_model2_minus_model1_ms"],
                  model2_mean - model1_mean, "runtime mean delta")
    for model_name, expected_mean, expected_p95 in (
        ("model1", model1_mean, model1_p95),
        ("model2", model2_mean, model2_p95),
    ):
        row = summary.loc[summary["model"] == model_name].iloc[0]
        _require(
            int(row["images_in_dataset"]) == len(quality["ordered_images"])
            and int(row["sample_requested"]) == expected_sample_size
            and int(row["unique_images_selected"]) == expected_sample_size
            and int(row["repeats"]) == expected_repeats
            and int(row["successful_observations"])
            == expected_sample_size * expected_repeats,
            f"Runtime {model_name} summary completeness is invalid.",
        )
        _assert_close(row["mean_compute_seconds"], expected_mean / 1000.0,
                      f"runtime {model_name} summary mean", atol=1e-12)
        _assert_close(row["p95_compute_seconds"], expected_p95 / 1000.0,
                      f"runtime {model_name} summary p95", atol=1e-12)
    runtime_ci = _runtime_bootstrap(pair_rows.to_dict("records"), bootstrap_samples, seed)
    for field, expected in (
        ("mean_delta_ci_lower_ms", runtime_ci["mean_lower"]),
        ("mean_delta_ci_upper_ms", runtime_ci["mean_upper"]),
        ("p95_delta_model2_minus_model1_ci_lower_ms", runtime_ci["p95_lower"]),
        ("p95_delta_model2_minus_model1_ci_upper_ms", runtime_ci["p95_upper"]),
    ):
        _assert_close(aggregate[field], expected, f"runtime {field}", atol=1e-8)
    _require(runtime_ci["mean_lower"] <= runtime_ci["mean_upper"]
             and runtime_ci["p95_lower"] <= runtime_ci["p95_upper"],
             "Runtime confidence interval bounds are reversed.")
    return {
        "manifest": manifest,
        "tables": tables,
        "pair_rows": pair_rows,
        "aggregate": aggregate,
        "model1_p95_ms": model1_p95,
        "model2_p95_ms": model2_p95,
        "model1_mean_ms": model1_mean,
        "model2_mean_ms": model2_mean,
        "p95_ci_lower": runtime_ci["p95_lower"],
        "p95_ci_upper": runtime_ci["p95_upper"],
        "mean_ci_lower": runtime_ci["mean_lower"],
        "mean_ci_upper": runtime_ci["mean_upper"],
    }


def validate_runtime_run(runtime_run, quality, bootstrap_samples, seed):
    directory = _absolute(runtime_run)
    _require(directory.is_dir(), f"Runtime run directory not found: {directory}")
    manifest, tables = _verify_runtime_manifest(
        directory, quality, bootstrap_samples, seed
    )
    evidence = _validate_runtime_tables(
        manifest, tables, quality, bootstrap_samples, seed
    )
    evidence["directory"] = directory
    return evidence


def _iou_vector(box, labels):
    if labels.size == 0:
        return np.empty(0, dtype=float)
    box = np.asarray(box, dtype=float)
    starts = np.maximum(box[:2], labels[:, :2])
    ends = np.minimum(box[:2] + box[2:], labels[:, :2] + labels[:, 2:])
    overlap = np.maximum(0.0, ends - starts)
    intersection = overlap[:, 0] * overlap[:, 1]
    box_area = max(0.0, box[2]) * max(0.0, box[3])
    label_area = np.maximum(0.0, labels[:, 2]) * np.maximum(0.0, labels[:, 3])
    union = box_area + label_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _match_image_class(predictions, labels, iou_threshold=0.50):
    """Match score-ranked predictions exactly once to same-class labels."""

    if predictions.empty:
        return np.empty(0, dtype=bool), np.empty(0, dtype=np.int64)
    scores = predictions["combined_confidence"].to_numpy(dtype=float)
    original = predictions["_row_order"].to_numpy(dtype=np.int64)
    order = np.lexsort((original, -scores))
    boxes = predictions[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(float)
    label_boxes = labels[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(float)
    consumed = np.zeros(len(label_boxes), dtype=bool)
    matched = np.zeros(len(order), dtype=bool)
    for ranked_position, prediction_index in enumerate(order):
        available = np.flatnonzero(~consumed)
        if available.size == 0:
            continue
        overlaps = _iou_vector(boxes[prediction_index], label_boxes[available])
        best_position = int(np.argmax(overlaps))
        best_label = int(available[best_position])
        if overlaps[best_position] >= iou_threshold:
            consumed[best_label] = True
            matched[ranked_position] = True
    return matched, order


def _safe_divide(numerator, denominator):
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator, dtype=float),
        where=np.asarray(denominator) != 0,
    )


def _f1_array(precision, recall):
    return np.divide(
        2.0 * precision * recall, precision + recall,
        out=np.zeros_like(precision, dtype=float), where=(precision + recall) != 0,
    )


def build_image_class_evidence(quality, model_name):
    """Build per-image/class counts and globally score-ranked match events."""

    images = quality["ordered_images"]
    classes = quality["classes"]
    image_count, class_count = len(images), len(classes)
    groups_for_image = [source_group_key(name) for name in images]
    source_groups = sorted(set(groups_for_image))
    group_index = {name: index for index, name in enumerate(source_groups)}
    image_group = np.asarray([group_index[name] for name in groups_for_image], dtype=np.int64)

    ground_truth = quality["ground_truth"]
    predictions = quality["predictions"][model_name].copy()
    predictions["_row_order"] = np.arange(len(predictions), dtype=np.int64)
    gt_counts = np.zeros((image_count, class_count), dtype=np.int64)
    if not ground_truth.empty:
        gt_grouped_counts = ground_truth.groupby(["image_index", "class_id"]).size()
        for (image_index, class_id), count in gt_grouped_counts.items():
            gt_counts[int(image_index), int(class_id)] = int(count)
    gt_lookup = {
        (int(image_index), int(class_id)): rows
        for (image_index, class_id), rows in ground_truth.groupby(
            ["image_index", "class_id"], sort=False
        )
    }

    low_tp = np.zeros_like(gt_counts)
    low_fp = np.zeros_like(gt_counts)
    deployment_tp = np.zeros_like(gt_counts)
    deployment_fp = np.zeros_like(gt_counts)
    event_scores = [[] for _ in classes]
    event_matches = [[] for _ in classes]
    event_groups = [[] for _ in classes]
    empty_labels = ground_truth.iloc[0:0]
    if not predictions.empty:
        grouped_predictions = predictions.sort_values(
            ["image_index_zero", "class_id", "_row_order"], kind="stable"
        ).groupby(["image_index_zero", "class_id"], sort=False)
        for (image_index, class_id), rows in grouped_predictions:
            image_index, class_id = int(image_index), int(class_id)
            labels = gt_lookup.get((image_index, class_id), empty_labels)
            matched, order = _match_image_class(rows, labels)
            ranked = rows.iloc[order]
            low_tp[image_index, class_id] = int(matched.sum())
            low_fp[image_index, class_id] = int(len(matched) - matched.sum())
            deployment_mask = (
                ranked["combined_confidence"].to_numpy(dtype=float) >= 0.50
            )
            deployment_matches = matched[deployment_mask]
            deployment_tp[image_index, class_id] = int(deployment_matches.sum())
            deployment_fp[image_index, class_id] = int(
                len(deployment_matches) - deployment_matches.sum()
            )
            event_scores[class_id].extend(
                ranked["combined_confidence"].to_numpy(dtype=float).tolist()
            )
            event_matches[class_id].extend(matched.astype(np.int8).tolist())
            event_groups[class_id].extend(
                [int(image_group[image_index])] * len(ranked)
            )

    low_fn = gt_counts - low_tp
    deployment_fn = gt_counts - deployment_tp
    _require((low_fn >= 0).all() and (deployment_fn >= 0).all(),
             f"{model_name} matching produced more true positives than labels.")

    events = []
    for class_id in range(class_count):
        scores = np.asarray(event_scores[class_id], dtype=float)
        matches = np.asarray(event_matches[class_id], dtype=np.int8)
        groups = np.asarray(event_groups[class_id], dtype=np.int64)
        order = np.argsort(-scores, kind="stable")
        events.append({
            "scores": scores[order], "matches": matches[order], "groups": groups[order]
        })

    group_count = len(source_groups)
    group_gt = np.zeros((group_count, class_count), dtype=np.int64)
    group_low_tp = np.zeros_like(group_gt)
    group_low_fp = np.zeros_like(group_gt)
    group_deployment_tp = np.zeros_like(group_gt)
    group_deployment_fp = np.zeros_like(group_gt)
    for image_index, group_id in enumerate(image_group):
        group_gt[group_id] += gt_counts[image_index]
        group_low_tp[group_id] += low_tp[image_index]
        group_low_fp[group_id] += low_fp[image_index]
        group_deployment_tp[group_id] += deployment_tp[image_index]
        group_deployment_fp[group_id] += deployment_fp[image_index]

    evidence_digest = _sha256_json({
        "model": model_name,
        "source_groups": source_groups,
        "gt": gt_counts.tolist(),
        "low_tp": low_tp.tolist(), "low_fp": low_fp.tolist(),
        "low_fn": low_fn.tolist(),
        "deployment_tp": deployment_tp.tolist(),
        "deployment_fp": deployment_fp.tolist(),
        "deployment_fn": deployment_fn.tolist(),
    })
    return {
        "model": model_name,
        "source_groups": source_groups,
        "image_group": image_group,
        "gt_counts": gt_counts,
        "low_tp": low_tp, "low_fp": low_fp, "low_fn": low_fn,
        "deployment_tp": deployment_tp, "deployment_fp": deployment_fp,
        "deployment_fn": deployment_fn, "events": events,
        "group_gt": group_gt, "group_low_tp": group_low_tp,
        "group_low_fp": group_low_fp,
        "group_deployment_tp": group_deployment_tp,
        "group_deployment_fp": group_deployment_fp,
        "evidence_sha256": evidence_digest,
    }


def _weighted_ap(event, group_weights, ground_truth_support):
    if ground_truth_support <= 0 or len(event["scores"]) == 0:
        return 0.0
    weights = group_weights[event["groups"]].astype(np.int64, copy=False)
    cumulative_predictions = np.cumsum(weights, dtype=np.int64)
    cumulative_tp = np.cumsum(weights * event["matches"], dtype=np.int64)
    precision = _safe_divide(cumulative_tp, cumulative_predictions)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = cumulative_tp / float(ground_truth_support)
    levels = np.linspace(0.0, 1.0, 101)
    positions = np.searchsorted(recall, levels, side="left")
    values = np.zeros(101, dtype=float)
    valid = positions < len(precision)
    values[valid] = precision[positions[valid]]
    return float(values.mean())


def quality_metrics_for_weights(evidence, group_weights):
    weights = np.asarray(group_weights, dtype=np.int64)
    _require(len(weights) == len(evidence["source_groups"]),
             "Source-group weight vector has the wrong length.")
    gt = weights @ evidence["group_gt"]
    ap = [
        _weighted_ap(evidence["events"][class_id], weights, int(gt[class_id]))
        for class_id in range(evidence["group_gt"].shape[1])
    ]
    tp = weights @ evidence["group_deployment_tp"]
    fp = weights @ evidence["group_deployment_fp"]
    fn = gt - tp
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _f1_array(precision, recall)
    return {
        "mAP50_101pt": float(np.mean(ap)),
        "deployment_macro_f1": float(np.mean(f1)),
        "per_class_ap": np.asarray(ap, dtype=float),
        "per_class_deployment_f1": f1,
        "deployment_tp": tp, "deployment_fp": fp, "deployment_fn": fn,
    }


def _weighted_ap_batch(event, group_weights, ground_truth_support):
    """Compute 101-point AP for several source-group draws at once."""

    batch_size = len(group_weights)
    result = np.zeros(batch_size, dtype=float)
    if len(event["scores"]) == 0:
        return result
    event_weights = group_weights[:, event["groups"]].astype(np.int64, copy=False)
    cumulative_predictions = np.cumsum(event_weights, axis=1, dtype=np.int64)
    cumulative_tp = np.cumsum(
        event_weights * event["matches"][None, :], axis=1, dtype=np.int64
    )
    precision = _safe_divide(cumulative_tp, cumulative_predictions)
    precision[:, ::-1] = np.maximum.accumulate(precision[:, ::-1], axis=1)
    levels = np.linspace(0.0, 1.0, 101)
    for row_index, support in enumerate(ground_truth_support):
        if support <= 0:
            continue
        recall = cumulative_tp[row_index] / float(support)
        positions = np.searchsorted(recall, levels, side="left")
        values = np.zeros(101, dtype=float)
        valid = positions < precision.shape[1]
        values[valid] = precision[row_index, positions[valid]]
        result[row_index] = float(values.mean())
    return result


def _quality_metrics_batch(evidence, group_weights):
    """Return mAP and deployment macro-F1 for a matrix of bootstrap weights."""

    weights = np.asarray(group_weights, dtype=np.int64)
    gt = weights @ evidence["group_gt"]
    map_values = np.zeros(len(weights), dtype=float)
    class_count = evidence["group_gt"].shape[1]
    for class_id in range(class_count):
        map_values += _weighted_ap_batch(
            evidence["events"][class_id], weights, gt[:, class_id]
        )
    map_values /= float(class_count)
    tp = weights @ evidence["group_deployment_tp"]
    fp = weights @ evidence["group_deployment_fp"]
    fn = gt - tp
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    macro_f1 = _f1_array(precision, recall).mean(axis=1)
    return map_values, macro_f1


def _validate_recomputed_metrics(quality, model_evidence, metrics):
    for model_name in ("model1", "model2"):
        evidence = model_evidence[model_name]
        current = metrics[model_name]
        aggregate = quality["aggregate"].loc[
            quality["aggregate"]["model"] == model_name
        ].iloc[0]
        _assert_close(current["mAP50_101pt"], aggregate["mAP50_101pt"],
                      f"{model_name} recomputed mAP50_101pt")
        _assert_close(current["deployment_macro_f1"], aggregate["deployment_macro_f1"],
                      f"{model_name} recomputed deployment macro F1")
        per_class = quality["per_class"].loc[
            quality["per_class"]["model"] == model_name
        ].sort_values("class_id")
        for class_id, row in enumerate(per_class.itertuples(index=False)):
            _assert_close(current["per_class_ap"][class_id], row.ap50_101pt,
                          f"{model_name} class {class_id} AP")
            _assert_close(current["per_class_deployment_f1"][class_id], row.deployment_f1,
                          f"{model_name} class {class_id} deployment F1")
            _require(int(evidence["gt_counts"][:, class_id].sum())
                     == int(row.ground_truth_count),
                     f"{model_name} class {class_id} ground-truth support mismatch.")
            _require(int(evidence["deployment_tp"][:, class_id].sum())
                     == int(row.deployment_true_positives)
                     and int(evidence["deployment_fp"][:, class_id].sum())
                     == int(row.deployment_false_positives)
                     and int(evidence["deployment_fn"][:, class_id].sum())
                     == int(row.deployment_false_negatives),
                     f"{model_name} class {class_id} deployment counts mismatch.")


def _percentile_interval(values):
    return (
        _linear_percentile(values, 2.5),
        _linear_percentile(values, 97.5),
    )


def paired_quality_bootstrap(model1, model2, samples=DEFAULT_BOOTSTRAP_SAMPLES,
                             seed=DEFAULT_SEED):
    _require(samples > 0, "Bootstrap samples must be positive.")
    _require(model1["source_groups"] == model2["source_groups"],
             "Model source-group mappings differ.")
    group_count = len(model1["source_groups"])
    _require(group_count > 0, "At least one source group is required.")
    maximum_class_events = max(
        [len(event["scores"]) for event in model1["events"] + model2["events"]]
        or [1]
    )
    batch_size = min(
        64,
        max(1, BOOTSTRAP_EVENT_CELL_BUDGET // max(1, maximum_class_events)),
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    rows, delta_ap, delta_f1 = [], [], []
    for first_replicate in range(1, samples + 1, batch_size):
        current_batch = min(batch_size, samples - first_replicate + 1)
        draws = rng.integers(
            0, group_count, size=(current_batch, group_count), dtype=np.int32
        )
        weights = np.zeros((current_batch, group_count), dtype=np.int64)
        row_indices = np.repeat(np.arange(current_batch), group_count)
        np.add.at(weights, (row_indices, draws.reshape(-1)), 1)
        first_ap, first_f1 = _quality_metrics_batch(model1, weights)
        second_ap, second_f1 = _quality_metrics_batch(model2, weights)
        batch_delta_ap = second_ap - first_ap
        batch_delta_f1 = second_f1 - first_f1
        delta_ap.extend(batch_delta_ap.tolist())
        delta_f1.extend(batch_delta_f1.tolist())
        for offset in range(current_batch):
            rows.append({
                "replicate": first_replicate + offset,
                "sampled_source_group_draw_sha256": hashlib.sha256(
                    draws[offset].tobytes()
                ).hexdigest(),
                "delta_mAP50_101pt": batch_delta_ap[offset],
                "delta_deployment_macro_f1": batch_delta_f1[offset],
            })
    ap_lower, ap_upper = _percentile_interval(delta_ap)
    f1_lower, f1_upper = _percentile_interval(delta_f1)
    return rows, {
        "delta_ap_ci_lower": ap_lower, "delta_ap_ci_upper": ap_upper,
        "delta_f1_ci_lower": f1_lower, "delta_f1_ci_upper": f1_upper,
    }


def interval_excludes_zero(lower, upper):
    return float(lower) > 0.0 or float(upper) < 0.0


def apply_selection_rule(quality_metrics, quality_intervals, runtime):
    """Apply the locked four-step rule with model2-minus-model1 signs."""

    delta_ap = (
        quality_metrics["model2"]["mAP50_101pt"]
        - quality_metrics["model1"]["mAP50_101pt"]
    )
    ap_lower = quality_intervals["delta_ap_ci_lower"]
    ap_upper = quality_intervals["delta_ap_ci_upper"]
    if delta_ap >= 0.01 and ap_lower > 0.0:
        return {"status": "selected", "step": 1, "selected_model": "model2",
                "selected_checkpoint": "B", "reason": "qualifying_low_floor_ap50"}
    if delta_ap <= -0.01 and ap_upper < 0.0:
        return {"status": "selected", "step": 1, "selected_model": "model1",
                "selected_checkpoint": "A", "reason": "qualifying_low_floor_ap50"}

    delta_f1 = (
        quality_metrics["model2"]["deployment_macro_f1"]
        - quality_metrics["model1"]["deployment_macro_f1"]
    )
    f1_lower = quality_intervals["delta_f1_ci_lower"]
    f1_upper = quality_intervals["delta_f1_ci_upper"]
    if delta_f1 >= 0.01 and f1_lower > 0.0:
        return {"status": "selected", "step": 2, "selected_model": "model2",
                "selected_checkpoint": "B", "reason": "qualifying_deployment_macro_f1"}
    if delta_f1 <= -0.01 and f1_upper < 0.0:
        return {"status": "selected", "step": 2, "selected_model": "model1",
                "selected_checkpoint": "A", "reason": "qualifying_deployment_macro_f1"}

    model1_p95, model2_p95 = runtime["model1_p95_ms"], runtime["model2_p95_ms"]
    p95_delta = model2_p95 - model1_p95
    slower = max(model1_p95, model2_p95)
    relative = 100.0 * abs(p95_delta) / slower if slower > 0 else 0.0
    if (model2_p95 < model1_p95 and relative >= 5.0
            and runtime["p95_ci_upper"] < 0.0):
        return {"status": "selected", "step": 3, "selected_model": "model2",
                "selected_checkpoint": "B", "reason": "qualifying_p95_compute_latency"}
    if (model1_p95 < model2_p95 and relative >= 5.0
            and runtime["p95_ci_lower"] > 0.0):
        return {"status": "selected", "step": 3, "selected_model": "model1",
                "selected_checkpoint": "A", "reason": "qualifying_p95_compute_latency"}

    selected = "model1" if runtime["model1_mean_ms"] <= runtime["model2_mean_ms"] else "model2"
    return {
        "status": "operationally_equivalent_under_protocol", "step": 4,
        "selected_model": selected,
        "selected_checkpoint": "A" if selected == "model1" else "B",
        "reason": "deterministic_lower_mean_latency_tiebreak",
        "quality_superiority_claim_permitted": False,
    }


def _selection_summary_rows(metrics, intervals, runtime):
    p95_delta = runtime["model2_p95_ms"] - runtime["model1_p95_ms"]
    slower = max(runtime["model1_p95_ms"], runtime["model2_p95_ms"])
    p95_relative = 100.0 * p95_delta / slower if slower > 0 else 0.0
    mean_delta = runtime["model2_mean_ms"] - runtime["model1_mean_ms"]
    mean_baseline = runtime["model1_mean_ms"]
    mean_relative = 100.0 * mean_delta / mean_baseline if mean_baseline > 0 else 0.0
    return [
        {
            "metric": "mAP50_101pt", "model1_value": metrics["model1"]["mAP50_101pt"],
            "model2_value": metrics["model2"]["mAP50_101pt"],
            "delta_model2_minus_model1": metrics["model2"]["mAP50_101pt"]
            - metrics["model1"]["mAP50_101pt"],
            "ci_lower": intervals["delta_ap_ci_lower"],
            "ci_upper": intervals["delta_ap_ci_upper"],
            "practical_threshold": 0.01, "relative_effect_pct": "",
            "selection_role": "step_1_primary_quality",
        },
        {
            "metric": "deployment_macro_f1",
            "model1_value": metrics["model1"]["deployment_macro_f1"],
            "model2_value": metrics["model2"]["deployment_macro_f1"],
            "delta_model2_minus_model1": metrics["model2"]["deployment_macro_f1"]
            - metrics["model1"]["deployment_macro_f1"],
            "ci_lower": intervals["delta_f1_ci_lower"],
            "ci_upper": intervals["delta_f1_ci_upper"],
            "practical_threshold": 0.01, "relative_effect_pct": "",
            "selection_role": "step_2_secondary_quality",
        },
        {
            "metric": "p95_compute_ms", "model1_value": runtime["model1_p95_ms"],
            "model2_value": runtime["model2_p95_ms"],
            "delta_model2_minus_model1": p95_delta,
            "ci_lower": runtime["p95_ci_lower"], "ci_upper": runtime["p95_ci_upper"],
            "practical_threshold": "5%_of_slower_p95",
            "relative_effect_pct": p95_relative,
            "selection_role": "step_3_subordinate_runtime",
        },
        {
            "metric": "mean_compute_ms", "model1_value": runtime["model1_mean_ms"],
            "model2_value": runtime["model2_mean_ms"],
            "delta_model2_minus_model1": mean_delta,
            "ci_lower": runtime["mean_ci_lower"], "ci_upper": runtime["mean_ci_upper"],
            "practical_threshold": "none_deterministic_tiebreak",
            "relative_effect_pct": mean_relative,
            "selection_role": "step_4_tiebreak_only",
        },
    ]


def _write_csv_atomic(path, columns, rows):
    destination = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _write_json_atomic(path, value):
    destination = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(value, temporary, indent=2, sort_keys=True, ensure_ascii=False)
            temporary.write("\n")
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _artifact_identity(path, columns):
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    _require(header == list(columns), f"Output schema mismatch: {source.name}")
    return {
        "sha256": _sha256_file(source), "size_bytes": source.stat().st_size,
        "rows": _csv_row_count(source), "columns": list(columns),
    }


def _verify_output_directory(directory, manifest):
    expected = {
        "selection_summary.csv": SUMMARY_COLUMNS,
        "bootstrap_replicates.csv": BOOTSTRAP_COLUMNS,
        "decision.json": None,
        "selection_manifest.json": None,
    }
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    _require(observed == set(expected), "Selection output set is incomplete.")
    for name, identity in manifest["artifacts"].items():
        path = directory / name
        _require(_sha256_file(path) == identity["sha256"], f"Output hash mismatch: {name}")
        _require(path.stat().st_size == identity["size_bytes"],
                 f"Output size mismatch: {name}")
        _require(_csv_row_count(path) == identity["rows"],
                 f"Output row mismatch: {name}")
    decision_identity = manifest["decision"]
    decision_path = directory / "decision.json"
    _require(_sha256_file(decision_path) == decision_identity["sha256"],
             "Decision hash mismatch.")
    _require(decision_path.stat().st_size == decision_identity["size_bytes"],
             "Decision size mismatch.")


def run_selection(
    quality_run, runtime_run, output_root, run_id,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES, seed=DEFAULT_SEED,
    expected_images=DEFAULT_EXPECTED_IMAGES, expected_labels=DEFAULT_EXPECTED_LABELS,
    locked_hashes=LOCKED_MODEL_HASHES,
):
    """Validate evidence, bootstrap quality, decide, and atomically promote output."""

    _require(bootstrap_samples > 0, "Bootstrap samples must be positive.")
    _require(bool(RUN_ID_PATTERN.fullmatch(str(run_id))),
             "run_id contains unsupported characters.")
    started = _utc_now()
    quality = validate_quality_run(
        quality_run, expected_images, expected_labels, locked_hashes=locked_hashes
    )
    runtime = validate_runtime_run(runtime_run, quality, bootstrap_samples, seed)
    model_evidence = {
        model_name: build_image_class_evidence(quality, model_name)
        for model_name in ("model1", "model2")
    }
    _require(
        model_evidence["model1"]["source_groups"]
        == model_evidence["model2"]["source_groups"],
        "Quality source-group mappings differ by model.",
    )
    unit_weights = np.ones(len(model_evidence["model1"]["source_groups"]), dtype=np.int64)
    point_metrics = {
        model_name: quality_metrics_for_weights(model_evidence[model_name], unit_weights)
        for model_name in ("model1", "model2")
    }
    _validate_recomputed_metrics(quality, model_evidence, point_metrics)
    bootstrap_rows, intervals = paired_quality_bootstrap(
        model_evidence["model1"], model_evidence["model2"],
        samples=bootstrap_samples, seed=seed,
    )
    decision = apply_selection_rule(point_metrics, intervals, runtime)
    decision.update({
        "integrity_status": "passed", "decision_rule": "locked_lexicographic_v1",
        "delta_sign": "model2_minus_model1", "quality_intervals": intervals,
        "quality_point_estimates": {
            model_name: {
                "mAP50_101pt": point_metrics[model_name]["mAP50_101pt"],
                "deployment_macro_f1": point_metrics[model_name]["deployment_macro_f1"],
            } for model_name in ("model1", "model2")
        },
        "runtime_point_estimates_ms": {
            "model1_p95": runtime["model1_p95_ms"],
            "model2_p95": runtime["model2_p95_ms"],
            "model1_mean": runtime["model1_mean_ms"],
            "model2_mean": runtime["model2_mean_ms"],
        },
        "runtime_delta_intervals_ms": {
            "p95": [runtime["p95_ci_lower"], runtime["p95_ci_upper"]],
            "mean": [runtime["mean_ci_lower"], runtime["mean_ci_upper"]],
        },
        "training_overlap": "unknown",
        "claim_scope": "relative performance on the available supplied corpus",
    })

    output_root = _absolute(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    final_directory = output_root / str(run_id)
    staging_directory = output_root / f".{run_id}.incomplete"
    if final_directory.exists():
        raise FileExistsError(f"Refusing to overwrite completed selection: {final_directory}")
    if staging_directory.exists():
        raise FileExistsError(f"Incomplete selection already exists: {staging_directory}")
    staging_directory.mkdir()
    try:
        summary_rows = _selection_summary_rows(point_metrics, intervals, runtime)
        summary_path = staging_directory / "selection_summary.csv"
        bootstrap_path = staging_directory / "bootstrap_replicates.csv"
        decision_path = staging_directory / "decision.json"
        _write_csv_atomic(summary_path, SUMMARY_COLUMNS, summary_rows)
        _write_csv_atomic(bootstrap_path, BOOTSTRAP_COLUMNS, bootstrap_rows)
        _write_json_atomic(decision_path, decision)
        artifacts = {
            "selection_summary.csv": _artifact_identity(summary_path, SUMMARY_COLUMNS),
            "bootstrap_replicates.csv": _artifact_identity(
                bootstrap_path, BOOTSTRAP_COLUMNS
            ),
        }
        source_mapping = [
            {"image_file": image, "source_group": source_group_key(image)}
            for image in quality["ordered_images"]
        ]
        selection_policy = {
            "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
            "bootstrap_generator": "NumPy PCG64",
            "bootstrap_unit": "source_group",
            "bootstrap_representation": (
                "source-group multiplicity weights on canonical score order"
            ),
            "source_group_rule": "prefix before '_jpg.rf.'; otherwise complete image_file",
            "map_iou_threshold": 0.50, "primary_ap_points": 101,
            "deployment_confidence": 0.50,
            "delta_sign": "model2_minus_model1",
            "lexicographic_rule": {
                "step_1": "abs(delta_mAP50_101pt)>=0.01 and paired CI excludes zero",
                "step_2": "abs(delta_deployment_macro_f1)>=0.01 and paired CI excludes zero",
                "step_3": "lower p95 by >=5% of slower p95 and paired CI excludes zero",
                "step_4": "operational equivalence; lower mean latency, then model1",
            },
        }
        manifest = {
            "schema_version": SELECTION_SCHEMA_VERSION, "status": "complete",
            "run_id": str(run_id), "started_utc": started, "completed_utc": _utc_now(),
            "quality_input": {
                "directory": str(quality["directory"]),
                "manifest_sha256": _sha256_file(quality["directory"] / "run_manifest.json"),
                "run_id": quality["manifest"].get("run_id"),
                "source_policy_sha256": quality["manifest"]["source_policy_sha256"],
            },
            "runtime_input": {
                "directory": str(runtime["directory"]),
                "manifest_sha256": _sha256_file(
                    runtime["directory"] / "inference_benchmark_manifest.json"
                ),
                "run_fingerprint_sha256": runtime["manifest"]["run_fingerprint_sha256"],
            },
            "corpus": {
                "images": expected_images, "labels": expected_labels,
                "classes": quality["classes"],
                "source_groups": len(model_evidence["model1"]["source_groups"]),
                "ordered_image_sha256": _sha256_json(quality["ordered_images"]),
                "image_source_group_sha256": _sha256_json(source_mapping),
            },
            "model_identities": {
                model_name: {
                    asset_name: quality["manifest"]["models"][model_name][asset_name]["sha256"]
                    for asset_name in ("weights", "cfg", "names")
                } for model_name in ("model1", "model2")
            },
            "image_class_evidence_sha256": {
                name: model_evidence[name]["evidence_sha256"]
                for name in ("model1", "model2")
            },
            "selection_policy": selection_policy,
            "selection_policy_sha256": _sha256_json(selection_policy),
            "environment": {
                "python_version": platform.python_version(),
                "numpy_version": np.__version__, "pandas_version": pd.__version__,
                "platform": platform.platform(),
            },
            "command": [str(value) for value in sys.argv],
            "artifacts": artifacts,
            "decision": {
                "sha256": _sha256_file(decision_path),
                "size_bytes": decision_path.stat().st_size,
            },
        }
        manifest_path = staging_directory / "selection_manifest.json"
        _write_json_atomic(manifest_path, manifest)
        _verify_output_directory(staging_directory, manifest)
        staging_directory.replace(final_directory)
    except Exception:
        raise
    print(f"[COMPLETE] Promoted verified selection: {final_directory}")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return final_directory, decision, manifest


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        description="Select a checkpoint from verified quality and paired-runtime evidence."
    )
    parser.add_argument("--quality-run", type=Path, required=True)
    parser.add_argument("--runtime-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bootstrap-samples", type=positive_int,
                        default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-images", type=positive_int,
                        default=DEFAULT_EXPECTED_IMAGES)
    parser.add_argument("--expected-labels", type=positive_int,
                        default=DEFAULT_EXPECTED_LABELS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_selection(
        quality_run=args.quality_run, runtime_run=args.runtime_run,
        output_root=args.output_root, run_id=args.run_id,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
        expected_images=args.expected_images, expected_labels=args.expected_labels,
    )


if __name__ == "__main__":
    main()
