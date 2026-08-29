"""Validate portable paths and verified upstream decisions for later experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_STORAGE_PREFIXES = {
    ("detector_service", "storage"),
}
MODEL_ASSET_PATHS = {
    "model1": {
        "weights": "yolo_model_1/yolov4-tiny-logistics_size_416_1.weights",
        "config": "yolo_model_1/yolov4-tiny-logistics_size_416_1.cfg",
        "classes": "yolo_model_1/logistics.names",
    },
    "model2": {
        "weights": "yolo_model_2/yolov4-tiny-logistics_size_416_2.weights",
        "config": "yolo_model_2/yolov4-tiny-logistics_size_416_2.cfg",
        "classes": "yolo_model_2/logistics.names",
    },
}
CHECKPOINT_BY_MODEL = {"model1": "A", "model2": "B"}
SELECTION_ARTIFACT_SCHEMAS = {
    "selection_summary.csv": [
        "metric",
        "model1_value",
        "model2_value",
        "delta_model2_minus_model1",
        "ci_lower",
        "ci_upper",
        "practical_threshold",
        "relative_effect_pct",
        "selection_role",
    ],
    "bootstrap_replicates.csv": [
        "replicate",
        "sampled_source_group_draw_sha256",
        "delta_mAP50_101pt",
        "delta_deployment_macro_f1",
    ],
}
SELECTION_PACKAGE_FILES = {
    *SELECTION_ARTIFACT_SCHEMAS,
    "decision.json",
    "selection_manifest.json",
}
SELECTION_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SELECTION_POLICY_RULE = {
    "step_1": "abs(delta_mAP50_101pt)>=0.01 and paired CI excludes zero",
    "step_2": "abs(delta_deployment_macro_f1)>=0.01 and paired CI excludes zero",
    "step_3": "lower p95 by >=5% of slower p95 and paired CI excludes zero",
    "step_4": "operational equivalence; lower mean latency, then model1",
}
OPERATING_POINT_EVIDENCE_SCHEMAS = {
    "nms_threshold_summary_sample5000.csv": [
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
    ],
    "duplicate_summary_by_threshold_sample5000.csv": [
        "duplicate_like_pairs_iou_gt_0_5",
        "images_with_duplicate_like_pairs",
        "mean_duplicate_like_pairs_per_image",
        "nms_threshold",
        "total_predictions_after_nms",
    ],
}
OPERATING_POINT_SELECTION_RULE = (
    "maximize threshold-constrained 11-point AP50; then minimize retained "
    "predictions; then choose the lower IoU threshold"
)
OPERATING_POINT_DATASET = "rare_aware_density_stratified_5000"


class EvidenceContractError(ValueError):
    """Raised when an upstream evidence or path contract is not satisfied."""


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise EvidenceContractError(f"{label} is not a lowercase SHA-256 digest.")
    return digest


def _require_exact_keys(value, expected, label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(expected):
        observed = set(value) if isinstance(value, dict) else set()
        raise EvidenceContractError(
            f"{label} fields do not match the contract: "
            f"missing={sorted(set(expected) - observed)}, "
            f"extra={sorted(observed - set(expected))}"
        )


def _finite_float(value, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise EvidenceContractError(f"{label} must be numeric.") from error
    if not math.isfinite(result):
        raise EvidenceContractError(f"{label} must be finite.")
    return result


def _nonnegative_integer(value, label: str) -> int:
    number = _finite_float(value, label)
    if number < 0 or not number.is_integer():
        raise EvidenceContractError(f"{label} must be a non-negative integer.")
    return int(number)


def _positive_integer(value, label: str) -> int:
    number = _nonnegative_integer(value, label)
    if number == 0:
        raise EvidenceContractError(f"{label} must be positive.")
    return number


def _require_close(observed, expected, label: str, tolerance: float = 1e-12) -> None:
    first = _finite_float(observed, label)
    second = _finite_float(expected, label)
    if not math.isclose(first, second, rel_tol=0.0, abs_tol=tolerance):
        raise EvidenceContractError(
            f"{label} mismatch: observed={first!r}, expected={second!r}."
        )


def _read_exact_csv(path: Path, columns: list[str], label: str) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != columns:
                raise EvidenceContractError(
                    f"{label} columns do not match the contract: "
                    f"observed={reader.fieldnames!r}, expected={columns!r}."
                )
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise EvidenceContractError(f"Unable to read {label}: {path}") from error


def _package_file(directory: Path, name: str, label: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
        raise EvidenceContractError(f"{label} must be a package-local basename: {name!r}")
    return _contained(directory / name, directory, label)


def _logical_parts(value) -> tuple[str, ...]:
    raw = str(value).strip()
    if not raw:
        raise EvidenceContractError("Indexed asset path cannot be empty.")
    parts = PurePosixPath(raw.replace("\\", "/")).parts
    if ".." in parts:
        raise EvidenceContractError(
            f"Indexed asset path cannot contain parent traversal: {value}"
        )
    return parts


def _contained(candidate: Path, root: Path, label: str) -> Path:
    resolved_candidate = candidate.expanduser().resolve(strict=False)
    resolved_root = root.expanduser().resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise EvidenceContractError(
            f"{label} resolves outside the permitted root {resolved_root}: "
            f"{resolved_candidate}"
        ) from error
    return resolved_candidate


def resolve_indexed_path(
    value,
    asset_root: Path | str | None = None,
    project_root: Path | str = PROJECT_ROOT,
) -> Path:
    """Resolve one indexed asset while enforcing root containment.

    Absolute paths are accepted only when they remain under ``asset_root`` (if
    supplied) or ``project_root``. Portable storage-prefixed paths are mapped to
    the external asset root. Other relative paths remain project-root relative.
    """

    direct = Path(str(value)).expanduser()
    project = Path(project_root)
    external = Path(asset_root) if asset_root is not None else None
    if direct.is_absolute():
        return _contained(direct, external or project, "Indexed asset path")

    parts = _logical_parts(value)
    if external is not None and tuple(parts[:2]) in SUPPORTED_STORAGE_PREFIXES:
        return _contained(external.joinpath(*parts[2:]), external, "Indexed asset path")
    return _contained(project.joinpath(*parts), project, "Indexed asset path")


def _validate_selection_summary(rows: list[dict], decision: dict) -> None:
    expected_roles = {
        "mAP50_101pt": "step_1_primary_quality",
        "deployment_macro_f1": "step_2_secondary_quality",
        "p95_compute_ms": "step_3_subordinate_runtime",
        "mean_compute_ms": "step_4_tiebreak_only",
    }
    if len(rows) != len(expected_roles):
        raise EvidenceContractError(
            "Checkpoint-selection summary must contain exactly four metric rows."
        )
    by_metric = {str(row["metric"]): row for row in rows}
    if set(by_metric) != set(expected_roles) or len(by_metric) != len(rows):
        raise EvidenceContractError(
            "Checkpoint-selection summary metric identities are incomplete or duplicated."
        )

    quality = decision.get("quality_point_estimates")
    intervals = decision.get("quality_intervals")
    runtime = decision.get("runtime_point_estimates_ms")
    runtime_intervals = decision.get("runtime_delta_intervals_ms")
    _require_exact_keys(quality, {"model1", "model2"}, "Quality point estimates")
    for model in ("model1", "model2"):
        _require_exact_keys(
            quality[model],
            {"mAP50_101pt", "deployment_macro_f1"},
            f"{model} quality point estimates",
        )
    _require_exact_keys(
        intervals,
        {
            "delta_ap_ci_lower",
            "delta_ap_ci_upper",
            "delta_f1_ci_lower",
            "delta_f1_ci_upper",
        },
        "Quality intervals",
    )
    _require_exact_keys(
        runtime,
        {"model1_p95", "model2_p95", "model1_mean", "model2_mean"},
        "Runtime point estimates",
    )
    _require_exact_keys(runtime_intervals, {"p95", "mean"}, "Runtime intervals")
    for name in ("p95", "mean"):
        values = runtime_intervals[name]
        if not isinstance(values, list) or len(values) != 2:
            raise EvidenceContractError(
                f"Runtime {name} interval must contain lower and upper bounds."
            )
        lower = _finite_float(values[0], f"Runtime {name} lower interval")
        upper = _finite_float(values[1], f"Runtime {name} upper interval")
        if lower > upper:
            raise EvidenceContractError(
                f"Runtime {name} interval lower bound exceeds its upper bound."
            )

    for model in ("model1", "model2"):
        for metric in ("mAP50_101pt", "deployment_macro_f1"):
            value = _finite_float(quality[model][metric], f"{model} {metric}")
            if not 0.0 <= value <= 1.0:
                raise EvidenceContractError(f"{model} {metric} must be within [0, 1].")
    for lower_name, upper_name in (
        ("delta_ap_ci_lower", "delta_ap_ci_upper"),
        ("delta_f1_ci_lower", "delta_f1_ci_upper"),
    ):
        lower = _finite_float(intervals[lower_name], lower_name)
        upper = _finite_float(intervals[upper_name], upper_name)
        if not -1.0 <= lower <= upper <= 1.0:
            raise EvidenceContractError(
                f"Quality interval {lower_name}/{upper_name} is invalid."
            )
    for name, value in runtime.items():
        if _finite_float(value, f"Runtime {name}") < 0.0:
            raise EvidenceContractError(f"Runtime {name} cannot be negative.")

    expected = {
        "mAP50_101pt": (
            quality["model1"]["mAP50_101pt"],
            quality["model2"]["mAP50_101pt"],
            intervals["delta_ap_ci_lower"],
            intervals["delta_ap_ci_upper"],
        ),
        "deployment_macro_f1": (
            quality["model1"]["deployment_macro_f1"],
            quality["model2"]["deployment_macro_f1"],
            intervals["delta_f1_ci_lower"],
            intervals["delta_f1_ci_upper"],
        ),
        "p95_compute_ms": (
            runtime["model1_p95"],
            runtime["model2_p95"],
            runtime_intervals["p95"][0],
            runtime_intervals["p95"][1],
        ),
        "mean_compute_ms": (
            runtime["model1_mean"],
            runtime["model2_mean"],
            runtime_intervals["mean"][0],
            runtime_intervals["mean"][1],
        ),
    }
    for metric, (model1, model2, lower, upper) in expected.items():
        row = by_metric[metric]
        if row["selection_role"] != expected_roles[metric]:
            raise EvidenceContractError(
                f"Checkpoint-selection role mismatch for {metric}."
            )
        _require_close(row["model1_value"], model1, f"{metric} model1 value")
        _require_close(row["model2_value"], model2, f"{metric} model2 value")
        _require_close(
            row["delta_model2_minus_model1"],
            _finite_float(model2, f"{metric} model2 value")
            - _finite_float(model1, f"{metric} model1 value"),
            f"{metric} delta",
        )
        _require_close(row["ci_lower"], lower, f"{metric} lower interval")
        _require_close(row["ci_upper"], upper, f"{metric} upper interval")


def _recompute_checkpoint_decision(decision: dict) -> dict:
    quality = decision["quality_point_estimates"]
    intervals = decision["quality_intervals"]
    runtime = decision["runtime_point_estimates_ms"]
    runtime_intervals = decision["runtime_delta_intervals_ms"]
    delta_ap = _finite_float(
        quality["model2"]["mAP50_101pt"], "model2 AP50"
    ) - _finite_float(quality["model1"]["mAP50_101pt"], "model1 AP50")
    ap_lower = _finite_float(intervals["delta_ap_ci_lower"], "AP lower interval")
    ap_upper = _finite_float(intervals["delta_ap_ci_upper"], "AP upper interval")
    if delta_ap >= 0.01 and ap_lower > 0.0:
        return {
            "status": "selected",
            "step": 1,
            "selected_model": "model2",
            "selected_checkpoint": "B",
            "reason": "qualifying_low_floor_ap50",
        }
    if delta_ap <= -0.01 and ap_upper < 0.0:
        return {
            "status": "selected",
            "step": 1,
            "selected_model": "model1",
            "selected_checkpoint": "A",
            "reason": "qualifying_low_floor_ap50",
        }

    delta_f1 = _finite_float(
        quality["model2"]["deployment_macro_f1"], "model2 deployment F1"
    ) - _finite_float(
        quality["model1"]["deployment_macro_f1"], "model1 deployment F1"
    )
    f1_lower = _finite_float(intervals["delta_f1_ci_lower"], "F1 lower interval")
    f1_upper = _finite_float(intervals["delta_f1_ci_upper"], "F1 upper interval")
    if delta_f1 >= 0.01 and f1_lower > 0.0:
        return {
            "status": "selected",
            "step": 2,
            "selected_model": "model2",
            "selected_checkpoint": "B",
            "reason": "qualifying_deployment_macro_f1",
        }
    if delta_f1 <= -0.01 and f1_upper < 0.0:
        return {
            "status": "selected",
            "step": 2,
            "selected_model": "model1",
            "selected_checkpoint": "A",
            "reason": "qualifying_deployment_macro_f1",
        }

    model1_p95 = _finite_float(runtime["model1_p95"], "model1 p95 latency")
    model2_p95 = _finite_float(runtime["model2_p95"], "model2 p95 latency")
    p95_delta = model2_p95 - model1_p95
    slower = max(model1_p95, model2_p95)
    relative = 100.0 * abs(p95_delta) / slower if slower > 0 else 0.0
    p95_lower = _finite_float(runtime_intervals["p95"][0], "p95 lower interval")
    p95_upper = _finite_float(runtime_intervals["p95"][1], "p95 upper interval")
    if model2_p95 < model1_p95 and relative >= 5.0 and p95_upper < 0.0:
        return {
            "status": "selected",
            "step": 3,
            "selected_model": "model2",
            "selected_checkpoint": "B",
            "reason": "qualifying_p95_compute_latency",
        }
    if model1_p95 < model2_p95 and relative >= 5.0 and p95_lower > 0.0:
        return {
            "status": "selected",
            "step": 3,
            "selected_model": "model1",
            "selected_checkpoint": "A",
            "reason": "qualifying_p95_compute_latency",
        }

    model1_mean = _finite_float(runtime["model1_mean"], "model1 mean latency")
    model2_mean = _finite_float(runtime["model2_mean"], "model2 mean latency")
    model = "model1" if model1_mean <= model2_mean else "model2"
    return {
        "status": "operationally_equivalent_under_protocol",
        "step": 4,
        "selected_model": model,
        "selected_checkpoint": CHECKPOINT_BY_MODEL[model],
        "reason": "deterministic_lower_mean_latency_tiebreak",
    }


def load_verified_checkpoint_selection(selection_run: Path | str) -> dict:
    """Validate a checkpoint-selection package and return its selected identity."""

    directory = Path(selection_run).expanduser().resolve(strict=False)
    if not directory.is_dir():
        raise NotADirectoryError(f"Checkpoint-selection package not found: {directory}")
    observed_entries = {entry.name for entry in directory.iterdir()}
    if observed_entries != SELECTION_PACKAGE_FILES:
        raise EvidenceContractError(
            "Checkpoint-selection package file set does not match the contract: "
            f"missing={sorted(SELECTION_PACKAGE_FILES - observed_entries)}, "
            f"extra={sorted(observed_entries - SELECTION_PACKAGE_FILES)}"
        )
    manifest_path = _package_file(
        directory, "selection_manifest.json", "Selection manifest"
    )
    decision_path = _package_file(directory, "decision.json", "Selection decision")
    if not manifest_path.is_file() or not decision_path.is_file():
        raise FileNotFoundError(
            "Checkpoint-selection package must contain regular decision and manifest files."
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceContractError(
            f"Checkpoint-selection package is not valid JSON: {directory}"
        ) from error
    if not isinstance(manifest, dict) or not isinstance(decision, dict):
        raise EvidenceContractError("Checkpoint-selection JSON roots must be objects.")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise EvidenceContractError("Checkpoint-selection manifest is not complete schema 1.")
    run_id = manifest.get("run_id")
    if (
        not isinstance(run_id, str)
        or not SELECTION_RUN_ID_PATTERN.fullmatch(run_id)
        or run_id != directory.name
    ):
        raise EvidenceContractError(
            "Checkpoint-selection run_id must be valid and match its package directory."
        )

    decision_record = manifest.get("decision")
    _require_exact_keys(
        decision_record, {"sha256", "size_bytes"}, "Selection decision identity"
    )
    expected_decision_hash = _require_sha256(
        decision_record["sha256"], "Selection decision hash"
    )
    if _nonnegative_integer(
        decision_record["size_bytes"], "Selection decision size"
    ) != decision_path.stat().st_size:
        raise EvidenceContractError("Checkpoint-selection decision size mismatch.")
    if expected_decision_hash != sha256_file(decision_path):
        raise EvidenceContractError("Checkpoint-selection decision hash mismatch.")

    artifacts = manifest.get("artifacts")
    _require_exact_keys(
        artifacts, SELECTION_ARTIFACT_SCHEMAS, "Selection artifact identities"
    )
    artifact_rows = {}
    for name, columns in SELECTION_ARTIFACT_SCHEMAS.items():
        identity = artifacts[name]
        _require_exact_keys(
            identity,
            {"columns", "rows", "sha256", "size_bytes"},
            f"Selection artifact identity for {name}",
        )
        if identity["columns"] != columns:
            raise EvidenceContractError(f"Selection artifact schema mismatch: {name}")
        artifact_path = _package_file(directory, name, f"Selection artifact {name}")
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Selection artifact is missing: {artifact_path}")
        rows = _read_exact_csv(artifact_path, columns, f"Selection artifact {name}")
        row_count = _nonnegative_integer(identity["rows"], f"{name} row count")
        if len(rows) != row_count:
            raise EvidenceContractError(f"Selection artifact row-count mismatch: {name}")
        size = _nonnegative_integer(identity["size_bytes"], f"{name} size")
        if artifact_path.stat().st_size != size:
            raise EvidenceContractError(f"Selection artifact size mismatch: {name}")
        digest = _require_sha256(identity["sha256"], f"{name} hash")
        if sha256_file(artifact_path) != digest:
            raise EvidenceContractError(f"Selection artifact hash mismatch: {name}")
        artifact_rows[name] = rows

    policy = manifest.get("selection_policy")
    if not isinstance(policy, dict):
        raise EvidenceContractError("Checkpoint-selection policy must be an object.")
    policy_digest = _require_sha256(
        manifest.get("selection_policy_sha256"), "Selection policy hash"
    )
    if _sha256_json(policy) != policy_digest:
        raise EvidenceContractError("Checkpoint-selection policy hash mismatch.")
    if (
        policy.get("lexicographic_rule") != SELECTION_POLICY_RULE
        or policy.get("delta_sign") != "model2_minus_model1"
        or policy.get("primary_ap_points") != 101
    ):
        raise EvidenceContractError("Checkpoint-selection policy is not the locked policy.")
    _require_close(policy.get("map_iou_threshold"), 0.5, "Selection AP IoU policy")
    _require_close(
        policy.get("deployment_confidence"), 0.5, "Selection deployment threshold"
    )
    bootstrap_samples = _positive_integer(
        policy.get("bootstrap_samples"), "Selection bootstrap sample count"
    )
    if bootstrap_samples != len(artifact_rows["bootstrap_replicates.csv"]):
        raise EvidenceContractError(
            "Selection bootstrap policy and artifact row counts disagree."
        )

    model_identities = manifest.get("model_identities")
    _require_exact_keys(model_identities, MODEL_ASSET_PATHS, "Model identities")
    for model, identity in model_identities.items():
        _require_exact_keys(identity, {"weights", "cfg", "names"}, f"{model} identity")
        for asset_name, digest in identity.items():
            _require_sha256(digest, f"{model} {asset_name} identity")

    if decision.get("integrity_status") != "passed":
        raise EvidenceContractError("Checkpoint-selection decision is not verified.")
    if decision.get("decision_rule") != "locked_lexicographic_v1":
        raise EvidenceContractError("Checkpoint-selection decision rule is not locked.")
    if decision.get("delta_sign") != policy["delta_sign"]:
        raise EvidenceContractError("Checkpoint-selection delta policies disagree.")
    _validate_selection_summary(artifact_rows["selection_summary.csv"], decision)
    expected_decision = _recompute_checkpoint_decision(decision)
    for field, expected in expected_decision.items():
        if decision.get(field) != expected:
            raise EvidenceContractError(
                f"Checkpoint-selection {field} conflicts with recomputed evidence."
            )
    selected_model = str(decision.get("selected_model", ""))
    if selected_model not in MODEL_ASSET_PATHS:
        raise EvidenceContractError(f"Unsupported selected model: {selected_model!r}")
    if decision.get("selected_checkpoint") != CHECKPOINT_BY_MODEL[selected_model]:
        raise EvidenceContractError(
            "Selected checkpoint does not correspond to the selected model."
        )
    identity = model_identities[selected_model]
    return {
        "selection_directory": directory,
        "selection_run_id": run_id,
        "selection_manifest_sha256": sha256_file(manifest_path),
        "decision_sha256": sha256_file(decision_path),
        "selected_checkpoint": decision["selected_checkpoint"],
        "selected_model": selected_model,
        "model_identity": dict(identity),
        "asset_paths": dict(MODEL_ASSET_PATHS[selected_model]),
    }


def resolve_selected_model_assets(
    asset_root: Path | str,
    selection_run: Path | str,
) -> dict:
    """Resolve and fingerprint the checkpoint selected by verified evidence."""

    selection = load_verified_checkpoint_selection(selection_run)
    root = Path(asset_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise NotADirectoryError(f"Asset root does not exist: {root}")

    paths = {
        name: _contained(root / relative, root, f"Selected-model {name}")
        for name, relative in selection["asset_paths"].items()
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Selected-model assets are missing:\n{details}")

    manifest_names = {"weights": "weights", "config": "cfg", "classes": "names"}
    for path_name, manifest_name in manifest_names.items():
        observed = sha256_file(paths[path_name])
        expected = selection["model_identity"][manifest_name]
        if observed != expected:
            raise EvidenceContractError(
                f"Selected-model {path_name} hash mismatch: {paths[path_name]}"
            )
    return {**selection, "resolved_paths": paths}


def threshold_tag(value: float) -> str:
    """Return a stable filename token for a numeric threshold."""

    return format(float(value), ".12g").replace("-", "neg_").replace(".", "_")


def _normalize_nms_summary(rows: list[dict], selected_model: str) -> dict[float, dict]:
    if not rows:
        raise EvidenceContractError("NMS threshold summary cannot be empty.")
    normalized = {}
    constants = {
        "model": set(),
        "dataset": set(),
        "score_threshold": set(),
        "map_iou_threshold": set(),
        "eval_type": set(),
        "total_ground_truth": set(),
    }
    for index, row in enumerate(rows, start=1):
        label = f"NMS summary row {index}"
        threshold = _finite_float(row["nms_threshold"], f"{label} threshold")
        if not 0.0 <= threshold <= 1.0:
            raise EvidenceContractError(f"{label} threshold must be within [0, 1].")
        if threshold in normalized:
            raise EvidenceContractError(
                f"NMS summary contains duplicate threshold {threshold}."
            )
        score = _finite_float(row["mAP@0.5_11_point"], f"{label} AP50")
        if not 0.0 <= score <= 1.0:
            raise EvidenceContractError(f"{label} AP50 must be within [0, 1].")
        ground_truth = _positive_integer(
            row["total_ground_truth"], f"{label} ground-truth count"
        )
        predictions = _nonnegative_integer(
            row["total_predictions_after_nms"], f"{label} prediction count"
        )
        evaluation_rows = _nonnegative_integer(
            row["evaluation_rows"], f"{label} evaluation-row count"
        )
        if evaluation_rows != predictions:
            raise EvidenceContractError(
                f"{label} evaluation rows must equal retained predictions."
            )
        score_threshold = _finite_float(
            row["score_threshold"], f"{label} score threshold"
        )
        map_iou_threshold = _finite_float(
            row["map_iou_threshold"], f"{label} AP IoU threshold"
        )
        if not 0.0 <= score_threshold <= 1.0 or not 0.0 <= map_iou_threshold <= 1.0:
            raise EvidenceContractError(f"{label} policy thresholds must be within [0, 1].")
        model = str(row["model"]).strip()
        dataset = str(row["dataset"]).strip()
        eval_type = str(row["eval_type"]).strip()
        if not model or not dataset or not eval_type:
            raise EvidenceContractError(f"{label} policy identities cannot be empty.")
        constants["model"].add(model)
        constants["dataset"].add(dataset)
        constants["score_threshold"].add(score_threshold)
        constants["map_iou_threshold"].add(map_iou_threshold)
        constants["eval_type"].add(eval_type)
        constants["total_ground_truth"].add(ground_truth)
        normalized[threshold] = {
            "mAP@0.5_11_point": score,
            "total_predictions_after_nms": predictions,
        }

    inconsistent = [name for name, values in constants.items() if len(values) != 1]
    if inconsistent:
        raise EvidenceContractError(
            "NMS summary policy columns are not constant: " + ", ".join(inconsistent)
        )
    if constants["model"] != {selected_model}:
        raise EvidenceContractError(
            "NMS summary model does not match checkpoint selection."
        )
    if constants["dataset"] != {OPERATING_POINT_DATASET}:
        raise EvidenceContractError("NMS summary dataset is not the locked sample.")
    if constants["eval_type"] != {"combined"}:
        raise EvidenceContractError("NMS summary evaluation policy must be combined.")
    _require_close(
        next(iter(constants["score_threshold"])), 0.5, "NMS score-threshold policy"
    )
    _require_close(
        next(iter(constants["map_iou_threshold"])), 0.5, "NMS AP IoU policy"
    )
    return normalized


def _normalize_duplicate_summary(rows: list[dict]) -> dict[float, dict]:
    if not rows:
        raise EvidenceContractError("NMS duplicate summary cannot be empty.")
    normalized = {}
    for index, row in enumerate(rows, start=1):
        label = f"NMS duplicate row {index}"
        threshold = _finite_float(row["nms_threshold"], f"{label} threshold")
        if not 0.0 <= threshold <= 1.0:
            raise EvidenceContractError(f"{label} threshold must be within [0, 1].")
        if threshold in normalized:
            raise EvidenceContractError(
                f"NMS duplicate summary contains duplicate threshold {threshold}."
            )
        duplicate_pairs = _nonnegative_integer(
            row["duplicate_like_pairs_iou_gt_0_5"], f"{label} duplicate-pair count"
        )
        duplicate_images = _nonnegative_integer(
            row["images_with_duplicate_like_pairs"], f"{label} duplicate-image count"
        )
        mean_pairs = _finite_float(
            row["mean_duplicate_like_pairs_per_image"], f"{label} mean duplicate pairs"
        )
        predictions = _nonnegative_integer(
            row["total_predictions_after_nms"], f"{label} prediction count"
        )
        if mean_pairs < 0.0:
            raise EvidenceContractError(f"{label} mean duplicate pairs cannot be negative.")
        if duplicate_images > duplicate_pairs:
            raise EvidenceContractError(
                f"{label} duplicate images cannot exceed duplicate pairs."
            )
        if duplicate_images > predictions:
            raise EvidenceContractError(
                f"{label} duplicate images cannot exceed retained predictions."
            )
        zero_flags = (duplicate_pairs == 0, duplicate_images == 0, mean_pairs == 0.0)
        if len(set(zero_flags)) != 1:
            raise EvidenceContractError(
                f"{label} zero-valued duplicate diagnostics are inconsistent."
            )
        normalized[threshold] = {
            "duplicate_like_pairs_iou_gt_0_5": duplicate_pairs,
            "total_predictions_after_nms": predictions,
        }
    return normalized


def load_verified_operating_point(
    path: Path | str,
    selection_run: Path | str,
) -> dict:
    """Validate and independently recompute a recorded NMS operating point."""

    source = Path(path).expanduser().resolve(strict=False)
    if not source.is_file():
        raise FileNotFoundError(f"NMS operating-point record not found: {source}")
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceContractError(
            f"NMS operating-point record is not valid JSON: {source}"
        ) from error
    if not isinstance(record, dict):
        raise EvidenceContractError("NMS operating-point JSON root must be an object.")
    if record.get("status") != "complete" or record.get("schema_version") != 1:
        raise EvidenceContractError("NMS operating-point record is not complete.")
    if record.get("selection_rule") != OPERATING_POINT_SELECTION_RULE:
        raise EvidenceContractError("NMS operating-point selection rule is not locked.")

    selection = load_verified_checkpoint_selection(selection_run)
    upstream = record.get("checkpoint_selection", {})
    expected_selection = {
        "run_id": selection["selection_run_id"],
        "manifest_sha256": selection["selection_manifest_sha256"],
        "decision_sha256": selection["decision_sha256"],
    }
    if upstream != expected_selection:
        raise EvidenceContractError(
            "NMS operating point does not reference the verified checkpoint selection."
        )
    if record.get("selected_model") != selection["selected_model"]:
        raise EvidenceContractError(
            "NMS operating point uses a different model than checkpoint selection."
        )
    threshold = _finite_float(
        record.get("selected_nms_iou_threshold"), "NMS operating-point threshold"
    )
    if not 0.0 <= threshold <= 1.0:
        raise EvidenceContractError(
            "NMS operating-point threshold must be between zero and one."
        )

    evidence = record.get("evidence")
    _require_exact_keys(
        evidence, OPERATING_POINT_EVIDENCE_SCHEMAS, "Operating-point evidence"
    )
    evidence_rows = {}
    for artifact_name, columns in OPERATING_POINT_EVIDENCE_SCHEMAS.items():
        identity = evidence[artifact_name]
        _require_exact_keys(
            identity,
            {"rows", "sha256"},
            f"Operating-point evidence identity for {artifact_name}",
        )
        artifact_path = _package_file(
            source.parent, artifact_name, f"Operating-point evidence {artifact_name}"
        )
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Operating-point evidence file is missing: {artifact_path}"
            )
        digest = _require_sha256(identity["sha256"], f"{artifact_name} hash")
        if digest != sha256_file(artifact_path):
            raise EvidenceContractError(
                f"Operating-point evidence hash mismatch: {artifact_path}"
            )
        rows = _read_exact_csv(
            artifact_path, columns, f"Operating-point evidence {artifact_name}"
        )
        expected_rows = _positive_integer(
            identity["rows"], f"{artifact_name} row count"
        )
        if len(rows) != expected_rows:
            raise EvidenceContractError(
                f"Operating-point evidence row-count mismatch: {artifact_name}"
            )
        evidence_rows[artifact_name] = rows

    summary = _normalize_nms_summary(
        evidence_rows["nms_threshold_summary_sample5000.csv"],
        selection["selected_model"],
    )
    duplicates = _normalize_duplicate_summary(
        evidence_rows["duplicate_summary_by_threshold_sample5000.csv"]
    )
    if set(summary) != set(duplicates):
        raise EvidenceContractError(
            "NMS evidence tables must contain exactly one row per shared threshold."
        )
    for shared_threshold in summary:
        if (
            summary[shared_threshold]["total_predictions_after_nms"]
            != duplicates[shared_threshold]["total_predictions_after_nms"]
        ):
            raise EvidenceContractError(
                "NMS evidence prediction totals disagree at threshold "
                f"{shared_threshold}."
            )

    selected_threshold, selected_summary = min(
        summary.items(),
        key=lambda item: (
            -item[1]["mAP@0.5_11_point"],
            item[1]["total_predictions_after_nms"],
            item[0],
        ),
    )
    _require_close(
        threshold, selected_threshold, "Selected NMS operating-point threshold"
    )
    selected_metrics = record.get("selected_metrics")
    _require_exact_keys(
        selected_metrics,
        {
            "mAP@0.5_11_point",
            "total_predictions_after_nms",
            "duplicate_like_pairs_iou_gt_0_5",
        },
        "Selected NMS metrics",
    )
    _require_close(
        selected_metrics["mAP@0.5_11_point"],
        selected_summary["mAP@0.5_11_point"],
        "Selected NMS AP50",
    )
    if _nonnegative_integer(
        selected_metrics["total_predictions_after_nms"],
        "Selected NMS prediction count",
    ) != selected_summary["total_predictions_after_nms"]:
        raise EvidenceContractError("Selected NMS prediction count is inconsistent.")
    expected_duplicates = duplicates[selected_threshold][
        "duplicate_like_pairs_iou_gt_0_5"
    ]
    if _nonnegative_integer(
        selected_metrics["duplicate_like_pairs_iou_gt_0_5"],
        "Selected NMS duplicate-pair count",
    ) != expected_duplicates:
        raise EvidenceContractError("Selected NMS duplicate-pair count is inconsistent.")
    return {**record, "path": source, "sha256": sha256_file(source)}
