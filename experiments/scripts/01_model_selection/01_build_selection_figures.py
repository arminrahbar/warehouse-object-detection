"""Build a verified, publication-quality Experiment 01 figure package.

This stage performs no inference. It verifies the immutable full-quality,
paired-runtime, and checkpoint-selection evidence packages before it renders
any figure. The destination directory is promoted atomically only after every
requested PNG has been written successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SELECTOR_PATH = Path(__file__).resolve().parent / "01_select_checkpoint.py"

MODEL_A = "model1"
MODEL_B = "model2"
MODEL_LABELS = {MODEL_A: "Checkpoint A", MODEL_B: "Checkpoint B"}
NAVY = "#1F4E79"
ORANGE = "#E67E22"
TEAL = "#009E73"
VERMILION = "#D55E00"
INK = "#20262E"
MUTED = "#5D6875"
GRID = "#D9E0E7"
PALE = "#F4F6F8"
NEUTRAL = "#8A96A3"

DEFAULT_EXPECTED_IMAGES = 9525
DEFAULT_EXPECTED_LABELS = 36721
DEFAULT_EXPECTED_CLASSES = 20
DEFAULT_EXPECTED_PAIRS = 1500

OUTPUT_STEMS = (
    "01_decision_summary",
    "02_class_ap50_dumbbell",
    "03_ap_delta_vs_support",
    "04_paired_latency_ecdf",
    "05_experiment_design",
)

SELECTION_FILES = {
    "selection_summary.csv",
    "bootstrap_replicates.csv",
    "decision.json",
    "selection_manifest.json",
}


class FigureEvidenceError(ValueError):
    """Raised when plotting evidence fails an integrity or semantic gate."""


def _load_selector_module():
    spec = importlib.util.spec_from_file_location(
        "checkpoint_selection_for_figures", SELECTOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load checkpoint selector: {SELECTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selector = _load_selector_module()


def _require(condition, message):
    if not condition:
        raise FigureEvidenceError(message)


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


def _read_json(path, context):
    source = Path(path)
    _require(source.is_file(), f"Missing {context}: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FigureEvidenceError(f"Unable to read {context}: {error}") from error
    _require(isinstance(value, dict), f"{context} must contain a JSON object.")
    return value


def _read_csv_exact(path, columns, context):
    source = Path(path)
    _require(source.is_file(), f"Missing {context}: {source}")
    try:
        table = pd.read_csv(source)
    except Exception as error:
        raise FigureEvidenceError(f"Unable to read {context}: {error}") from error
    _require(
        table.columns.tolist() == list(columns),
        f"{context} schema does not match its locked ordered columns.",
    )
    return table


def _csv_row_count(path):
    with Path(path).open("r", encoding="utf-8", newline="") as source:
        return max(0, sum(1 for _ in source) - 1)


def _verify_csv_identity(path, identity, columns, context):
    source = Path(path)
    _require(isinstance(identity, dict), f"{context} identity must be an object.")
    _require(source.is_file(), f"Missing {context}: {source}")
    _require(
        int(identity.get("size_bytes", -1)) == source.stat().st_size,
        f"{context} byte size does not match its manifest.",
    )
    _require(
        _sha256_file(source) == str(identity.get("sha256", "")),
        f"{context} hash mismatch.",
    )
    _require(
        int(identity.get("rows", -1)) == _csv_row_count(source),
        f"{context} row count does not match its manifest.",
    )
    _require(
        identity.get("columns") == list(columns),
        f"{context} manifest schema is invalid.",
    )


def _numeric(table, columns, context):
    result = table.copy()
    for column in columns:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except Exception as error:
            raise FigureEvidenceError(f"{context}.{column} must be numeric.") from error
        _require(
            np.isfinite(result[column].to_numpy(dtype=float)).all(),
            f"{context}.{column} contains a non-finite value.",
        )
    return result


def _close(observed, expected, context, atol=1e-10):
    _require(
        math.isclose(float(observed), float(expected), rel_tol=1e-9, abs_tol=atol),
        f"{context} is inconsistent with its source evidence.",
    )


def _validate_selection_run(
    selection_run, quality, runtime, expected_images, expected_labels,
    expected_classes,
):
    directory = _absolute(selection_run)
    _require(directory.is_dir(), f"Selection run directory not found: {directory}")
    observed_files = {path.name for path in directory.iterdir() if path.is_file()}
    _require(
        observed_files == SELECTION_FILES,
        "Selection run must contain exactly its four verified evidence files.",
    )
    manifest_path = directory / "selection_manifest.json"
    manifest = _read_json(manifest_path, "selection manifest")
    _require(manifest.get("schema_version") == 1, "Unsupported selection schema.")
    _require(manifest.get("status") == "complete", "Selection run is not complete.")
    policy = manifest.get("selection_policy")
    _require(isinstance(policy, dict), "Selection policy is missing.")
    _require(
        manifest.get("selection_policy_sha256") == _sha256_json(policy),
        "Selection policy hash mismatch.",
    )

    artifacts = manifest.get("artifacts")
    _require(
        isinstance(artifacts, dict)
        and set(artifacts) == {"selection_summary.csv", "bootstrap_replicates.csv"},
        "Selection artifact set is invalid.",
    )
    _verify_csv_identity(
        directory / "selection_summary.csv", artifacts["selection_summary.csv"],
        selector.SUMMARY_COLUMNS, "selection summary",
    )
    _verify_csv_identity(
        directory / "bootstrap_replicates.csv",
        artifacts["bootstrap_replicates.csv"], selector.BOOTSTRAP_COLUMNS,
        "selection bootstrap evidence",
    )
    decision_path = directory / "decision.json"
    decision_identity = manifest.get("decision")
    _require(isinstance(decision_identity, dict), "Selection decision identity is missing.")
    _require(
        decision_path.stat().st_size == int(decision_identity.get("size_bytes", -1)),
        "Selection decision byte size mismatch.",
    )
    _require(
        _sha256_file(decision_path) == decision_identity.get("sha256"),
        "Selection decision hash mismatch.",
    )
    decision = _read_json(decision_path, "selection decision")
    _require(decision.get("integrity_status") == "passed", "Selection integrity did not pass.")
    _require(decision.get("selected_model") in {MODEL_A, MODEL_B},
             "Selection decision names an invalid model.")
    _require(decision.get("selected_checkpoint") in {"A", "B"},
             "Selection decision names an invalid checkpoint.")

    quality_input = manifest.get("quality_input", {})
    runtime_input = manifest.get("runtime_input", {})
    _require(
        _sha256_file(quality["directory"] / "run_manifest.json")
        == quality_input.get("manifest_sha256"),
        "Selection-to-quality manifest hash mismatch.",
    )
    _require(
        quality_input.get("source_policy_sha256")
        == quality["manifest"].get("source_policy_sha256"),
        "Selection-to-quality policy identity mismatch.",
    )
    _require(
        _sha256_file(runtime["directory"] / "inference_benchmark_manifest.json")
        == runtime_input.get("manifest_sha256"),
        "Selection-to-runtime manifest hash mismatch.",
    )
    _require(
        runtime_input.get("run_fingerprint_sha256")
        == runtime["manifest"].get("run_fingerprint_sha256"),
        "Selection-to-runtime fingerprint mismatch.",
    )

    corpus = manifest.get("corpus", {})
    _require(int(corpus.get("images", -1)) == expected_images,
             "Selection corpus image count is invalid.")
    _require(int(corpus.get("labels", -1)) == expected_labels,
             "Selection corpus label count is invalid.")
    _require(corpus.get("classes") == quality["classes"],
             "Selection class vocabulary differs from quality evidence.")
    _require(len(corpus.get("classes", [])) == expected_classes,
             "Selection class count is invalid.")

    expected_models = {
        model: {
            asset: quality["manifest"]["models"][model][asset]["sha256"]
            for asset in ("weights", "cfg", "names")
        }
        for model in (MODEL_A, MODEL_B)
    }
    _require(manifest.get("model_identities") == expected_models,
             "Selection model identities differ from quality evidence.")

    summary = _read_csv_exact(
        directory / "selection_summary.csv", selector.SUMMARY_COLUMNS,
        "selection summary",
    )
    _require(
        summary["metric"].astype(str).tolist()
        == ["mAP50_101pt", "deployment_macro_f1", "p95_compute_ms", "mean_compute_ms"],
        "Selection summary metric order is invalid.",
    )
    summary = _numeric(
        summary,
        ["model1_value", "model2_value", "delta_model2_minus_model1",
         "ci_lower", "ci_upper"],
        "selection summary",
    )
    _require((summary["ci_lower"] <= summary["ci_upper"]).all(),
             "Selection summary contains a reversed confidence interval.")
    _require(
        np.allclose(
            summary["delta_model2_minus_model1"],
            summary["model2_value"] - summary["model1_value"],
            rtol=1e-10, atol=1e-12,
        ),
        "Selection summary deltas are inconsistent.",
    )

    aggregate = quality["aggregate"].set_index("model")
    lookup = summary.set_index("metric")
    for metric in ("mAP50_101pt", "deployment_macro_f1"):
        _close(lookup.loc[metric, "model1_value"], aggregate.loc[MODEL_A, metric], metric)
        _close(lookup.loc[metric, "model2_value"], aggregate.loc[MODEL_B, metric], metric)
    p95 = lookup.loc["p95_compute_ms"]
    _close(p95["model1_value"], runtime["model1_p95_ms"], "Checkpoint A p95")
    _close(p95["model2_value"], runtime["model2_p95_ms"], "Checkpoint B p95")
    mean = lookup.loc["mean_compute_ms"]
    _close(mean["model1_value"], runtime["model1_mean_ms"], "Checkpoint A mean latency")
    _close(mean["model2_value"], runtime["model2_mean_ms"], "Checkpoint B mean latency")

    bootstrap = _read_csv_exact(
        directory / "bootstrap_replicates.csv", selector.BOOTSTRAP_COLUMNS,
        "selection bootstrap evidence",
    )
    _require(
        len(bootstrap) == int(policy.get("bootstrap_samples", -1)),
        "Selection bootstrap count differs from its policy.",
    )
    return {
        "directory": directory,
        "manifest": manifest,
        "summary": summary,
        "bootstrap": bootstrap,
        "decision": decision,
    }


def load_verified_evidence(
    quality_run, runtime_run, selection_run, *,
    expected_images=DEFAULT_EXPECTED_IMAGES,
    expected_labels=DEFAULT_EXPECTED_LABELS,
    expected_classes=DEFAULT_EXPECTED_CLASSES,
    expected_pairs=DEFAULT_EXPECTED_PAIRS,
    locked_hashes=None,
):
    """Verify the three immutable evidence packages and return plotting tables."""

    selection_manifest = _read_json(
        _absolute(selection_run) / "selection_manifest.json", "selection manifest"
    )
    policy = selection_manifest.get("selection_policy", {})
    bootstrap_samples = int(policy.get("bootstrap_samples", -1))
    bootstrap_seed = int(policy.get("bootstrap_seed", -1))
    _require(bootstrap_samples > 0, "Selection bootstrap count is invalid.")
    _require(bootstrap_seed >= 0, "Selection bootstrap seed is invalid.")
    try:
        quality = selector.validate_quality_run(
            quality_run, expected_images=expected_images,
            expected_labels=expected_labels,
            locked_hashes=(
                selector.LOCKED_MODEL_HASHES
                if locked_hashes is None
                else locked_hashes
            ),
        )
        runtime = selector.validate_runtime_run(
            runtime_run,
            quality,
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
    except selector.IntegrityError as error:
        raise FigureEvidenceError(str(error)) from error
    _require(
        len(runtime["pair_rows"]) == expected_pairs,
        f"Runtime pair count must be exactly {expected_pairs}.",
    )
    _require(len(quality["classes"]) == expected_classes,
             f"Quality evidence must contain exactly {expected_classes} classes.")
    selection = _validate_selection_run(
        selection_run, quality, runtime, expected_images, expected_labels,
        expected_classes,
    )
    return {"quality": quality, "runtime": runtime, "selection": selection}


def select_delta_labels(class_table, max_labels=6, materiality=0.01):
    """Return deterministic class labels for the largest meaningful AP deltas."""

    _require(max_labels >= 0, "Maximum label count must be non-negative.")
    required = {"class_id", "class_name", "ground_truth_count", "ap_delta"}
    _require(required.issubset(class_table.columns),
             "Class-delta table is missing required columns.")
    candidates = class_table.copy()
    candidates["absolute_delta"] = candidates["ap_delta"].abs()
    candidates = candidates.loc[
        (candidates["ground_truth_count"] > 0)
        & (candidates["absolute_delta"] >= float(materiality))
    ]
    candidates = candidates.sort_values(
        ["absolute_delta", "ground_truth_count", "class_id", "class_name"],
        ascending=[False, True, True, True], kind="mergesort",
    )
    return candidates.head(max_labels)["class_name"].astype(str).tolist()


def _class_comparison_table(evidence):
    per_class = evidence["quality"]["per_class"].copy()
    for column in ("class_id", "ground_truth_count", "ap50_101pt"):
        per_class[column] = pd.to_numeric(per_class[column], errors="raise")
    a = per_class.loc[per_class["model"] == MODEL_A].copy()
    b = per_class.loc[per_class["model"] == MODEL_B].copy()
    key = ["class_id", "class_name"]
    merged = a.merge(b, on=key, suffixes=("_a", "_b"), validate="one_to_one")
    _require(
        np.array_equal(
            merged["ground_truth_count_a"].to_numpy(),
            merged["ground_truth_count_b"].to_numpy(),
        ),
        "Per-class support differs between checkpoints.",
    )
    merged = merged.rename(columns={"ground_truth_count_a": "ground_truth_count"})
    merged["ap_a"] = merged["ap50_101pt_a"].astype(float)
    merged["ap_b"] = merged["ap50_101pt_b"].astype(float)
    merged["ap_delta"] = merged["ap_b"] - merged["ap_a"]
    supported = merged.loc[merged["ground_truth_count"] > 0].copy()
    _require(not supported.empty, "No class has ground-truth support.")
    _require(
        (supported[["ap_a", "ap_b"]] >= 0).all().all()
        and (supported[["ap_a", "ap_b"]] <= 1).all().all(),
        "Per-class AP50 values must be within [0, 1].",
    )
    return supported


def _style_context():
    return plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.75,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "svg.hashsalt": "warehouse-object-detection-experiment-01-v1",
    })


def _header(fig, title, subtitle):
    fig.text(0.06, 0.965, title, fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.915, subtitle, fontsize=10.5, color=MUTED, va="top")


def _clean_axis(axis, grid_axis="x"):
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis=grid_axis)
    axis.set_axisbelow(True)


def _metric_materiality(row):
    raw = str(row["practical_threshold"])
    try:
        return float(raw)
    except ValueError:
        if raw == "5%_of_slower_p95":
            return 0.05 * max(float(row["model1_value"]), float(row["model2_value"]))
        return 0.0


def _decision_summary_figure(evidence):
    summary = evidence["selection"]["summary"].set_index("metric")
    quality_manifest = evidence["quality"]["manifest"]
    pairs = len(evidence["runtime"]["pair_rows"])
    decision = evidence["selection"]["decision"]
    specs = [
        ("mAP50_101pt", "mAP50", 100.0, "percentage points"),
        ("deployment_macro_f1", "Deployment macro F1", 100.0, "percentage points"),
        ("p95_compute_ms", "p95 compute latency", 1.0, "milliseconds"),
    ]
    fig = plt.figure(figsize=(14.2, 6.2))
    grid = fig.add_gridspec(
        2, 3, left=0.06, right=0.98, bottom=0.11, top=0.79,
        height_ratios=[1.9, 1.1], hspace=0.34, wspace=0.28,
    )
    _header(
        fig,
        "Checkpoint decision summary",
        f"Quality: {quality_manifest['dataset']['selected_images']:,} images, "
        f"{quality_manifest['dataset']['selected_labels']:,} objects; runtime: "
        f"{pairs:,} paired comparisons ({pairs * 2:,} checkpoint observations). "
        "AP50 uses IoU 0.50; latency excludes image decode.",
    )
    selected = decision["selected_checkpoint"]
    fig.text(
        0.98, 0.965, f"Selected: Checkpoint {selected}", ha="right", va="top",
        fontsize=12, fontweight="bold", color=TEAL,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EAF6F2", "edgecolor": TEAL},
    )
    for column, (metric, title, scale, unit) in enumerate(specs):
        row = summary.loc[metric]
        values = np.array([row["model1_value"], row["model2_value"]], dtype=float) * scale
        delta = float(row["delta_model2_minus_model1"]) * scale
        lower = float(row["ci_lower"]) * scale
        upper = float(row["ci_upper"]) * scale
        materiality = _metric_materiality(row) * scale

        top = fig.add_subplot(grid[0, column])
        lo, hi = float(values.min()), float(values.max())
        span = max(hi - lo, 0.04 * max(abs(hi), abs(lo), 1.0))
        margin = max(span * 0.85, 0.015 * max(abs(hi), abs(lo), 1.0))
        top.set_xlim(lo - margin, hi + margin)
        top.plot(values, [0, 0], color=GRID, linewidth=3, zorder=1)
        top.scatter(values[0], 0, s=115, color=NAVY, zorder=3)
        top.scatter(values[1], 0, s=115, color=ORANGE, zorder=3)
        decimals = 1 if scale == 100 else 2
        top.annotate(
            f"A  {values[0]:.{decimals}f}", (values[0], 0), xytext=(0, 17),
            textcoords="offset points", ha="center", color=NAVY, fontweight="bold",
        )
        top.annotate(
            f"B  {values[1]:.{decimals}f}", (values[1], 0), xytext=(0, -24),
            textcoords="offset points", ha="center", color=ORANGE, fontweight="bold",
        )
        top.set_title(title, loc="left", fontweight="bold", color=INK)
        top.set_yticks([])
        top.tick_params(axis="x", labelbottom=False, bottom=False)
        top.spines[:].set_visible(False)

        bottom = fig.add_subplot(grid[1, column])
        extent = max(abs(lower), abs(upper), abs(delta), materiality, 1e-9)
        bottom.axvspan(-materiality, materiality, color=PALE, zorder=0)
        bottom.axvline(0, color=INK, linewidth=0.9)
        color = TEAL if delta > 0 else VERMILION if delta < 0 else NEUTRAL
        bottom.hlines(0, lower, upper, color=color, linewidth=2)
        bottom.vlines([lower, upper], -0.06, 0.06, color=color, linewidth=1.5)
        bottom.scatter(delta, 0, color=color, s=48, zorder=3)
        bottom.set_xlim(-1.3 * extent, 1.3 * extent)
        bottom.set_yticks([])
        bottom.set_xlabel(f"Δ B − A ({unit})")
        bottom.text(
            0.02, 0.92,
            f"Δ {delta:+.{2 if scale == 1 else 1}f}  |  95% CI "
            f"[{lower:+.{2 if scale == 1 else 1}f}, {upper:+.{2 if scale == 1 else 1}f}]",
            transform=bottom.transAxes, va="top", color=color, fontsize=9,
        )
        if materiality > 0:
            bottom.text(
                0.98, 0.08, f"materiality ±{materiality:.2g}",
                transform=bottom.transAxes, ha="right", va="bottom",
                fontsize=8.5, color=MUTED,
            )
        _clean_axis(bottom)
    return fig


def _class_dumbbell_figure(evidence, class_table):
    ordered = class_table.copy()
    ordered["mean_ap"] = (ordered["ap_a"] + ordered["ap_b"]) / 2.0
    ordered = ordered.sort_values(
        ["mean_ap", "ground_truth_count", "class_name"],
        ascending=[True, True, True], kind="mergesort",
    ).reset_index(drop=True)
    count = len(ordered)
    fig_height = max(7.0, 0.42 * count + 2.8)
    fig, axis = plt.subplots(figsize=(13.2, fig_height))
    fig.subplots_adjust(left=0.27, right=0.96, bottom=0.10, top=0.84)
    total_classes = len(evidence["quality"]["classes"])
    _header(
        fig,
        "AP50 by class",
        f"{count} of {total_classes} classes have ground-truth support; sorted by mean AP50. "
        "Support (n) is the full-corpus ground-truth object count.",
    )
    y = np.arange(count)
    a = ordered["ap_a"].to_numpy(dtype=float) * 100.0
    b = ordered["ap_b"].to_numpy(dtype=float) * 100.0
    for position, left, right in zip(y, a, b):
        axis.plot([left, right], [position, position], color=GRID, linewidth=2.2, zorder=1)
    axis.scatter(a, y, s=54, color=NAVY, zorder=3)
    axis.scatter(b, y, s=54, color=ORANGE, zorder=3)
    labels = [
        f"{row.class_name}  (n={int(row.ground_truth_count):,})"
        for row in ordered.itertuples(index=False)
    ]
    axis.set_yticks(y, labels)
    axis.set_ylim(-0.7, count + 0.5)
    axis.set_xlim(-1, 119)
    axis.set_xticks(np.arange(0, 101, 20))
    axis.set_xlabel("AP50 (101-point interpolation, %)")
    axis.text(103, count - 0.15, "A", color=NAVY, fontweight="bold", ha="center")
    axis.text(112, count - 0.15, "B", color=ORANGE, fontweight="bold", ha="center")
    for position, value_a, value_b in zip(y, a, b):
        axis.text(103, position, f"{value_a:5.1f}", color=NAVY, ha="center", va="center", fontsize=8.5)
        axis.text(112, position, f"{value_b:5.1f}", color=ORANGE, ha="center", va="center", fontsize=8.5)
    axis.axvline(100, color=GRID, linewidth=0.8)
    _clean_axis(axis)
    return fig


def _delta_support_figure(evidence, class_table):
    fig, axis = plt.subplots(figsize=(11.2, 6.8))
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.13, top=0.82)
    total_classes = len(evidence["quality"]["classes"])
    _header(
        fig,
        "Where checkpoint AP50 differs",
        f"Supported classes: {len(class_table)}/{total_classes}. X-axis is log-scaled class support; "
        "the shaded ±1 pp band is the selector's recorded quality materiality threshold. "
        "Labels show material shifts and the strongest counterexample.",
    )
    delta_pp = class_table["ap_delta"].to_numpy(dtype=float) * 100.0
    support = class_table["ground_truth_count"].to_numpy(dtype=float)
    colors = np.where(delta_pp > 1.0, TEAL, np.where(delta_pp < -1.0, VERMILION, NEUTRAL))
    axis.axhspan(-1.0, 1.0, color=PALE, zorder=0)
    axis.axhline(0, color=INK, linewidth=0.9)
    axis.scatter(support, delta_pp, s=64, c=colors, edgecolor="white", linewidth=0.7, zorder=3)
    axis.set_xscale("log")
    axis.set_xlabel("Ground-truth object count (log scale)")
    axis.set_ylabel("AP50 delta, B − A (percentage points)")
    axis.text(0.99, 0.97, "B higher", transform=axis.transAxes, ha="right", va="top",
              color=TEAL, fontweight="bold")
    axis.text(0.99, 0.03, "A higher", transform=axis.transAxes, ha="right", va="bottom",
              color=VERMILION, fontweight="bold")
    selected = set(select_delta_labels(class_table, max_labels=6, materiality=0.01))
    negative_rows = class_table[class_table["ap_delta"] < 0]
    if not negative_rows.empty:
        strongest_counterexample = negative_rows.loc[
            negative_rows["ap_delta"].idxmin(), "class_name"
        ]
        selected.add(str(strongest_counterexample))
    for row in class_table.itertuples(index=False):
        if str(row.class_name) not in selected:
            continue
        offset = (7, 7) if row.ap_delta >= 0 else (7, -13)
        axis.annotate(
            str(row.class_name),
            (float(row.ground_truth_count), float(row.ap_delta) * 100.0),
            xytext=offset, textcoords="offset points", fontsize=9, color=INK,
        )
    _clean_axis(axis, grid_axis="both")
    return fig


def _ecdf(values):
    x = np.sort(np.asarray(values, dtype=float))
    _require(len(x) > 0 and np.isfinite(x).all() and (x >= 0).all(),
             "Latency evidence must contain finite non-negative values.")
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def _latency_ecdf_figure(evidence):
    pairs = evidence["runtime"]["pair_rows"]
    values = {
        MODEL_A: pairs["model1_compute_ms"].to_numpy(dtype=float),
        MODEL_B: pairs["model2_compute_ms"].to_numpy(dtype=float),
    }
    fig, axis = plt.subplots(figsize=(11.4, 6.8))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.13, top=0.82)
    policy = evidence["runtime"]["manifest"]["policy"]
    _header(
        fig,
        "Paired compute-latency distribution",
        f"{len(pairs):,} paired comparisons ({len(pairs) * 2:,} checkpoint observations); "
        "each image is decoded once and both checkpoints run on the same frame. "
        f"Scope: {policy['timing_scope']}.",
    )
    for model, color in ((MODEL_A, NAVY), (MODEL_B, ORANGE)):
        x, y = _ecdf(values[model])
        median = float(np.percentile(x, 50, method="linear"))
        p95 = float(np.percentile(x, 95, method="linear"))
        axis.step(x, y, where="post", color=color, linewidth=2.5)
        axis.scatter([median, p95], [0.50, 0.95], color=color, s=48, zorder=4)
        axis.vlines([median, p95], [0, 0], [0.50, 0.95], color=color,
                    linewidth=1, linestyles="dotted", alpha=0.8)
        label_y = 0.88 if model == MODEL_A else 0.76
        axis.text(
            0.025, label_y,
            f"{MODEL_LABELS[model]} — median {median:.2f} ms; p95 {p95:.2f} ms",
            transform=axis.transAxes, color=color, fontweight="bold", fontsize=9.5,
            ha="left", va="top",
        )
    axis.set_xlabel("Compute time per image (ms)")
    axis.set_ylabel("Cumulative share of observations")
    axis.set_ylim(0, 1.01)
    axis.set_yticks(np.linspace(0, 1, 6), [f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    _clean_axis(axis, grid_axis="both")
    return fig


def _design_figure(evidence):
    quality_manifest = evidence["quality"]["manifest"]
    decision = evidence["selection"]["decision"]
    pairs = len(evidence["runtime"]["pair_rows"])
    fig = plt.figure(figsize=(13.6, 5.7))
    axis = fig.add_axes([0, 0, 1, 1])
    axis.set_axis_off()
    _header(
        fig,
        "Experiment design at a glance",
        "A compact record of what entered the comparison, what stayed fixed, and how the decision was made.",
    )
    cards = [
        (
            "INPUT", INK,
            [
                f"{quality_manifest['dataset']['selected_images']:,} labeled images",
                f"{quality_manifest['dataset']['selected_labels']:,} objects / "
                f"{len(evidence['quality']['classes'])} classes",
                "Two YOLOv4-tiny logistics checkpoints",
                f"{pairs:,} paired runtime comparisons "
                f"({pairs * 2:,} checkpoint observations)",
            ],
        ),
        (
            "CONTROLLED", MUTED,
            [
                "Same corpus and class vocabulary",
                "Candidate floor 0.001 for AP50",
                "Deployment confidence 0.50",
                "Class-aware NMS IoU 0.30",
                "Same-frame paired timing; decode excluded",
            ],
        ),
        (
            "DECISION", TEAL,
            [
                "1  mAP50: primary quality gate",
                "2  Macro F1: secondary quality gate",
                "3  p95 latency: subordinate gate",
                "4  Mean latency: deterministic tie-break",
                f"Selected Checkpoint {decision['selected_checkpoint']} at step {decision['step']}",
            ],
        ),
    ]
    lefts = [0.055, 0.37, 0.685]
    for left, (heading, color, bullets) in zip(lefts, cards):
        card = FancyBboxPatch(
            (left, 0.12), 0.27, 0.65,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="white", edgecolor=GRID, linewidth=1.2,
            transform=axis.transAxes,
        )
        axis.add_patch(card)
        axis.add_patch(Rectangle(
            (left, 0.70), 0.27, 0.07, transform=axis.transAxes,
            facecolor=color, edgecolor="none",
        ))
        axis.text(left + 0.018, 0.735, heading, transform=axis.transAxes,
                  color="white", fontsize=12, fontweight="bold", va="center")
        y = 0.645
        for bullet in bullets:
            wrapped = textwrap.fill(str(bullet), width=35)
            axis.text(left + 0.022, y, "•", transform=axis.transAxes,
                      color=color, fontsize=14, va="top")
            axis.text(left + 0.044, y, wrapped, transform=axis.transAxes,
                      color=INK, fontsize=10.3, va="top", linespacing=1.25)
            y -= 0.105 if "\n" not in wrapped else 0.135
    return fig


def _save_figure_png(fig, directory, stem):
    png = Path(directory) / f"{stem}.png"
    fig.savefig(
        png, dpi=200, bbox_inches="tight", facecolor="white",
        metadata={"Software": "Matplotlib; Experiment 01 verified figure builder"},
    )
    plt.close(fig)
    _require(png.is_file() and png.stat().st_size > 0, f"PNG was not written: {png}")


def build_figure_package(evidence, output_dir, asset_root=None):
    """Render five required figures and atomically promote their directory."""

    destination = _absolute(output_dir)
    _require(not destination.exists(), f"Refusing to overwrite output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.incomplete"
    _require(not staging.exists(), f"Incomplete figure build already exists: {staging}")
    staging.mkdir()
    class_table = _class_comparison_table(evidence)
    builders = (
        (OUTPUT_STEMS[0], lambda: _decision_summary_figure(evidence)),
        (OUTPUT_STEMS[1], lambda: _class_dumbbell_figure(evidence, class_table)),
        (OUTPUT_STEMS[2], lambda: _delta_support_figure(evidence, class_table)),
        (OUTPUT_STEMS[3], lambda: _latency_ecdf_figure(evidence)),
        (OUTPUT_STEMS[4], lambda: _design_figure(evidence)),
    )
    with _style_context():
        for stem, builder in builders:
            _save_figure_png(builder(), staging, stem)
    expected = {f"{stem}.png" for stem in OUTPUT_STEMS}
    observed = {path.name for path in staging.iterdir() if path.is_file()}
    _require(observed == expected, "Figure package output set is incomplete.")
    if asset_root is not None:
        root = _absolute(asset_root)
        _require(root.is_dir(), f"Optional asset root not found: {root}")
        print("[INFO] Qualitative case figure omitted: the five verified analytical panels "
              "fully cover the selection evidence without introducing a subjective case pick.")
    staging.replace(destination)
    print(f"[COMPLETE] Promoted verified figure package: {destination}")
    for name in sorted(expected):
        print(f"[WRITE] {destination / name}")
    return destination


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build verified publication figures for Experiment 01."
    )
    parser.add_argument("--quality-run", type=Path, required=True,
                        help="Verified strict full-corpus quality run directory.")
    parser.add_argument("--runtime-run", type=Path, required=True,
                        help="Verified paired runtime benchmark directory.")
    parser.add_argument("--selection-run", type=Path, required=True,
                        help="Verified checkpoint-selection run directory.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory to receive the atomic figure package.")
    parser.add_argument(
        "--asset-root", type=Path,
        help="Optional external asset root. Qualitative rendering is intentionally omitted "
             "unless a future evidence protocol predeclares a case-selection rule.",
    )
    parser.add_argument("--expected-images", type=positive_int,
                        default=DEFAULT_EXPECTED_IMAGES)
    parser.add_argument("--expected-labels", type=positive_int,
                        default=DEFAULT_EXPECTED_LABELS)
    parser.add_argument("--expected-classes", type=positive_int,
                        default=DEFAULT_EXPECTED_CLASSES)
    parser.add_argument("--expected-pairs", type=positive_int,
                        default=DEFAULT_EXPECTED_PAIRS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    evidence = load_verified_evidence(
        args.quality_run, args.runtime_run, args.selection_run,
        expected_images=args.expected_images,
        expected_labels=args.expected_labels,
        expected_classes=args.expected_classes,
        expected_pairs=args.expected_pairs,
    )
    return build_figure_package(evidence, args.output_dir, asset_root=args.asset_root)


if __name__ == "__main__":
    main()
