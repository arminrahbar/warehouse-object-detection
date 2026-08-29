"""Build the locked publication figure package for Experiment 03.

This stage performs no inference and does not recompute NMS or AP. It validates
the preserved threshold-sweep tables, derives five deterministic presentation
views, and atomically promotes a new output directory only after all five PNGs
are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_SCRIPT_DIR = SCRIPT_DIR.parent
if str(SHARED_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPT_DIR))

from report_figure_style import (  # noqa: E402
    GRID,
    INK,
    MUTED,
    NAVY,
    NEUTRAL,
    ORANGE,
    PALE,
    TEAL,
    VERMILION,
    add_header,
    build_atomic_package,
    clean_axis,
    require,
    three_panel_figure,
)


MODEL_NAME = "model2"
DATASET_NAME = "rare_aware_density_stratified_5000"
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7)
SELECTED_THRESHOLD = 0.3
EXPECTED_IMAGES = 5000
EXPECTED_LABELS = 19196
EXPECTED_CLASSES = 20
EXPECTED_CROWDED_IMAGES = 1021
EXPECTED_CROWDED_LABELS = 11411
EXPECTED_DETECTED_IMAGES = 3323

DEFAULT_EVIDENCE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "outputs"
    / "03_nms_thresholding"
    / "01_threshold_sweep"
)

SUMMARY_NAME = "nms_threshold_summary_sample5000.csv"
PER_CLASS_NAME = "per_class_ap_by_threshold_sample5000.csv"
DUPLICATE_NAME = "duplicate_summary_by_threshold_sample5000.csv"
SUBSET_NAME = "subset_summary_by_threshold_sample5000.csv"

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

CANONICAL_HASHES = {
    SUMMARY_NAME: "5e6c266f2efe87da32bb095e4edf1106d7e8df8487d538e4a8500254d1422f3c",
    PER_CLASS_NAME: "ccea15a1b293f90345517a069a5e557801296a556ba27db428ae5e3455560ad3",
    DUPLICATE_NAME: "61b04ec850f26b804de7a35834ead472ad16a9a8a7b1c7722fb1fcdf5415e033",
    SUBSET_NAME: "958e84a50d7c6d17d8ed628a89e6895550c590007514b53d82810da1990b00a6",
}

CANONICAL_MAP = {
    0.2: 0.4015335479763172,
    0.3: 0.40157294334629645,
    0.4: 0.4015134042249227,
    0.5: 0.401490051424598,
    0.55: 0.40133304269039805,
    0.6: 0.40080118172547696,
    0.7: 0.39715903293884636,
}
CANONICAL_PREDICTIONS = {
    0.2: 7685,
    0.3: 7727,
    0.4: 7744,
    0.5: 7758,
    0.55: 7785,
    0.6: 7824,
    0.7: 8032,
}
CANONICAL_DUPLICATE_PAIRS = {
    0.2: 0,
    0.3: 0,
    0.4: 0,
    0.5: 0,
    0.55: 28,
    0.6: 69,
    0.7: 301,
}
CANONICAL_DUPLICATE_IMAGES = {
    0.2: 0,
    0.3: 0,
    0.4: 0,
    0.5: 0,
    0.55: 25,
    0.6: 63,
    0.7: 234,
}
CANONICAL_CROWDED_MAP = {
    0.2: 0.24475410752952081,
    0.3: 0.24477791974946794,
    0.4: 0.2447729968313524,
    0.5: 0.2447701183711562,
    0.55: 0.2447647297143835,
    0.6: 0.2446170242119571,
    0.7: 0.24442044618500458,
}

OUTPUT_STEMS = (
    "01_experiment_design",
    "02_quality_sensitivity",
    "03_quality_output_frontier",
    "04_redundancy_onset",
    "05_class_ap_impact",
)


def _sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv_exact(path, columns, context):
    source = Path(path)
    require(source.is_file(), f"Missing {context}: {source}")
    try:
        table = pd.read_csv(source)
    except Exception as error:
        raise ValueError(f"Unable to read {context}: {error}") from error
    require(
        table.columns.tolist() == list(columns),
        f"{context} schema does not match its locked ordered columns.",
    )
    return table


def _numeric(table, columns, context):
    normalized = table.copy()
    for column in columns:
        try:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        except Exception as error:
            raise ValueError(f"{context}.{column} must be numeric.") from error
        require(
            np.isfinite(normalized[column].to_numpy(dtype=float)).all(),
            f"{context}.{column} contains a non-finite value.",
        )
    return normalized


def _close(observed, expected, context, atol=1e-12):
    require(
        math.isclose(float(observed), float(expected), rel_tol=1e-10, abs_tol=atol),
        f"{context} differs from the locked evidence.",
    )


def _validate_threshold_rows(table, context):
    observed = tuple(table["nms_threshold"].astype(float).tolist())
    require(observed == THRESHOLDS, f"{context} threshold order is invalid.")


def _validate_summary(table):
    require(len(table) == len(THRESHOLDS), "Threshold summary must contain seven rows.")
    table = _numeric(
        table,
        [
            "nms_threshold",
            "mAP@0.5_11_point",
            "total_ground_truth",
            "total_predictions_after_nms",
            "evaluation_rows",
            "score_threshold",
            "map_iou_threshold",
        ],
        "threshold summary",
    )
    _validate_threshold_rows(table, "Threshold summary")
    require(set(table["model"].astype(str)) == {MODEL_NAME}, "Unexpected model identity.")
    require(set(table["dataset"].astype(str)) == {DATASET_NAME}, "Unexpected dataset identity.")
    require(set(table["eval_type"].astype(str)) == {"combined"}, "Unexpected score semantics.")
    require((table["total_ground_truth"] == EXPECTED_LABELS).all(), "Ground-truth total changed.")
    require((table["score_threshold"] == 0.5).all(), "Score threshold changed.")
    require((table["map_iou_threshold"] == 0.5).all(), "Match IoU changed.")
    require(
        np.array_equal(
            table["evaluation_rows"].to_numpy(dtype=int),
            table["total_predictions_after_nms"].to_numpy(dtype=int),
        ),
        "Every retained prediction must enter evaluation.",
    )
    for row in table.itertuples(index=False):
        threshold = float(row.nms_threshold)
        _close(getattr(row, "_3"), CANONICAL_MAP[threshold], f"AP at {threshold:g}")
        require(
            int(row.total_predictions_after_nms) == CANONICAL_PREDICTIONS[threshold],
            f"Prediction count at {threshold:g} changed.",
        )
    require(
        float(table.loc[table["mAP@0.5_11_point"].idxmax(), "nms_threshold"])
        == SELECTED_THRESHOLD,
        "The locked full-sample nominal maximum is no longer 0.30.",
    )
    low = table[table["nms_threshold"] <= 0.5]["mAP@0.5_11_point"]
    _close(float(low.max() - low.min()), 0.00008289192169846915, "0.20-0.50 AP span")
    return table


def _validate_duplicates(table, summary):
    require(len(table) == len(THRESHOLDS), "Duplicate summary must contain seven rows.")
    table = _numeric(table, DUPLICATE_COLUMNS, "duplicate summary")
    _validate_threshold_rows(table, "Duplicate summary")
    summary_lookup = summary.set_index("nms_threshold")
    for row in table.itertuples(index=False):
        threshold = float(row.nms_threshold)
        require(
            int(row.total_predictions_after_nms)
            == int(summary_lookup.loc[threshold, "total_predictions_after_nms"]),
            f"Duplicate summary prediction count differs at {threshold:g}.",
        )
        pair_count = int(row.duplicate_like_pairs_iou_gt_0_5)
        image_count = int(row.images_with_duplicate_like_pairs)
        require(pair_count == CANONICAL_DUPLICATE_PAIRS[threshold], f"Pair count changed at {threshold:g}.")
        require(image_count == CANONICAL_DUPLICATE_IMAGES[threshold], f"Affected-image count changed at {threshold:g}.")
        _close(
            row.mean_duplicate_like_pairs_per_image,
            pair_count / EXPECTED_DETECTED_IMAGES,
            f"Duplicate-pair denominator at {threshold:g}",
        )
    require(
        (table.loc[table["nms_threshold"] <= 0.5, "duplicate_like_pairs_iou_gt_0_5"] == 0).all(),
        "The structural zero region is invalid.",
    )
    return table


def _validate_subsets(table, summary, duplicates):
    require(len(table) == 2 * len(THRESHOLDS), "Subset summary must contain fourteen rows.")
    table = _numeric(
        table,
        [column for column in SUBSET_COLUMNS if column != "subset_name"],
        "subset summary",
    )
    require(set(table["subset_name"].astype(str)) == {"all_selected", "crowded_any_overlap"},
            "Subset identities changed.")
    for subset_name, group in table.groupby("subset_name", sort=False):
        _validate_threshold_rows(group.reset_index(drop=True), f"{subset_name} summary")

    all_selected = table[table["subset_name"] == "all_selected"].reset_index(drop=True)
    crowded = table[table["subset_name"] == "crowded_any_overlap"].reset_index(drop=True)
    require((all_selected["image_count"] == EXPECTED_IMAGES).all(), "Full image count changed.")
    require((all_selected["ground_truth_count"] == EXPECTED_LABELS).all(), "Full label count changed.")
    require((crowded["image_count"] == EXPECTED_CROWDED_IMAGES).all(), "Crowded image count changed.")
    require((crowded["ground_truth_count"] == EXPECTED_CROWDED_LABELS).all(), "Crowded label count changed.")

    for row in all_selected.itertuples(index=False):
        threshold = float(row.nms_threshold)
        _close(getattr(row, "_4"), CANONICAL_MAP[threshold], f"Full subset AP at {threshold:g}")
    for row in crowded.itertuples(index=False):
        threshold = float(row.nms_threshold)
        _close(getattr(row, "_4"), CANONICAL_CROWDED_MAP[threshold], f"Crowded AP at {threshold:g}")
    require(
        float(crowded.loc[crowded["mAP@0.5_11_point"].idxmax(), "nms_threshold"])
        == SELECTED_THRESHOLD,
        "The crowded nominal maximum is no longer 0.30.",
    )

    summary_columns = ["nms_threshold", "mAP@0.5_11_point", "total_predictions_after_nms", "evaluation_rows"]
    pd.testing.assert_frame_equal(
        all_selected[summary_columns],
        summary[summary_columns].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    duplicate_columns = [
        "nms_threshold",
        "duplicate_like_pairs_iou_gt_0_5",
        "images_with_duplicate_like_pairs",
        "mean_duplicate_like_pairs_per_image",
        "total_predictions_after_nms",
    ]
    pd.testing.assert_frame_equal(
        all_selected[duplicate_columns],
        duplicates[duplicate_columns],
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    return table


def _validate_per_class(table, summary):
    expected_rows = len(THRESHOLDS) * EXPECTED_CLASSES
    require(len(table) == expected_rows, f"Per-class table must contain {expected_rows} rows.")
    table = _numeric(
        table,
        ["nms_threshold", "class_id", "ground_truth_count", "prediction_count", "ap_11_point"],
        "per-class table",
    )
    require(set(table["model"].astype(str)) == {MODEL_NAME}, "Unexpected per-class model identity.")
    require(set(table["dataset"].astype(str)) == {DATASET_NAME}, "Unexpected per-class dataset identity.")
    require(table["ap_11_point"].between(0, 1).all(), "Per-class AP must be between zero and one.")
    baseline_identity = None
    summary_lookup = summary.set_index("nms_threshold")
    for threshold, group in table.groupby("nms_threshold", sort=False):
        group = group.sort_values("class_id").reset_index(drop=True)
        require(group["class_id"].astype(int).tolist() == list(range(EXPECTED_CLASSES)),
                f"Class IDs are incomplete at {float(threshold):g}.")
        identity = tuple(zip(group["class_id"].astype(int), group["class_name"].astype(str)))
        if baseline_identity is None:
            baseline_identity = identity
        require(identity == baseline_identity, "Class vocabulary changes across thresholds.")
        require(int(group["ground_truth_count"].sum()) == EXPECTED_LABELS,
                f"Class support does not sum to {EXPECTED_LABELS} at {float(threshold):g}.")
        require(
            int(group["prediction_count"].sum())
            == int(summary_lookup.loc[float(threshold), "total_predictions_after_nms"]),
            f"Class prediction counts do not reconcile at {float(threshold):g}.",
        )
    return table.sort_values(["nms_threshold", "class_id"]).reset_index(drop=True)


def load_verified_evidence(evidence_dir, *, locked_hashes=CANONICAL_HASHES):
    """Validate the exact preserved sweep tables used by the publication report."""

    directory = Path(evidence_dir).expanduser().absolute()
    require(directory.is_dir(), f"Evidence directory not found: {directory}")
    schemas = {
        SUMMARY_NAME: SUMMARY_COLUMNS,
        PER_CLASS_NAME: PER_CLASS_COLUMNS,
        DUPLICATE_NAME: DUPLICATE_COLUMNS,
        SUBSET_NAME: SUBSET_COLUMNS,
    }
    if locked_hashes is not None:
        require(set(locked_hashes) == set(schemas), "Locked hash set is incomplete.")
        for name, expected_hash in locked_hashes.items():
            path = directory / name
            require(path.is_file(), f"Missing locked evidence: {path}")
            require(_sha256_file(path) == expected_hash, f"Locked evidence hash mismatch: {name}")

    summary = _validate_summary(_read_csv_exact(directory / SUMMARY_NAME, SUMMARY_COLUMNS, "threshold summary"))
    duplicates = _validate_duplicates(
        _read_csv_exact(directory / DUPLICATE_NAME, DUPLICATE_COLUMNS, "duplicate summary"),
        summary,
    )
    subsets = _validate_subsets(
        _read_csv_exact(directory / SUBSET_NAME, SUBSET_COLUMNS, "subset summary"),
        summary,
        duplicates,
    )
    per_class = _validate_per_class(
        _read_csv_exact(directory / PER_CLASS_NAME, PER_CLASS_COLUMNS, "per-class table"),
        summary,
    )
    return {
        "directory": directory,
        "summary": summary,
        "duplicates": duplicates,
        "subsets": subsets,
        "per_class": per_class,
    }


def _selected_value(table, value_column, subset_name=None):
    source = table
    if subset_name is not None:
        source = source[source["subset_name"] == subset_name]
    return float(source.loc[source["nms_threshold"] == SELECTED_THRESHOLD, value_column].iloc[0])


def derive_class_impact(per_class):
    """Return deterministic 0.30-to-0.70 class AP changes with support."""

    baseline = per_class[per_class["nms_threshold"] == SELECTED_THRESHOLD][
        ["class_id", "class_name", "ground_truth_count", "prediction_count", "ap_11_point"]
    ].rename(columns={"prediction_count": "predictions_030", "ap_11_point": "ap_030"})
    permissive = per_class[per_class["nms_threshold"] == 0.7][
        ["class_id", "class_name", "ground_truth_count", "prediction_count", "ap_11_point"]
    ].rename(columns={"prediction_count": "predictions_070", "ap_11_point": "ap_070"})
    merged = baseline.merge(
        permissive,
        on=["class_id", "class_name", "ground_truth_count"],
        validate="one_to_one",
    )
    merged["delta_pp"] = 100.0 * (merged["ap_070"] - merged["ap_030"])
    return merged.sort_values(["delta_pp", "class_id"]).reset_index(drop=True)


def pareto_thresholds(summary):
    """Return thresholds not dominated on higher AP and lower output count."""

    rows = summary[["nms_threshold", "mAP@0.5_11_point", "total_predictions_after_nms"]]
    result = []
    for row in rows.itertuples(index=False):
        dominated = False
        for other in rows.itertuples(index=False):
            no_worse = (
                float(getattr(other, "_1")) >= float(getattr(row, "_1"))
                and int(other.total_predictions_after_nms) <= int(row.total_predictions_after_nms)
            )
            strictly_better = (
                float(getattr(other, "_1")) > float(getattr(row, "_1"))
                or int(other.total_predictions_after_nms) < int(row.total_predictions_after_nms)
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(float(row.nms_threshold))
    return tuple(result)


def _design_figure(evidence):
    return three_panel_figure(
        "Experiment design at a glance",
        "One checkpoint and one deterministic workload; only the same-class suppression boundary changed.",
        [
            {
                "heading": "Input",
                "bullets": [
                    "Checkpoint B configuration",
                    "Fixed 5,000-image workload",
                    "19,196 labels across 20 classes",
                    "Reused decoded-candidate table",
                ],
            },
            {
                "heading": "Fixed controls",
                "bullets": [
                    "Confidence floor fixed at 0.50",
                    "Seven class-aware NMS IoU thresholds",
                    "Same-class matching at IoU 0.50",
                    "11-point AP50 over 20 classes",
                ],
            },
            {
                "heading": "Decision",
                "bullets": [
                    "Keep 0.30 as the provisional default",
                    "Treat 0.20 as a compact near-equivalent",
                    "Require new evidence above 0.50",
                    "Use 0.30 in downstream analyses",
                ],
            },
        ],
    )


def _quality_sensitivity_figure(evidence):
    table = evidence["subsets"].copy()
    labels = {"all_selected": "All selected images", "crowded_any_overlap": "Crowded sensitivity view"}
    colors = {"all_selected": NAVY, "crowded_any_overlap": ORANGE}
    fig, axis = plt.subplots(figsize=(11.8, 6.8))
    fig.subplots_adjust(left=0.11, right=0.90, bottom=0.14, top=0.80)
    add_header(
        fig,
        "Detection quality is stable through NMS IoU 0.50",
        "Change from each view's score at 0.30; shared scale in AP percentage points.",
    )
    axis.axhline(0, color=INK, linewidth=0.9)
    axis.axvspan(0.50, 0.72, color=PALE, zorder=0)
    for subset_name in ("all_selected", "crowded_any_overlap"):
        group = table[table["subset_name"] == subset_name].sort_values("nms_threshold")
        baseline = _selected_value(table, "mAP@0.5_11_point", subset_name)
        values = 100.0 * (group["mAP@0.5_11_point"].to_numpy(dtype=float) - baseline)
        thresholds = group["nms_threshold"].to_numpy(dtype=float)
        color = colors[subset_name]
        axis.plot(thresholds, values, color=color, linewidth=2.2, marker="o", markersize=5.5)
        selected_position = int(np.flatnonzero(np.isclose(thresholds, SELECTED_THRESHOLD))[0])
        axis.scatter(
            [thresholds[selected_position]], [values[selected_position]],
            color=TEAL, s=72, edgecolor="white", linewidth=1.0, zorder=4,
        )
        endpoint = float(values[-1])
        axis.annotate(
            f"{labels[subset_name]}  {endpoint:+.3f} pp",
            (float(thresholds[-1]), endpoint),
            xytext=(10, 0), textcoords="offset points", color=color,
            fontsize=9.5, fontweight="bold", ha="left", va="center",
        )
    axis.text(
        0.325,
        0.018,
        "Selected 0.30\n(common nominal maximum)",
        color=TEAL,
        fontsize=9.2,
        fontweight="bold",
        ha="left",
    )
    axis.text(0.61, -0.47, "More permissive region", color=MUTED, ha="center", fontsize=9)
    axis.set_xlim(0.18, 0.77)
    axis.set_ylim(-0.50, 0.045)
    axis.set_xticks(THRESHOLDS, [f"{value:.2f}" for value in THRESHOLDS])
    axis.set_xlabel("Class-aware NMS IoU threshold")
    axis.set_ylabel("AP50 change from 0.30 (percentage points)")
    clean_axis(axis, grid_axis="both", keep_left=True)
    return fig


def _frontier_figure(evidence):
    table = evidence["summary"].sort_values("nms_threshold").copy()
    baseline_map = _selected_value(table, "mAP@0.5_11_point")
    baseline_predictions = int(_selected_value(table, "total_predictions_after_nms"))
    table["delta_pp"] = 100.0 * (table["mAP@0.5_11_point"] - baseline_map)
    table["prediction_delta"] = table["total_predictions_after_nms"] - baseline_predictions
    pareto = set(pareto_thresholds(table))

    fig, axis = plt.subplots(figsize=(11.5, 6.8))
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.14, top=0.80)
    add_header(
        fig,
        "Two operating points define the tested quality/output frontier",
        "Higher is better for AP50; farther left emits fewer predictions. Values are relative to 0.30.",
    )
    axis.axhline(0, color=GRID, linewidth=0.9)
    axis.axvline(0, color=GRID, linewidth=0.9)
    for row in table.itertuples(index=False):
        threshold = float(row.nms_threshold)
        if threshold == SELECTED_THRESHOLD:
            color, size = TEAL, 92
        elif threshold in pareto:
            color, size = NAVY, 82
        elif threshold > 0.5:
            color, size = VERMILION, 70
        else:
            color, size = NEUTRAL, 66
        x = float(row.prediction_delta)
        y = float(row.delta_pp)
        axis.scatter([x], [y], s=size, color=color, edgecolor="white", linewidth=0.9, zorder=3)
        if threshold in {0.2, 0.3}:
            continue
        offset = {
            0.4: (8, -15), 0.5: (8, 8),
            0.55: (8, 8), 0.6: (8, 8), 0.7: (-7, 8),
        }[threshold]
        axis.annotate(
            f"{threshold:.2f}", (x, y), xytext=offset, textcoords="offset points",
            fontsize=9.2, color=INK,
            ha="right" if threshold in {0.2, 0.7} else "left",
        )
    axis.text(
        -8,
        0.022,
        "0.20 · compact option",
        color=NAVY,
        fontsize=9.2,
        fontweight="bold",
        ha="right",
    )
    axis.text(
        8,
        0.022,
        "0.30 · selected maximum",
        color=TEAL,
        fontsize=9.2,
        fontweight="bold",
        ha="left",
    )
    axis.set_xlim(-75, 335)
    axis.set_ylim(-0.49, 0.045)
    axis.set_xlabel("Change in retained predictions vs 0.30")
    axis.set_ylabel("AP50 change vs 0.30 (percentage points)")
    clean_axis(axis, grid_axis="both", keep_left=True)
    return fig


def _redundancy_figure(evidence):
    table = evidence["duplicates"].sort_values("nms_threshold").copy()
    x = table["nms_threshold"].to_numpy(dtype=float)
    pairs = table["duplicate_like_pairs_iou_gt_0_5"].to_numpy(dtype=float)
    images = table["images_with_duplicate_like_pairs"].to_numpy(dtype=float)
    offset = 0.008

    fig, axis = plt.subplots(figsize=(11.5, 6.8))
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.14, top=0.80)
    add_header(
        fig,
        "High-overlap same-class pairs emerge only above the 0.50 boundary",
        "Counts are a geometric redundancy diagnostic, not proof that two boxes describe one object.",
    )
    axis.axvspan(0.18, 0.50, color=PALE, zorder=0)
    axis.text(0.34, 315, "Structural zero region", color=MUTED, ha="center", fontsize=9.2)
    axis.vlines(x - offset, 0, pairs, color=VERMILION, linewidth=2.2)
    axis.vlines(x + offset, 0, images, color=NAVY, linewidth=2.2)
    axis.scatter(x - offset, pairs, color=VERMILION, s=55, zorder=3)
    axis.scatter(x + offset, images, color=NAVY, s=55, zorder=3, marker="s")
    for threshold, pair_count, image_count in zip(x, pairs, images):
        if pair_count <= 0:
            continue
        axis.annotate(
            f"{int(pair_count)}", (threshold - offset, pair_count),
            xytext=(-5, 8), textcoords="offset points", color=VERMILION,
            fontsize=9, ha="right", fontweight="bold",
        )
        axis.annotate(
            f"{int(image_count)}", (threshold + offset, image_count),
            xytext=(5, 8), textcoords="offset points", color=NAVY,
            fontsize=9, ha="left", fontweight="bold",
        )
    axis.annotate(
        "Same-class pairs", (0.7 - offset, pairs[-1]), xytext=(-8, 26),
        textcoords="offset points", color=VERMILION, fontsize=9.2,
        fontweight="bold", ha="right",
    )
    axis.annotate(
        "Affected images", (0.7 + offset, images[-1]), xytext=(8, -18),
        textcoords="offset points", color=NAVY, fontsize=9.2,
        fontweight="bold", ha="left",
    )
    axis.set_xlim(0.18, 0.76)
    axis.set_ylim(0, 340)
    axis.set_xticks(THRESHOLDS, [f"{value:.2f}" for value in THRESHOLDS])
    axis.set_xlabel("Class-aware NMS IoU threshold")
    axis.set_ylabel("Count")
    clean_axis(axis, grid_axis="y", keep_left=True)
    return fig


def _class_impact_figure(evidence):
    table = derive_class_impact(evidence["per_class"])
    fig, axis = plt.subplots(figsize=(11.8, 8.6))
    fig.subplots_adjust(left=0.28, right=0.95, bottom=0.10, top=0.84)
    add_header(
        fig,
        "Class-level effect of loosening NMS from 0.30 to 0.70",
        "14 classes declined, 6 were unchanged, and none improved; support is the selected-sample label count.",
    )
    y = np.arange(len(table))
    values = table["delta_pp"].to_numpy(dtype=float)
    colors = np.where(values < -1e-10, VERMILION, NEUTRAL)
    axis.axvline(0, color=INK, linewidth=0.9)
    axis.hlines(y, 0, values, color=GRID, linewidth=2.0, zorder=1)
    axis.scatter(values, y, color=colors, s=52, edgecolor="white", linewidth=0.7, zorder=3)
    labels = [
        f"{row.class_name}  (n={int(row.ground_truth_count):,})"
        for row in table.itertuples(index=False)
    ]
    axis.set_yticks(y, labels)
    axis.set_ylim(-0.7, len(table) - 0.3)
    axis.set_xlim(-2.9, 0.18)
    axis.set_xlabel("11-point AP50 change, 0.70 − 0.30 (percentage points)")
    for position, row in enumerate(table.itertuples(index=False)):
        if position >= 5:
            break
        axis.annotate(
            f"{float(row.delta_pp):.2f} pp",
            (float(row.delta_pp), position), xytext=(-7, 0), textcoords="offset points",
            color=VERMILION, fontsize=9, fontweight="bold", ha="right", va="center",
        )
    clean_axis(axis, grid_axis="x")
    return fig


def build_figure_package(evidence, output_dir):
    builders = (
        (OUTPUT_STEMS[0], lambda: _design_figure(evidence)),
        (OUTPUT_STEMS[1], lambda: _quality_sensitivity_figure(evidence)),
        (OUTPUT_STEMS[2], lambda: _frontier_figure(evidence)),
        (OUTPUT_STEMS[3], lambda: _redundancy_figure(evidence)),
        (OUTPUT_STEMS[4], lambda: _class_impact_figure(evidence)),
    )
    return build_atomic_package(
        output_dir,
        builders,
        hash_salt="warehouse-object-detection-exp03-publication-v1",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build locked publication figures for Experiment 03."
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
        help="Directory containing the four locked Experiment 03 summary CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory to receive the atomic five-figure PNG package.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    evidence = load_verified_evidence(args.evidence_dir)
    destination = build_figure_package(evidence, args.output_dir)
    print(f"[COMPLETE] Promoted verified Experiment 03 figures: {destination}")
    for path in sorted(destination.iterdir()):
        print(f"[WRITE] {path}")
    return destination


if __name__ == "__main__":
    main()
