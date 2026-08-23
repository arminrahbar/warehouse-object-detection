"""Build verified publication figures for Experiment 05 error-review queues."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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

import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "outputs" / "05_hard_negative_mining"
)
DEFAULT_COMPONENT_DIR = DEFAULT_EXPERIMENT_DIR / "01_error_components"
DEFAULT_QUEUE_DIR = DEFAULT_EXPERIMENT_DIR / "02_review_queues"

PROFILE_ORDER = (
    "balanced",
    "localization",
    "matched_confidence",
    "false_positive",
    "false_negative",
)
DISPLAY_ORDER = (
    "Mixed-error",
    "Localization",
    "Matched confidence",
    "False positive",
    "False negative",
)
COMPONENTS = (
    "localization_error",
    "confidence_error",
    "false_positive_rate",
    "false_negative_rate",
)
COMPONENT_LABELS = {
    "localization_error": "Localization",
    "confidence_error": "Matched confidence",
    "false_positive_rate": "False-positive rate",
    "false_negative_rate": "False-negative rate",
}
DENSITY_ORDER = ("1", "2-4", "5-9", "10-14", "15-19", "20+")
OUTPUT_STEMS = (
    "01_experiment_design",
    "02_error_component_distribution",
    "03_component_correlation",
    "04_queue_error_profiles",
    "05_queue_overlap",
    "06_queue_scene_profile",
    "07_class_presence",
)

SCHEMAS = {
    "image_error_components_sample5000.csv": (
        "image_file", "image_path", "num_objects", "density_bucket",
        "class_names_present", "localization_error", "confidence_error",
        "false_positive_rate", "false_negative_rate", "prediction_count",
        "ground_truth_count", "matched_prediction_count",
        "false_positive_prediction_count", "matched_gt_count", "missed_gt_count",
        "mean_matched_iou", "mean_matched_confidence",
    ),
    "review_queue_profiles.csv": (
        "profile_name", "display_name", "eligibility_rule",
        "weight_localization_error", "weight_confidence_error",
        "weight_false_positive_rate", "weight_false_negative_rate",
    ),
    "top_images_by_profile.csv": (
        "image_file", "image_path", "num_objects", "density_bucket",
        "class_names_present", "localization_error", "confidence_error",
        "false_positive_rate", "false_negative_rate", "prediction_count",
        "ground_truth_count", "matched_prediction_count",
        "false_positive_prediction_count", "matched_gt_count", "missed_gt_count",
        "mean_matched_iou", "mean_matched_confidence",
        "contribution_localization_error", "contribution_confidence_error",
        "contribution_false_positive_rate", "contribution_false_negative_rate",
        "error_score", "dominant_component", "zero_prediction", "profile_name",
        "display_name", "eligibility_rule", "weight_localization_error",
        "weight_confidence_error", "weight_false_positive_rate",
        "weight_false_negative_rate", "rank",
    ),
    "profile_summary.csv": (
        "profile_name", "display_name", "eligibility_rule", "top_n",
        "mean_error_score", "median_error_score", "mean_num_objects",
        "mean_prediction_count", "mean_false_positive_count",
        "mean_missed_gt_count", "zero_prediction_images",
        "weight_localization_error", "mean_localization_error",
        "weight_confidence_error", "mean_confidence_error",
        "weight_false_positive_rate", "mean_false_positive_rate",
        "weight_false_negative_rate", "mean_false_negative_rate",
    ),
    "profile_overlap.csv": (
        "profile_a", "profile_b", "intersection_count", "jaccard_overlap",
    ),
    "density_by_profile.csv": (
        "profile_name", "display_name", "density_bucket", "image_count", "image_share",
    ),
    "class_presence_by_profile.csv": (
        "profile_name", "display_name", "class_name", "image_count", "image_share",
    ),
}


def _read_exact(directory, name):
    path = Path(directory) / name
    require(path.is_file(), f"Missing Experiment 05 evidence: {path}")
    table = pd.read_csv(path)
    require(
        table.columns.tolist() == list(SCHEMAS[name]),
        f"{name} does not match its locked ordered schema.",
    )
    return table


def _numeric(table, columns, context):
    result = table.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
        require(
            np.isfinite(result[column].to_numpy(dtype=float)).all(),
            f"{context}.{column} contains a non-finite value.",
        )
    return result


def load_verified_evidence(component_dir, queue_dir=None):
    """Load and reconcile the complete Experiment 05 evidence package."""

    component_directory = Path(component_dir).expanduser().absolute()
    queue_directory = (
        Path(queue_dir).expanduser().absolute()
        if queue_dir is not None
        else component_directory
    )
    require(
        component_directory.is_dir(),
        f"Component evidence directory not found: {component_directory}",
    )
    require(
        queue_directory.is_dir(),
        f"Review-queue evidence directory not found: {queue_directory}",
    )
    tables = {}
    for name in SCHEMAS:
        directory = (
            component_directory
            if name == "image_error_components_sample5000.csv"
            else queue_directory
        )
        tables[name] = _read_exact(directory, name)

    components = _numeric(
        tables["image_error_components_sample5000.csv"],
        [
            *COMPONENTS, "num_objects", "prediction_count", "ground_truth_count",
            "matched_prediction_count", "false_positive_prediction_count",
            "matched_gt_count", "missed_gt_count", "mean_matched_iou",
            "mean_matched_confidence",
        ],
        "components",
    )
    require(len(components) == 5000, "Component evidence must contain 5,000 images.")
    require(components["image_file"].is_unique, "Component image identities must be unique.")
    require(int(components["ground_truth_count"].sum()) == 19196, "Unexpected label total.")
    require(int(components["prediction_count"].sum()) == 7727, "Unexpected prediction total.")
    require((components[list(COMPONENTS)] >= 0).all().all(), "Error components must be >= 0.")
    require((components[list(COMPONENTS)] <= 1).all().all(), "Error components must be <= 1.")
    require(
        components["matched_prediction_count"].equals(components["matched_gt_count"]),
        "Matched prediction and label counts differ.",
    )
    require(
        components["false_positive_prediction_count"].equals(
            components["prediction_count"] - components["matched_prediction_count"]
        ),
        "False-positive count identity failed.",
    )
    require(
        components["missed_gt_count"].equals(
            components["ground_truth_count"] - components["matched_gt_count"]
        ),
        "Missed-label count identity failed.",
    )
    require(
        int((components["prediction_count"] == 0).sum()) == 1677,
        "Unexpected complete-miss image count.",
    )

    profiles = _numeric(
        tables["review_queue_profiles.csv"],
        [f"weight_{component}" for component in COMPONENTS],
        "profiles",
    )
    require(profiles["profile_name"].tolist() == list(PROFILE_ORDER), "Profile order changed.")
    require(profiles["display_name"].tolist() == list(DISPLAY_ORDER), "Display labels changed.")

    top = _numeric(
        tables["top_images_by_profile.csv"],
        [
            *COMPONENTS, "num_objects", "prediction_count", "ground_truth_count",
            "matched_prediction_count", "false_positive_prediction_count",
            "matched_gt_count", "missed_gt_count", "mean_matched_iou",
            "mean_matched_confidence", "error_score", "zero_prediction", "rank",
            *[f"weight_{component}" for component in COMPONENTS],
        ],
        "top_images",
    )
    require(len(top) == 1250, "Top-image evidence must contain five queues of 250.")
    for profile in PROFILE_ORDER:
        group = top.loc[top["profile_name"] == profile]
        require(len(group) == 250, f"{profile} queue must contain 250 images.")
        require(group["image_file"].is_unique, f"{profile} queue contains duplicate images.")
        require(group["rank"].astype(int).tolist() == list(range(1, 251)), f"{profile} ranks changed.")

    summary = _numeric(
        tables["profile_summary.csv"],
        [
            "top_n", "mean_error_score", "median_error_score", "mean_num_objects",
            "mean_prediction_count", "mean_false_positive_count",
            "mean_missed_gt_count", "zero_prediction_images",
            *[f"mean_{component}" for component in COMPONENTS],
        ],
        "profile_summary",
    )
    require(summary["profile_name"].tolist() == list(PROFILE_ORDER), "Summary profile order changed.")
    require((summary["top_n"] == 250).all(), "Every queue summary must represent 250 images.")
    for row in summary.itertuples(index=False):
        group = top.loc[top["profile_name"] == row.profile_name]
        checks = {
            "mean_error_score": group["error_score"].mean(),
            "mean_num_objects": group["num_objects"].mean(),
            "mean_prediction_count": group["prediction_count"].mean(),
            "mean_false_positive_count": group["false_positive_prediction_count"].mean(),
            "mean_missed_gt_count": group["missed_gt_count"].mean(),
        }
        for name, expected in checks.items():
            require(
                np.isclose(float(getattr(row, name)), float(expected), rtol=1e-9, atol=1e-10),
                f"profile_summary.{name} does not reconcile for {row.profile_name}.",
            )

    overlap = _numeric(
        tables["profile_overlap.csv"],
        ["intersection_count", "jaccard_overlap"],
        "profile_overlap",
    )
    require(len(overlap) == 25, "Overlap evidence must contain a 5x5 matrix.")
    overlap_pivot = overlap.pivot(index="profile_a", columns="profile_b", values="jaccard_overlap")
    overlap_pivot = overlap_pivot.loc[list(PROFILE_ORDER), list(PROFILE_ORDER)]
    require(np.allclose(overlap_pivot, overlap_pivot.T), "Queue overlap must be symmetric.")
    require(np.allclose(np.diag(overlap_pivot), 1.0), "Queue overlap diagonal must be one.")

    density = _numeric(tables["density_by_profile.csv"], ["image_count", "image_share"], "density")
    require(len(density) == 30, "Density evidence must contain 30 profile-bucket rows.")
    for profile in PROFILE_ORDER:
        group = density.loc[density["profile_name"] == profile]
        require(set(group["density_bucket"]) == set(DENSITY_ORDER), f"Density buckets changed for {profile}.")
        require(int(group["image_count"].sum()) == 250, f"Density counts do not sum to 250 for {profile}.")
        require(np.isclose(group["image_share"].sum(), 1.0), f"Density shares do not sum to one for {profile}.")

    presence = _numeric(
        tables["class_presence_by_profile.csv"], ["image_count", "image_share"], "class_presence"
    )
    require(set(presence["profile_name"]) == set(PROFILE_ORDER), "Class-presence profiles changed.")
    require((presence["image_count"] >= 0).all(), "Class-presence counts must be non-negative.")
    require((presence["image_share"].between(0, 1)).all(), "Class-presence shares must be bounded.")

    tables.update(
        {
            "components": components,
            "profiles": profiles,
            "top": top,
            "summary": summary,
            "overlap": overlap,
            "density": density,
            "presence": presence,
        }
    )
    return tables


def _design_figure(evidence):
    components = evidence["components"]
    complete_misses = int((components["prediction_count"] == 0).sum())
    missed_objects = int(components.loc[components["prediction_count"] == 0, "ground_truth_count"].sum())
    return three_panel_figure(
        "Error-review design at a glance",
        "One reusable image-level evidence table supports five deterministic review objectives.",
        [
            {
                "heading": "Input",
                "bullets": [
                    "5,000 selected evaluation images",
                    "19,196 labels and 7,727 retained predictions",
                    "Checkpoint B at confidence 0.50",
                    "Class-aware NMS IoU 0.30",
                ],
            },
            {
                "heading": "Scoring",
                "bullets": [
                    "Same-class one-to-one matching at IoU 0.50",
                    "Four bounded, interpretable error components",
                    "Eligibility masks distinguish missing from good measurements",
                    "Deterministic count and filename tie-breakers",
                ],
            },
            {
                "heading": "Output",
                "bullets": [
                    "Five separate top-250 review queues",
                    f"{complete_misses:,} complete-miss images preserved",
                    f"{missed_objects:,} labels in complete-miss images",
                    "Triage candidates for manual diagnosis—not automatic retraining",
                ],
            },
        ],
    )


def _eligible_component_statistics(components):
    masks = {
        "localization_error": components["matched_prediction_count"] > 0,
        "confidence_error": components["matched_prediction_count"] > 0,
        "false_positive_rate": components["prediction_count"] > 0,
        "false_negative_rate": components["ground_truth_count"] > 0,
    }
    rows = []
    for component in COMPONENTS:
        values = components.loc[masks[component], component]
        rows.append(
            {
                "component": component,
                "eligible": len(values),
                "p50": values.quantile(0.50),
                "p75": values.quantile(0.75),
                "p95": values.quantile(0.95),
            }
        )
    return pd.DataFrame(rows)


def _distribution_figure(evidence):
    stats = _eligible_component_statistics(evidence["components"])
    stats = stats.iloc[::-1].reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(11.4, 6.6))
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.14, top=0.80)
    add_header(
        fig,
        "Eligible error-component distributions",
        "Dots show the median, 75th percentile, and 95th percentile; unavailable match-based values are excluded.",
    )
    y = np.arange(len(stats))
    axis.hlines(y, stats["p50"], stats["p95"], color=GRID, linewidth=7, zorder=1)
    series = (
        ("p50", "Median", NAVY, "o", -0.09, 75),
        ("p75", "75th percentile", ORANGE, "s", 0.00, 75),
        ("p95", "95th percentile", TEAL, "D", 0.09, 82),
    )
    for column, label, color, marker, offset, size in series:
        axis.scatter(stats[column], y + offset, s=size, color=color, marker=marker, label=label, zorder=3)
        for row_index, value in enumerate(stats[column].to_numpy(dtype=float)):
            near_right = value >= 0.94
            axis.text(
                value - 0.014 if near_right else value + 0.014,
                row_index + offset,
                f"{value:.2f}",
                ha="right" if near_right else "left",
                va="center",
                fontsize=8.5,
                color=INK,
            )
    labels = [f"{COMPONENT_LABELS[row.component]}  (n={int(row.eligible):,})" for row in stats.itertuples()]
    axis.set_yticks(y, labels)
    axis.set_xlim(-0.02, 1.04)
    axis.set_xlabel("Bounded image-level error")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.23), ncol=3)
    clean_axis(axis, grid_axis="x")
    return fig


def _component_correlation(evidence):
    values = evidence["components"][list(COMPONENTS)].copy()
    values.loc[
        evidence["components"]["matched_prediction_count"] == 0,
        ["localization_error", "confidence_error"],
    ] = np.nan
    return values.corr(method="spearman", min_periods=50)


def _correlation_figure(evidence):
    correlation = _component_correlation(evidence)
    labels = [COMPONENT_LABELS[name] for name in COMPONENTS]
    fig, axis = plt.subplots(figsize=(9.2, 7.4))
    fig.subplots_adjust(left=0.24, right=0.91, bottom=0.20, top=0.80)
    add_header(
        fig,
        "Error components are related, not interchangeable",
        "Spearman association; match-dependent components are masked when no match exists.",
    )
    image = axis.imshow(correlation.to_numpy(), cmap="BrBG", vmin=-1, vmax=1)
    axis.set_xticks(range(4), labels, rotation=28, ha="right")
    axis.set_yticks(range(4), labels)
    for row in range(4):
        for column in range(4):
            value = float(correlation.iloc[row, column])
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) >= 0.65 else INK,
                fontweight="bold",
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Spearman correlation")
    return fig


def _queue_profile_figure(evidence):
    summary = evidence["summary"].set_index("profile_name").loc[list(PROFILE_ORDER)]
    matrix = summary[[f"mean_{component}" for component in COMPONENTS]].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(10.8, 7.2))
    fig.subplots_adjust(left=0.21, right=0.93, bottom=0.20, top=0.80)
    add_header(
        fig,
        "Each review queue has a distinct error signature",
        "Cells are mean bounded component values among the 250 images selected by each queue policy.",
    )
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(4), [COMPONENT_LABELS[name] for name in COMPONENTS], rotation=25, ha="right")
    axis.set_yticks(range(5), list(DISPLAY_ORDER))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.70 else INK,
                fontweight="bold",
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean bounded error")
    return fig


def _overlap_figure(evidence):
    overlap = evidence["overlap"].pivot(index="profile_a", columns="profile_b", values="jaccard_overlap")
    overlap = overlap.loc[list(PROFILE_ORDER), list(PROFILE_ORDER)]
    values = overlap.to_numpy(dtype=float)
    mask = np.triu(np.ones_like(values, dtype=bool), k=0)
    display = np.ma.array(values, mask=mask)
    off_diagonal_max = float(values[~np.eye(len(values), dtype=bool)].max())
    fig, axis = plt.subplots(figsize=(9.6, 7.4))
    fig.subplots_adjust(left=0.22, right=0.90, bottom=0.20, top=0.80)
    add_header(
        fig,
        "Targeted queues surface different image populations",
        "Lower triangle shows Jaccard overlap between the five deterministic top-250 image sets.",
    )
    cmap = plt.get_cmap("Blues").with_extremes(bad=PALE)
    image = axis.imshow(display, cmap=cmap, vmin=0, vmax=max(0.30, off_diagonal_max), aspect="equal")
    axis.set_xticks(range(5), list(DISPLAY_ORDER), rotation=28, ha="right")
    axis.set_yticks(range(5), list(DISPLAY_ORDER))
    for row in range(5):
        for column in range(5):
            if row == column:
                axis.text(column, row, "—", ha="center", va="center", color=MUTED)
            elif row > column:
                value = float(values[row, column])
                axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.20 else INK,
                    fontweight="bold",
                )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Jaccard overlap")
    return fig


def _scene_profile_figure(evidence):
    density = evidence["density"].pivot(index="display_name", columns="density_bucket", values="image_share")
    density = density.reindex(index=list(DISPLAY_ORDER), columns=list(DENSITY_ORDER), fill_value=0.0)
    summary = evidence["summary"].set_index("display_name").loc[list(DISPLAY_ORDER)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.8, 7.2), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.18, top=0.78, wspace=0.30)
    add_header(
        fig,
        "Review queues imply different investigation workloads",
        "Left: labeled-object density mix. Right: average labels, predictions, false positives, and misses per selected image.",
    )
    palette = ("#EAF2F8", "#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", NAVY)
    starts = np.zeros(len(density))
    for bucket_index, (bucket, color) in enumerate(zip(DENSITY_ORDER, palette)):
        values = density[bucket].to_numpy(dtype=float)
        left.barh(np.arange(len(density)), values, left=starts, color=color, height=0.63, label=bucket)
        for row_index, (start, value) in enumerate(zip(starts, values)):
            if value >= 0.12:
                left.text(
                    start + value / 2,
                    row_index,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if bucket_index >= 4 else INK,
                    fontweight="bold",
                )
        starts += values
    left.set_yticks(range(5), list(DISPLAY_ORDER))
    left.invert_yaxis()
    left.set_xlim(0, 1)
    left.set_xlabel("Share of queue images")
    left.set_title("Scene density (labels per image)", loc="left", fontweight="bold")
    left.legend(title="Labels", loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3)
    clean_axis(left, grid_axis="x")

    metrics = (
        ("mean_num_objects", "Labels", NAVY, "o"),
        ("mean_prediction_count", "Predictions", ORANGE, "s"),
        ("mean_false_positive_count", "False positives", NEUTRAL, "D"),
        ("mean_missed_gt_count", "Misses", VERMILION, "^"),
    )
    y = np.arange(len(summary))
    offsets = (-0.18, -0.06, 0.06, 0.18)
    for (column, label, color, marker), offset in zip(metrics, offsets):
        values = summary[column].to_numpy(dtype=float)
        right.scatter(values, y + offset, color=color, marker=marker, s=58, label=label, zorder=3)
        for row_index, value in enumerate(values):
            near_right = value >= 19.0
            right.text(
                value - 0.22 if near_right else value + 0.22,
                row_index + offset,
                f"{value:.1f}",
                ha="right" if near_right else "left",
                va="center",
                fontsize=8.2,
                color=INK,
            )
    right.set_yticks(y, list(DISPLAY_ORDER))
    right.invert_yaxis()
    right.set_xlabel("Mean count per selected image")
    right.set_title("Average image profile", loc="left", fontweight="bold")
    right.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2)
    clean_axis(right, grid_axis="x")
    return fig


def _class_presence_figure(evidence):
    presence = evidence["presence"]
    top_classes = presence.groupby("class_name")["image_count"].sum().nlargest(10).index.tolist()
    pivot = presence.loc[presence["class_name"].isin(top_classes)].pivot_table(
        index="class_name", columns="display_name", values="image_share", fill_value=0.0
    )
    pivot = pivot.reindex(index=top_classes, columns=list(DISPLAY_ORDER), fill_value=0.0)
    matrix = pivot.to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(11.5, 7.6))
    fig.subplots_adjust(left=0.18, right=0.92, bottom=0.22, top=0.80)
    add_header(
        fig,
        "Class composition differs across review objectives",
        "Share of each top-250 queue containing the ten classes with the highest aggregate queue presence.",
    )
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=max(0.55, float(matrix.max())), aspect="auto")
    axis.set_xticks(range(5), list(DISPLAY_ORDER), rotation=28, ha="right")
    axis.set_yticks(range(len(top_classes)), top_classes)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            axis.text(
                column,
                row,
                f"{value:.0%}",
                ha="center",
                va="center",
                color="white" if value >= 0.35 else INK,
                fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Share of queue images")
    return fig


def build_figure_package(evidence, output_dir):
    builders = (
        (OUTPUT_STEMS[0], lambda: _design_figure(evidence)),
        (OUTPUT_STEMS[1], lambda: _distribution_figure(evidence)),
        (OUTPUT_STEMS[2], lambda: _correlation_figure(evidence)),
        (OUTPUT_STEMS[3], lambda: _queue_profile_figure(evidence)),
        (OUTPUT_STEMS[4], lambda: _overlap_figure(evidence)),
        (OUTPUT_STEMS[5], lambda: _scene_profile_figure(evidence)),
        (OUTPUT_STEMS[6], lambda: _class_presence_figure(evidence)),
    )
    return build_atomic_package(
        output_dir,
        builders,
        hash_salt="warehouse-object-detection-experiment-05-v1",
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component-dir",
        type=Path,
        default=DEFAULT_COMPONENT_DIR,
        help="Directory containing the image-level error-component table.",
    )
    parser.add_argument(
        "--queue-dir",
        type=Path,
        default=DEFAULT_QUEUE_DIR,
        help="Directory containing the six review-queue evidence tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory to receive the atomic seven-figure PNG/SVG package.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    evidence = load_verified_evidence(args.component_dir, args.queue_dir)
    destination = build_figure_package(evidence, args.output_dir)
    print(f"[COMPLETE] Promoted verified Experiment 05 figures: {destination}")
    for path in sorted(destination.iterdir()):
        print(f"[WRITE] {path}")


if __name__ == "__main__":
    main()
