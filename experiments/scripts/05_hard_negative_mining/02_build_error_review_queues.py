"""Create targeted detector-error review queues from image-level evidence."""

import argparse
import ast
import json
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
    score_error_components,
)


DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "outputs" / "05_hard_negative_mining"
)
DEFAULT_COMPONENT_DIR = DEFAULT_EXPERIMENT_DIR / "01_error_components"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "02_review_queues"
DEFAULT_COMPONENT_PATH = DEFAULT_COMPONENT_DIR / "image_error_components_sample5000.csv"
DEFAULT_FIGURE_DIR = (
    PROJECT_ROOT / "scratch" / "diagnostic-figures" / "05_hard_negative_mining"
)
DEFAULT_TOP_N = 250
DENSITY_BUCKETS = ("1", "2-4", "5-9", "10-14", "15-19", "20+")
FIGURE_FILENAMES = (
    "01_error_component_distribution.png",
    "02_error_component_correlation.png",
    "03_review_queue_overlap.png",
    "04_scene_density_by_queue.png",
    "05_image_profile_by_queue.png",
    "06_error_profile_by_queue.png",
    "07_class_presence_by_queue.png",
)

COMPONENT_LABELS = {
    "localization_error": "Localization",
    "confidence_error": "Matched confidence",
    "false_positive_rate": "False-positive rate",
    "false_negative_rate": "False-negative rate",
}

PROFILES = (
    {
        "profile_name": "balanced",
        "display_name": "Mixed-error",
        "eligibility": "all",
        "weights": dict.fromkeys(ERROR_COMPONENT_COLUMNS, 1.0),
        "tie_breakers": ("missed_gt_count", "false_positive_prediction_count"),
    },
    {
        "profile_name": "localization",
        "display_name": "Localization",
        "eligibility": "matched",
        "weights": {
            "localization_error": 1.0,
            "confidence_error": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
        },
        "tie_breakers": ("matched_prediction_count",),
    },
    {
        "profile_name": "matched_confidence",
        "display_name": "Matched confidence",
        "eligibility": "matched",
        "weights": {
            "localization_error": 0.0,
            "confidence_error": 1.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
        },
        "tie_breakers": ("matched_prediction_count",),
    },
    {
        "profile_name": "false_positive",
        "display_name": "False positive",
        "eligibility": "predictions",
        "weights": {
            "localization_error": 0.0,
            "confidence_error": 0.0,
            "false_positive_rate": 1.0,
            "false_negative_rate": 0.0,
        },
        "tie_breakers": ("false_positive_prediction_count",),
    },
    {
        "profile_name": "false_negative",
        "display_name": "False negative",
        "eligibility": "ground_truth",
        "weights": {
            "localization_error": 0.0,
            "confidence_error": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 1.0,
        },
        "tie_breakers": ("zero_prediction", "missed_gt_count"),
    },
)

BASE_COMPONENT_COLUMNS = [
    "image_file",
    "image_path",
    "num_objects",
    "density_bucket",
    "class_names_present",
    *ERROR_COMPONENT_COLUMNS,
    "prediction_count",
    "ground_truth_count",
    "matched_prediction_count",
    "false_positive_prediction_count",
    "matched_gt_count",
    "missed_gt_count",
    "mean_matched_iou",
    "mean_matched_confidence",
]
PROFILE_COLUMNS = [
    "profile_name",
    "display_name",
    "eligibility_rule",
    *[f"weight_{component}" for component in ERROR_COMPONENT_COLUMNS],
]
SUMMARY_COLUMNS = [
    "profile_name",
    "display_name",
    "eligibility_rule",
    "top_n",
    "mean_error_score",
    "median_error_score",
    "mean_num_objects",
    "mean_prediction_count",
    "mean_false_positive_count",
    "mean_missed_gt_count",
    "zero_prediction_images",
]
for _component in ERROR_COMPONENT_COLUMNS:
    SUMMARY_COLUMNS.extend([f"weight_{_component}", f"mean_{_component}"])
OVERLAP_COLUMNS = ["profile_a", "profile_b", "intersection_count", "jaccard_overlap"]
DENSITY_COLUMNS = [
    "profile_name",
    "display_name",
    "density_bucket",
    "image_count",
    "image_share",
]
CLASS_PRESENCE_COLUMNS = [
    "profile_name",
    "display_name",
    "class_name",
    "image_count",
    "image_share",
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


def parse_class_names(value):
    """Decode class-presence metadata written as JSON or a Python list."""

    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [value]


def load_component_table(path):
    """Load image-level components and verify their count/error invariants."""

    source = Path(path).expanduser().absolute()
    if not source.is_file():
        raise FileNotFoundError(
            f"Component file not found: {source}. Run 01_build_error_components.py first."
        )
    table = pd.read_csv(source)
    _required_columns(table, BASE_COMPONENT_COLUMNS, "Component table")
    if table.empty:
        raise ValueError("Component table is empty.")
    table = table[BASE_COMPONENT_COLUMNS].copy()
    if table["image_file"].isna().any() or table["image_file"].duplicated().any():
        raise ValueError("Component table must contain unique image identifiers.")
    numeric_columns = [
        "num_objects",
        *ERROR_COMPONENT_COLUMNS,
        "prediction_count",
        "ground_truth_count",
        "matched_prediction_count",
        "false_positive_prediction_count",
        "matched_gt_count",
        "missed_gt_count",
        "mean_matched_iou",
        "mean_matched_confidence",
    ]
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if not np.isfinite(table[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Component table contains non-finite numeric values.")
    count_columns = [
        "num_objects",
        "prediction_count",
        "ground_truth_count",
        "matched_prediction_count",
        "false_positive_prediction_count",
        "matched_gt_count",
        "missed_gt_count",
    ]
    for column in count_columns:
        table[column] = _integer_series(table[column], column)
    for column in [*ERROR_COMPONENT_COLUMNS, "mean_matched_iou", "mean_matched_confidence"]:
        if not table[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} values must be within [0, 1].")
    identities = (
        table["num_objects"].equals(table["ground_truth_count"])
        and table["matched_prediction_count"].equals(table["matched_gt_count"])
        and table["false_positive_prediction_count"].equals(
            table["prediction_count"] - table["matched_prediction_count"]
        )
        and table["missed_gt_count"].equals(
            table["ground_truth_count"] - table["matched_gt_count"]
        )
    )
    if not identities:
        raise ValueError("Component table count identities are inconsistent.")
    if not set(table["density_bucket"].astype(str)).issubset(DENSITY_BUCKETS):
        raise ValueError("Component table contains an unsupported density bucket.")
    if table["class_names_present"].isna().any():
        raise ValueError("Component table contains missing class-presence metadata.")
    no_matches = table["matched_prediction_count"] == 0
    if not (
        table.loc[no_matches, ["localization_error", "confidence_error"]] == 0.0
    ).all().all():
        raise ValueError("Match-dependent errors must be zero when no match exists.")
    return table


def _eligible_rows(scored, eligibility):
    if eligibility == "all":
        return scored
    if eligibility == "matched":
        return scored[scored["matched_prediction_count"] > 0]
    if eligibility == "predictions":
        return scored[scored["prediction_count"] > 0]
    if eligibility == "ground_truth":
        return scored[scored["ground_truth_count"] > 0]
    raise ValueError(f"Unsupported eligibility rule: {eligibility}")


def build_top_samples(component_table, top_n):
    """Score, filter, and deterministically rank each targeted review queue."""

    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    samples = []
    for profile in PROFILES:
        scored = score_error_components(component_table, profile["weights"])
        scored["zero_prediction"] = (scored["prediction_count"] == 0).astype(int)
        scored = _eligible_rows(scored, profile["eligibility"]).copy()
        scored["profile_name"] = profile["profile_name"]
        scored["display_name"] = profile["display_name"]
        scored["eligibility_rule"] = profile["eligibility"]
        for component in ERROR_COMPONENT_COLUMNS:
            scored[f"weight_{component}"] = profile["weights"][component]
        sort_columns = ["error_score", *profile["tie_breakers"], "image_file"]
        ascending = [False] * (len(sort_columns) - 1) + [True]
        scored = scored.sort_values(
            sort_columns,
            ascending=ascending,
            kind="stable",
        ).head(top_n).copy()
        scored["rank"] = np.arange(1, len(scored) + 1, dtype=int)
        samples.append(scored)
    return pd.concat(samples, ignore_index=True)


def build_profile_table():
    rows = []
    for profile in PROFILES:
        rows.append(
            {
                "profile_name": profile["profile_name"],
                "display_name": profile["display_name"],
                "eligibility_rule": profile["eligibility"],
                **{
                    f"weight_{component}": profile["weights"][component]
                    for component in ERROR_COMPONENT_COLUMNS
                },
            }
        )
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


def summarize_profiles(top_samples):
    rows = []
    for profile in PROFILES:
        group = top_samples[top_samples["profile_name"] == profile["profile_name"]]
        row = {
            "profile_name": profile["profile_name"],
            "display_name": profile["display_name"],
            "eligibility_rule": profile["eligibility"],
            "top_n": len(group),
            "mean_error_score": group["error_score"].mean(),
            "median_error_score": group["error_score"].median(),
            "mean_num_objects": group["num_objects"].mean(),
            "mean_prediction_count": group["prediction_count"].mean(),
            "mean_false_positive_count": group["false_positive_prediction_count"].mean(),
            "mean_missed_gt_count": group["missed_gt_count"].mean(),
            "zero_prediction_images": int((group["prediction_count"] == 0).sum()),
        }
        for component in ERROR_COMPONENT_COLUMNS:
            row[f"weight_{component}"] = profile["weights"][component]
            row[f"mean_{component}"] = group[component].mean()
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_overlap(top_samples):
    selected = {
        profile: set(group["image_file"])
        for profile, group in top_samples.groupby("profile_name", sort=True)
    }
    rows = []
    for profile_a, images_a in selected.items():
        for profile_b, images_b in selected.items():
            intersection = len(images_a & images_b)
            union = len(images_a | images_b)
            rows.append(
                {
                    "profile_a": profile_a,
                    "profile_b": profile_b,
                    "intersection_count": intersection,
                    "jaccard_overlap": intersection / union if union else 0.0,
                }
            )
    return pd.DataFrame(rows, columns=OVERLAP_COLUMNS)


def build_density(top_samples):
    rows = []
    for profile in PROFILES:
        group = top_samples[top_samples["profile_name"] == profile["profile_name"]]
        counts = group["density_bucket"].value_counts()
        for bucket in DENSITY_BUCKETS:
            count = int(counts.get(bucket, 0))
            rows.append(
                {
                    "profile_name": profile["profile_name"],
                    "display_name": profile["display_name"],
                    "density_bucket": bucket,
                    "image_count": count,
                    "image_share": count / len(group) if len(group) else 0.0,
                }
            )
    return pd.DataFrame(rows, columns=DENSITY_COLUMNS)


def build_class_presence(top_samples):
    rows = []
    for profile in PROFILES:
        group = top_samples[top_samples["profile_name"] == profile["profile_name"]]
        counts = {}
        for value in group["class_names_present"]:
            for class_name in parse_class_names(value):
                counts[class_name] = counts.get(class_name, 0) + 1
        for class_name, count in counts.items():
            rows.append(
                {
                    "profile_name": profile["profile_name"],
                    "display_name": profile["display_name"],
                    "class_name": class_name,
                    "image_count": count,
                    "image_share": count / len(group) if len(group) else 0.0,
                }
            )
    return pd.DataFrame(rows, columns=CLASS_PRESENCE_COLUMNS)


def build_artifacts(component_table, top_n):
    top_samples = build_top_samples(component_table, top_n)
    return {
        "review_queue_profiles.csv": build_profile_table(),
        "top_images_by_profile.csv": top_samples,
        "profile_summary.csv": summarize_profiles(top_samples),
        "profile_overlap.csv": build_overlap(top_samples),
        "density_by_profile.csv": build_density(top_samples),
        "class_presence_by_profile.csv": build_class_presence(top_samples),
    }


def _profile_names():
    return [profile["profile_name"] for profile in PROFILES]


def _display_names():
    return [profile["display_name"] for profile in PROFILES]


def build_figures(component_table, artifacts, top_n, figure_dir):
    """Render seven complementary diagnostics from validated queue evidence."""

    import matplotlib.pyplot as plt

    directory = Path(figure_dir).expanduser().absolute()
    directory.mkdir(parents=True, exist_ok=True)
    paths = []

    eligible_values = {
        "localization_error": component_table.loc[
            component_table["matched_prediction_count"] > 0,
            "localization_error",
        ],
        "confidence_error": component_table.loc[
            component_table["matched_prediction_count"] > 0,
            "confidence_error",
        ],
        "false_positive_rate": component_table.loc[
            component_table["prediction_count"] > 0,
            "false_positive_rate",
        ],
        "false_negative_rate": component_table.loc[
            component_table["ground_truth_count"] > 0,
            "false_negative_rate",
        ],
    }
    statistics = pd.DataFrame(
        {
            "Median": {name: values.median() for name, values in eligible_values.items()},
            "75th percentile": {
                name: values.quantile(0.75) for name, values in eligible_values.items()
            },
            "95th percentile": {
                name: values.quantile(0.95) for name, values in eligible_values.items()
            },
        }
    )
    statistics.index = [COMPONENT_LABELS[name] for name in statistics.index]
    axis = statistics.plot(kind="bar", figsize=(10, 6))
    axis.set_ylabel("Bounded image-level error")
    axis.set_xlabel("Error component")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Detector error component distribution")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = directory / FIGURE_FILENAMES[0]
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    paths.append(path)

    components = list(ERROR_COMPONENT_COLUMNS)
    correlation_input = component_table[components].copy()
    correlation_input.loc[
        component_table["matched_prediction_count"] == 0,
        ["localization_error", "confidence_error"],
    ] = np.nan
    correlation = correlation_input.corr(method="spearman", min_periods=50)
    labels = [COMPONENT_LABELS[component] for component in components]
    plt.figure(figsize=(8, 7))
    plt.imshow(correlation.values, aspect="auto", vmin=-1, vmax=1)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(label="Spearman correlation")
    plt.title("Association between image-level error components")
    for row_index in range(len(components)):
        for column_index in range(len(components)):
            plt.text(
                column_index,
                row_index,
                f"{correlation.values[row_index, column_index]:.2f}",
                ha="center",
                va="center",
            )
    plt.tight_layout()
    path = directory / FIGURE_FILENAMES[1]
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    paths.append(path)

    overlap = artifacts["profile_overlap.csv"]
    profile_names = _profile_names()
    display_names = _display_names()
    overlap_matrix = overlap.pivot(
        index="profile_a",
        columns="profile_b",
        values="jaccard_overlap",
    ).loc[profile_names, profile_names]
    plt.figure(figsize=(8, 7))
    plt.imshow(overlap_matrix.values, aspect="auto", vmin=0, vmax=1)
    plt.xticks(range(len(display_names)), display_names, rotation=40, ha="right")
    plt.yticks(range(len(display_names)), display_names)
    plt.colorbar(label="Jaccard overlap")
    plt.title(f"Top-{top_n} overlap across review queues")
    for row_index in range(len(profile_names)):
        for column_index in range(len(profile_names)):
            plt.text(
                column_index,
                row_index,
                f"{overlap_matrix.values[row_index, column_index]:.2f}",
                ha="center",
                va="center",
            )
    plt.tight_layout()
    path = directory / FIGURE_FILENAMES[2]
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    paths.append(path)

    density = artifacts["density_by_profile.csv"]
    density_pivot = density.pivot(
        index="display_name",
        columns="density_bucket",
        values="image_share",
    ).fillna(0.0).loc[display_names, DENSITY_BUCKETS]
    axis = density_pivot.plot(kind="bar", stacked=True, figsize=(11, 6))
    axis.set_ylabel("Share of selected images")
    axis.set_xlabel("Review queue")
    axis.set_title(f"Scene density in top-{top_n} review queues")
    axis.legend(title="Objects per image", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path = directory / FIGURE_FILENAMES[3]
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    paths.append(path)

    summary = artifacts["profile_summary.csv"].set_index("profile_name").loc[profile_names]
    x_values = np.arange(len(display_names))
    width = 0.2
    plt.figure(figsize=(12, 6))
    for position, (column, label) in enumerate(
        (
            ("mean_num_objects", "Labeled objects"),
            ("mean_prediction_count", "Predictions"),
            ("mean_false_positive_count", "False positives"),
            ("mean_missed_gt_count", "Missed objects"),
        )
    ):
        plt.bar(x_values + (position - 1.5) * width, summary[column], width, label=label)
    plt.xticks(x_values, display_names, rotation=25, ha="right")
    plt.ylabel("Mean count per selected image")
    plt.title(f"Image profile of top-{top_n} review queues")
    plt.legend()
    plt.tight_layout()
    path = directory / FIGURE_FILENAMES[4]
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    paths.append(path)

    error_profile = summary[
        [f"mean_{component}" for component in ERROR_COMPONENT_COLUMNS]
    ].copy()
    error_profile.columns = [COMPONENT_LABELS[item] for item in ERROR_COMPONENT_COLUMNS]
    error_profile.index = display_names
    axis = error_profile.plot(kind="bar", figsize=(12, 6))
    axis.set_ylabel("Mean bounded error")
    axis.set_xlabel("Review queue")
    axis.set_ylim(0.0, 1.05)
    axis.set_title(f"Error profile of top-{top_n} review queues")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path = directory / FIGURE_FILENAMES[5]
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    paths.append(path)

    class_presence = artifacts["class_presence_by_profile.csv"]
    top_classes = (
        class_presence.groupby("class_name")["image_count"].sum().nlargest(10).index
    )
    presence_pivot = class_presence[
        class_presence["class_name"].isin(top_classes)
    ].pivot_table(
        index="class_name",
        columns="display_name",
        values="image_share",
        fill_value=0.0,
    ).loc[:, display_names]
    plt.figure(figsize=(11, 7))
    plt.imshow(presence_pivot.values, aspect="auto", vmin=0, vmax=1)
    plt.xticks(range(len(display_names)), display_names, rotation=35, ha="right")
    plt.yticks(range(len(presence_pivot.index)), presence_pivot.index)
    plt.colorbar(label="Share of selected images containing class")
    plt.title(f"Class presence in top-{top_n} review queues")
    for row_index in range(presence_pivot.shape[0]):
        for column_index in range(presence_pivot.shape[1]):
            plt.text(
                column_index,
                row_index,
                f"{presence_pivot.values[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    plt.tight_layout()
    path = directory / FIGURE_FILENAMES[6]
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    paths.append(path)
    return paths


def write_artifacts(output_dir, artifacts):
    paths = {}
    for filename, table in artifacts.items():
        paths[filename] = _write_csv(Path(output_dir) / filename, table)
    return paths


def load_artifacts(output_dir):
    directory = Path(output_dir).expanduser().absolute()
    filenames = (
        "review_queue_profiles.csv",
        "top_images_by_profile.csv",
        "profile_summary.csv",
        "profile_overlap.csv",
        "density_by_profile.csv",
        "class_presence_by_profile.csv",
    )
    missing = [directory / name for name in filenames if not (directory / name).is_file()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing review-queue artifacts:\n{details}")
    return {filename: pd.read_csv(directory / filename) for filename in filenames}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build targeted image-review queues from detector errors."
    )
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--top-n", type=positive_int, default=DEFAULT_TOP_N)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    components = load_component_table(args.components)
    if args.figures_only:
        artifacts = load_artifacts(args.output_dir)
        figures = build_figures(components, artifacts, args.top_n, args.figure_dir)
        for path in figures:
            print(f"[WRITE] {path}")
        return artifacts, {}, figures
    artifacts = build_artifacts(components, args.top_n)
    paths = write_artifacts(args.output_dir, artifacts)
    figures = []
    if not args.skip_figures:
        figures = build_figures(components, artifacts, args.top_n, args.figure_dir)
    print("\nREVIEW QUEUE SUMMARY")
    print(artifacts["profile_summary.csv"].to_string(index=False))
    print("\nREVIEW QUEUE OVERLAP")
    print(
        artifacts["profile_overlap.csv"].pivot(
            index="profile_a",
            columns="profile_b",
            values="jaccard_overlap",
        ).loc[_profile_names(), _profile_names()].round(3).to_string()
    )
    for path in [*paths.values(), *figures]:
        print(f"[WRITE] {path}")
    return artifacts, paths, figures


if __name__ == "__main__":
    main()
