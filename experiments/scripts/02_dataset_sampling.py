"""Compare deterministic dataset samples and preserve rare-class coverage."""

import argparse
import math
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY_DIR = (
    PROJECT_ROOT / "experiments" / "outputs" / "00_dataset_inventory"
)
DEFAULT_ANALYSIS_ROOT = (
    PROJECT_ROOT / "experiments" / "outputs" / "02_dataset_analysis"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ANALYSIS_ROOT / "02_sample_selection"
DEFAULT_FIGURE_DIR = (
    PROJECT_ROOT / "scratch" / "diagnostic-figures" / "02_dataset_analysis"
)
DEFAULT_DATASET_INDEX = DEFAULT_INVENTORY_DIR / "dataset_index.csv"
DEFAULT_CLASS_DISTRIBUTION = DEFAULT_INVENTORY_DIR / "class_distribution.csv"

SAMPLE_SIZE = 5000
RANDOM_SEED = 42
RARE_CLASS_COUNT = 8
RARE_CLASS_MINIMUM_IMAGES = 100
DENSITY_BUCKET_ORDER = ["1", "2-4", "5-9", "10-14", "15-19", "20+"]


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def clean_column_name(name):
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).lower())
    return normalized.strip("_")


def density_bucket(object_count):
    count = int(object_count)
    if count < 0:
        raise ValueError("Object count must be non-negative.")
    if count <= 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 9:
        return "5-9"
    if count <= 14:
        return "10-14"
    if count <= 19:
        return "15-19"
    return "20+"


def _require_columns(table, required, label):
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _validated_integer_series(series, label):
    numeric = pd.to_numeric(series, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain only finite values.")
    if (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label} must contain non-negative integers.")
    return numeric.astype(np.int64)


def load_and_validate_inputs(dataset_index_path, class_distribution_path, sample_size):
    """Load index sources and verify all sampling-relevant class totals."""

    index_path = Path(dataset_index_path).expanduser().absolute()
    class_path = Path(class_distribution_path).expanduser().absolute()
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Dataset index not found: {index_path}. "
            "Run experiments/scripts/00_build_dataset_inventory.py first."
        )
    if not class_path.is_file():
        raise FileNotFoundError(
            f"Class distribution not found: {class_path}. "
            "Run experiments/scripts/00_build_dataset_inventory.py first."
        )

    index = pd.read_csv(index_path)
    classes = pd.read_csv(class_path)
    _require_columns(
        index,
        ["image_file", "image_path", "label_path", "num_objects"],
        "Dataset index",
    )
    _require_columns(
        classes,
        ["class_id", "class_name", "object_count", "image_count"],
        "Class distribution",
    )
    if len(index) == 0:
        raise RuntimeError("Dataset index is empty.")
    if len(classes) == 0:
        raise RuntimeError("Class distribution is empty.")
    if sample_size > len(index):
        raise ValueError(
            f"Sample size {sample_size} exceeds dataset size {len(index)}."
        )
    if index["image_file"].duplicated().any():
        raise ValueError("Dataset index contains duplicate image_file values.")
    if index["image_path"].duplicated().any():
        raise ValueError("Dataset index contains duplicate image_path values.")

    classes = classes.copy()
    classes["class_id"] = _validated_integer_series(classes["class_id"], "class_id")
    expected_ids = np.arange(len(classes), dtype=np.int64)
    if not np.array_equal(classes["class_id"].to_numpy(), expected_ids):
        raise ValueError("Class identifiers must be contiguous and zero-based.")
    if classes["class_name"].isna().any() or (classes["class_name"].str.strip() == "").any():
        raise ValueError("Class names must be non-empty.")
    if classes["class_name"].duplicated().any():
        raise ValueError("Class names must be unique.")

    classes["object_count"] = _validated_integer_series(
        classes["object_count"],
        "class object_count",
    )
    classes["image_count"] = _validated_integer_series(
        classes["image_count"],
        "class image_count",
    )

    index = index.copy()
    index["num_objects"] = _validated_integer_series(
        index["num_objects"],
        "num_objects",
    )
    count_columns = []
    for row in classes.itertuples(index=False):
        suffix = clean_column_name(row.class_name)
        if not suffix:
            raise ValueError(f"Class {row.class_id} cannot form a count column.")
        column = f"count_{suffix}"
        if column in count_columns:
            raise ValueError(f"Duplicate class count column: {column}")
        if column not in index.columns:
            raise ValueError(f"Dataset index is missing required column: {column}")
        index[column] = _validated_integer_series(index[column], column)
        count_columns.append(column)

    row_totals = index[count_columns].sum(axis=1)
    if not np.array_equal(row_totals.to_numpy(), index["num_objects"].to_numpy()):
        raise ValueError("num_objects does not equal the per-class count sum.")

    for row, column in zip(classes.itertuples(index=False), count_columns):
        object_count = int(index[column].sum())
        image_count = int((index[column] > 0).sum())
        if object_count != int(row.object_count):
            raise ValueError(f"Object total mismatch for class {row.class_id}.")
        if image_count != int(row.image_count):
            raise ValueError(f"Image total mismatch for class {row.class_id}.")

    index["density_bucket"] = index["num_objects"].map(density_bucket)
    return index, classes


def proportional_targets(group_counts, sample_size):
    """Allocate a fixed sample with deterministic largest-remainder rounding."""

    counts = pd.Series(group_counts, dtype=np.int64)
    if len(counts) == 0 or int(counts.sum()) <= 0:
        raise ValueError("Group counts must describe a non-empty dataset.")
    if sample_size <= 0 or sample_size > int(counts.sum()):
        raise ValueError("Sample size must be within the grouped dataset size.")

    raw = counts.astype(float) / int(counts.sum()) * sample_size
    targets = np.floor(raw).astype(np.int64)
    remainder = sample_size - int(targets.sum())
    if remainder:
        order = sorted(
            range(len(counts)),
            key=lambda position: (
                -(float(raw.iloc[position]) - int(targets.iloc[position])),
                position,
            ),
        )
        for position in order[:remainder]:
            targets.iloc[position] += 1
    return targets


def proportional_sample(table, group_column, sample_size, seed):
    """Sample within every group according to largest-remainder targets."""

    if group_column not in table.columns:
        raise ValueError(f"Missing sampling group column: {group_column}")
    if table[group_column].isna().any():
        raise ValueError(f"Sampling group column contains missing values: {group_column}")

    group_counts = table[group_column].value_counts().sort_index()
    targets = proportional_targets(group_counts, sample_size)
    generator = np.random.default_rng(seed)
    pieces = []

    for group, target in targets.items():
        group_table = table[table[group_column] == group]
        count = min(int(target), len(group_table))
        if count:
            pieces.append(
                group_table.sample(
                    n=count,
                    random_state=int(generator.integers(0, 1_000_000)),
                )
            )

    sample = pd.concat(pieces, axis=0) if pieces else table.iloc[0:0].copy()
    if len(sample) < sample_size:
        remaining = table.drop(index=sample.index)
        fill_count = min(sample_size - len(sample), len(remaining))
        if fill_count:
            sample = pd.concat(
                [
                    sample,
                    remaining.sample(
                        n=fill_count,
                        random_state=int(generator.integers(0, 1_000_000)),
                    ),
                ],
                axis=0,
            )
    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=seed)
    if len(sample) != sample_size:
        raise RuntimeError(
            f"Expected proportional sample of {sample_size}, got {len(sample)}."
        )
    if sample["image_path"].duplicated().any():
        raise RuntimeError("Proportional sample contains duplicate image paths.")
    return sample


def class_distribution(table, classes, label):
    total_objects = int(table["num_objects"].sum())
    total_images = len(table)
    rows = []
    for row in classes.itertuples(index=False):
        column = f"count_{clean_column_name(row.class_name)}"
        object_count = int(table[column].sum())
        image_count = int((table[column] > 0).sum())
        rows.append(
            {
                "dataset": label,
                "class_id": int(row.class_id),
                "class_name": row.class_name,
                "object_count": object_count,
                "image_count": image_count,
                "object_share_pct": (
                    100.0 * object_count / total_objects if total_objects else 0.0
                ),
                "image_share_pct": (
                    100.0 * image_count / total_images if total_images else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def density_distribution(table, label):
    counts = (
        table["density_bucket"]
        .value_counts()
        .reindex(DENSITY_BUCKET_ORDER, fill_value=0)
    )
    return pd.DataFrame(
        {
            "dataset": label,
            "density_bucket": DENSITY_BUCKET_ORDER,
            "image_count": counts.to_numpy(dtype=np.int64),
            "image_share_pct": 100.0 * counts.to_numpy(dtype=float) / len(table),
        }
    )


def dataset_summary(table, label):
    return {
        "dataset": label,
        "images": int(len(table)),
        "total_objects": int(table["num_objects"].sum()),
        "mean_objects_per_image": round(float(table["num_objects"].mean()), 3),
        "median_objects_per_image": float(table["num_objects"].median()),
        "max_objects_per_image": int(table["num_objects"].max()),
        "images_ge_5_objects": int((table["num_objects"] >= 5).sum()),
        "images_ge_10_objects": int((table["num_objects"] >= 10).sum()),
        "images_ge_15_objects": int((table["num_objects"] >= 15).sum()),
        "images_ge_20_objects": int((table["num_objects"] >= 20).sum()),
    }


def rare_class_targets(classes, sample_fraction):
    """Set proportional-but-protected image targets for the rarest classes."""

    rare = (
        classes.sort_values(
            ["object_count", "class_id"],
            ascending=[True, True],
            kind="mergesort",
        )
        .head(min(RARE_CLASS_COUNT, len(classes)))
        .copy()
    )
    rare["target_image_count"] = rare["image_count"].map(
        lambda count: min(
            int(count),
            max(
                int(math.ceil(int(count) * sample_fraction)),
                min(RARE_CLASS_MINIMUM_IMAGES, int(count)),
            ),
        )
    )
    return rare


def _rare_columns(rare_targets):
    return [
        f"count_{clean_column_name(name)}"
        for name in rare_targets["class_name"].tolist()
    ]


def enforce_rare_class_targets(
    base_sample,
    full_table,
    rare_targets,
    sample_size,
    seed,
):
    """Add missing rare coverage, trim non-rare rows, and prove all targets."""

    generator = np.random.default_rng(seed)
    sample = base_sample.copy()
    for row in rare_targets.itertuples(index=False):
        column = f"count_{clean_column_name(row.class_name)}"
        target = int(row.target_image_count)
        current = int((sample[column] > 0).sum())
        needed = target - current
        if needed <= 0:
            continue

        selected_paths = set(sample["image_path"])
        candidates = full_table[
            ~full_table["image_path"].isin(selected_paths)
            & (full_table[column] > 0)
        ]
        if len(candidates) < needed:
            raise RuntimeError(
                f"Insufficient candidates for rare class {row.class_name}: "
                f"needed {needed}, available {len(candidates)}."
            )
        additions = candidates.sample(
            n=needed,
            random_state=int(generator.integers(0, 1_000_000)),
        )
        sample = pd.concat([sample, additions], axis=0)

    sample = sample.drop_duplicates(subset=["image_path"], keep="first").copy()
    if len(sample) > sample_size:
        rare_columns = _rare_columns(rare_targets)
        sample["_rare_object_count"] = sample[rare_columns].sum(axis=1)
        excess = len(sample) - sample_size
        removable = sample[sample["_rare_object_count"] == 0]

        remove_count = min(excess, len(removable))
        if remove_count:
            remove_indices = removable.sample(
                n=remove_count,
                random_state=seed,
            ).index
            sample = sample.drop(index=remove_indices)

        excess = len(sample) - sample_size
        if excess:
            # This fallback is deterministic, but the target assertions below
            # prevent it from silently sacrificing protected class coverage.
            remove_indices = sample.sample(n=excess, random_state=seed).index
            sample = sample.drop(index=remove_indices)
        sample = sample.drop(columns=["_rare_object_count"], errors="ignore")

    if len(sample) < sample_size:
        selected_paths = set(sample["image_path"])
        remaining = full_table[~full_table["image_path"].isin(selected_paths)]
        needed = sample_size - len(sample)
        if len(remaining) < needed:
            raise RuntimeError("Insufficient unique images to fill the selected sample.")
        sample = pd.concat(
            [sample, remaining.sample(n=needed, random_state=seed)],
            axis=0,
        )

    sample = sample.drop_duplicates(subset=["image_path"], keep="first").copy()
    if len(sample) != sample_size or sample["image_path"].nunique() != sample_size:
        raise RuntimeError(
            f"Expected {sample_size} unique images, got {len(sample)}."
        )

    failures = []
    for row in rare_targets.itertuples(index=False):
        column = f"count_{clean_column_name(row.class_name)}"
        actual = int((sample[column] > 0).sum())
        if actual < int(row.target_image_count):
            failures.append(
                f"{row.class_name}: target={int(row.target_image_count)}, actual={actual}"
            )
    if failures:
        raise RuntimeError("Rare-class targets were not preserved: " + "; ".join(failures))

    return sample.sort_values("image_file", kind="mergesort").reset_index(drop=True)


def compare_sample(full_table, sample, classes, name, rare_targets):
    full_class = class_distribution(full_table, classes, "full")
    sample_class = class_distribution(sample, classes, name)
    class_comparison = full_class.merge(
        sample_class,
        on=["class_id", "class_name"],
        suffixes=("_full", "_sample"),
        validate="one_to_one",
    )
    class_error = (
        class_comparison["object_share_pct_sample"]
        - class_comparison["object_share_pct_full"]
    ).abs()

    full_density = density_distribution(full_table, "full")
    sample_density = density_distribution(sample, name)
    density_comparison = full_density.merge(
        sample_density,
        on="density_bucket",
        suffixes=("_full", "_sample"),
        validate="one_to_one",
    )
    density_error = (
        density_comparison["image_share_pct_sample"]
        - density_comparison["image_share_pct_full"]
    ).abs()

    rare_names = set(rare_targets["class_name"])
    rare_comparison = class_comparison[
        class_comparison["class_name"].isin(rare_names)
    ].copy()
    retention = (
        100.0
        * rare_comparison["image_count_sample"]
        / rare_comparison["image_count_full"]
    )
    return {
        "sample_name": name,
        "images": int(len(sample)),
        "total_objects": int(sample["num_objects"].sum()),
        "mean_objects_per_image": round(float(sample["num_objects"].mean()), 3),
        "class_object_share_mae_pp": round(float(class_error.mean()), 4),
        "class_object_share_max_error_pp": round(float(class_error.max()), 4),
        "density_share_mae_pp": round(float(density_error.mean()), 4),
        "density_share_max_error_pp": round(float(density_error.max()), 4),
        "min_rare_class_image_retention_pct": round(float(retention.min()), 2),
        "images_ge_10_objects": int((sample["num_objects"] >= 10).sum()),
        "images_ge_20_objects": int((sample["num_objects"] >= 20).sum()),
    }


def build_sampling_evidence(index, classes, sample_size, seed):
    """Build candidate samples and every downstream CSV table in memory."""

    sample_fraction = sample_size / len(index)
    rare_targets = rare_class_targets(classes, sample_fraction)
    random_name = f"random_{sample_size}"
    density_name = f"density_stratified_{sample_size}"
    selected_name = f"rare_aware_density_stratified_{sample_size}"

    random_sample = index.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    density_sample = proportional_sample(
        index,
        "density_bucket",
        sample_size,
        seed,
    )
    selected = enforce_rare_class_targets(
        base_sample=density_sample,
        full_table=index,
        rare_targets=rare_targets,
        sample_size=sample_size,
        seed=seed,
    )
    candidates = {
        random_name: random_sample,
        density_name: density_sample,
        selected_name: selected,
    }
    quality = pd.DataFrame(
        [
            compare_sample(index, sample, classes, name, rare_targets)
            for name, sample in candidates.items()
        ]
    )

    summary = pd.DataFrame(
        [
            dataset_summary(index, "full_dataset"),
            dataset_summary(selected, selected_name),
        ]
    )
    full_class = class_distribution(index, classes, "full_dataset")
    selected_class = class_distribution(selected, classes, selected_name)
    class_comparison = full_class.merge(
        selected_class,
        on=["class_id", "class_name"],
        suffixes=("_full", "_sample"),
        validate="one_to_one",
    )
    class_comparison["object_share_diff_pp"] = (
        class_comparison["object_share_pct_sample"]
        - class_comparison["object_share_pct_full"]
    )
    class_comparison["image_share_diff_pp"] = (
        class_comparison["image_share_pct_sample"]
        - class_comparison["image_share_pct_full"]
    )

    rare_coverage = class_comparison[
        class_comparison["class_name"].isin(set(rare_targets["class_name"]))
    ].copy()
    rare_coverage = rare_coverage.merge(
        rare_targets[["class_name", "target_image_count"]],
        on="class_name",
        how="left",
        validate="one_to_one",
    )
    rare_coverage["sample_image_retention_pct"] = (
        100.0
        * rare_coverage["image_count_sample"]
        / rare_coverage["image_count_full"]
    )

    full_density = density_distribution(index, "full_dataset")
    selected_density = density_distribution(selected, selected_name)
    density_comparison = full_density.merge(
        selected_density,
        on="density_bucket",
        suffixes=("_full", "_sample"),
        validate="one_to_one",
    )
    density_comparison["image_share_diff_pp"] = (
        density_comparison["image_share_pct_sample"]
        - density_comparison["image_share_pct_full"]
    )

    artifacts = {
        "rare_class_targets.csv": rare_targets,
        "candidate_sample_quality.csv": quality,
        "selected_sample_index.csv": selected,
        "sample_summary.csv": summary,
        "class_distribution_comparison.csv": class_comparison,
        "rare_class_coverage.csv": rare_coverage,
        "density_distribution_comparison.csv": density_comparison,
    }
    return artifacts, candidates, selected_name


def _write_dataframe_atomic(path, table):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            table.to_csv(temporary, index=False)
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_sampling_artifacts(output_dir, artifacts):
    destination = Path(output_dir).expanduser().absolute()
    paths = []
    for filename, table in artifacts.items():
        path = destination / filename
        _write_dataframe_atomic(path, table)
        paths.append(path)
    return paths


def build_figures(artifacts, figure_dir):
    import matplotlib.pyplot as plt

    destination = Path(figure_dir).expanduser().absolute()
    destination.mkdir(parents=True, exist_ok=True)
    quality = artifacts["candidate_sample_quality.csv"]
    class_comparison = artifacts["class_distribution_comparison.csv"]
    density_comparison = artifacts["density_distribution_comparison.csv"]
    figure_paths = []

    plt.figure(figsize=(9, 6))
    plt.bar(quality["sample_name"], quality["class_object_share_mae_pp"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Mean absolute class object-share error (percentage points)")
    plt.title("Candidate sample class-distribution error")
    plt.tight_layout()
    path = destination / "01_class_distribution_error.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    figure_paths.append(path)

    plot_table = class_comparison.sort_values(
        ["object_count_full", "class_id"],
        ascending=[True, True],
        kind="mergesort",
    )
    y_positions = np.arange(len(plot_table))
    bar_height = 0.38
    plt.figure(figsize=(11, 9))
    plt.barh(
        y_positions - bar_height / 2,
        plot_table["object_share_pct_full"],
        height=bar_height,
        label="Full dataset",
    )
    plt.barh(
        y_positions + bar_height / 2,
        plot_table["object_share_pct_sample"],
        height=bar_height,
        label="Selected sample",
    )
    plt.yticks(y_positions, plot_table["class_name"])
    plt.xlabel("Object share (%)")
    plt.ylabel("Class")
    plt.title("Full dataset vs selected sample class distribution")
    plt.legend()
    plt.tight_layout()
    path = destination / "02_class_distribution_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    figure_paths.append(path)

    x_positions = np.arange(len(density_comparison))
    bar_width = 0.38
    plt.figure(figsize=(9, 6))
    plt.bar(
        x_positions - bar_width / 2,
        density_comparison["image_share_pct_full"],
        width=bar_width,
        label="Full dataset",
    )
    plt.bar(
        x_positions + bar_width / 2,
        density_comparison["image_share_pct_sample"],
        width=bar_width,
        label="Selected sample",
    )
    plt.xticks(x_positions, density_comparison["density_bucket"])
    plt.xlabel("Objects per image bucket")
    plt.ylabel("Image share (%)")
    plt.title("Full dataset vs selected sample object-density distribution")
    plt.legend()
    plt.tight_layout()
    path = destination / "03_density_distribution.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    figure_paths.append(path)
    return figure_paths


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare and select deterministic coverage-preserving samples."
    )
    parser.add_argument("--dataset-index", type=Path, default=DEFAULT_DATASET_INDEX)
    parser.add_argument(
        "--class-distribution",
        type=Path,
        default=DEFAULT_CLASS_DISTRIBUTION,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--sample-size", type=positive_int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Write analytical CSVs without importing Matplotlib.",
    )
    return parser


def _project_relative(path):
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main(argv=None):
    args = build_parser().parse_args(argv)
    index, classes = load_and_validate_inputs(
        _project_relative(args.dataset_index),
        _project_relative(args.class_distribution),
        args.sample_size,
    )
    artifacts, candidates, selected_name = build_sampling_evidence(
        index,
        classes,
        args.sample_size,
        args.seed,
    )
    output_paths = write_sampling_artifacts(
        _project_relative(args.output_dir),
        artifacts,
    )
    figure_paths = []
    if not args.skip_figures:
        figure_paths = build_figures(artifacts, _project_relative(args.figure_dir))

    print("RARE-CLASS TARGETS")
    print(artifacts["rare_class_targets.csv"].to_string(index=False))
    print("\nCANDIDATE SAMPLE QUALITY")
    print(artifacts["candidate_sample_quality.csv"].to_string(index=False))
    print(f"\n[SELECTED] {selected_name}: {len(candidates[selected_name])} images")
    for path in output_paths + figure_paths:
        print(f"[WRITE] {path}")
    return artifacts, candidates, selected_name, output_paths, figure_paths


if __name__ == "__main__":
    main()
