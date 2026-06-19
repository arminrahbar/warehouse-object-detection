from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "outputs"
FIG = ROOT / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 5000
RANDOM_SEED = 42


def clean_col(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def density_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 4:
        return "2-4"
    if n <= 9:
        return "5-9"
    if n <= 14:
        return "10-14"
    if n <= 19:
        return "15-19"
    return "20+"


def proportional_sample(df: pd.DataFrame, group_col: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    group_counts = df[group_col].value_counts().sort_index()
    raw_targets = group_counts / len(df) * n
    targets = np.floor(raw_targets).astype(int)

    remainder = n - int(targets.sum())
    if remainder > 0:
        add_order = (raw_targets - targets).sort_values(ascending=False).index.tolist()
        for group in add_order[:remainder]:
            targets[group] += 1

    sampled = []
    for group, group_n in targets.items():
        group_df = df[df[group_col] == group]
        group_n = min(int(group_n), len(group_df))
        if group_n > 0:
            sampled.append(group_df.sample(n=group_n, random_state=int(rng.integers(0, 1_000_000))))

    sample = pd.concat(sampled, axis=0)

    if len(sample) < n:
        remaining = df.drop(index=sample.index)
        fill_n = min(n - len(sample), len(remaining))
        sample = pd.concat([
            sample,
            remaining.sample(n=fill_n, random_state=int(rng.integers(0, 1_000_000)))
        ])

    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)

    return sample


def class_distribution(df: pd.DataFrame, cls: pd.DataFrame, label: str) -> pd.DataFrame:
    total_objects = int(df["num_objects"].sum())
    total_images = len(df)
    rows = []

    for _, row in cls.iterrows():
        class_name = row["class_name"]
        col = f"count_{clean_col(class_name)}"

        object_count = int(df[col].sum())
        image_count = int((df[col] > 0).sum())

        rows.append({
            "dataset": label,
            "class_id": int(row["class_id"]),
            "class_name": class_name,
            "object_count": object_count,
            "image_count": image_count,
            "object_share_pct": 100 * object_count / total_objects if total_objects else 0,
            "image_share_pct": 100 * image_count / total_images if total_images else 0,
        })

    return pd.DataFrame(rows)


def density_distribution(df: pd.DataFrame, label: str) -> pd.DataFrame:
    bucket_order = ["1", "2-4", "5-9", "10-14", "15-19", "20+"]
    counts = df["density_bucket"].value_counts().reindex(bucket_order, fill_value=0)

    return pd.DataFrame({
        "dataset": label,
        "density_bucket": bucket_order,
        "image_count": counts.values,
        "image_share_pct": 100 * counts.values / len(df),
    })


def dataset_summary(df: pd.DataFrame, label: str) -> dict:
    return {
        "dataset": label,
        "images": int(len(df)),
        "total_objects": int(df["num_objects"].sum()),
        "mean_objects_per_image": round(float(df["num_objects"].mean()), 3),
        "median_objects_per_image": float(df["num_objects"].median()),
        "max_objects_per_image": int(df["num_objects"].max()),
        "images_ge_5_objects": int((df["num_objects"] >= 5).sum()),
        "images_ge_10_objects": int((df["num_objects"] >= 10).sum()),
        "images_ge_15_objects": int((df["num_objects"] >= 15).sum()),
        "images_ge_20_objects": int((df["num_objects"] >= 20).sum()),
    }


def rare_class_targets(cls: pd.DataFrame, sample_fraction: float) -> pd.DataFrame:
    rare = cls.sort_values("object_count", ascending=True).head(8).copy()

    # Target is roughly proportional to the sample size, with a small floor.
    # This protects rare classes without forcing 100% retention or badly distorting the distribution.
    rare["target_image_count"] = rare["image_count"].apply(
        lambda x: min(int(x), max(int(np.ceil(x * sample_fraction)), min(100, int(x))))
    )

    return rare


def enforce_rare_class_targets(
    base_sample: pd.DataFrame,
    full_df: pd.DataFrame,
    rare_targets: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sample = base_sample.copy()

    for _, row in rare_targets.iterrows():
        class_name = row["class_name"]
        target = int(row["target_image_count"])
        col = f"count_{clean_col(class_name)}"

        current = int((sample[col] > 0).sum())
        needed = target - current

        if needed <= 0:
            continue

        selected_paths = set(sample["image_path"])
        outside = full_df[~full_df["image_path"].isin(selected_paths)]
        candidates = outside[outside[col] > 0]

        if len(candidates) == 0:
            continue

        add_n = min(needed, len(candidates))
        additions = candidates.sample(n=add_n, random_state=int(rng.integers(0, 1_000_000)))
        sample = pd.concat([sample, additions], axis=0)

    sample = sample.drop_duplicates(subset=["image_path"], keep="first").copy()

    # Trim back to SAMPLE_SIZE by removing non-rare images first.
    if len(sample) > SAMPLE_SIZE:
        rare_cols = [f"count_{clean_col(c)}" for c in rare_targets["class_name"].tolist()]
        sample = sample.copy()
        sample["_rare_object_count"] = sample[rare_cols].sum(axis=1)

        excess = len(sample) - SAMPLE_SIZE
        removable = sample[sample["_rare_object_count"] == 0]

        if len(removable) >= excess:
            remove_idx = removable.sample(n=excess, random_state=seed).index
            sample = sample.drop(index=remove_idx)
        else:
            sample = sample.drop(index=removable.index)
            excess = len(sample) - SAMPLE_SIZE
            if excess > 0:
                remove_idx = sample.sample(n=excess, random_state=seed).index
                sample = sample.drop(index=remove_idx)

        sample = sample.drop(columns=["_rare_object_count"], errors="ignore")

    # Fill if short, using only unused images.
    if len(sample) < SAMPLE_SIZE:
        selected_paths = set(sample["image_path"])
        remaining = full_df[~full_df["image_path"].isin(selected_paths)]
        fill = remaining.sample(n=SAMPLE_SIZE - len(sample), random_state=seed)
        sample = pd.concat([sample, fill], axis=0)

    sample = sample.drop_duplicates(subset=["image_path"], keep="first").copy()

    if len(sample) != SAMPLE_SIZE:
        raise ValueError(f"Expected {SAMPLE_SIZE} unique images, got {len(sample)}")

    if sample["image_path"].nunique() != SAMPLE_SIZE:
        raise ValueError("Duplicate image_path values remain in selected sample.")

    return sample.sort_values("image_file").reset_index(drop=True)


def compare_sample(full_df: pd.DataFrame, sample_df: pd.DataFrame, cls: pd.DataFrame, name: str, rare_targets: pd.DataFrame) -> dict:
    full_class = class_distribution(full_df, cls, "full")
    sample_class = class_distribution(sample_df, cls, name)

    class_cmp = full_class.merge(
        sample_class,
        on=["class_id", "class_name"],
        suffixes=("_full", "_sample")
    )

    class_cmp["object_share_abs_error_pp"] = (
        class_cmp["object_share_pct_sample"] - class_cmp["object_share_pct_full"]
    ).abs()

    full_density = density_distribution(full_df, "full")
    sample_density = density_distribution(sample_df, name)

    density_cmp = full_density.merge(
        sample_density,
        on="density_bucket",
        suffixes=("_full", "_sample")
    )

    density_cmp["image_share_abs_error_pp"] = (
        density_cmp["image_share_pct_sample"] - density_cmp["image_share_pct_full"]
    ).abs()

    rare_cmp = class_cmp[class_cmp["class_name"].isin(rare_targets["class_name"])].copy()
    rare_cmp["rare_image_retention_pct"] = (
        100 * rare_cmp["image_count_sample"] / rare_cmp["image_count_full"]
    )

    return {
        "sample_name": name,
        "images": int(len(sample_df)),
        "total_objects": int(sample_df["num_objects"].sum()),
        "mean_objects_per_image": round(float(sample_df["num_objects"].mean()), 3),
        "class_object_share_mae_pp": round(float(class_cmp["object_share_abs_error_pp"].mean()), 4),
        "class_object_share_max_error_pp": round(float(class_cmp["object_share_abs_error_pp"].max()), 4),
        "density_share_mae_pp": round(float(density_cmp["image_share_abs_error_pp"].mean()), 4),
        "density_share_max_error_pp": round(float(density_cmp["image_share_abs_error_pp"].max()), 4),
        "min_rare_class_image_retention_pct": round(float(rare_cmp["rare_image_retention_pct"].min()), 2),
        "images_ge_10_objects": int((sample_df["num_objects"] >= 10).sum()),
        "images_ge_20_objects": int((sample_df["num_objects"] >= 20).sum()),
    }


def main():
    idx = pd.read_csv(OUT / "task2_dataset_index.csv")
    cls = pd.read_csv(OUT / "task2_class_distribution_full.csv")

    idx["density_bucket"] = idx["num_objects"].apply(density_bucket)

    sample_fraction = SAMPLE_SIZE / len(idx)
    rare_targets = rare_class_targets(cls, sample_fraction)
    rare_targets.to_csv(OUT / "task2_rare_class_targets.csv", index=False)

    random_sample = idx.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
    density_sample = proportional_sample(idx, "density_bucket", SAMPLE_SIZE, RANDOM_SEED)

    rare_density_sample = enforce_rare_class_targets(
        base_sample=density_sample,
        full_df=idx,
        rare_targets=rare_targets,
        seed=RANDOM_SEED,
    )

    candidates = {
        "random_5000": random_sample,
        "density_stratified_5000": density_sample,
        "rare_aware_density_stratified_5000": rare_density_sample,
    }

    quality = pd.DataFrame([
        compare_sample(idx, sample, cls, name, rare_targets)
        for name, sample in candidates.items()
    ])

    quality.to_csv(OUT / "task2_candidate_sample_quality.csv", index=False)

    selected_name = "rare_aware_density_stratified_5000"
    selected = candidates[selected_name].copy()
    selected.to_csv(OUT / "task2_selected_sample_index.csv", index=False)

    summary = pd.DataFrame([
        dataset_summary(idx, "full_dataset"),
        dataset_summary(selected, selected_name),
    ])
    summary.to_csv(OUT / "task2_full_vs_selected_sample_summary.csv", index=False)

    full_class = class_distribution(idx, cls, "full_dataset")
    selected_class = class_distribution(selected, cls, selected_name)

    class_compare = full_class.merge(
        selected_class,
        on=["class_id", "class_name"],
        suffixes=("_full", "_sample")
    )

    class_compare["object_share_diff_pp"] = (
        class_compare["object_share_pct_sample"] - class_compare["object_share_pct_full"]
    )
    class_compare["image_share_diff_pp"] = (
        class_compare["image_share_pct_sample"] - class_compare["image_share_pct_full"]
    )

    class_compare.to_csv(OUT / "task2_full_vs_selected_class_distribution.csv", index=False)

    rare_coverage = class_compare[class_compare["class_name"].isin(rare_targets["class_name"])].copy()
    rare_coverage = rare_coverage.merge(
        rare_targets[["class_name", "target_image_count"]],
        on="class_name",
        how="left"
    )
    rare_coverage["sample_image_retention_pct"] = (
        100 * rare_coverage["image_count_sample"] / rare_coverage["image_count_full"]
    )

    rare_coverage.to_csv(OUT / "task2_selected_rare_class_coverage.csv", index=False)

    full_density = density_distribution(idx, "full_dataset")
    selected_density = density_distribution(selected, selected_name)

    density_compare = full_density.merge(
        selected_density,
        on="density_bucket",
        suffixes=("_full", "_sample")
    )
    density_compare["image_share_diff_pp"] = (
        density_compare["image_share_pct_sample"] - density_compare["image_share_pct_full"]
    )

    density_compare.to_csv(OUT / "task2_full_vs_selected_density_distribution.csv", index=False)

    # Figure 1: class distribution full vs selected.
    plot_df = class_compare.sort_values("object_count_full", ascending=True)
    y = np.arange(len(plot_df))
    bar_height = 0.38

    plt.figure(figsize=(11, 9))
    plt.barh(y - bar_height / 2, plot_df["object_share_pct_full"], height=bar_height, label="Full dataset")
    plt.barh(y + bar_height / 2, plot_df["object_share_pct_sample"], height=bar_height, label="Selected sample")
    plt.yticks(y, plot_df["class_name"])
    plt.xlabel("Object share (%)")
    plt.ylabel("Class")
    plt.title("Task 2: Full dataset vs selected sample class distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "task2_class_distribution_full_vs_selected_sample.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Figure 2: density distribution full vs selected.
    density_plot = density_compare.copy()
    x = np.arange(len(density_plot))
    bar_width = 0.38

    plt.figure(figsize=(9, 6))
    plt.bar(x - bar_width / 2, density_plot["image_share_pct_full"], width=bar_width, label="Full dataset")
    plt.bar(x + bar_width / 2, density_plot["image_share_pct_sample"], width=bar_width, label="Selected sample")
    plt.xticks(x, density_plot["density_bucket"])
    plt.xlabel("Objects per image bucket")
    plt.ylabel("Image share (%)")
    plt.title("Task 2: Full dataset vs selected sample object-density distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "task2_density_distribution_full_vs_selected_sample.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Figure 3: candidate quality by class-distribution error.
    plt.figure(figsize=(9, 6))
    plt.bar(quality["sample_name"], quality["class_object_share_mae_pp"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Mean absolute class object-share error (percentage points)")
    plt.title("Task 2: Candidate sample class-distribution error")
    plt.tight_layout()
    plt.savefig(FIG / "task2_candidate_class_distribution_error.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Rare-class targets:")
    print(rare_targets.to_string(index=False))

    print("\nCandidate sample quality:")
    print(quality.to_string(index=False))

    print("\nSelected sample:", selected_name)
    print("Selected sample images:", len(selected))

    print("\nWrote:")
    print(OUT / "task2_rare_class_targets.csv")
    print(OUT / "task2_candidate_sample_quality.csv")
    print(OUT / "task2_selected_sample_index.csv")
    print(OUT / "task2_full_vs_selected_sample_summary.csv")
    print(OUT / "task2_full_vs_selected_class_distribution.csv")
    print(OUT / "task2_selected_rare_class_coverage.csv")
    print(OUT / "task2_full_vs_selected_density_distribution.csv")
    print(FIG / "task2_class_distribution_full_vs_selected_sample.png")
    print(FIG / "task2_density_distribution_full_vs_selected_sample.png")
    print(FIG / "task2_candidate_class_distribution_error.png")


if __name__ == "__main__":
    main()
