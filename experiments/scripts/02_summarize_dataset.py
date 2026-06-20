from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

OUTPUT_DIR = EXPERIMENTS_DIR / "outputs"
SUMMARY_OUTPUT_DIR = OUTPUT_DIR / "dataset_sampling"
SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_INDEX = OUTPUT_DIR / "dataset_index.csv"
CLASS_DISTRIBUTION = OUTPUT_DIR / "class_distribution.csv"
OBJECT_COUNT_DISTRIBUTION = OUTPUT_DIR / "object_count_distribution.csv"


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


def main():
    if not DATASET_INDEX.exists():
        raise FileNotFoundError(
            f"Dataset index not found: {DATASET_INDEX}. "
            "Run experiments/scripts/02_build_dataset_index.py first."
        )

    if not CLASS_DISTRIBUTION.exists():
        raise FileNotFoundError(
            f"Class distribution not found: {CLASS_DISTRIBUTION}. "
            "Run experiments/scripts/02_build_dataset_index.py first."
        )

    if not OBJECT_COUNT_DISTRIBUTION.exists():
        raise FileNotFoundError(
            f"Object-count distribution not found: {OBJECT_COUNT_DISTRIBUTION}. "
            "Run experiments/scripts/02_build_dataset_index.py first."
        )

    idx = pd.read_csv(DATASET_INDEX)
    cls = pd.read_csv(CLASS_DISTRIBUTION)
    obj = pd.read_csv(OBJECT_COUNT_DISTRIBUTION)

    summary = pd.DataFrame([{
        "dataset": "full_dataset",
        "images": len(idx),
        "total_objects": int(cls["object_count"].sum()),
        "images_with_zero_objects": int((idx["num_objects"] == 0).sum()),
        "mean_objects_per_image": round(idx["num_objects"].mean(), 3),
        "median_objects_per_image": idx["num_objects"].median(),
        "max_objects_per_image": int(idx["num_objects"].max()),
        "images_ge_5_objects": int((idx["num_objects"] >= 5).sum()),
        "images_ge_10_objects": int((idx["num_objects"] >= 10).sum()),
        "images_ge_15_objects": int((idx["num_objects"] >= 15).sum()),
        "images_ge_20_objects": int((idx["num_objects"] >= 20).sum()),
    }])

    summary_path = SUMMARY_OUTPUT_DIR / "full_dataset_summary.csv"
    summary.to_csv(summary_path, index=False)

    cls_enriched = cls.copy()
    total_objects = cls_enriched["object_count"].sum()
    total_images = len(idx)

    cls_enriched["object_share_pct"] = (
        100 * cls_enriched["object_count"] / total_objects
    ).round(4)
    cls_enriched["image_share_pct"] = (
        100 * cls_enriched["image_count"] / total_images
    ).round(4)

    cls_enriched_path = SUMMARY_OUTPUT_DIR / "class_distribution_enriched.csv"
    cls_enriched.to_csv(cls_enriched_path, index=False)

    top_classes = cls_enriched.sort_values("object_count", ascending=False).head(10)
    bottom_classes = cls_enriched.sort_values("object_count", ascending=True).head(10)

    top_path = SUMMARY_OUTPUT_DIR / "top10_classes_by_object_count.csv"
    bottom_path = SUMMARY_OUTPUT_DIR / "bottom10_classes_by_object_count.csv"

    top_classes.to_csv(top_path, index=False)
    bottom_classes.to_csv(bottom_path, index=False)

    idx_with_density = idx.copy()
    idx_with_density["density_bucket"] = idx_with_density["num_objects"].apply(density_bucket)

    bucket_order = ["1", "2-4", "5-9", "10-14", "15-19", "20+"]
    density_summary = (
        idx_with_density["density_bucket"]
        .value_counts()
        .reindex(bucket_order, fill_value=0)
        .rename_axis("density_bucket")
        .reset_index(name="image_count")
    )

    density_summary["image_share_pct"] = (
        100 * density_summary["image_count"] / len(idx)
    ).round(4)

    density_path = SUMMARY_OUTPUT_DIR / "density_bucket_distribution.csv"
    density_summary.to_csv(density_path, index=False)

    dense_counts = pd.DataFrame([
        {"threshold": ">=5 objects", "image_count": int((idx["num_objects"] >= 5).sum())},
        {"threshold": ">=10 objects", "image_count": int((idx["num_objects"] >= 10).sum())},
        {"threshold": ">=15 objects", "image_count": int((idx["num_objects"] >= 15).sum())},
        {"threshold": ">=20 objects", "image_count": int((idx["num_objects"] >= 20).sum())},
    ])

    dense_counts["image_share_pct"] = (
        100 * dense_counts["image_count"] / len(idx)
    ).round(4)

    dense_path = SUMMARY_OUTPUT_DIR / "dense_image_counts.csv"
    dense_counts.to_csv(dense_path, index=False)

    print("DATASET SUMMARY")
    print("---------------")
    print(summary.to_string(index=False))

    print("\nTOP 10 CLASSES BY OBJECT COUNT")
    print(top_classes.to_string(index=False))

    print("\nBOTTOM 10 CLASSES BY OBJECT COUNT")
    print(bottom_classes.to_string(index=False))

    print("\nDENSITY BUCKET DISTRIBUTION")
    print(density_summary.to_string(index=False))

    print("\nOBJECT COUNT DISTRIBUTION")
    print(obj.to_string(index=False))

    print("\nDENSE IMAGE COUNTS")
    print(dense_counts.to_string(index=False))

    print("\nWrote:")
    for path in [
        summary_path,
        cls_enriched_path,
        top_path,
        bottom_path,
        density_path,
        dense_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()