import pandas as pd
from pathlib import Path

out = Path("analysis/outputs")
out.mkdir(parents=True, exist_ok=True)

idx = pd.read_csv(out / "task2_dataset_index.csv")
cls = pd.read_csv(out / "task2_class_distribution_full.csv")
obj = pd.read_csv(out / "task2_object_count_distribution_full.csv")

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

summary_path = out / "task2_full_dataset_summary.csv"
summary.to_csv(summary_path, index=False)

cls_enriched = cls.copy()
total_objects = cls_enriched["object_count"].sum()
total_images = len(idx)

cls_enriched["object_share_pct"] = (100 * cls_enriched["object_count"] / total_objects).round(4)
cls_enriched["image_share_pct"] = (100 * cls_enriched["image_count"] / total_images).round(4)

cls_enriched_path = out / "task2_class_distribution_full_enriched.csv"
cls_enriched.to_csv(cls_enriched_path, index=False)

top_classes = cls_enriched.sort_values("object_count", ascending=False).head(10)
bottom_classes = cls_enriched.sort_values("object_count", ascending=True).head(10)

top_path = out / "task2_top10_classes_by_object_count.csv"
bottom_path = out / "task2_bottom10_classes_by_object_count.csv"

top_classes.to_csv(top_path, index=False)
bottom_classes.to_csv(bottom_path, index=False)

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

density_summary["image_share_pct"] = (100 * density_summary["image_count"] / len(idx)).round(4)

density_path = out / "task2_density_bucket_distribution_full.csv"
density_summary.to_csv(density_path, index=False)

dense_counts = pd.DataFrame([
    {"threshold": ">=5 objects", "image_count": int((idx["num_objects"] >= 5).sum())},
    {"threshold": ">=10 objects", "image_count": int((idx["num_objects"] >= 10).sum())},
    {"threshold": ">=15 objects", "image_count": int((idx["num_objects"] >= 15).sum())},
    {"threshold": ">=20 objects", "image_count": int((idx["num_objects"] >= 20).sum())},
])

dense_counts["image_share_pct"] = (100 * dense_counts["image_count"] / len(idx)).round(4)

dense_path = out / "task2_dense_image_counts_full.csv"
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

print("\nDENSE IMAGE COUNTS")
print(dense_counts.to_string(index=False))

print("\nWrote:")
print(summary_path)
print(cls_enriched_path)
print(top_path)
print(bottom_path)
print(density_path)
print(dense_path)
