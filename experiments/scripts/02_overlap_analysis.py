from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

OUTPUT_DIR = EXPERIMENTS_DIR / "outputs"
SAMPLING_OUTPUT_DIR = OUTPUT_DIR / "dataset_sampling"
FIGURE_DIR = EXPERIMENTS_DIR / "figures" / "02_dataset_sampling"

SAMPLING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from techtrack.modules.utils.metrics import calculate_iou


DATASET_INDEX = OUTPUT_DIR / "dataset_index.csv"
SELECTED_SAMPLE_INDEX = SAMPLING_OUTPUT_DIR / "selected_sample_index.csv"


def yolo_to_xywh(cx, cy, w, h):
    """
    Convert YOLO normalized center-format boxes into top-left xywh format.
    This matches the box format used by the model evaluation pipeline.
    """
    x = cx - w / 2
    y = cy - h / 2
    return [x, y, w, h]


def crowding_bucket(pairs_gt_01):
    if pairs_gt_01 == 0:
        return "0"
    if pairs_gt_01 <= 4:
        return "1-4"
    if pairs_gt_01 <= 19:
        return "5-19"
    return "20+"


def compute_overlap_for_label(label_path):
    text = label_path.read_text().strip()
    boxes = []

    if text:
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            boxes.append(yolo_to_xywh(cx, cy, w, h))

    n = len(boxes)

    if n < 2:
        return {
            "pair_count": 0,
            "max_pairwise_iou": 0.0,
            "mean_pairwise_iou": 0.0,
            "pairs_iou_gt_0_1": 0,
            "pairs_iou_gt_0_3": 0,
            "pairs_iou_gt_0_5": 0,
        }

    ious = []

    for i in range(n):
        for j in range(i + 1, n):
            iou = float(calculate_iou(boxes[i], boxes[j]))
            iou = min(1.0, max(0.0, iou))
            ious.append(iou)

    ious = np.array(ious)

    return {
        "pair_count": int(len(ious)),
        "max_pairwise_iou": float(ious.max()) if len(ious) else 0.0,
        "mean_pairwise_iou": float(ious.mean()) if len(ious) else 0.0,
        "pairs_iou_gt_0_1": int((ious > 0.1).sum()),
        "pairs_iou_gt_0_3": int((ious > 0.3).sum()),
        "pairs_iou_gt_0_5": int((ious > 0.5).sum()),
    }


def summarize_overlap(df, label):
    return {
        "dataset": label,
        "images": int(len(df)),
        "mean_pair_count": round(float(df["pair_count"].mean()), 3),
        "mean_max_pairwise_iou": round(float(df["max_pairwise_iou"].mean()), 4),
        "mean_pairs_iou_gt_0_1": round(float(df["pairs_iou_gt_0_1"].mean()), 3),
        "images_with_any_iou_gt_0_1": int((df["pairs_iou_gt_0_1"] > 0).sum()),
        "images_with_any_iou_gt_0_3": int((df["pairs_iou_gt_0_3"] > 0).sum()),
        "images_with_any_iou_gt_0_5": int((df["pairs_iou_gt_0_5"] > 0).sum()),
        "images_with_20plus_iou_gt_0_1_pairs": int((df["pairs_iou_gt_0_1"] >= 20).sum()),
    }


def crowding_distribution(df, label):
    order = ["0", "1-4", "5-19", "20+"]
    counts = df["crowding_bucket"].value_counts().reindex(order, fill_value=0)

    return pd.DataFrame({
        "dataset": label,
        "crowding_bucket": order,
        "image_count": counts.values,
        "image_share_pct": 100 * counts.values / len(df),
    })


def main():
    if not DATASET_INDEX.exists():
        raise FileNotFoundError(
            f"Dataset index not found: {DATASET_INDEX}. "
            "Run experiments/scripts/02_build_dataset_index.py first."
        )

    if not SELECTED_SAMPLE_INDEX.exists():
        raise FileNotFoundError(
            f"Selected sample index not found: {SELECTED_SAMPLE_INDEX}. "
            "Run experiments/scripts/02_dataset_sampling.py first."
        )

    idx = pd.read_csv(DATASET_INDEX)
    selected = pd.read_csv(SELECTED_SAMPLE_INDEX)

    rows = []

    for row_num, row in idx.iterrows():
        label_path = PROJECT_ROOT / row["label_path"]
        metrics = compute_overlap_for_label(label_path)

        rows.append({
            "image_file": row["image_file"],
            "image_path": row["image_path"],
            "label_path": row["label_path"],
            "num_objects": int(row["num_objects"]),
            **metrics,
        })

        if (row_num + 1) % 1000 == 0:
            print(f"Processed {row_num + 1}/{len(idx)} images")

    overlap = pd.DataFrame(rows)
    overlap["crowding_bucket"] = overlap["pairs_iou_gt_0_1"].apply(crowding_bucket)

    overlap_path = SAMPLING_OUTPUT_DIR / "overlap_profile.csv"
    overlap.to_csv(overlap_path, index=False)

    selected_files = set(selected["image_file"])
    selected_overlap = overlap[overlap["image_file"].isin(selected_files)].copy()

    selected_name = "rare_aware_density_stratified_5000"

    summary = pd.DataFrame([
        summarize_overlap(overlap, "full_dataset"),
        summarize_overlap(selected_overlap, selected_name),
    ])

    summary_path = SAMPLING_OUTPUT_DIR / "overlap_summary.csv"
    summary.to_csv(summary_path, index=False)

    full_crowding = crowding_distribution(overlap, "full_dataset")
    selected_crowding = crowding_distribution(selected_overlap, selected_name)

    crowding_compare = full_crowding.merge(
        selected_crowding,
        on="crowding_bucket",
        suffixes=("_full", "_sample"),
    )

    crowding_compare["image_share_diff_pp"] = (
        crowding_compare["image_share_pct_sample"] - crowding_compare["image_share_pct_full"]
    )

    crowding_path = SAMPLING_OUTPUT_DIR / "crowding_distribution_comparison.csv"
    crowding_compare.to_csv(crowding_path, index=False)

    x = np.arange(len(crowding_compare))
    width = 0.38

    plt.figure(figsize=(9, 6))
    plt.bar(x - width / 2, crowding_compare["image_share_pct_full"], width=width, label="Full dataset")
    plt.bar(x + width / 2, crowding_compare["image_share_pct_sample"], width=width, label="Selected sample")
    plt.xticks(x, crowding_compare["crowding_bucket"])
    plt.xlabel("Box-pair overlap bucket: number of pairs with IoU > 0.1")
    plt.ylabel("Image share (%)")
    plt.title("Full dataset vs selected sample crowding distribution")
    plt.legend()
    plt.tight_layout()

    figure_path = FIGURE_DIR / "04_crowding_distribution.png"
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close()

    print()
    print("OVERLAP SUMMARY")
    print(summary.to_string(index=False))

    print()
    print("CROWDING DISTRIBUTION")
    print(crowding_compare.to_string(index=False))

    print()
    print("Wrote:")
    print(overlap_path)
    print(summary_path)
    print(crowding_path)
    print(figure_path)


if __name__ == "__main__":
    main()