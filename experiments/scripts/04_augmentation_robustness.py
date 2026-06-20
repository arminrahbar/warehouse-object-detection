from pathlib import Path
import sys
import argparse
import json
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

OUTPUT_DIR = EXPERIMENTS_DIR / "outputs"
SAMPLING_OUTPUT_DIR = OUTPUT_DIR / "dataset_sampling"
AUGMENTATION_OUTPUT_DIR = OUTPUT_DIR / "augmentation_robustness"
FIGURE_DIR = EXPERIMENTS_DIR / "figures" / "04_augmentation_robustness"

AUGMENTATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from techtrack.modules.inference.model import Detector
from techtrack.modules.inference.nms import NMS
from techtrack.modules.rectification.augmentation import Augmenter
from techtrack.modules.utils.metrics import (
    match_detections,
    calculate_precision_recall_curve,
    calculate_map_x_point_interpolated,
)

SAMPLE_INDEX = SAMPLING_OUTPUT_DIR / "selected_sample_index.csv"
CLASS_FILE = PROJECT_ROOT / "techtrack" / "storage" / "yolo_model_2" / "logistics.names"

MODEL2 = {
    "weights": PROJECT_ROOT / "techtrack/storage/yolo_model_2/yolov4-tiny-logistics_size_416_2.weights",
    "cfg": PROJECT_ROOT / "techtrack/storage/yolo_model_2/yolov4-tiny-logistics_size_416_2.cfg",
    "names": PROJECT_ROOT / "techtrack/storage/yolo_model_2/logistics.names",
}

MODEL_NAME = "model2"
DATASET_NAME = "rare_aware_density_stratified_5000"

SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
MAP_IOU_THRESHOLD = 0.5
EVAL_TYPE = "combined"

CONDITIONS = [
    {
        "tag": "original",
        "display": "Original",
        "type": "none",
    },
    {
        "tag": "gaussian_blur_k9",
        "display": "Gaussian blur",
        "type": "gaussian_blur",
        "kernel_size": 9,
        "sigma": 0,
    },
    {
        "tag": "vertical_flip",
        "display": "Vertical flip",
        "type": "vertical_flip",
    },
    {
        "tag": "brightness_increase",
        "display": "Brightness increase",
        "type": "brightness",
        "alpha": 1.15,
        "beta": 35,
    },
    {
        "tag": "brightness_decrease",
        "display": "Brightness decrease",
        "type": "brightness",
        "alpha": 0.85,
        "beta": -35,
    },
]

RAW_COLUMNS = [
    "model",
    "dataset",
    "augmentation_condition",
    "augmentation_display",
    "image_file",
    "image_path",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "class_id",
    "class_name",
    "object_score",
    "predicted_class_score",
    "combined_confidence",
    "class_scores_json",
]

PRED_COLUMNS = RAW_COLUMNS + ["nms_threshold"]

GT_COLUMNS = [
    "dataset",
    "augmentation_condition",
    "augmentation_display",
    "image_file",
    "image_path",
    "class_id",
    "class_name",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
]


def load_classes():
    return [line.strip() for line in CLASS_FILE.read_text().splitlines() if line.strip()]


def apply_condition_to_image(image_bgr, condition):
    condition_type = condition["type"]

    if condition_type == "none":
        return image_bgr

    if condition_type == "gaussian_blur":
        return Augmenter.gaussian_blur(
            image=image_bgr,
            kernel_size=condition.get("kernel_size", 9),
            sigma=condition.get("sigma", 0),
        )

    if condition_type == "vertical_flip":
        return Augmenter.vertical_flip(image=image_bgr)

    if condition_type == "brightness":
        return Augmenter.change_brightness(
            image=image_bgr,
            alpha=condition.get("alpha", 1.0),
            beta=condition.get("beta", 0),
        )

    raise ValueError(f"Unsupported condition type: {condition_type}")


def yolo_label_to_xywh_for_condition(label_path: Path, image_w: int, image_h: int, classes, condition):
    rows = []
    text = label_path.read_text().strip()

    if not text:
        return rows

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))
        if class_id < 0 or class_id >= len(classes):
            continue

        x_center = float(parts[1]) * image_w
        y_center = float(parts[2]) * image_h
        box_w = float(parts[3]) * image_w
        box_h = float(parts[4]) * image_h

        x = x_center - box_w / 2
        y = y_center - box_h / 2

        if condition["type"] == "vertical_flip":
            # Convert top-left y for vertically flipped image.
            y = image_h - y - box_h

        # Clip only to protect against tiny floating point boundary issues.
        x = float(np.clip(x, 0, image_w))
        y = float(np.clip(y, 0, image_h))
        box_w = float(np.clip(box_w, 0, image_w))
        box_h = float(np.clip(box_h, 0, image_h))

        rows.append({
            "class_id": class_id,
            "class_name": classes[class_id],
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": box_w,
            "bbox_h": box_h,
        })

    return rows


def build_ground_truth(idx, classes, condition, run_label, force=False):
    tag = condition["tag"]
    gt_path = AUGMENTATION_OUTPUT_DIR / f"ground_truth_{tag}_{run_label}.csv"

    if gt_path.exists() and not force:
        print(f"[SKIP] Ground truth already exists: {gt_path}")
        return pd.read_csv(gt_path)

    print("=" * 100)
    print(f"Building ground-truth table for condition: {condition['display']}")
    print("=" * 100)

    rows = []

    for row_num, (_, row) in enumerate(idx.iterrows(), start=1):
        image_path = PROJECT_ROOT / row["image_path"]
        label_path = PROJECT_ROOT / row["label_path"]

        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[WARN] Could not read image for ground-truth conversion: {image_path}")
            continue

        image_h, image_w = frame.shape[:2]

        for gt in yolo_label_to_xywh_for_condition(label_path, image_w, image_h, classes, condition):
            rows.append({
                "dataset": DATASET_NAME,
                "augmentation_condition": tag,
                "augmentation_display": condition["display"],
                "image_file": row["image_file"],
                "image_path": row["image_path"],
                "class_id": gt["class_id"],
                "class_name": gt["class_name"],
                "bbox_x": gt["bbox_x"],
                "bbox_y": gt["bbox_y"],
                "bbox_w": gt["bbox_w"],
                "bbox_h": gt["bbox_h"],
            })

        if row_num % 1000 == 0:
            print(f"[GT {tag}] processed {row_num}/{len(idx)} images")

    gt_df = pd.DataFrame(rows, columns=GT_COLUMNS)
    gt_df.to_csv(gt_path, index=False)

    print(f"[WRITE] {gt_path} rows={len(gt_df)}")
    return gt_df


def run_raw_inference(idx, classes, condition, run_label, force=False):
    tag = condition["tag"]
    raw_path = AUGMENTATION_OUTPUT_DIR / f"{MODEL_NAME}_raw_predictions_{tag}_{run_label}.csv"

    if raw_path.exists() and not force:
        print(f"[SKIP] Raw predictions already exist: {raw_path}")
        return pd.read_csv(raw_path)

    print("=" * 100)
    print(f"Running inference for condition: {condition['display']}")
    print("=" * 100)

    detector = Detector(
        str(MODEL2["weights"]),
        str(MODEL2["cfg"]),
        str(MODEL2["names"]),
        score_threshold=SCORE_THRESHOLD,
    )

    rows = []
    start = time.time()

    for row_num, (_, row) in enumerate(idx.iterrows(), start=1):
        image_path = PROJECT_ROOT / row["image_path"]
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        frame = apply_condition_to_image(frame, condition)

        outputs = detector.predict(frame)
        bboxes, class_ids, scores, class_scores = detector.post_process(outputs)

        for det_idx, bbox in enumerate(bboxes):
            class_id = int(class_ids[det_idx])
            object_score = float(scores[det_idx])

            score_vector = np.asarray(class_scores[det_idx], dtype=float).ravel()
            score_vector = [float(v) for v in score_vector]

            if 0 <= class_id < len(score_vector):
                predicted_class_score = float(score_vector[class_id])
            else:
                predicted_class_score = 0.0

            combined_confidence = object_score * predicted_class_score

            rows.append({
                "model": MODEL_NAME,
                "dataset": DATASET_NAME,
                "augmentation_condition": tag,
                "augmentation_display": condition["display"],
                "image_file": row["image_file"],
                "image_path": row["image_path"],
                "bbox_x": float(bbox[0]),
                "bbox_y": float(bbox[1]),
                "bbox_w": float(bbox[2]),
                "bbox_h": float(bbox[3]),
                "class_id": class_id,
                "class_name": classes[class_id] if 0 <= class_id < len(classes) else "unknown",
                "object_score": object_score,
                "predicted_class_score": predicted_class_score,
                "combined_confidence": combined_confidence,
                "class_scores_json": json.dumps(score_vector),
            })

        if row_num % 250 == 0:
            elapsed = time.time() - start
            print(
                f"[RAW {tag}] processed {row_num}/{len(idx)} images | "
                f"raw detections={len(rows)} | elapsed={elapsed:.1f}s"
            )

    raw_df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    raw_df.to_csv(raw_path, index=False)

    elapsed = time.time() - start
    print(f"[WRITE] {raw_path} rows={len(raw_df)}")
    print(f"[DONE] raw inference elapsed seconds: {elapsed:.2f}")

    return raw_df


def apply_fixed_nms(raw_df, idx, classes, condition, run_label, force=False):
    tag = condition["tag"]
    pred_path = AUGMENTATION_OUTPUT_DIR / f"{MODEL_NAME}_predictions_{tag}_nms_0_5_{run_label}.csv"

    if pred_path.exists() and not force:
        print(f"[SKIP] NMS predictions already exist: {pred_path}")
        return pd.read_csv(pred_path)

    print("=" * 100)
    print(f"Applying fixed NMS threshold = {NMS_THRESHOLD} for condition: {condition['display']}")
    print("=" * 100)

    nms = NMS(
        score_threshold=SCORE_THRESHOLD,
        nms_iou_threshold=NMS_THRESHOLD,
    )

    raw_groups = {k: v for k, v in raw_df.groupby("image_file")} if len(raw_df) else {}
    rows = []

    for row_num, (_, row) in enumerate(idx.iterrows(), start=1):
        image_file = row["image_file"]
        group = raw_groups.get(image_file)

        if group is None or len(group) == 0:
            continue

        bboxes = group[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].astype(float).values.tolist()
        class_ids = group["class_id"].astype(int).tolist()
        scores = group["object_score"].astype(float).tolist()
        class_scores = [
            json.loads(value)
            for value in group["class_scores_json"].tolist()
        ]

        filtered_boxes, filtered_classes, filtered_scores, filtered_class_scores = nms.filter(
            bboxes,
            class_ids,
            scores,
            class_scores,
        )

        for det_idx, bbox in enumerate(filtered_boxes):
            class_id = int(filtered_classes[det_idx])
            object_score = float(filtered_scores[det_idx])

            score_vector = np.asarray(filtered_class_scores[det_idx], dtype=float).ravel()
            score_vector = [float(v) for v in score_vector]

            if 0 <= class_id < len(score_vector):
                predicted_class_score = float(score_vector[class_id])
            else:
                predicted_class_score = 0.0

            combined_confidence = object_score * predicted_class_score

            rows.append({
                "model": MODEL_NAME,
                "dataset": DATASET_NAME,
                "augmentation_condition": tag,
                "augmentation_display": condition["display"],
                "image_file": row["image_file"],
                "image_path": row["image_path"],
                "bbox_x": float(bbox[0]),
                "bbox_y": float(bbox[1]),
                "bbox_w": float(bbox[2]),
                "bbox_h": float(bbox[3]),
                "class_id": class_id,
                "class_name": classes[class_id] if 0 <= class_id < len(classes) else "unknown",
                "object_score": object_score,
                "predicted_class_score": predicted_class_score,
                "combined_confidence": combined_confidence,
                "class_scores_json": json.dumps(score_vector),
                "nms_threshold": NMS_THRESHOLD,
            })

        if row_num % 1000 == 0:
            print(f"[NMS {tag}] processed {row_num}/{len(idx)} images")

    pred_df = pd.DataFrame(rows, columns=PRED_COLUMNS)
    pred_df.to_csv(pred_path, index=False)

    print(f"[WRITE] {pred_path} rows={len(pred_df)}")
    return pred_df


def build_metric_lists(idx, pred_df, gt_df):
    pred_groups = {k: v for k, v in pred_df.groupby("image_file")} if len(pred_df) else {}
    gt_groups = {k: v for k, v in gt_df.groupby("image_file")} if len(gt_df) else {}

    boxes = []
    pred_classes = []
    scores = []
    cls_scores = []

    gt_boxes = []
    gt_classes = []

    for _, row in idx.iterrows():
        image_file = row["image_file"]

        pred_group = pred_groups.get(image_file)
        if pred_group is None:
            boxes.append([])
            pred_classes.append([])
            scores.append([])
            cls_scores.append([])
        else:
            boxes.append(pred_group[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].values.tolist())
            pred_classes.append(pred_group["class_id"].astype(int).tolist())
            scores.append(pred_group["object_score"].astype(float).tolist())
            cls_scores.append([
                json.loads(value)
                for value in pred_group["class_scores_json"].tolist()
            ])

        gt_group = gt_groups.get(image_file)
        if gt_group is None:
            gt_boxes.append([])
            gt_classes.append([])
        else:
            gt_boxes.append(gt_group[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].values.tolist())
            gt_classes.append(gt_group["class_id"].astype(int).tolist())

    return boxes, pred_classes, scores, cls_scores, gt_boxes, gt_classes


def evaluate_with_metrics_py(idx, pred_df, gt_df, classes, condition):
    boxes, pred_classes, scores, cls_scores, gt_boxes, gt_classes = build_metric_lists(
        idx,
        pred_df,
        gt_df,
    )

    y_true, pred_scores = match_detections(
        boxes=boxes,
        classes=pred_classes,
        scores=scores,
        cls_scores=cls_scores,
        gt_boxes=gt_boxes,
        gt_classes=gt_classes,
        map_iou_threshold=MAP_IOU_THRESHOLD,
        eval_type=EVAL_TYPE,
    )

    precision, recall, thresholds = calculate_precision_recall_curve(
        y_true,
        pred_scores,
        num_classes=len(classes),
    )

    precision_recall_points = {
        class_id: list(zip(recall[class_id], precision[class_id]))
        for class_id in range(len(classes))
    }

    map_score = calculate_map_x_point_interpolated(
        precision_recall_points,
        num_classes=len(classes),
        num_interpolated_points=11,
    )

    per_class_rows = []

    for class_id, class_name in enumerate(classes):
        per_class_ap = calculate_map_x_point_interpolated(
            {0: precision_recall_points[class_id]},
            num_classes=1,
            num_interpolated_points=11,
        )

        gt_count = int((gt_df["class_id"] == class_id).sum())
        prediction_count = int((pred_df["class_id"] == class_id).sum()) if len(pred_df) else 0

        per_class_rows.append({
            "model": MODEL_NAME,
            "dataset": DATASET_NAME,
            "augmentation_condition": condition["tag"],
            "augmentation_display": condition["display"],
            "class_id": class_id,
            "class_name": class_name,
            "ground_truth_count": gt_count,
            "prediction_count": prediction_count,
            "ap_11_point": per_class_ap,
        })

    summary = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "augmentation_condition": condition["tag"],
        "augmentation_display": condition["display"],
        "mAP@0.5_11_point": map_score,
        "total_ground_truth": int(len(gt_df)),
        "total_predictions_after_nms": int(len(pred_df)),
        "evaluation_rows": int(len(y_true)),
        "score_threshold": SCORE_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "map_iou_threshold": MAP_IOU_THRESHOLD,
        "eval_type": EVAL_TYPE,
    }

    return summary, pd.DataFrame(per_class_rows)


def add_change_from_original(summary_df, per_class_df):
    original_map = float(
        summary_df.loc[
            summary_df["augmentation_condition"] == "original",
            "mAP@0.5_11_point"
        ].iloc[0]
    )

    original_predictions = int(
        summary_df.loc[
            summary_df["augmentation_condition"] == "original",
            "total_predictions_after_nms"
        ].iloc[0]
    )

    summary_df = summary_df.copy()
    summary_df["mAP_change_vs_original"] = summary_df["mAP@0.5_11_point"] - original_map
    summary_df["mAP_percent_change_vs_original"] = (
        summary_df["mAP_change_vs_original"] / original_map * 100
        if original_map != 0 else 0.0
    )
    summary_df["prediction_change_vs_original"] = (
        summary_df["total_predictions_after_nms"] - original_predictions
    )

    original_ap = (
        per_class_df[per_class_df["augmentation_condition"] == "original"]
        [["class_id", "ap_11_point"]]
        .rename(columns={"ap_11_point": "original_ap_11_point"})
    )

    per_class_df = per_class_df.merge(original_ap, on="class_id", how="left")
    per_class_df["ap_change_vs_original"] = (
        per_class_df["ap_11_point"] - per_class_df["original_ap_11_point"]
    )

    return summary_df, per_class_df


def build_figures(summary_df, per_class_df):
    ordered = summary_df.copy()
    condition_order = [c["tag"] for c in CONDITIONS]
    display_order = [c["display"] for c in CONDITIONS]
    order_map = {tag: i for i, tag in enumerate(condition_order)}
    ordered["condition_order"] = ordered["augmentation_condition"].map(order_map)
    ordered = ordered.sort_values("condition_order")

    plt.figure(figsize=(9, 5))
    plt.bar(ordered["augmentation_display"], ordered["mAP@0.5_11_point"])
    plt.xlabel("Image condition")
    plt.ylabel("mAP@0.5, 11-point interpolated")
    plt.title("mAP@0.5 by image condition")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig1 = FIGURE_DIR / "02_map_by_condition.png"
    plt.savefig(fig1, dpi=200, bbox_inches="tight")
    plt.close()

    drop_df = ordered[ordered["augmentation_condition"] != "original"].copy()
    drop_df["mAP_drop_vs_original"] = -drop_df["mAP_change_vs_original"]

    plt.figure(figsize=(9, 5))
    plt.bar(drop_df["augmentation_display"], drop_df["mAP_drop_vs_original"])
    plt.xlabel("Image condition")
    plt.ylabel("mAP@0.5 drop vs original")
    plt.title("Robustness degradation under image augmentations")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig2 = FIGURE_DIR / "03_map_drop_vs_baseline.png"
    plt.savefig(fig2, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(ordered["augmentation_display"], ordered["total_predictions_after_nms"])
    plt.xlabel("Image condition")
    plt.ylabel("Predictions retained after NMS")
    plt.title("Post-NMS prediction count by augmentation condition")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig3 = FIGURE_DIR / "04_prediction_count_by_condition.png"
    plt.savefig(fig3, dpi=200, bbox_inches="tight")
    plt.close()

    non_original = per_class_df[per_class_df["augmentation_condition"] != "original"].copy()
    non_original["ap_drop_vs_original"] = -non_original["ap_change_vs_original"]

    top_drops = (
        non_original
        .sort_values("ap_drop_vs_original", ascending=False)
        .head(12)
        .copy()
    )
    top_drops["label"] = top_drops["augmentation_display"] + " / " + top_drops["class_name"]
    plot_df = top_drops.sort_values("ap_drop_vs_original", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["label"], plot_df["ap_drop_vs_original"])
    plt.xlabel("AP@0.5 drop vs original")
    plt.ylabel("Augmentation / class")
    plt.title("Largest per-class AP drops under augmentation")
    plt.tight_layout()
    fig4 = FIGURE_DIR / "05_largest_per_class_ap_drops.png"
    plt.savefig(fig4, dpi=200, bbox_inches="tight")
    plt.close()

    print("[WRITE]", fig1)
    print("[WRITE]", fig2)
    print("[WRITE]", fig3)
    print("[WRITE]", fig4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not SAMPLE_INDEX.exists():
        raise FileNotFoundError(
            f"Missing selected sample index: {SAMPLE_INDEX}. "
            "Run experiments/scripts/02_dataset_sampling.py first."
        )

    classes = load_classes()
    idx = pd.read_csv(SAMPLE_INDEX)

    if args.max_images is not None:
        idx = idx.head(args.max_images).copy()
        run_label = f"first_{args.max_images}"
        print(f"[INFO] Running augmentation robustness sweep on first {args.max_images} selected sample images.")
    else:
        run_label = "sample5000"
        print("[INFO] Running augmentation robustness sweep on full selected 5,000-image sample.")

    print(f"[INFO] Images selected: {len(idx)}")
    print(f"[INFO] Classes: {len(classes)}")
    print(f"[INFO] Model held constant: {MODEL_NAME}")
    print(f"[INFO] Dataset held constant: {DATASET_NAME}")
    print(f"[INFO] Score threshold held constant: {SCORE_THRESHOLD}")
    print(f"[INFO] NMS threshold held constant: {NMS_THRESHOLD}")
    print(f"[INFO] Evaluation IoU threshold held constant: {MAP_IOU_THRESHOLD}")
    print(f"[INFO] Augmentation conditions tested: {[c['tag'] for c in CONDITIONS]}")

    summary_rows = []
    per_class_all = []

    for condition in CONDITIONS:
        gt_df = build_ground_truth(
            idx,
            classes,
            condition,
            run_label=run_label,
            force=args.force,
        )

        raw_df = run_raw_inference(
            idx,
            classes,
            condition,
            run_label=run_label,
            force=args.force,
        )

        pred_df = apply_fixed_nms(
            raw_df,
            idx,
            classes,
            condition,
            run_label=run_label,
            force=args.force,
        )

        summary, per_class_df = evaluate_with_metrics_py(
            idx,
            pred_df,
            gt_df,
            classes,
            condition,
        )

        summary_rows.append(summary)
        per_class_all.append(per_class_df)

    summary_df = pd.DataFrame(summary_rows)
    per_class_df = pd.concat(per_class_all, axis=0, ignore_index=True)

    summary_df, per_class_df = add_change_from_original(summary_df, per_class_df)

    summary_path = AUGMENTATION_OUTPUT_DIR / f"summary_by_condition_{run_label}.csv"
    per_class_path = AUGMENTATION_OUTPUT_DIR / f"per_class_ap_by_condition_{run_label}.csv"

    summary_df.to_csv(summary_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)

    print()
    print("AUGMENTATION ROBUSTNESS SUMMARY")
    print(summary_df.to_string(index=False))

    print()
    print("LARGEST PER-CLASS AP DROPS")
    print(
        per_class_df[per_class_df["augmentation_condition"] != "original"]
        .sort_values("ap_change_vs_original")
        .head(15)
        .to_string(index=False)
    )

    print()
    print("[WRITE]", summary_path)
    print("[WRITE]", per_class_path)

    build_figures(summary_df, per_class_df)


if __name__ == "__main__":
    main()
