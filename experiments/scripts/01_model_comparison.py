from pathlib import Path
import sys
import argparse
import json
import time

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR = EXPERIMENTS_DIR / "outputs"
MODEL_SELECTION_DIR = OUTPUT_DIR / "model_selection"
MODEL_SELECTION_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from techtrack.modules.inference.model import Detector
from techtrack.modules.inference.nms import NMS
from techtrack.modules.utils.metrics import (
    match_detections,
    calculate_precision_recall_curve,
    calculate_map_x_point_interpolated,
)


DATASET_INDEX = OUTPUT_DIR / "dataset_index.csv"
CLASS_FILE = PROJECT_ROOT / "techtrack" / "storage" / "yolo_model_1" / "logistics.names"

SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
MAP_IOU_THRESHOLD = 0.5
EVAL_TYPE = "combined"

MODELS = {
    "model1": {
        "weights": PROJECT_ROOT / "techtrack/storage/yolo_model_1/yolov4-tiny-logistics_size_416_1.weights",
        "cfg": PROJECT_ROOT / "techtrack/storage/yolo_model_1/yolov4-tiny-logistics_size_416_1.cfg",
        "names": PROJECT_ROOT / "techtrack/storage/yolo_model_1/logistics.names",
    },
    "model2": {
        "weights": PROJECT_ROOT / "techtrack/storage/yolo_model_2/yolov4-tiny-logistics_size_416_2.weights",
        "cfg": PROJECT_ROOT / "techtrack/storage/yolo_model_2/yolov4-tiny-logistics_size_416_2.cfg",
        "names": PROJECT_ROOT / "techtrack/storage/yolo_model_2/logistics.names",
    },
}

PRED_COLUMNS = [
    "model",
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

GT_COLUMNS = [
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


def yolo_label_to_xywh(label_path: Path, image_w: int, image_h: int, classes):
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

        rows.append({
            "class_id": class_id,
            "class_name": classes[class_id],
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": box_w,
            "bbox_h": box_h,
        })

    return rows


def build_ground_truth(idx, classes, run_label, force=False):
    gt_path = MODEL_SELECTION_DIR / f"ground_truth_{run_label}.csv"

    if gt_path.exists() and not force:
        print(f"[SKIP] Ground truth already exists: {gt_path}")
        return pd.read_csv(gt_path)

    print("=" * 80)
    print("Building ground-truth table")
    print("=" * 80)

    rows = []

    for row_num, (_, row) in enumerate(idx.iterrows(), start=1):
        image_path = PROJECT_ROOT / row["image_path"]
        label_path = PROJECT_ROOT / row["label_path"]

        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[WARN] Could not read image for ground-truth conversion: {image_path}")
            continue

        image_h, image_w = frame.shape[:2]

        for gt in yolo_label_to_xywh(label_path, image_w, image_h, classes):
            rows.append({
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
            print(f"[GT] processed {row_num}/{len(idx)} images")

    gt_df = pd.DataFrame(rows, columns=GT_COLUMNS)
    gt_df.to_csv(gt_path, index=False)

    print(f"[WRITE] {gt_path} rows={len(gt_df)}")
    return gt_df


def run_inference_for_model(model_name, paths, idx, classes, run_label, force=False):
    pred_path = MODEL_SELECTION_DIR / f"{model_name}_predictions_{run_label}.csv"

    if pred_path.exists() and not force:
        print(f"[SKIP] Predictions already exist: {pred_path}")
        return pd.read_csv(pred_path)

    print("=" * 80)
    print(f"Running inference: {model_name}")
    print("=" * 80)

    detector = Detector(
        str(paths["weights"]),
        str(paths["cfg"]),
        str(paths["names"]),
        score_threshold=SCORE_THRESHOLD,
    )

    nms = NMS(
        score_threshold=SCORE_THRESHOLD,
        nms_iou_threshold=NMS_THRESHOLD,
    )

    rows = []
    start = time.time()

    for row_num, (_, row) in enumerate(idx.iterrows(), start=1):
        image_path = PROJECT_ROOT / row["image_path"]
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        outputs = detector.predict(frame)
        bboxes, class_ids, scores, class_scores = detector.post_process(outputs)

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
                "model": model_name,
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

        if row_num % 500 == 0:
            elapsed = time.time() - start
            print(
                f"[{model_name}] processed {row_num}/{len(idx)} images | "
                f"detections={len(rows)} | elapsed={elapsed:.1f}s"
            )

    pred_df = pd.DataFrame(rows, columns=PRED_COLUMNS)
    pred_df.to_csv(pred_path, index=False)

    elapsed = time.time() - start
    print(f"[WRITE] {pred_path} rows={len(pred_df)}")
    print(f"[DONE] {model_name} elapsed seconds: {elapsed:.2f}")

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


def evaluate_with_metrics_py(model_name, idx, pred_df, gt_df, classes):
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
            "model": model_name,
            "class_id": class_id,
            "class_name": class_name,
            "ground_truth_count": gt_count,
            "prediction_count": prediction_count,
            "ap_11_point": per_class_ap,
        })

    per_class_df = pd.DataFrame(per_class_rows)

    summary = {
        "model": model_name,
        "mAP@0.5_11_point": map_score,
        "total_ground_truth": int(len(gt_df)),
        "total_predictions_after_nms": int(len(pred_df)),
        "evaluation_rows": int(len(y_true)),
        "score_threshold": SCORE_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "map_iou_threshold": MAP_IOU_THRESHOLD,
        "eval_type": EVAL_TYPE,
    }

    return summary, per_class_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not DATASET_INDEX.exists():
        raise FileNotFoundError(
            f"Dataset index not found: {DATASET_INDEX}. "
            "Run experiments/scripts/02_build_dataset_index.py first."
        )

    classes = load_classes()
    idx = pd.read_csv(DATASET_INDEX)

    if args.max_images is not None:
        idx = idx.head(args.max_images).copy()
        run_label = f"first_{args.max_images}"
        print(f"[INFO] Running model comparison on first {args.max_images} images.")
    else:
        run_label = "full"
        print("[INFO] Running model comparison on full dataset.")

    print(f"[INFO] Images selected: {len(idx)}")
    print(f"[INFO] Classes: {len(classes)}")

    gt_df = build_ground_truth(idx, classes, run_label=run_label, force=args.force)

    summaries = []
    per_class_all = []

    for model_name, paths in MODELS.items():
        pred_df = run_inference_for_model(
            model_name,
            paths,
            idx,
            classes,
            run_label=run_label,
            force=args.force,
        )

        summary, per_class_df = evaluate_with_metrics_py(
            model_name,
            idx,
            pred_df,
            gt_df,
            classes,
        )

        summaries.append(summary)
        per_class_all.append(per_class_df)

    summary_df = pd.DataFrame(summaries)
    per_class_df = pd.concat(per_class_all, axis=0)

    comparison = (
        per_class_df.pivot(
            index=["class_id", "class_name", "ground_truth_count"],
            columns="model",
            values="ap_11_point",
        )
        .reset_index()
        .rename(columns={"model1": "model1_ap", "model2": "model2_ap"})
    )

    comparison["ap_difference_model2_minus_model1"] = (
        comparison["model2_ap"] - comparison["model1_ap"]
    )

    comparison["better_model"] = np.where(
        comparison["ap_difference_model2_minus_model1"] > 0,
        "model2",
        np.where(
            comparison["ap_difference_model2_minus_model1"] < 0,
            "model1",
            "tie",
        ),
    )

    summary_path = MODEL_SELECTION_DIR / f"model_summary_{run_label}.csv"
    per_class_path = MODEL_SELECTION_DIR / f"per_class_metrics_{run_label}.csv"
    comparison_path = MODEL_SELECTION_DIR / f"per_class_ap_comparison_{run_label}.csv"

    summary_df.to_csv(summary_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    print()
    print("MODEL SELECTION SUMMARY")
    print(summary_df.to_string(index=False))

    print()
    print("PER-CLASS AP COMPARISON")
    print(comparison.sort_values("ap_difference_model2_minus_model1", ascending=False).to_string(index=False))

    print()
    print("[WRITE]", summary_path)
    print("[WRITE]", per_class_path)
    print("[WRITE]", comparison_path)


if __name__ == "__main__":
    main()