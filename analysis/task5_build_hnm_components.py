from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "analysis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_PATH = OUT / "task2_selected_sample_index.csv"
RAW_PRED_PATH = OUT / "task3_nts_model2_raw_predictions_selected_sample.csv"
GT_PATH = OUT / "task3_nts_ground_truth_selected_sample.csv"

COMPONENT_PATH = OUT / "task5_hnm_image_loss_components_selected_sample.csv"

IOU_THRESHOLD = 0.5
NUM_CLASSES = 20


def xywh_to_xyxy_array(df):
    if df is None or len(df) == 0:
        return np.empty((0, 4), dtype=float)

    x1 = df["bbox_x"].astype(float).to_numpy()
    y1 = df["bbox_y"].astype(float).to_numpy()
    x2 = x1 + df["bbox_w"].astype(float).to_numpy()
    y2 = y1 + df["bbox_h"].astype(float).to_numpy()

    return np.stack([x1, y1, x2, y2], axis=1)


def iou_xyxy_one_to_many(box, boxes):
    if boxes.size == 0:
        return np.empty((0,), dtype=float)

    ax1, ay1, ax2, ay2 = box

    bx1 = boxes[:, 0]
    by1 = boxes[:, 1]
    bx2 = boxes[:, 2]
    by2 = boxes[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)

    union = area_a + area_b - intersection
    return np.where(union > 0, intersection / union, 0.0)


def parse_class_scores(value):
    if isinstance(value, str):
        scores = json.loads(value)
    else:
        scores = []

    if len(scores) < NUM_CLASSES:
        scores = scores + [0.0] * (NUM_CLASSES - len(scores))
    elif len(scores) > NUM_CLASSES:
        scores = scores[:NUM_CLASSES]

    return np.asarray(scores, dtype=float)


def compute_image_components(image_row, pred_group, gt_group):
    loc_loss = 0.0
    conf_loss_obj = 0.0
    conf_loss_noobj = 0.0
    class_loss = 0.0

    raw_prediction_count = 0 if pred_group is None else len(pred_group)
    gt_count = 0 if gt_group is None else len(gt_group)

    matched_prediction_count = 0
    false_positive_prediction_count = 0
    matched_gt_indices = set()

    if pred_group is None or len(pred_group) == 0:
        return {
            "loc_loss": 0.0,
            "conf_loss_obj": 0.0,
            "conf_loss_noobj": 0.0,
            "class_loss": 0.0,
            "raw_prediction_count": 0,
            "ground_truth_count": gt_count,
            "matched_prediction_count": 0,
            "false_positive_prediction_count": 0,
            "matched_gt_count": 0,
            "missed_gt_count": gt_count,
        }

    pred_boxes = xywh_to_xyxy_array(pred_group)
    pred_objectness = pred_group["object_score"].astype(float).to_numpy()
    pred_class_scores = [parse_class_scores(v) for v in pred_group["class_scores_json"].tolist()]

    if gt_group is None or len(gt_group) == 0:
        conf_loss_noobj = float(np.sum(pred_objectness ** 2))
        return {
            "loc_loss": 0.0,
            "conf_loss_obj": 0.0,
            "conf_loss_noobj": conf_loss_noobj,
            "class_loss": 0.0,
            "raw_prediction_count": raw_prediction_count,
            "ground_truth_count": 0,
            "matched_prediction_count": 0,
            "false_positive_prediction_count": raw_prediction_count,
            "matched_gt_count": 0,
            "missed_gt_count": 0,
        }

    gt_boxes = xywh_to_xyxy_array(gt_group)
    gt_classes = gt_group["class_id"].astype(int).to_numpy()

    for pred_idx in range(len(pred_group)):
        pred_box = pred_boxes[pred_idx]
        objectness = float(pred_objectness[pred_idx])
        class_scores = pred_class_scores[pred_idx]

        ious = iou_xyxy_one_to_many(pred_box, gt_boxes)

        if len(ious) == 0:
            conf_loss_noobj += objectness ** 2
            false_positive_prediction_count += 1
            continue

        best_gt_idx = int(np.argmax(ious))
        best_iou = float(ious[best_gt_idx])

        if best_iou >= IOU_THRESHOLD:
            target_box = gt_boxes[best_gt_idx]
            target_class = int(gt_classes[best_gt_idx])

            loc_loss += float(np.sum((pred_box - target_box) ** 2))
            conf_loss_obj += float((1.0 - objectness) ** 2)

            correct_class_score = float(class_scores[target_class]) if 0 <= target_class < len(class_scores) else 0.0
            class_loss += float((1.0 - correct_class_score) ** 2)

            matched_prediction_count += 1
            matched_gt_indices.add(best_gt_idx)
        else:
            conf_loss_noobj += float(objectness ** 2)
            false_positive_prediction_count += 1

    matched_gt_count = len(matched_gt_indices)
    missed_gt_count = max(0, gt_count - matched_gt_count)

    return {
        "loc_loss": loc_loss,
        "conf_loss_obj": conf_loss_obj,
        "conf_loss_noobj": conf_loss_noobj,
        "class_loss": class_loss,
        "raw_prediction_count": raw_prediction_count,
        "ground_truth_count": gt_count,
        "matched_prediction_count": matched_prediction_count,
        "false_positive_prediction_count": false_positive_prediction_count,
        "matched_gt_count": matched_gt_count,
        "missed_gt_count": missed_gt_count,
    }


def main():
    sample = pd.read_csv(SAMPLE_PATH)
    raw = pd.read_csv(RAW_PRED_PATH)
    gt = pd.read_csv(GT_PATH)

    pred_groups = {k: v for k, v in raw.groupby("image_path")} if len(raw) else {}
    gt_groups = {k: v for k, v in gt.groupby("image_path")} if len(gt) else {}

    rows = []

    for i, (_, image_row) in enumerate(sample.iterrows(), start=1):
        image_path = image_row["image_path"]
        pred_group = pred_groups.get(image_path)
        gt_group = gt_groups.get(image_path)

        components = compute_image_components(image_row, pred_group, gt_group)

        rows.append({
            "image_file": image_row["image_file"],
            "image_path": image_row["image_path"],
            "num_objects": int(image_row["num_objects"]),
            "density_bucket": image_row["density_bucket"],
            "class_names_present": image_row["class_names_present"],
            **components,
        })

        if i % 1000 == 0:
            print(f"[COMPONENTS] processed {i}/{len(sample)} images")

    df = pd.DataFrame(rows)
    df.to_csv(COMPONENT_PATH, index=False)

    print()
    print("[WRITE]", COMPONENT_PATH)
    print("shape:", df.shape)

    print()
    print("=" * 100)
    print("COMPONENT SCALE SUMMARY")
    print("=" * 100)

    component_cols = [
        "loc_loss",
        "conf_loss_obj",
        "conf_loss_noobj",
        "class_loss",
        "raw_prediction_count",
        "ground_truth_count",
        "matched_prediction_count",
        "false_positive_prediction_count",
        "matched_gt_count",
        "missed_gt_count",
    ]

    print(df[component_cols].describe().T.to_string())

    print()
    print("=" * 100)
    print("NONZERO COUNTS")
    print("=" * 100)
    for col in ["loc_loss", "conf_loss_obj", "conf_loss_noobj", "class_loss"]:
        print(f"{col:20s}", int((df[col] > 0).sum()))

    print()
    print("=" * 100)
    print("ZERO-PREDICTION IMAGES")
    print("=" * 100)
    zero_pred = df[df["raw_prediction_count"] == 0]
    print("count:", len(zero_pred))
    print("mean GT objects:", zero_pred["ground_truth_count"].mean())
    print("median GT objects:", zero_pred["ground_truth_count"].median())
    print("max GT objects:", zero_pred["ground_truth_count"].max())

    print()
    print("=" * 100)
    print("TOP 10 BY EACH RAW COMPONENT")
    print("=" * 100)

    for col in ["loc_loss", "conf_loss_obj", "conf_loss_noobj", "class_loss", "missed_gt_count"]:
        print()
        print(f"--- Top 10 by {col} ---")
        top = df.sort_values(col, ascending=False).head(10)
        print(top[[
            "image_file",
            "density_bucket",
            "num_objects",
            "raw_prediction_count",
            "matched_prediction_count",
            "false_positive_prediction_count",
            "matched_gt_count",
            "missed_gt_count",
            "loc_loss",
            "conf_loss_obj",
            "conf_loss_noobj",
            "class_loss",
            "class_names_present",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
