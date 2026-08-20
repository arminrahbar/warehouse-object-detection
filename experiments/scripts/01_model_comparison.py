"""Compare two detector checkpoints under one fixed evaluation policy."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR = EXPERIMENTS_DIR / "outputs"
MODEL_SELECTION_DIR = OUTPUT_DIR / "model_selection"
FIGURE_DIR = EXPERIMENTS_DIR / "figures" / "01_model_selection"
MODEL_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from detector_service.modules.inference.nms import NMS
from detector_service.modules.utils.metrics import (
    calculate_map_x_point_interpolated,
    calculate_precision_recall_curve,
    match_detections,
)


DATASET_INDEX = OUTPUT_DIR / "dataset_index.csv"

DETECTOR_OBJECTNESS_THRESHOLD = 0.5
NMS_CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3
MAP_IOU_THRESHOLD = 0.5
EVAL_TYPE = "combined"

MODELS = {
    "model1": {
        "weights": Path(
            "detector_service/storage/yolo_model_1/"
            "yolov4-tiny-logistics_size_416_1.weights"
        ),
        "cfg": Path(
            "detector_service/storage/yolo_model_1/"
            "yolov4-tiny-logistics_size_416_1.cfg"
        ),
        "names": Path("detector_service/storage/yolo_model_1/logistics.names"),
    },
    "model2": {
        "weights": Path(
            "detector_service/storage/yolo_model_2/"
            "yolov4-tiny-logistics_size_416_2.weights"
        ),
        "cfg": Path(
            "detector_service/storage/yolo_model_2/"
            "yolov4-tiny-logistics_size_416_2.cfg"
        ),
        "names": Path("detector_service/storage/yolo_model_2/logistics.names"),
    },
}

RAW_COLUMNS = [
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
PRED_COLUMNS = RAW_COLUMNS + ["nms_threshold"]
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


def _cache_path(stem, run_label):
    return MODEL_SELECTION_DIR / f"{stem}_{run_label}.csv"


def _read_cache(path, force, description):
    if path.exists() and not force:
        print(f"[SKIP] {description} already exists: {path}")
        return pd.read_csv(path)
    return None


def _groups_by_image(table):
    if len(table) == 0:
        return {}
    return {image_file: group for image_file, group in table.groupby("image_file")}


def load_classes(asset_root, run_label):
    """Load class names from assets or reconstruct them from cached evidence."""

    class_file = asset_root / MODELS["model1"]["names"]
    if class_file.exists():
        return [
            name.strip()
            for name in class_file.read_text(encoding="utf-8").splitlines()
            if name.strip()
        ]

    cache_candidates = [
        _cache_path("ground_truth", run_label),
        _cache_path("model1_raw_predictions", run_label),
        _cache_path("model2_raw_predictions", run_label),
    ]
    available_mappings = [
        pd.read_csv(path, usecols=["class_id", "class_name"])
        for path in cache_candidates
        if path.exists()
    ]
    if not available_mappings:
        raise FileNotFoundError(
            f"Missing class-name file: {class_file}. No raw cache is available "
            "to recover the class mapping."
        )

    class_names = {}
    unique_rows = pd.concat(available_mappings, ignore_index=True).drop_duplicates()
    for row in unique_rows.itertuples():
        class_id = int(row.class_id)
        class_name = str(row.class_name)
        previous_name = class_names.get(class_id)
        if previous_name is not None and previous_name != class_name:
            raise ValueError(
                f"Conflicting names for class {class_id}: "
                f"{previous_name!r} and {class_name!r}"
            )
        class_names[class_id] = class_name

    if not class_names:
        raise ValueError("Cached model-comparison evidence has no class mapping")

    contiguous_ids = list(range(max(class_names) + 1))
    if sorted(class_names) != contiguous_ids:
        raise ValueError(
            "Cached class mapping is incomplete; expected contiguous class IDs "
            f"from 0 through {contiguous_ids[-1]}"
        )

    print("[INFO] Loaded class names from cached model-comparison evidence.")
    return [class_names[class_id] for class_id in contiguous_ids]


def yolo_label_to_xywh(label_path: Path, image_w: int, image_h: int, classes):
    """Convert normalized YOLO labels to pixel-space xywh dictionaries."""

    contents = label_path.read_text(encoding="utf-8").strip()
    if not contents:
        return []

    converted = []
    for label_line in contents.splitlines():
        fields = label_line.strip().split()
        if len(fields) < 5:
            continue

        class_id = int(float(fields[0]))
        if not 0 <= class_id < len(classes):
            continue

        center_x = float(fields[1]) * image_w
        center_y = float(fields[2]) * image_h
        width = float(fields[3]) * image_w
        height = float(fields[4]) * image_h
        converted.append(
            {
                "class_id": class_id,
                "class_name": classes[class_id],
                "bbox_x": center_x - width / 2,
                "bbox_y": center_y - height / 2,
                "bbox_w": width,
                "bbox_h": height,
            }
        )
    return converted


def build_ground_truth(idx, classes, asset_root, run_label, force=False):
    """Build or reuse the pixel-space ground-truth evidence table."""

    ground_truth_path = _cache_path("ground_truth", run_label)
    cached = _read_cache(ground_truth_path, force, "Ground truth")
    if cached is not None:
        return cached

    import cv2

    print("=" * 80)
    print("Building ground-truth table")
    print("=" * 80)
    records = []

    for row_number, (_, image) in enumerate(idx.iterrows(), start=1):
        image_path = asset_root / image["image_path"]
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(
                "[WARN] Could not read image for ground-truth conversion: "
                f"{image_path}"
            )
            continue

        image_height, image_width = frame.shape[:2]
        labels = yolo_label_to_xywh(
            asset_root / image["label_path"],
            image_width,
            image_height,
            classes,
        )
        for label in labels:
            records.append(
                {
                    "image_file": image["image_file"],
                    "image_path": image["image_path"],
                    **label,
                }
            )

        if row_number % 1000 == 0:
            print(f"[GT] processed {row_number}/{len(idx)} images")

    ground_truth = pd.DataFrame(records, columns=GT_COLUMNS)
    ground_truth.to_csv(ground_truth_path, index=False)
    print(f"[WRITE] {ground_truth_path} rows={len(ground_truth)}")
    return ground_truth


def serialize_detection(
    model_name,
    image_row,
    bbox,
    class_id,
    object_score,
    score_vector,
    classes,
):
    """Create a stable raw-evidence record for one decoded prediction."""

    class_id = int(class_id)
    object_score = float(object_score)
    probabilities = np.asarray(score_vector, dtype=float).ravel().tolist()
    probabilities = [float(probability) for probability in probabilities]
    class_probability = (
        float(probabilities[class_id])
        if 0 <= class_id < len(probabilities)
        else 0.0
    )

    return {
        "model": model_name,
        "image_file": image_row["image_file"],
        "image_path": image_row["image_path"],
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_w": float(bbox[2]),
        "bbox_h": float(bbox[3]),
        "class_id": class_id,
        "class_name": classes[class_id] if 0 <= class_id < len(classes) else "unknown",
        "object_score": object_score,
        "predicted_class_score": class_probability,
        "combined_confidence": object_score * class_probability,
        "class_scores_json": json.dumps(probabilities),
    }


def _resolve_model_assets(paths, asset_root):
    resolved = {
        asset_name: asset_root / relative_path
        for asset_name, relative_path in paths.items()
    }
    missing = [path for path in resolved.values() if not path.exists()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing model assets:\n{details}")
    return resolved


def run_raw_inference_for_model(
    model_name,
    paths,
    idx,
    classes,
    asset_root,
    run_label,
    force=False,
):
    """Run a checkpoint once and persist every objectness-qualified candidate."""

    raw_path = _cache_path(f"{model_name}_raw_predictions", run_label)
    cached = _read_cache(raw_path, force, "Raw predictions")
    if cached is not None:
        return cached

    import cv2

    from detector_service.modules.inference.model import Detector

    assets = _resolve_model_assets(paths, asset_root)
    print("=" * 80)
    print(f"Running raw inference: {model_name}")
    print("=" * 80)

    detector = Detector(
        str(assets["weights"]),
        str(assets["cfg"]),
        str(assets["names"]),
        score_threshold=DETECTOR_OBJECTNESS_THRESHOLD,
    )
    records = []
    started_at = time.time()

    for row_number, (_, image) in enumerate(idx.iterrows(), start=1):
        image_path = asset_root / image["image_path"]
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        decoded = detector.post_process(detector.predict(frame))
        for bbox, class_id, object_score, probabilities in zip(*decoded):
            records.append(
                serialize_detection(
                    model_name,
                    image,
                    bbox,
                    class_id,
                    object_score,
                    probabilities,
                    classes,
                )
            )

        if row_number % 500 == 0:
            elapsed = time.time() - started_at
            print(
                f"[{model_name}] processed {row_number}/{len(idx)} images | "
                f"raw detections={len(records)} | elapsed={elapsed:.1f}s"
            )

    raw_predictions = pd.DataFrame(records, columns=RAW_COLUMNS)
    raw_predictions.to_csv(raw_path, index=False)
    elapsed = time.time() - started_at
    print(f"[WRITE] {raw_path} rows={len(raw_predictions)}")
    print(f"[DONE] {model_name} raw inference seconds: {elapsed:.2f}")
    return raw_predictions


def apply_nms_for_model(
    model_name,
    raw_df,
    idx,
    classes,
    run_label,
    force=False,
):
    """Apply the fixed class-aware NMS policy to cached raw candidates."""

    threshold_tag = str(NMS_THRESHOLD).replace(".", "_")
    prediction_path = _cache_path(
        f"{model_name}_predictions_class_aware_nms_{threshold_tag}",
        run_label,
    )
    cached = _read_cache(prediction_path, force, "Post-NMS predictions")
    if cached is not None:
        return cached

    print("=" * 80)
    print(f"Applying class-aware NMS: {model_name}")
    print("=" * 80)
    nms = NMS(NMS_CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    candidates_by_image = _groups_by_image(raw_df)
    retained_records = []

    for row_number, (_, image) in enumerate(idx.iterrows(), start=1):
        candidates = candidates_by_image.get(image["image_file"])
        if candidates is None or len(candidates) == 0:
            continue

        boxes = candidates[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].astype(float).values.tolist()
        class_ids = candidates["class_id"].astype(int).tolist()
        objectness = candidates["object_score"].astype(float).tolist()
        class_probabilities = [
            json.loads(serialized)
            for serialized in candidates["class_scores_json"].tolist()
        ]

        retained = nms.filter(boxes, class_ids, objectness, class_probabilities)
        for bbox, class_id, object_score, probabilities in zip(*retained):
            record = serialize_detection(
                model_name,
                image,
                bbox,
                class_id,
                object_score,
                probabilities,
                classes,
            )
            record["nms_threshold"] = NMS_THRESHOLD
            retained_records.append(record)

        if row_number % 1000 == 0:
            print(
                f"[{model_name}] applied NMS to {row_number}/{len(idx)} images | "
                f"retained detections={len(retained_records)}"
            )

    predictions = pd.DataFrame(retained_records, columns=PRED_COLUMNS)
    predictions.to_csv(prediction_path, index=False)
    print(f"[WRITE] {prediction_path} rows={len(predictions)}")
    return predictions


def build_metric_lists(idx, pred_df, gt_df):
    """Align prediction and label lists to dataset-index image order."""

    predictions_by_image = _groups_by_image(pred_df)
    labels_by_image = _groups_by_image(gt_df)
    boxes, predicted_classes, objectness, class_probabilities = [], [], [], []
    ground_truth_boxes, ground_truth_classes = [], []

    for _, image in idx.iterrows():
        image_file = image["image_file"]
        predictions = predictions_by_image.get(image_file)
        if predictions is None:
            boxes.append([])
            predicted_classes.append([])
            objectness.append([])
            class_probabilities.append([])
        else:
            boxes.append(
                predictions[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].values.tolist()
            )
            predicted_classes.append(predictions["class_id"].astype(int).tolist())
            objectness.append(predictions["object_score"].astype(float).tolist())
            class_probabilities.append(
                [
                    json.loads(serialized)
                    for serialized in predictions["class_scores_json"].tolist()
                ]
            )

        labels = labels_by_image.get(image_file)
        if labels is None:
            ground_truth_boxes.append([])
            ground_truth_classes.append([])
        else:
            ground_truth_boxes.append(
                labels[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].values.tolist()
            )
            ground_truth_classes.append(labels["class_id"].astype(int).tolist())

    return (
        boxes,
        predicted_classes,
        objectness,
        class_probabilities,
        ground_truth_boxes,
        ground_truth_classes,
    )


def evaluate_with_metrics_py(model_name, idx, pred_df, gt_df, classes):
    """Evaluate post-NMS predictions with the project metric implementation."""

    metric_inputs = build_metric_lists(idx, pred_df, gt_df)
    matches, ground_truth_counts = match_detections(
        boxes=metric_inputs[0],
        classes=metric_inputs[1],
        scores=metric_inputs[2],
        cls_scores=metric_inputs[3],
        gt_boxes=metric_inputs[4],
        gt_classes=metric_inputs[5],
        map_iou_threshold=MAP_IOU_THRESHOLD,
        eval_type=EVAL_TYPE,
    )
    precision, recall, _ = calculate_precision_recall_curve(
        matches,
        ground_truth_counts,
        num_classes=len(classes),
    )
    points = {
        class_id: list(zip(recall[class_id], precision[class_id]))
        for class_id in range(len(classes))
    }
    mean_ap = calculate_map_x_point_interpolated(
        points,
        num_classes=len(classes),
        num_interpolated_points=11,
    )

    class_records = []
    for class_id, class_name in enumerate(classes):
        class_ap = calculate_map_x_point_interpolated(
            {0: points[class_id]},
            num_classes=1,
            num_interpolated_points=11,
        )
        class_records.append(
            {
                "model": model_name,
                "class_id": class_id,
                "class_name": class_name,
                "ground_truth_count": int((gt_df["class_id"] == class_id).sum()),
                "prediction_count": (
                    int((pred_df["class_id"] == class_id).sum())
                    if len(pred_df)
                    else 0
                ),
                "ap_11_point": class_ap,
            }
        )

    summary = {
        "model": model_name,
        "mAP@0.5_11_point": mean_ap,
        "total_ground_truth": int(len(gt_df)),
        "total_predictions_after_nms": int(len(pred_df)),
        "evaluation_rows": int(sum(len(records) for records in matches.values())),
        "candidate_objectness_threshold": DETECTOR_OBJECTNESS_THRESHOLD,
        "nms_confidence_threshold": NMS_CONFIDENCE_THRESHOLD,
        "nms_iou_threshold": NMS_THRESHOLD,
        "map_iou_threshold": MAP_IOU_THRESHOLD,
        "eval_type": EVAL_TYPE,
    }
    return summary, pd.DataFrame(class_records)


def build_figures(comparison):
    """Write per-class checkpoint comparison and delta figures."""

    import matplotlib.pyplot as plt

    ordered = comparison.sort_values("model2_ap", ascending=True)
    positions = np.arange(len(ordered))
    bar_height = 0.38

    plt.figure(figsize=(11, 10))
    plt.barh(positions - bar_height / 2, ordered["model1_ap"], height=bar_height, label="Checkpoint A")
    plt.barh(positions + bar_height / 2, ordered["model2_ap"], height=bar_height, label="Checkpoint B")
    plt.yticks(positions, ordered["class_name"])
    plt.xlim(0.0, 1.0)
    plt.xlabel("11-point AP at IoU 0.5 (fixed confidence filter)")
    plt.ylabel("Class")
    plt.title("Per-class detection score by checkpoint")
    plt.grid(axis="x", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    per_class_path = FIGURE_DIR / "01_per_class_ap.png"
    plt.savefig(per_class_path, dpi=200, bbox_inches="tight")
    plt.close()

    delta_name = "ap_difference_model2_minus_model1"
    delta_ordered = comparison.sort_values(delta_name, ascending=True)
    deltas = delta_ordered[delta_name].to_numpy(float)
    colors = np.where(
        deltas > 1e-12,
        "#2e7d32",
        np.where(deltas < -1e-12, "#c62828", "#757575"),
    )
    lower_limit = min(-0.02, float(np.min(deltas)) * 1.15)
    upper_limit = max(0.02, float(np.max(deltas)) * 1.15)

    plt.figure(figsize=(11, 10))
    plt.barh(delta_ordered["class_name"], deltas, color=colors)
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlim(lower_limit, upper_limit)
    plt.xlabel("AP difference: Checkpoint B minus Checkpoint A")
    plt.ylabel("Class")
    plt.title("Per-class score change between checkpoints")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    delta_path = FIGURE_DIR / "02_per_class_ap_delta.png"
    plt.savefig(delta_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("[WRITE]", per_class_path)
    print("[WRITE]", delta_path)


def _comparison_table(per_class_df):
    comparison = (
        per_class_df.pivot(
            index=["class_id", "class_name", "ground_truth_count"],
            columns="model",
            values="ap_11_point",
        )
        .reset_index()
        .rename(columns={"model1": "model1_ap", "model2": "model2_ap"})
    )
    delta = comparison["model2_ap"] - comparison["model1_ap"]
    comparison["ap_difference_model2_minus_model1"] = delta
    comparison["better_model"] = np.where(
        delta > 0,
        "model2",
        np.where(delta < 0, "model1", "tie"),
    )
    return comparison


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=PROJECT_ROOT,
        help=(
            "Root containing detector_service/storage. Use this to reuse dataset "
            "and model assets stored outside the current repository copy."
        ),
    )
    parser.add_argument(
        "--refresh-postprocessing",
        action="store_true",
        help="Reapply NMS and evaluation while reusing raw inference caches.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild ground truth and raw inference before evaluation.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not DATASET_INDEX.exists():
        raise FileNotFoundError(
            f"Dataset index not found: {DATASET_INDEX}. "
            "Run experiments/scripts/02_build_dataset_index.py first."
        )

    dataset_index = pd.read_csv(DATASET_INDEX)
    if args.max_images is None:
        run_label = "full"
        print("[INFO] Running model comparison on full dataset.")
    else:
        dataset_index = dataset_index.head(args.max_images).copy()
        run_label = f"first_{args.max_images}"
        print(f"[INFO] Running model comparison on first {args.max_images} images.")

    asset_root = args.asset_root.expanduser().resolve()
    if not asset_root.is_dir():
        raise NotADirectoryError(f"Asset root does not exist: {asset_root}")
    classes = load_classes(asset_root, run_label)

    print(f"[INFO] Images selected: {len(dataset_index)}")
    print(f"[INFO] Classes: {len(classes)}")
    print(f"[INFO] Asset root: {asset_root}")
    print(
        "[INFO] Confidence policy: "
        f"objectness>{DETECTOR_OBJECTNESS_THRESHOLD}, "
        f"combined confidence>={NMS_CONFIDENCE_THRESHOLD}"
    )
    print(f"[INFO] Class-aware NMS IoU threshold: {NMS_THRESHOLD}")

    ground_truth = build_ground_truth(
        dataset_index,
        classes,
        asset_root,
        run_label,
        force=args.force,
    )
    refresh_postprocessing = args.force or args.refresh_postprocessing
    summaries, class_tables = [], []

    for model_name, paths in MODELS.items():
        raw_predictions = run_raw_inference_for_model(
            model_name,
            paths,
            dataset_index,
            classes,
            asset_root,
            run_label,
            force=args.force,
        )
        predictions = apply_nms_for_model(
            model_name,
            raw_predictions,
            dataset_index,
            classes,
            run_label,
            force=refresh_postprocessing,
        )
        summary, class_table = evaluate_with_metrics_py(
            model_name,
            dataset_index,
            predictions,
            ground_truth,
            classes,
        )
        summaries.append(summary)
        class_tables.append(class_table)

    summary_table = pd.DataFrame(summaries)
    per_class_table = pd.concat(class_tables, axis=0)
    comparison = _comparison_table(per_class_table)

    summary_path = _cache_path("model_summary", run_label)
    per_class_path = _cache_path("per_class_metrics", run_label)
    comparison_path = _cache_path("per_class_ap_comparison", run_label)
    summary_table.to_csv(summary_path, index=False)
    per_class_table.to_csv(per_class_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    build_figures(comparison)

    print("\nMODEL SELECTION SUMMARY")
    print(summary_table.to_string(index=False))
    print("\nPER-CLASS AP COMPARISON")
    print(
        comparison.sort_values(
            "ap_difference_model2_minus_model1",
            ascending=False,
        ).to_string(index=False)
    )
    print("\n[WRITE]", summary_path)
    print("[WRITE]", per_class_path)
    print("[WRITE]", comparison_path)


if __name__ == "__main__":
    main()
