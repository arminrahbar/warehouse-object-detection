from pathlib import Path
import sys
import time

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR = EXPERIMENTS_DIR / "outputs"

sys.path.insert(0, str(PROJECT_ROOT))

from techtrack.modules.inference.model import Detector
from techtrack.modules.inference.nms import NMS


DATASET_INDEX = OUTPUT_DIR / "dataset_index.csv"
SAMPLE_SIZE = 100
IMAGE_COUNT_ESTIMATE = 9525

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


def main():
    if not DATASET_INDEX.exists():
        raise FileNotFoundError(
            f"Dataset index not found: {DATASET_INDEX}. "
            "Run experiments/scripts/02_build_dataset_index.py first."
        )

    idx = pd.read_csv(DATASET_INDEX).head(SAMPLE_SIZE)

    for model_name, paths in MODELS.items():
        print("=" * 80)
        print(f"Benchmarking {model_name}")
        print("=" * 80)

        detector = Detector(
            str(paths["weights"]),
            str(paths["cfg"]),
            str(paths["names"]),
            score_threshold=0.5,
        )

        nms = NMS(score_threshold=0.5, nms_iou_threshold=0.4)

        start = time.time()
        total_detections = 0
        images_processed = 0

        for _, row in idx.iterrows():
            image_path = PROJECT_ROOT / row["image_path"]
            frame = cv2.imread(str(image_path))

            if frame is None:
                print(f"WARNING: could not read {image_path}")
                continue

            outputs = detector.predict(frame)
            bboxes, class_ids, scores, class_scores = detector.post_process(outputs)
            filtered_boxes, filtered_classes, filtered_scores, filtered_class_scores = nms.filter(
                bboxes, class_ids, scores, class_scores
            )

            total_detections += len(filtered_boxes)
            images_processed += 1

        if images_processed == 0:
            raise RuntimeError("No images were processed. Check dataset paths in the index.")

        elapsed = time.time() - start
        sec_per_image = elapsed / images_processed
        estimated_full_minutes_one_model = (sec_per_image * IMAGE_COUNT_ESTIMATE) / 60
        estimated_full_minutes_two_models = estimated_full_minutes_one_model * 2

        print(f"Images processed: {images_processed}")
        print(f"Total detections after NMS: {total_detections}")
        print(f"Elapsed seconds: {elapsed:.2f}")
        print(f"Seconds per image: {sec_per_image:.4f}")
        print(f"Estimated full-dataset runtime for this model: {estimated_full_minutes_one_model:.1f} minutes")
        print(f"Estimated two-model full-dataset runtime at this speed: {estimated_full_minutes_two_models:.1f} minutes")


if __name__ == "__main__":
    main()