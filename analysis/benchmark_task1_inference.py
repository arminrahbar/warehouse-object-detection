from pathlib import Path
import sys
import time
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from techtrack.modules.inference.model import Detector
from techtrack.modules.inference.nms import NMS
OUT = ROOT / "analysis" / "outputs"
DATASET_INDEX = OUT / "dataset_index.csv"

idx = pd.read_csv(DATASET_INDEX).head(100)

MODELS = {
    "model1": {
        "weights": ROOT / "techtrack/storage/yolo_model_1/yolov4-tiny-logistics_size_416_1.weights",
        "cfg": ROOT / "techtrack/storage/yolo_model_1/yolov4-tiny-logistics_size_416_1.cfg",
        "names": ROOT / "techtrack/storage/yolo_model_1/logistics.names",
    },
    "model2": {
        "weights": ROOT / "techtrack/storage/yolo_model_2/yolov4-tiny-logistics_size_416_2.weights",
        "cfg": ROOT / "techtrack/storage/yolo_model_2/yolov4-tiny-logistics_size_416_2.cfg",
        "names": ROOT / "techtrack/storage/yolo_model_2/logistics.names",
    },
}

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
        image_path = ROOT / row["image_path"]
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"WARNING: could not read {image_path}")
            continue

        outputs = detector.predict(frame)
        bboxes, class_ids, scores, class_scores = detector.post_process(outputs)
        fbboxes, fclasses, fscores, fclass_scores = nms.filter(
            bboxes, class_ids, scores, class_scores
        )

        total_detections += len(fbboxes)
        images_processed += 1

    elapsed = time.time() - start
    sec_per_image = elapsed / images_processed
    estimated_full_minutes_one_model = (sec_per_image * 9525) / 60
    estimated_full_minutes_two_models = estimated_full_minutes_one_model * 2

    print(f"Images processed: {images_processed}")
    print(f"Total detections after NMS: {total_detections}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Seconds per image: {sec_per_image:.4f}")
    print(f"Estimated full-dataset runtime for this model: {estimated_full_minutes_one_model:.1f} minutes")
    print(f"Estimated two-model full-dataset runtime at this speed: {estimated_full_minutes_two_models:.1f} minutes")
