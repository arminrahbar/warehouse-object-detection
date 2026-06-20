from pathlib import Path
import csv
import json
import re
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR = PROJECT_ROOT / "techtrack" / "storage"
DATA_DIR = STORAGE_DIR / "logistics"
NAMES_FILE = STORAGE_DIR / "yolo_model_1" / "logistics.names"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR = EXPERIMENTS_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_column_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def load_classes():
    return [line.strip() for line in NAMES_FILE.read_text().splitlines() if line.strip()]


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    if not NAMES_FILE.exists():
        raise FileNotFoundError(f"Class names file not found: {NAMES_FILE}")

    classes = load_classes()
    class_cols = [f"count_{clean_column_name(class_name)}" for class_name in classes]

    image_paths = sorted(DATA_DIR.glob("*.jpg"))

    rows = []
    object_counter = Counter()
    image_counter = Counter()
    object_count_distribution = Counter()
    missing_labels = []

    for image_path in image_paths:
        label_path = image_path.with_suffix(".txt")

        if not label_path.exists():
            missing_labels.append(str(image_path))
            continue

        class_counts = Counter()
        total_objects = 0

        label_text = label_path.read_text().strip()
        if label_text:
            for line in label_text.splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(float(parts[0]))
                if 0 <= class_id < len(classes):
                    class_counts[class_id] += 1
                    object_counter[class_id] += 1
                    total_objects += 1

        for class_id in class_counts:
            image_counter[class_id] += 1

        object_count_distribution[total_objects] += 1

        row = {
            "image_path": str(image_path.relative_to(PROJECT_ROOT)),
            "label_path": str(label_path.relative_to(PROJECT_ROOT)),
            "image_file": image_path.name,
            "label_file": label_path.name,
            "num_objects": total_objects,
            "class_ids_present": json.dumps(sorted(class_counts.keys())),
            "class_names_present": json.dumps([classes[i] for i in sorted(class_counts.keys())]),
        }

        for class_id, col in enumerate(class_cols):
            row[col] = class_counts.get(class_id, 0)

        rows.append(row)

    if not rows:
        raise RuntimeError(f"No image-label pairs were indexed from {DATA_DIR}")

    dataset_index_path = OUTPUT_DIR / "dataset_index.csv"
    with dataset_index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    class_summary_path = OUTPUT_DIR / "class_distribution.csv"
    with class_summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "class_name", "object_count", "image_count"],
        )
        writer.writeheader()
        for class_id, class_name in enumerate(classes):
            writer.writerow({
                "class_id": class_id,
                "class_name": class_name,
                "object_count": object_counter[class_id],
                "image_count": image_counter[class_id],
            })

    object_dist_path = OUTPUT_DIR / "object_count_distribution.csv"
    with object_dist_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["num_objects", "image_count"])
        writer.writeheader()
        for num_objects, image_count in sorted(object_count_distribution.items()):
            writer.writerow({
                "num_objects": num_objects,
                "image_count": image_count,
            })

    print("Dataset index written:", dataset_index_path)
    print("Class distribution written:", class_summary_path)
    print("Object-count distribution written:", object_dist_path)
    print("Images indexed:", len(rows))
    print("Missing labels:", len(missing_labels))
    print("Total labeled objects:", sum(object_counter.values()))


if __name__ == "__main__":
    main()