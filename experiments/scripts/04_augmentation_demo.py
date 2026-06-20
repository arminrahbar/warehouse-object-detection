from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

OUTPUT_DIR = EXPERIMENTS_DIR / "outputs"
SAMPLING_OUTPUT_DIR = OUTPUT_DIR / "dataset_sampling"
FIGURE_DIR = EXPERIMENTS_DIR / "figures" / "04_augmentation_robustness"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from techtrack.modules.rectification.augmentation import Augmenter


SAMPLE_INDEX = SAMPLING_OUTPUT_DIR / "selected_sample_index.csv"


def main():
    if not SAMPLE_INDEX.exists():
        raise FileNotFoundError(
            f"Selected sample index not found: {SAMPLE_INDEX}. "
            "Run experiments/scripts/02_dataset_sampling.py first."
        )

    sample = pd.read_csv(SAMPLE_INDEX)

    if "num_objects" in sample.columns:
        row = sample.sort_values("num_objects", ascending=False).iloc[0]
    else:
        row = sample.iloc[0]

    image_path = Path(row["image_path"])
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    vertical_flip = Augmenter.vertical_flip(image=image_rgb)
    blurred = Augmenter.gaussian_blur(image=image_rgb, kernel_size=9)
    brighter = Augmenter.change_brightness(image=image_rgb, alpha=1.15, beta=35)
    darker = Augmenter.change_brightness(image=image_rgb, alpha=0.85, beta=-35)

    panels = [
        ("Original", image_rgb),
        ("Vertical flip", vertical_flip),
        ("Gaussian blur", blurred),
        ("Brighter / higher contrast", brighter),
        ("Darker / lower contrast", darker),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(18, 5))

    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle("Visual examples of augmentation functions", fontsize=14)
    plt.tight_layout()

    output_path = FIGURE_DIR / "01_augmentation_examples.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Source image: {image_path}")


if __name__ == "__main__":
    main()