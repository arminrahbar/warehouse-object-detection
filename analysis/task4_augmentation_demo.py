from pathlib import Path
import sys
import pandas as pd
import cv2
import matplotlib.pyplot as plt

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from techtrack.modules.rectification.augmentation import Augmenter

OUT = ROOT / "analysis" / "outputs"
FIG = ROOT / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sample_index_path = OUT / "task2_selected_sample_index.csv"
if not sample_index_path.exists():
    raise FileNotFoundError(f"Missing selected sample index: {sample_index_path}")

sample = pd.read_csv(sample_index_path)

# Pick a reasonably object-rich image for visual demonstration.
if "object_count" in sample.columns:
    row = sample.sort_values("object_count", ascending=False).iloc[0]
else:
    row = sample.iloc[0]

image_path = Path(row["image_path"])
if not image_path.is_absolute():
    image_path = ROOT / image_path

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

fig.suptitle("Task 4: Visual examples of TechTrack augmentation functions", fontsize=14)
plt.tight_layout()

output_path = FIG / "task4_augmentation_examples.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")
print(f"Source image: {image_path}")
