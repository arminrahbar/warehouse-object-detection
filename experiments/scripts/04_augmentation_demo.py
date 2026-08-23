"""Render a deterministic visual comparison of supported image perturbations."""

import argparse
import sys
import tempfile
from pathlib import Path, PurePosixPath

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector_service.modules.rectification.augmentation import Augmenter


DEFAULT_SAMPLE_INDEX = (
    PROJECT_ROOT
    / "experiments"
    / "outputs"
    / "02_dataset_analysis"
    / "02_sample_selection"
    / "selected_sample_index.csv"
)
DEFAULT_FIGURE_DIR = (
    PROJECT_ROOT
    / "scratch"
    / "diagnostic-figures"
    / "04_augmentation_robustness"
)
DEFAULT_OUTPUT = (
    DEFAULT_FIGURE_DIR
    / "01_augmentation_examples.png"
)

PANEL_DEFINITIONS = (
    ("Original", "original", {}),
    ("Vertical flip", "vertical_flip", {}),
    ("Gaussian blur", "gaussian_blur", {"kernel_size": 9, "sigma": 0}),
    (
        "Brighter / higher contrast",
        "change_brightness",
        {"alpha": 1.15, "beta": 35},
    ),
    (
        "Darker / lower contrast",
        "change_brightness",
        {"alpha": 0.85, "beta": -35},
    ),
)


def _required_columns(table, columns, label):
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def load_sample_index(path):
    """Load and validate the selected-image manifest used for the example."""

    source = Path(path).expanduser().absolute()
    if not source.is_file():
        raise FileNotFoundError(
            f"Selected sample index not found: {source}. "
            "Run experiments/scripts/02_dataset_sampling.py first."
        )
    sample = pd.read_csv(source)
    _required_columns(sample, ["image_file", "image_path", "num_objects"], "Sample index")
    if sample.empty:
        raise ValueError("Sample index is empty.")
    if sample[["image_file", "image_path"]].isna().any().any():
        raise ValueError("Sample index contains missing image identifiers.")
    numeric = pd.to_numeric(sample["num_objects"], errors="coerce")
    if numeric.isna().any() or (numeric < 0).any() or (numeric % 1 != 0).any():
        raise ValueError("num_objects must contain non-negative integers.")
    sample = sample.copy()
    sample["num_objects"] = numeric.astype("int64")
    return sample


def select_example(sample):
    """Choose the densest image with a deterministic filename tie-breaker."""

    _required_columns(sample, ["image_file", "image_path", "num_objects"], "Sample index")
    ordered = sample.sort_values(
        ["num_objects", "image_file"],
        ascending=[False, True],
        kind="stable",
    )
    return ordered.iloc[0]


def resolve_image_path(value, asset_root=None, project_root=PROJECT_ROOT):
    """Resolve absolute and portable storage paths without dereferencing symlinks."""

    raw = str(value).strip()
    if not raw:
        raise ValueError("Image path cannot be empty.")
    direct = Path(raw).expanduser()
    if direct.is_absolute():
        return direct
    parts = PurePosixPath(raw.replace("\\", "/")).parts
    if asset_root is not None and tuple(parts[:2]) in {
        ("detector_service", "storage"),
        ("techtrack", "storage"),
    }:
        return Path(asset_root).expanduser().absolute().joinpath(*parts[2:])
    return Path(project_root).joinpath(*parts)


def build_panels(image_rgb, augmenter=Augmenter):
    """Apply each fixed condition to the same source image."""

    panels = []
    for title, operation, parameters in PANEL_DEFINITIONS:
        if operation == "original":
            transformed = image_rgb.copy()
        else:
            transformed = getattr(augmenter, operation)(
                image=image_rgb,
                **parameters,
            )
        panels.append((title, transformed))
    return panels


def render_figure(panels, output_path):
    """Write the five-panel comparison atomically and return its destination."""

    import matplotlib.pyplot as plt

    destination = Path(output_path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(panels), figsize=(18, 5))
    if len(panels) == 1:
        axes = [axes]
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("Visual examples of augmentation functions", fontsize=14)
    figure.tight_layout()

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            prefix=f".{destination.stem}-",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        figure.savefig(temporary_path, dpi=200, bbox_inches="tight")
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)
    return destination


def build_parser():
    parser = argparse.ArgumentParser(
        description="Render fixed augmentation examples from a selected dataset image."
    )
    parser.add_argument("--sample-index", type=Path, default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv=None):
    import cv2

    args = build_parser().parse_args(argv)
    sample = load_sample_index(args.sample_index)
    selected = select_example(sample)
    image_path = resolve_image_path(selected["image_path"], args.asset_root)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not decode selected image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    output_path = render_figure(build_panels(image_rgb), args.output)
    print(f"[WRITE] {output_path}")
    print(f"[INFO] Source image: {image_path}")
    print(f"[INFO] Source objects: {int(selected['num_objects'])}")
    return output_path, image_path


if __name__ == "__main__":
    main()
