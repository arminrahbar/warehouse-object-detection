"""Build the verified publication figure package for Experiment 04.

This stage performs no model inference. It verifies the selected-sample,
augmentation-robustness, and Experiment 03 NMS evidence before rendering a new
directory containing the six publication figures. The destination is promoted
only after every PNG has been written successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import sys
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "experiments" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_figure_style import (  # noqa: E402
    GRID,
    INK,
    MUTED,
    NAVY,
    ORANGE,
    PALE,
    TEAL,
    VERMILION,
    WHITE,
    FigureBuildError,
    add_header,
    build_atomic_package,
    clean_axis,
    require,
    three_panel_figure,
)


ROBUSTNESS_SCRIPT = SCRIPT_DIR / "04_augmentation_robustness.py"
DEFAULT_ROBUSTNESS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "outputs"
    / "04_augmentation_robustness"
    / "01_condition_evaluation"
)
DEFAULT_SAMPLE_INDEX = (
    PROJECT_ROOT
    / "experiments"
    / "outputs"
    / "02_dataset_analysis"
    / "02_sample_selection"
    / "selected_sample_index.csv"
)
DEFAULT_NMS_SUMMARY = (
    PROJECT_ROOT
    / "experiments"
    / "outputs"
    / "03_nms_thresholding"
    / "01_threshold_sweep"
    / "nms_threshold_summary_sample5000.csv"
)
DEFAULT_ASSET_ROOT = PROJECT_ROOT / "detector_service" / "storage"

EXPECTED_IMAGES = 5000
EXPECTED_LABELS = 19196
EXPECTED_CLASSES = 20
EXPECTED_SOURCE_GROUPS = 4778
SOURCE_GROUP_MARKER = "_jpg.rf."
CLASS_DROP_PRESENTATION_THRESHOLD = 0.01

OUTPUT_STEMS = (
    "01_experiment_design",
    "02_condition_examples",
    "03_condition_quality",
    "04_prediction_retention",
    "05_class_impact_breadth",
    "06_largest_class_drops",
)

CONDITION_ORDER = (
    "original",
    "brightness_increase",
    "brightness_decrease",
    "gaussian_blur_k9",
    "vertical_flip",
)
CONDITION_LABELS = {
    "original": "Original",
    "brightness_increase": "Brighter / higher contrast",
    "brightness_decrease": "Darker / lower contrast",
    "gaussian_blur_k9": "Gaussian blur",
    "vertical_flip": "Vertical flip",
}
CONDITION_SHORT_LABELS = {
    "original": "Original",
    "brightness_increase": "Brighter",
    "brightness_decrease": "Darker",
    "gaussian_blur_k9": "Gaussian blur",
    "vertical_flip": "Vertical flip",
}
CONDITION_COLORS = {
    "original": NAVY,
    "brightness_increase": TEAL,
    "brightness_decrease": ORANGE,
    "gaussian_blur_k9": VERMILION,
    "vertical_flip": NAVY,
}


def _filesystem_path(path):
    """Return an extended-length Windows path at the filesystem boundary."""

    normal = Path(path).expanduser().absolute()
    if os.name != "nt":
        return normal
    raw = str(normal)
    if raw.startswith("\\\\?\\"):
        return normal
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw[2:])
    return Path("\\\\?\\" + raw)

SAMPLE_COLUMNS = (
    "image_path",
    "label_path",
    "image_file",
    "label_file",
    "num_objects",
    "class_ids_present",
    "class_names_present",
    "count_barcode",
    "count_car",
    "count_cardboard_box",
    "count_fire",
    "count_forklift",
    "count_freight_container",
    "count_gloves",
    "count_helmet",
    "count_ladder",
    "count_license_plate",
    "count_person",
    "count_qr_code",
    "count_road_sign",
    "count_safety_vest",
    "count_smoke",
    "count_traffic_cone",
    "count_traffic_light",
    "count_truck",
    "count_van",
    "count_wood_pallet",
    "density_bucket",
)
NMS_COLUMNS = (
    "model",
    "dataset",
    "nms_threshold",
    "mAP@0.5_11_point",
    "total_ground_truth",
    "total_predictions_after_nms",
    "evaluation_rows",
    "score_threshold",
    "map_iou_threshold",
    "eval_type",
)

LOCKED_HASHES = {
    "sample_index": "cebdceb80fccd6f87e14a414db93d3c84f98770b0c9cdf263228027a7b768073",
    "nms_summary": "5e6c266f2efe87da32bb095e4edf1106d7e8df8487d538e4a8500254d1422f3c",
    "summary": "99a21593f86d13f07c8f083de7f860f231fe46fbe77c85a33c21d29a06dc3c4e",
    "per_class": "6cf6143ceb23e9a9e7dfa96659a7b16711d6ede00d91b3eda1e9bdb5c599fc3b",
    "predictions_original": "b8e53384764cfa1508591f08a6bc7766e0485a3db9a2aedfcb473eb50fae842b",
    "predictions_gaussian_blur_k9": "85a07a7bcd39a172dfa0a8d85184956b0d5c2a145c1ecdbc27457541706227e4",
    "predictions_vertical_flip": "f1939a8ee85491c6e7c9306f2fc8a8b06888f34543cb614a28df44473a8b0adb",
    "predictions_brightness_increase": "cd27d28bee4c95a4b0901d26e1c97088668d32d8ebf7215b680366e85846b752",
    "predictions_brightness_decrease": "60015a7a57ee7cfe862040a8587ed6fff981df95f3f9fd36b869c9031ba36d0e",
}


def _load_robustness_module():
    spec = importlib.util.spec_from_file_location(
        "augmentation_robustness_for_report_figures", ROBUSTNESS_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load robustness implementation: {ROBUSTNESS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


robustness = _load_robustness_module()


def _sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv_exact(path, columns, context):
    source = Path(path).expanduser().absolute()
    require(source.is_file(), f"Missing {context}: {source}")
    try:
        table = pd.read_csv(source)
    except Exception as error:
        raise FigureBuildError(f"Unable to read {context}: {error}") from error
    require(
        table.columns.tolist() == list(columns),
        f"{context} schema does not match its locked ordered columns.",
    )
    return table


def _numeric(table, columns, context):
    result = table.copy()
    for column in columns:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except Exception as error:
            raise FigureBuildError(f"{context}.{column} must be numeric.") from error
        require(
            np.isfinite(result[column].to_numpy(dtype=float)).all(),
            f"{context}.{column} contains a non-finite value.",
        )
    return result


def _close(observed, expected, context, atol=1e-12):
    require(
        math.isclose(float(observed), float(expected), rel_tol=1e-9, abs_tol=atol),
        f"{context} is inconsistent with its source evidence.",
    )


def _verify_hash(path, key, locked_hashes):
    require(key in locked_hashes, f"No locked hash was supplied for {key}.")
    observed = _sha256_file(path)
    require(observed == str(locked_hashes[key]).lower(), f"Hash mismatch for {key}.")


def source_group_key(image_file):
    value = str(image_file).strip()
    require(bool(value), "image_file must not be empty.")
    prefix, marker, _ = value.partition(SOURCE_GROUP_MARKER)
    return prefix if marker and prefix else value


def _prediction_filename(condition):
    return (
        f"model2_predictions_{condition}_class_aware_nms_0_3_sample5000.csv"
    )


def _validate_sample(path, locked_hashes, expected_images, expected_labels, expected_source_groups):
    _verify_hash(path, "sample_index", locked_hashes)
    sample = _read_csv_exact(path, SAMPLE_COLUMNS, "selected-sample index")
    require(len(sample) == expected_images, "Selected-sample image count is invalid.")
    require(not sample["image_file"].duplicated().any(), "Sample image_file values must be unique.")
    require(not sample["image_path"].duplicated().any(), "Sample image_path values must be unique.")
    sample = _numeric(sample, ["num_objects"], "selected-sample index")
    require((sample["num_objects"] >= 0).all(), "Sample object counts cannot be negative.")
    require((sample["num_objects"] % 1 == 0).all(), "Sample object counts must be integers.")
    require(int(sample["num_objects"].sum()) == expected_labels, "Selected-sample label total is invalid.")
    groups = sample["image_file"].map(source_group_key)
    require(groups.nunique() == expected_source_groups, "Selected-sample source-group count is invalid.")
    return sample


def _validate_summary(directory, locked_hashes, expected_labels):
    summary_path = directory / "summary_by_condition_sample5000.csv"
    per_class_path = directory / "per_class_ap_by_condition_sample5000.csv"
    _verify_hash(summary_path, "summary", locked_hashes)
    _verify_hash(per_class_path, "per_class", locked_hashes)
    summary = _read_csv_exact(summary_path, robustness.SUMMARY_COLUMNS, "robustness summary")
    per_class = _read_csv_exact(per_class_path, robustness.PER_CLASS_COLUMNS, "per-class robustness evidence")

    require(len(summary) == 5, "Robustness summary must contain exactly five conditions.")
    require(set(summary["augmentation_condition"]) == set(CONDITION_ORDER), "Robustness condition set is invalid.")
    require(not summary["augmentation_condition"].duplicated().any(), "Robustness conditions must be unique.")
    summary = _numeric(
        summary,
        [
            "mAP@0.5_11_point",
            "total_ground_truth",
            "total_predictions_after_nms",
            "evaluation_rows",
            "candidate_objectness_threshold",
            "nms_confidence_threshold",
            "nms_iou_threshold",
            "map_iou_threshold",
            "mAP_change_vs_original",
            "mAP_percent_change_vs_original",
            "prediction_change_vs_original",
        ],
        "robustness summary",
    )
    require(set(summary["model"].astype(str)) == {"model2"}, "Robustness model must be model2.")
    require(
        set(summary["dataset"].astype(str)) == {"rare_aware_density_stratified_5000"},
        "Robustness dataset identity is invalid.",
    )
    require(set(summary["eval_type"].astype(str)) == {"combined"}, "Robustness score type is invalid.")
    require((summary["total_ground_truth"] == expected_labels).all(), "Ground-truth totals differ by condition.")
    require((summary["evaluation_rows"] == summary["total_predictions_after_nms"]).all(), "Evaluation rows must equal retained predictions.")
    for column, expected in (
        ("candidate_objectness_threshold", 0.5),
        ("nms_confidence_threshold", 0.5),
        ("nms_iou_threshold", 0.3),
        ("map_iou_threshold", 0.5),
    ):
        require(np.allclose(summary[column], expected), f"Locked {column} changed.")

    baseline = summary.set_index("augmentation_condition").loc["original"]
    for row in summary.itertuples(index=False):
        _close(
            row.mAP_change_vs_original,
            row._4 - baseline["mAP@0.5_11_point"],
            f"{row.augmentation_condition} AP change",
        )
        expected_percent = (
            row.mAP_change_vs_original / baseline["mAP@0.5_11_point"] * 100.0
        )
        _close(
            row.mAP_percent_change_vs_original,
            expected_percent,
            f"{row.augmentation_condition} relative AP change",
        )
        _close(
            row.prediction_change_vs_original,
            row.total_predictions_after_nms - baseline["total_predictions_after_nms"],
            f"{row.augmentation_condition} prediction change",
        )
    return summary, per_class


def _validate_per_class(per_class, expected_classes, expected_labels):
    require(len(per_class) == 5 * expected_classes, "Per-class evidence row count is invalid.")
    per_class = _numeric(
        per_class,
        [
            "class_id",
            "ground_truth_count",
            "prediction_count",
            "ap_11_point",
            "original_ap_11_point",
            "ap_change_vs_original",
        ],
        "per-class robustness evidence",
    )
    require(set(per_class["augmentation_condition"]) == set(CONDITION_ORDER), "Per-class condition set is invalid.")
    expected_ids = set(range(expected_classes))
    for condition, rows in per_class.groupby("augmentation_condition", sort=False):
        require(len(rows) == expected_classes, f"{condition} must contain every class once.")
        require(set(rows["class_id"].astype(int)) == expected_ids, f"{condition} class IDs are invalid.")
        require(not rows["class_id"].duplicated().any(), f"{condition} contains duplicate classes.")
        require(int(rows["ground_truth_count"].sum()) == expected_labels, f"{condition} class support is invalid.")
    baseline = (
        per_class[per_class["augmentation_condition"] == "original"]
        .set_index("class_id")
        .sort_index()
    )
    require(np.allclose(baseline["ap_change_vs_original"], 0.0), "Original per-class AP changes must be zero.")
    require(np.allclose(baseline["ap_11_point"], baseline["original_ap_11_point"]), "Original AP identity is invalid.")
    for condition, rows in per_class.groupby("augmentation_condition", sort=False):
        ordered = rows.set_index("class_id").sort_index()
        require(
            ordered["class_name"].astype(str).tolist()
            == baseline["class_name"].astype(str).tolist(),
            f"{condition} class vocabulary differs from the baseline.",
        )
        require(
            np.array_equal(
                ordered["ground_truth_count"].to_numpy(dtype=int),
                baseline["ground_truth_count"].to_numpy(dtype=int),
            ),
            f"{condition} class support differs from the baseline.",
        )
        require(
            np.allclose(ordered["original_ap_11_point"], baseline["ap_11_point"]),
            f"{condition} stored baseline AP values are invalid.",
        )
        require(
            np.allclose(
                ordered["ap_change_vs_original"],
                ordered["ap_11_point"] - baseline["ap_11_point"],
            ),
            f"{condition} per-class AP deltas are invalid.",
        )
    return per_class


def _validate_predictions(directory, summary, sample, locked_hashes, expected_classes):
    expected_files = {
        "summary_by_condition_sample5000.csv",
        "per_class_ap_by_condition_sample5000.csv",
        *(_prediction_filename(condition) for condition in CONDITION_ORDER),
    }
    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    require(actual_files == expected_files, "Robustness evidence directory has an unexpected file set.")
    selected_images = set(sample["image_file"].astype(str))
    summary_lookup = summary.set_index("augmentation_condition")
    for condition in CONDITION_ORDER:
        path = directory / _prediction_filename(condition)
        _verify_hash(path, f"predictions_{condition}", locked_hashes)
        table = _read_csv_exact(path, robustness.PREDICTION_COLUMNS, f"{condition} predictions")
        require(
            len(table) == int(summary_lookup.loc[condition, "total_predictions_after_nms"]),
            f"{condition} prediction row count differs from the summary.",
        )
        require(set(table["augmentation_condition"].astype(str)) == {condition}, f"{condition} prediction identity is invalid.")
        require(set(table["model"].astype(str)) == {"model2"}, f"{condition} model identity is invalid.")
        require(set(table["image_file"].astype(str)).issubset(selected_images), f"{condition} predictions contain unknown images.")
        table = _numeric(
            table,
            [
                "class_id",
                "object_score",
                "predicted_class_score",
                "combined_confidence",
                "nms_threshold",
            ],
            f"{condition} predictions",
        )
        require(table["class_id"].between(0, expected_classes - 1).all(), f"{condition} class IDs are invalid.")
        require(np.allclose(table["nms_threshold"], 0.3), f"{condition} NMS identity is invalid.")
        require(
            np.allclose(
                table["combined_confidence"],
                table["object_score"] * table["predicted_class_score"],
                rtol=1e-10,
                atol=1e-12,
            ),
            f"{condition} combined-confidence values are invalid.",
        )


def _validate_nms_summary(path, summary, locked_hashes, expected_labels):
    _verify_hash(path, "nms_summary", locked_hashes)
    nms = _read_csv_exact(path, NMS_COLUMNS, "Experiment 03 NMS summary")
    nms = _numeric(
        nms,
        [
            "nms_threshold",
            "mAP@0.5_11_point",
            "total_ground_truth",
            "total_predictions_after_nms",
            "evaluation_rows",
            "score_threshold",
            "map_iou_threshold",
        ],
        "Experiment 03 NMS summary",
    )
    require(len(nms) == 7, "Experiment 03 NMS summary must contain seven thresholds.")
    require(
        np.allclose(sorted(nms["nms_threshold"]), [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7]),
        "Experiment 03 NMS threshold set is invalid.",
    )
    selected = nms[np.isclose(nms["nms_threshold"], 0.3)]
    require(len(selected) == 1, "Experiment 03 must contain one NMS 0.30 row.")
    selected = selected.iloc[0]
    original = summary.set_index("augmentation_condition").loc["original"]
    for field in (
        "mAP@0.5_11_point",
        "total_ground_truth",
        "total_predictions_after_nms",
        "evaluation_rows",
        "map_iou_threshold",
    ):
        _close(original[field], selected[field], f"Experiment 03/04 baseline {field}")
    _close(original["nms_confidence_threshold"], selected["score_threshold"], "Experiment 03/04 confidence threshold")
    require(str(selected["model"]) == str(original["model"]), "Experiment 03/04 model identity differs.")
    require(str(selected["dataset"]) == str(original["dataset"]), "Experiment 03/04 dataset identity differs.")
    require(str(selected["eval_type"]) == str(original["eval_type"]), "Experiment 03/04 score type differs.")
    require(int(selected["total_ground_truth"]) == expected_labels, "Experiment 03 label count is invalid.")
    return nms


def load_verified_evidence(
    robustness_dir,
    sample_index,
    nms_summary,
    *,
    locked_hashes=LOCKED_HASHES,
    expected_images=EXPECTED_IMAGES,
    expected_labels=EXPECTED_LABELS,
    expected_classes=EXPECTED_CLASSES,
    expected_source_groups=EXPECTED_SOURCE_GROUPS,
):
    """Verify all report inputs and return their normalized plotting tables."""

    directory = Path(robustness_dir).expanduser().absolute()
    require(directory.is_dir(), f"Robustness evidence directory not found: {directory}")
    sample = _validate_sample(
        Path(sample_index),
        locked_hashes,
        expected_images,
        expected_labels,
        expected_source_groups,
    )
    summary, per_class = _validate_summary(directory, locked_hashes, expected_labels)
    per_class = _validate_per_class(per_class, expected_classes, expected_labels)
    _validate_predictions(directory, summary, sample, locked_hashes, expected_classes)
    nms = _validate_nms_summary(Path(nms_summary), summary, locked_hashes, expected_labels)
    return {
        "directory": directory,
        "sample": sample,
        "summary": summary,
        "per_class": per_class,
        "nms": nms,
        "expected_images": expected_images,
        "expected_labels": expected_labels,
        "expected_classes": expected_classes,
        "expected_source_groups": expected_source_groups,
    }


def resolve_image_path(value, asset_root):
    raw = str(value).strip()
    require(bool(raw), "Example image path cannot be empty.")
    direct = Path(raw).expanduser()
    if direct.is_absolute():
        return direct
    parts = PurePosixPath(raw.replace("\\", "/")).parts
    if tuple(parts[:2]) == ("detector_service", "storage"):
        return Path(asset_root).expanduser().absolute().joinpath(*parts[2:])
    return PROJECT_ROOT.joinpath(*parts)


def select_example(sample):
    return sample.sort_values(
        ["num_objects", "image_file"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]


def apply_visual_condition(image, condition):
    """Apply the declared deterministic condition for the qualitative panel.

    The analytical evidence is never recomputed here. This NumPy implementation
    mirrors the production transform parameters so the publication builder does
    not require an OpenCV binary merely to redraw the qualitative example.
    """

    source = np.asarray(image, dtype=np.uint8)
    condition_type = condition["type"]
    if condition_type == "none":
        return source.copy()
    if condition_type == "vertical_flip":
        return np.flipud(source).copy()
    if condition_type == "brightness":
        adjusted = source.astype(np.float32) * float(condition["alpha"])
        adjusted += float(condition["beta"])
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    if condition_type == "gaussian_blur":
        size = int(condition["kernel_size"])
        require(size > 0 and size % 2 == 1, "Gaussian kernel must be positive and odd.")
        sigma = float(condition["sigma"])
        if sigma <= 0.0:
            sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
        radius = size // 2
        coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(coordinates**2) / (2.0 * sigma**2))
        kernel /= kernel.sum()
        floating = source.astype(np.float32)
        horizontal_pad = np.pad(
            floating,
            ((0, 0), (radius, radius), (0, 0)),
            mode="reflect",
        )
        horizontal = sum(
            kernel[position] * horizontal_pad[:, position : position + source.shape[1], :]
            for position in range(size)
        )
        vertical_pad = np.pad(
            horizontal,
            ((radius, radius), (0, 0), (0, 0)),
            mode="reflect",
        )
        vertical = sum(
            kernel[position] * vertical_pad[position : position + source.shape[0], :, :]
            for position in range(size)
        )
        return np.clip(np.rint(vertical), 0, 255).astype(np.uint8)
    raise FigureBuildError(f"Unsupported visual condition: {condition_type}")


def load_example_panels(sample, asset_root):
    selected = select_example(sample)
    path = resolve_image_path(selected["image_path"], asset_root)
    require(path.is_file(), f"Example image not found: {path}")
    try:
        image = np.asarray(Image.open(path).convert("RGB"))
    except Exception as error:
        raise FigureBuildError(f"Unable to decode example image: {path}") from error
    definitions = (
        ("original", "Original", "Reference image"),
        ("brightness_increase", "Brighter", "gain 1.15 · offset +35"),
        ("brightness_decrease", "Darker", "gain 0.85 · offset −35"),
        ("gaussian_blur_k9", "Gaussian blur", "9 × 9 kernel · automatic σ"),
        ("vertical_flip", "Vertical flip", "Vertical orientation stress test"),
    )
    panels = []
    for tag, title, parameters in definitions:
        condition = next(value for value in robustness.CONDITIONS if value["tag"] == tag)
        transformed = apply_visual_condition(image, condition)
        panels.append((tag, title, parameters, transformed))
    return selected, path, panels


def class_impact_counts(per_class, threshold=CLASS_DROP_PRESENTATION_THRESHOLD):
    changed = per_class[per_class["augmentation_condition"] != "original"].copy()
    rows = []
    for condition in CONDITION_ORDER[1:]:
        values = changed.loc[
            changed["augmentation_condition"] == condition,
            "ap_change_vs_original",
        ].to_numpy(dtype=float)
        rows.append(
            {
                "condition": condition,
                "drop": int(np.sum(values < -threshold)),
                "within": int(np.sum(np.abs(values) <= threshold)),
                "gain": int(np.sum(values > threshold)),
            }
        )
    return pd.DataFrame(rows)


def largest_class_drops(per_class, count=10):
    changed = per_class[per_class["augmentation_condition"] != "original"].copy()
    changed["ap_drop"] = -changed["ap_change_vs_original"]
    changed["condition_order"] = changed["augmentation_condition"].map(
        {tag: position for position, tag in enumerate(CONDITION_ORDER)}
    )
    return changed.sort_values(
        ["ap_drop", "condition_order", "class_id"],
        ascending=[False, True, True],
        kind="stable",
    ).head(count)


def _design_figure(evidence):
    return three_panel_figure(
        "Controlled input-shift diagnostic",
        "One selected detector and one fixed operating point; only decoded pixels changed between conditions",
        (
            {
                "heading": "Input",
                "bullets": [
                    "Checkpoint B (model2)",
                    f"{evidence['expected_images']:,} selected images",
                    f"{evidence['expected_labels']:,} labels across {evidence['expected_classes']} classes",
                    f"{evidence['expected_source_groups']:,} source groups",
                ],
            },
            {
                "heading": "Controlled",
                "bullets": [
                    "Same image identities and ground truth",
                    "One deterministic perturbation per run",
                    "Objectness > 0.50; combined confidence ≥ 0.50",
                    "NMS IoU 0.30; evaluation IoU 0.50",
                ],
            },
            {
                "heading": "Output",
                "bullets": [
                    "Threshold-constrained 11-point AP50",
                    "Retained-output context and per-class deltas",
                    "Synthetic blur became the first field-validation hypothesis",
                    "Vertical flip: orientation diagnostic",
                ],
            },
        ),
    )


def _examples_figure(evidence, asset_root):
    selected, _, panels = load_example_panels(evidence["sample"], asset_root)
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 4.8))
    add_header(
        fig,
        "One image, five controlled input conditions",
        f"Deterministic transformation check on the highest-density selected image ({int(selected['num_objects']):,} labels); qualitative only",
        title_x=0.035,
    )
    for axis, (_, title, parameters, image) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title, fontsize=11, fontweight="bold", pad=22)
        axis.text(
            0.5,
            1.025,
            parameters,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            color=MUTED,
            fontsize=8.8,
        )
        axis.set_axis_off()
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.06, top=0.79, wspace=0.06)
    return fig


def _quality_figure(evidence):
    summary = evidence["summary"].set_index("augmentation_condition")
    baseline = float(summary.loc["original", "mAP@0.5_11_point"])
    tags = list(CONDITION_ORDER[1:])
    fig, axis = plt.subplots(figsize=(11.8, 6.2))
    add_header(
        fig,
        "Synthetic blur and inversion produced the largest score losses",
        "Threshold-constrained 11-point AP50 on the same 5,000 images; values are descriptive point estimates",
    )
    y = np.arange(len(tags))
    for position, tag in zip(y, tags):
        value = float(summary.loc[tag, "mAP@0.5_11_point"])
        relative = float(summary.loc[tag, "mAP_percent_change_vs_original"])
        axis.hlines(position, value, baseline, color=GRID, linewidth=7, zorder=1)
        axis.scatter(value, position, s=95, color=CONDITION_COLORS[tag], zorder=2)
        axis.text(
            value - 0.006,
            position,
            f"{value:.4f}  ({relative:.2f}%)",
            ha="right",
            va="center",
            color=INK,
            fontsize=10,
        )
    axis.axvline(baseline, color=NAVY, linewidth=1.4)
    axis.text(
        baseline,
        -0.62,
        f"Original {baseline:.4f}",
        ha="right",
        va="bottom",
        color=NAVY,
        fontsize=9.5,
        fontweight="bold",
    )
    axis.set_yticks(y, [CONDITION_SHORT_LABELS[tag] for tag in tags])
    axis.invert_yaxis()
    axis.set_xlim(0.14, 0.43)
    axis.set_xlabel("Threshold-constrained 11-point AP50")
    clean_axis(axis, "x")
    fig.subplots_adjust(left=0.19, right=0.96, bottom=0.13, top=0.79)
    return fig


def _prediction_figure(evidence):
    summary = evidence["summary"].set_index("augmentation_condition")
    baseline = float(summary.loc["original", "total_predictions_after_nms"])
    tags = list(CONDITION_ORDER)
    values = np.asarray(
        [float(summary.loc[tag, "total_predictions_after_nms"]) / baseline * 100 for tag in tags]
    )
    counts = [int(summary.loc[tag, "total_predictions_after_nms"]) for tag in tags]
    fig, axis = plt.subplots(figsize=(11.8, 6.4))
    add_header(
        fig,
        "Retained output contracted under blur and inversion",
        "Percentage of the original post-NMS count; output volume is behavioral context, not a recall metric",
    )
    y = np.arange(len(tags))
    axis.barh(y, values, color=[CONDITION_COLORS[tag] for tag in tags], height=0.58)
    for position, value, count in zip(y, values, counts):
        axis.text(
            min(value + 1.4, 101.5),
            position,
            f"{count:,}  |  {value:.1f}%",
            ha="left",
            va="center",
            color=INK,
            fontsize=10,
        )
    axis.set_yticks(y, [CONDITION_SHORT_LABELS[tag] for tag in tags])
    axis.invert_yaxis()
    axis.set_xlim(0, 116)
    axis.set_xlabel("Retained predictions (% of original condition)")
    clean_axis(axis, "x")
    fig.subplots_adjust(left=0.19, right=0.96, bottom=0.13, top=0.79)
    return fig


def _breadth_figure(evidence):
    counts = class_impact_counts(evidence["per_class"])
    tags = counts["condition"].tolist()
    drops = counts["drop"].to_numpy(dtype=int)
    within = counts["within"].to_numpy(dtype=int)
    gains = counts["gain"].to_numpy(dtype=int)
    require(np.all(drops + within + gains == evidence["expected_classes"]), "Class-impact counts do not reconcile.")
    fig, axis = plt.subplots(figsize=(11.8, 6.2))
    add_header(
        fig,
        "Blur and inversion affected most classes, not only isolated categories",
        "A 0.01 AP threshold is used only to summarize effect breadth; it is not a significance test",
    )
    y = np.arange(len(tags))
    axis.barh(y, drops, color=VERMILION, height=0.58, label="AP drop > 0.01")
    axis.barh(y, within, left=drops, color=PALE, edgecolor=GRID, linewidth=0.8, height=0.58, label="Within ±0.01")
    if np.any(gains):
        axis.barh(y, gains, left=drops + within, color=TEAL, height=0.58, label="AP gain > 0.01")
    for position, drop, stable in zip(y, drops, within):
        if drop:
            axis.text(drop / 2, position, str(drop), ha="center", va="center", color=WHITE, fontweight="bold")
        if stable:
            axis.text(drop + stable / 2, position, str(stable), ha="center", va="center", color=INK, fontweight="bold")
    axis.set_yticks(y, [CONDITION_SHORT_LABELS[tag] for tag in tags])
    axis.invert_yaxis()
    axis.set_xlim(0, evidence["expected_classes"])
    axis.set_xlabel(f"Classes ({evidence['expected_classes']} total)")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    clean_axis(axis, "x")
    fig.subplots_adjust(left=0.19, right=0.96, bottom=0.21, top=0.79)
    return fig


def _largest_drops_figure(evidence):
    largest = largest_class_drops(evidence["per_class"], 10).copy()
    largest = largest.sort_values("ap_drop", ascending=True, kind="stable")
    labels = [
        f"{CONDITION_SHORT_LABELS[row.augmentation_condition]} · {row.class_name} (n={int(row.ground_truth_count):,})"
        for row in largest.itertuples(index=False)
    ]
    values = largest["ap_drop"].to_numpy(dtype=float)
    colors = [CONDITION_COLORS[tag] for tag in largest["augmentation_condition"]]
    fig, axis = plt.subplots(figsize=(12.2, 7.4))
    add_header(
        fig,
        "Largest class-level AP losses occurred under blur and inversion",
        "Ten largest observed condition–class changes; n is ground-truth object support in the selected sample",
    )
    y = np.arange(len(largest))
    axis.barh(y, values, color=colors, height=0.6)
    for position, value in zip(y, values):
        axis.text(value + 0.008, position, f"−{value:.3f}", ha="left", va="center", color=INK, fontsize=9.5)
    axis.set_yticks(y, labels)
    axis.set_xlim(0, max(values) * 1.16)
    axis.set_xlabel("Observed AP50 drop from original images")
    clean_axis(axis, "x")
    fig.subplots_adjust(left=0.33, right=0.96, bottom=0.11, top=0.82)
    return fig


def build_figure_package(evidence, asset_root, output_dir):
    builders = (
        (OUTPUT_STEMS[0], lambda: _design_figure(evidence)),
        (OUTPUT_STEMS[1], lambda: _examples_figure(evidence, asset_root)),
        (OUTPUT_STEMS[2], lambda: _quality_figure(evidence)),
        (OUTPUT_STEMS[3], lambda: _prediction_figure(evidence)),
        (OUTPUT_STEMS[4], lambda: _breadth_figure(evidence)),
        (OUTPUT_STEMS[5], lambda: _largest_drops_figure(evidence)),
    )
    return build_atomic_package(
        output_dir,
        builders,
        hash_salt="warehouse-object-detection-experiment-04-v1",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Verify Experiment 04 evidence and render its publication figure package."
    )
    parser.add_argument("--robustness-dir", type=Path, default=DEFAULT_ROBUSTNESS_DIR)
    parser.add_argument("--sample-index", type=Path, default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument("--nms-summary", type=Path, default=DEFAULT_NMS_SUMMARY)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory to receive the atomic six-figure PNG package.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    evidence = load_verified_evidence(
        _filesystem_path(args.robustness_dir),
        _filesystem_path(args.sample_index),
        _filesystem_path(args.nms_summary),
    )
    destination = build_figure_package(
        evidence,
        _filesystem_path(args.asset_root),
        _filesystem_path(args.output_dir),
    )
    print(f"[VERIFIED] Images: {evidence['expected_images']:,}")
    print(f"[VERIFIED] Labels: {evidence['expected_labels']:,}")
    print(f"[VERIFIED] Source groups: {evidence['expected_source_groups']:,}")
    print(f"[WRITE] Publication figure package: {destination}")
    for path in sorted(destination.iterdir()):
        print(f"  {path.name}")
    return destination


if __name__ == "__main__":
    main()
