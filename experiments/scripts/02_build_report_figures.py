"""Build the verified Experiment 02 publication figure package.

This stage performs no sampling and no model inference. It validates the
canonical dataset-index, characterization, sampling, and overlap CSV evidence
before rendering six PNG figures. The destination is promoted only
after the complete package has been written to a staging directory.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_figure_style import (  # noqa: E402
    GRID,
    INK,
    MUTED,
    NAVY,
    NEUTRAL,
    TEAL,
    FigureBuildError,
    add_header,
    build_atomic_package,
    clean_axis,
    three_panel_figure,
)


DEFAULT_INDEX_DIR = (
    PROJECT_ROOT / "experiments" / "outputs" / "00_dataset_inventory"
)
DEFAULT_ANALYSIS_ROOT = (
    PROJECT_ROOT / "experiments" / "outputs" / "02_dataset_analysis"
)
DEFAULT_SUMMARY_DIR = DEFAULT_ANALYSIS_ROOT / "01_dataset_summary"
DEFAULT_SAMPLING_DIR = DEFAULT_ANALYSIS_ROOT / "02_sample_selection"
DEFAULT_OVERLAP_DIR = DEFAULT_ANALYSIS_ROOT / "03_overlap_analysis"

EXPECTED_IMAGES = 9525
EXPECTED_LABELS = 36721
EXPECTED_CLASSES = 20
EXPECTED_SELECTED_IMAGES = 5000
EXPECTED_SELECTED_LABELS = 19196
EXPECTED_RARE_CLASSES = 8

BASE_INDEX_COLUMNS = [
    "image_path",
    "label_path",
    "image_file",
    "label_file",
    "num_objects",
    "class_ids_present",
    "class_names_present",
]
CLASS_COLUMNS = ["class_id", "class_name", "object_count", "image_count"]
OBJECT_DISTRIBUTION_COLUMNS = ["num_objects", "image_count"]
SUMMARY_COLUMNS = [
    "dataset",
    "images",
    "total_objects",
    "images_with_zero_objects",
    "mean_objects_per_image",
    "median_objects_per_image",
    "max_objects_per_image",
    "images_ge_5_objects",
    "images_ge_10_objects",
    "images_ge_15_objects",
    "images_ge_20_objects",
]
ENRICHED_CLASS_COLUMNS = CLASS_COLUMNS + ["object_share_pct", "image_share_pct"]
DENSITY_COLUMNS = ["density_bucket", "image_count", "image_share_pct"]
DENSE_COUNT_COLUMNS = ["threshold", "image_count", "image_share_pct"]
CANDIDATE_COLUMNS = [
    "sample_name",
    "images",
    "total_objects",
    "mean_objects_per_image",
    "class_object_share_mae_pp",
    "class_object_share_max_error_pp",
    "density_share_mae_pp",
    "density_share_max_error_pp",
    "min_rare_class_image_retention_pct",
    "images_ge_10_objects",
    "images_ge_20_objects",
]
SAMPLE_SUMMARY_COLUMNS = [
    "dataset",
    "images",
    "total_objects",
    "mean_objects_per_image",
    "median_objects_per_image",
    "max_objects_per_image",
    "images_ge_5_objects",
    "images_ge_10_objects",
    "images_ge_15_objects",
    "images_ge_20_objects",
]
CLASS_COMPARISON_COLUMNS = [
    "dataset_full",
    "class_id",
    "class_name",
    "object_count_full",
    "image_count_full",
    "object_share_pct_full",
    "image_share_pct_full",
    "dataset_sample",
    "object_count_sample",
    "image_count_sample",
    "object_share_pct_sample",
    "image_share_pct_sample",
    "object_share_diff_pp",
    "image_share_diff_pp",
]
RARE_TARGET_COLUMNS = [
    "class_id",
    "class_name",
    "object_count",
    "image_count",
    "target_image_count",
]
RARE_COVERAGE_COLUMNS = CLASS_COMPARISON_COLUMNS + [
    "target_image_count",
    "sample_image_retention_pct",
]
DENSITY_COMPARISON_COLUMNS = [
    "dataset_full",
    "density_bucket",
    "image_count_full",
    "image_share_pct_full",
    "dataset_sample",
    "image_count_sample",
    "image_share_pct_sample",
    "image_share_diff_pp",
]
OVERLAP_PROFILE_COLUMNS = [
    "image_file",
    "image_path",
    "label_path",
    "num_objects",
    "pair_count",
    "max_pairwise_iou",
    "mean_pairwise_iou",
    "pairs_iou_gt_0_1",
    "pairs_iou_gt_0_3",
    "pairs_iou_gt_0_5",
    "crowding_bucket",
]
OVERLAP_SUMMARY_COLUMNS = [
    "dataset",
    "images",
    "mean_pair_count",
    "mean_max_pairwise_iou",
    "mean_pairs_iou_gt_0_1",
    "images_with_any_iou_gt_0_1",
    "images_with_any_iou_gt_0_3",
    "images_with_any_iou_gt_0_5",
    "images_with_20plus_iou_gt_0_1_pairs",
]
CROWDING_COMPARISON_COLUMNS = [
    "dataset_full",
    "crowding_bucket",
    "image_count_full",
    "image_share_pct_full",
    "dataset_sample",
    "image_count_sample",
    "image_share_pct_sample",
    "image_share_diff_pp",
]

DENSITY_BUCKET_ORDER = ["1", "2-4", "5-9", "10-14", "15-19", "20+"]
CROWDING_BUCKET_ORDER = ["0", "1-4", "5-19", "20+"]
CANDIDATE_ORDER = [
    "random_5000",
    "density_stratified_5000",
    "rare_aware_density_stratified_5000",
]
SELECTED_NAME = "rare_aware_density_stratified_5000"

INDEX_FILES = {
    "dataset_index.csv",
    "class_distribution.csv",
    "object_count_distribution.csv",
}
SUMMARY_FILES = {
    "full_dataset_summary.csv",
    "class_distribution_enriched.csv",
    "top10_classes_by_object_count.csv",
    "bottom10_classes_by_object_count.csv",
    "density_bucket_distribution.csv",
    "dense_image_counts.csv",
}
SAMPLING_FILES = {
    "candidate_sample_quality.csv",
    "selected_sample_index.csv",
    "sample_summary.csv",
    "class_distribution_comparison.csv",
    "rare_class_targets.csv",
    "rare_class_coverage.csv",
    "density_distribution_comparison.csv",
}
OVERLAP_FILES = {
    "overlap_profile.csv",
    "overlap_summary.csv",
    "crowding_distribution_comparison.csv",
}

OUTPUT_STEMS = (
    "01_workload_design",
    "02_class_inventory",
    "03_candidate_scorecard",
    "04_rare_class_coverage",
    "05_class_composition_fidelity",
    "06_scene_structure_fidelity",
)

# The selected evidence predates a run manifest. These hashes lock the default
# production paths to the exact CSVs interpreted in the publication report.
LOCKED_HASHES = {
    "index/class_distribution.csv": "354496866ee8668f37ad61bcc6ac95cec73197080a374d74e799cf7064555fc3",
    "index/dataset_index.csv": "f94e010ecf2884ed2743d3ecacff7401b9a19d703d8954ef1aada7908563fda0",
    "index/object_count_distribution.csv": "fdf74e1152003e4e9fa168c9ebb5a23df61ff6d44e7b3b9f7ae7a415ca9b7718",
    "summary/bottom10_classes_by_object_count.csv": "c686f992fdd9e17c9365fedc530c0634c968e3025888766de21896d5fe1ddde9",
    "summary/class_distribution_enriched.csv": "e8e13d345d3c58f3ac624053d58a8b2d9ab38e78219e618ff56f2456cad0537a",
    "summary/dense_image_counts.csv": "d53d89b6fbaa3126ec787e9b5e22c352ba47d5d5d58577807ba5cb138bc8c11d",
    "summary/density_bucket_distribution.csv": "d813ec2f79b2d2d1d6234c91faf1f4c43b776573d7d2c51811fa64073e05cea1",
    "summary/full_dataset_summary.csv": "ea31081e3fdac8c9a11db6fc0c13b79ed2a6b4d9510f4bcf8c7eb74d7adc30b3",
    "summary/top10_classes_by_object_count.csv": "8d07abef2a7a5f017c086c6bd220aa76f3db063343f4ae35690da4075ca38f3b",
    "sampling/candidate_sample_quality.csv": "ff0c8b1f7f3b9e0725d9af54896a67cdea8cdbaf40351a2c969d9615157dc526",
    "sampling/class_distribution_comparison.csv": "da62189f9a7a5d65e926424779c062d9fce380705251fb3ea1df18aa274688b6",
    "sampling/density_distribution_comparison.csv": "7826aabc0a1f54cb261d48c8b44174b327a5f24ed37f404861a0a88a04a74027",
    "sampling/rare_class_coverage.csv": "879932bacec6d71e0824138151df6cbecf96ff8ec200dde4555ff5472aa1912c",
    "sampling/rare_class_targets.csv": "309d7c374efa1e08fe928a595c0a265ea2be9d733393c705edcbaa8279e432c1",
    "sampling/sample_summary.csv": "82eb5f42e22c68dbe9e87a0fb53d05dbbe34be03bdbcdb794bcc72d898430399",
    "sampling/selected_sample_index.csv": "cebdceb80fccd6f87e14a414db93d3c84f98770b0c9cdf263228027a7b768073",
    "overlap/crowding_distribution_comparison.csv": "c59e8616b62f957982560fe9741e938f2da7cf57d702452a544b10eb298ea24f",
    "overlap/overlap_profile.csv": "a493d18994c9fc3b95747a97826bd7640cb181e9337046a00722956d85380d94",
    "overlap/overlap_summary.csv": "a921fdd83ec9af08a0f7fa2e9090015f15690e15437943b2c61e018b7bf18163",
}


class FigureEvidenceError(FigureBuildError):
    """Raised when Experiment 02 evidence fails a publication gate."""


def _require(condition, message):
    if not condition:
        raise FigureEvidenceError(message)


def _sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_column_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _read_csv_exact(path, columns, context):
    source = Path(path)
    _require(source.is_file(), f"Missing {context}: {source}")
    try:
        table = pd.read_csv(source)
    except Exception as error:
        raise FigureEvidenceError(f"Unable to read {context}: {error}") from error
    _require(
        table.columns.tolist() == list(columns),
        f"{context} schema does not match its ordered column contract.",
    )
    return table


def _numeric(table, columns, context, *, integer=False, nonnegative=False):
    result = table.copy()
    for column in columns:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except Exception as error:
            raise FigureEvidenceError(f"{context}.{column} must be numeric.") from error
        values = result[column].to_numpy(dtype=float)
        _require(np.isfinite(values).all(), f"{context}.{column} is non-finite.")
        if nonnegative:
            _require((values >= 0).all(), f"{context}.{column} must be non-negative.")
        if integer:
            _require(
                np.equal(values, np.floor(values)).all(),
                f"{context}.{column} must contain integers.",
            )
            result[column] = result[column].astype(np.int64)
    return result


def _close(observed, expected, context, atol=1e-9):
    _require(
        math.isclose(float(observed), float(expected), rel_tol=1e-9, abs_tol=atol),
        f"{context} is inconsistent with its granular source evidence.",
    )


def _verify_directory(directory, expected, label):
    path = Path(directory).expanduser().absolute()
    _require(path.is_dir(), f"{label} directory not found: {path}")
    actual = {item.name for item in path.iterdir() if item.is_file()}
    _require(actual == set(expected), f"{label} directory has an unexpected file set.")
    return path


def _verify_locked_hashes(directories, locked_hashes):
    if not locked_hashes:
        return
    expected_keys = {
        f"{group}/{name}"
        for group, (_, names) in directories.items()
        for name in names
    }
    _require(
        set(locked_hashes) == expected_keys,
        "Locked evidence hash set does not match the required CSV set.",
    )
    for key, expected_hash in locked_hashes.items():
        group, name = key.split("/", 1)
        observed = _sha256_file(directories[group][0] / name)
        _require(observed == expected_hash, f"Evidence hash mismatch: {key}")


def _density_bucket(count):
    value = int(count)
    if value <= 1:
        return "1"
    if value <= 4:
        return "2-4"
    if value <= 9:
        return "5-9"
    if value <= 14:
        return "10-14"
    if value <= 19:
        return "15-19"
    return "20+"


def _crowding_bucket(count):
    value = int(count)
    if value == 0:
        return "0"
    if value <= 4:
        return "1-4"
    if value <= 19:
        return "5-19"
    return "20+"


def _class_count_columns(classes):
    columns = [f"count_{_clean_column_name(name)}" for name in classes]
    _require(all(column != "count_" for column in columns), "Invalid class name.")
    _require(len(columns) == len(set(columns)), "Class names collide as count columns.")
    return columns


def _validate_index(index_dir, expected_images, expected_labels, expected_classes):
    classes = _read_csv_exact(
        index_dir / "class_distribution.csv", CLASS_COLUMNS, "class distribution"
    )
    classes = _numeric(
        classes,
        ["class_id", "object_count", "image_count"],
        "class distribution",
        integer=True,
        nonnegative=True,
    )
    _require(len(classes) == expected_classes, "Unexpected class count.")
    _require(
        classes["class_id"].tolist() == list(range(expected_classes)),
        "Class identifiers must be contiguous and zero-based.",
    )
    _require(
        classes["class_name"].notna().all()
        and (classes["class_name"].astype(str).str.strip() != "").all(),
        "Class names must be populated.",
    )
    _require(not classes["class_name"].duplicated().any(), "Class names must be unique.")
    count_columns = _class_count_columns(classes["class_name"].astype(str).tolist())

    index = _read_csv_exact(
        index_dir / "dataset_index.csv",
        BASE_INDEX_COLUMNS + count_columns,
        "dataset index",
    )
    _require(len(index) == expected_images, "Unexpected full-index image count.")
    for column in ("image_file", "image_path", "label_path"):
        _require(index[column].notna().all(), f"Dataset index {column} is missing.")
        _require(not index[column].duplicated().any(), f"Dataset index {column} is duplicated.")
    index = _numeric(
        index,
        ["num_objects"] + count_columns,
        "dataset index",
        integer=True,
        nonnegative=True,
    )
    _require(
        np.array_equal(
            index[count_columns].sum(axis=1).to_numpy(),
            index["num_objects"].to_numpy(),
        ),
        "Dataset index object totals do not equal their class-count sums.",
    )
    _require(int(index["num_objects"].sum()) == expected_labels, "Unexpected label total.")
    for row, column in zip(classes.itertuples(index=False), count_columns):
        _require(int(index[column].sum()) == int(row.object_count),
                 f"Object total mismatch for {row.class_name}.")
        _require(int((index[column] > 0).sum()) == int(row.image_count),
                 f"Image total mismatch for {row.class_name}.")

    supplied_distribution = _read_csv_exact(
        index_dir / "object_count_distribution.csv",
        OBJECT_DISTRIBUTION_COLUMNS,
        "object-count distribution",
    )
    supplied_distribution = _numeric(
        supplied_distribution,
        OBJECT_DISTRIBUTION_COLUMNS,
        "object-count distribution",
        integer=True,
        nonnegative=True,
    )
    observed = Counter(index["num_objects"].astype(int))
    supplied = dict(
        zip(
            supplied_distribution["num_objects"].astype(int),
            supplied_distribution["image_count"].astype(int),
        )
    )
    _require(supplied == dict(sorted(observed.items())),
             "Object-count distribution does not reconcile with the index.")
    return {"index": index, "classes": classes, "count_columns": count_columns}


def _validate_summary(summary_dir, base):
    index = base["index"]
    classes = base["classes"]
    summary = _read_csv_exact(
        summary_dir / "full_dataset_summary.csv", SUMMARY_COLUMNS, "dataset summary"
    )
    _require(len(summary) == 1, "Dataset summary must contain one row.")
    summary = _numeric(
        summary,
        [column for column in SUMMARY_COLUMNS if column != "dataset"],
        "dataset summary",
        nonnegative=True,
    )
    row = summary.iloc[0]
    _require(str(row["dataset"]) == "full_dataset", "Dataset summary label is invalid.")
    exact = {
        "images": len(index),
        "total_objects": int(index["num_objects"].sum()),
        "images_with_zero_objects": int((index["num_objects"] == 0).sum()),
        "median_objects_per_image": float(index["num_objects"].median()),
        "max_objects_per_image": int(index["num_objects"].max()),
        "images_ge_5_objects": int((index["num_objects"] >= 5).sum()),
        "images_ge_10_objects": int((index["num_objects"] >= 10).sum()),
        "images_ge_15_objects": int((index["num_objects"] >= 15).sum()),
        "images_ge_20_objects": int((index["num_objects"] >= 20).sum()),
    }
    for key, value in exact.items():
        _close(row[key], value, f"dataset summary {key}")
    _close(row["mean_objects_per_image"], round(float(index["num_objects"].mean()), 3),
           "dataset summary mean")

    enriched = _read_csv_exact(
        summary_dir / "class_distribution_enriched.csv",
        ENRICHED_CLASS_COLUMNS,
        "enriched class distribution",
    )
    _require(len(enriched) == len(classes), "Enriched class row count is invalid.")
    enriched = _numeric(
        enriched,
        ["class_id", "object_count", "image_count", "object_share_pct", "image_share_pct"],
        "enriched class distribution",
        nonnegative=True,
    )
    _require(
        enriched[["class_id", "class_name"]].astype(str).to_records(index=False).tolist()
        == classes[["class_id", "class_name"]].astype(str).to_records(index=False).tolist(),
        "Enriched class vocabulary differs from the index.",
    )
    for enriched_row, class_row in zip(
        enriched.itertuples(index=False), classes.itertuples(index=False)
    ):
        _close(enriched_row.object_count, class_row.object_count, "enriched object count")
        _close(enriched_row.image_count, class_row.image_count, "enriched image count")
        _close(
            enriched_row.object_share_pct,
            round(100.0 * int(class_row.object_count) / int(index["num_objects"].sum()), 4),
            f"{class_row.class_name} object share",
            atol=5e-5,
        )
        _close(
            enriched_row.image_share_pct,
            round(100.0 * int(class_row.image_count) / len(index), 4),
            f"{class_row.class_name} image share",
            atol=5e-5,
        )

    for filename, ascending, context in (
        ("top10_classes_by_object_count.csv", False, "top class table"),
        ("bottom10_classes_by_object_count.csv", True, "bottom class table"),
    ):
        table = _read_csv_exact(summary_dir / filename, ENRICHED_CLASS_COLUMNS, context)
        expected = enriched.sort_values(
            ["object_count", "class_id"], ascending=[ascending, True], kind="mergesort"
        ).head(min(10, len(enriched)))
        _require(
            table["class_name"].astype(str).tolist() == expected["class_name"].astype(str).tolist(),
            f"{context} ordering is inconsistent.",
        )

    density = _read_csv_exact(
        summary_dir / "density_bucket_distribution.csv",
        DENSITY_COLUMNS,
        "density distribution",
    )
    _require(density["density_bucket"].astype(str).tolist() == DENSITY_BUCKET_ORDER,
             "Density bucket order is invalid.")
    density = _numeric(density, ["image_count", "image_share_pct"],
                       "density distribution", nonnegative=True)
    observed_density = index["num_objects"].map(_density_bucket).value_counts()
    for density_row in density.itertuples(index=False):
        expected_count = int(observed_density.get(str(density_row.density_bucket), 0))
        _close(density_row.image_count, expected_count, "density image count")
        _close(density_row.image_share_pct, round(100 * expected_count / len(index), 4),
               "density image share", atol=5e-5)

    dense = _read_csv_exact(
        summary_dir / "dense_image_counts.csv", DENSE_COUNT_COLUMNS, "dense image counts"
    )
    _require(dense["threshold"].astype(str).tolist()
             == [">=5 objects", ">=10 objects", ">=15 objects", ">=20 objects"],
             "Dense thresholds are invalid.")
    return {"summary": summary, "enriched": enriched, "density": density, "dense": dense}


def _validate_sampling(sampling_dir, base, expected_selected_images,
                       expected_selected_labels, expected_rare_classes):
    index = base["index"]
    classes = base["classes"]
    count_columns = base["count_columns"]
    selected = _read_csv_exact(
        sampling_dir / "selected_sample_index.csv",
        BASE_INDEX_COLUMNS + count_columns + ["density_bucket"],
        "selected sample index",
    )
    _require(len(selected) == expected_selected_images, "Unexpected selected image count.")
    _require(selected["image_file"].notna().all(), "Selected image identifiers are missing.")
    _require(not selected["image_file"].duplicated().any(), "Selected image files are duplicated.")
    _require(not selected["image_path"].duplicated().any(), "Selected image paths are duplicated.")
    _require(
        selected["image_file"].astype(str).tolist()
        == sorted(selected["image_file"].astype(str).tolist()),
        "Selected index must be filename-sorted.",
    )
    selected = _numeric(
        selected,
        ["num_objects"] + count_columns,
        "selected sample index",
        integer=True,
        nonnegative=True,
    )
    _require(int(selected["num_objects"].sum()) == expected_selected_labels,
             "Unexpected selected label total.")
    _require(np.array_equal(selected[count_columns].sum(axis=1), selected["num_objects"]),
             "Selected object totals do not equal class-count sums.")
    full_by_file = index.set_index("image_file", drop=False)
    _require(set(selected["image_file"]).issubset(set(index["image_file"])),
             "Selected index is not a subset of the full index.")
    aligned = full_by_file.loc[selected["image_file"]].reset_index(drop=True)
    for column in BASE_INDEX_COLUMNS + count_columns:
        left = selected[column].reset_index(drop=True)
        right = aligned[column].reset_index(drop=True)
        _require(left.equals(right), f"Selected {column} differs from the full index.")
    _require(
        selected["density_bucket"].astype(str).tolist()
        == selected["num_objects"].map(_density_bucket).astype(str).tolist(),
        "Selected density buckets are inconsistent.",
    )

    candidates = _read_csv_exact(
        sampling_dir / "candidate_sample_quality.csv", CANDIDATE_COLUMNS,
        "candidate quality table",
    )
    _require(candidates["sample_name"].astype(str).tolist() == CANDIDATE_ORDER,
             "Candidate policy order is invalid.")
    numeric_candidate_columns = [column for column in CANDIDATE_COLUMNS if column != "sample_name"]
    candidates = _numeric(candidates, numeric_candidate_columns, "candidate quality", nonnegative=True)
    _require((candidates["images"] == expected_selected_images).all(),
             "Every candidate must use the same image budget.")

    sample_summary = _read_csv_exact(
        sampling_dir / "sample_summary.csv", SAMPLE_SUMMARY_COLUMNS, "sample summary"
    )
    _require(sample_summary["dataset"].astype(str).tolist()
             == ["full_dataset", SELECTED_NAME], "Sample summary labels are invalid.")
    sample_summary = _numeric(
        sample_summary,
        [column for column in SAMPLE_SUMMARY_COLUMNS if column != "dataset"],
        "sample summary",
        nonnegative=True,
    )
    selected_summary = sample_summary.iloc[1]
    selected_exact = {
        "images": len(selected),
        "total_objects": int(selected["num_objects"].sum()),
        "max_objects_per_image": int(selected["num_objects"].max()),
        "images_ge_5_objects": int((selected["num_objects"] >= 5).sum()),
        "images_ge_10_objects": int((selected["num_objects"] >= 10).sum()),
        "images_ge_15_objects": int((selected["num_objects"] >= 15).sum()),
        "images_ge_20_objects": int((selected["num_objects"] >= 20).sum()),
    }
    for key, value in selected_exact.items():
        _close(selected_summary[key], value, f"selected summary {key}")
    selected_candidate = candidates.set_index("sample_name").loc[SELECTED_NAME]
    for key in ("images", "total_objects", "mean_objects_per_image",
                "images_ge_10_objects", "images_ge_20_objects"):
        _close(selected_candidate[key], selected_summary[key], f"selected candidate {key}", atol=5e-4)

    comparison = _read_csv_exact(
        sampling_dir / "class_distribution_comparison.csv",
        CLASS_COMPARISON_COLUMNS,
        "class comparison",
    )
    _require(len(comparison) == len(classes), "Class comparison row count is invalid.")
    comparison = _numeric(
        comparison,
        [column for column in CLASS_COMPARISON_COLUMNS
         if column not in ("dataset_full", "class_name", "dataset_sample")],
        "class comparison",
        nonnegative=False,
    )
    _require(comparison["class_name"].astype(str).tolist()
             == classes["class_name"].astype(str).tolist(),
             "Class comparison vocabulary differs from the index.")
    full_total = int(index["num_objects"].sum())
    selected_total = int(selected["num_objects"].sum())
    for row, column in zip(comparison.itertuples(index=False), count_columns):
        full_objects = int(index[column].sum())
        sample_objects = int(selected[column].sum())
        full_images = int((index[column] > 0).sum())
        sample_images = int((selected[column] > 0).sum())
        _close(row.object_count_full, full_objects, "full class object count")
        _close(row.object_count_sample, sample_objects, "sample class object count")
        _close(row.image_count_full, full_images, "full class image count")
        _close(row.image_count_sample, sample_images, "sample class image count")
        full_object_share = 100.0 * full_objects / full_total
        sample_object_share = 100.0 * sample_objects / selected_total
        full_image_share = 100.0 * full_images / len(index)
        sample_image_share = 100.0 * sample_images / len(selected)
        _close(row.object_share_pct_full, full_object_share, "full object share")
        _close(row.object_share_pct_sample, sample_object_share, "sample object share")
        _close(row.image_share_pct_full, full_image_share, "full image share")
        _close(row.image_share_pct_sample, sample_image_share, "sample image share")
        _close(row.object_share_diff_pp, sample_object_share - full_object_share,
               "object-share difference")
        _close(row.image_share_diff_pp, sample_image_share - full_image_share,
               "image-share difference")

    rare_targets = _read_csv_exact(
        sampling_dir / "rare_class_targets.csv", RARE_TARGET_COLUMNS, "rare targets"
    )
    _require(len(rare_targets) == expected_rare_classes, "Unexpected protected-class count.")
    rare_targets = _numeric(
        rare_targets,
        ["class_id", "object_count", "image_count", "target_image_count"],
        "rare targets",
        integer=True,
        nonnegative=True,
    )
    expected_rare = classes.sort_values(
        ["object_count", "class_id"], ascending=[True, True], kind="mergesort"
    ).head(expected_rare_classes)
    _require(rare_targets["class_name"].astype(str).tolist()
             == expected_rare["class_name"].astype(str).tolist(),
             "Protected classes are not the lowest object-count classes.")
    sample_fraction = expected_selected_images / len(index)
    expected_targets = rare_targets["image_count"].map(
        lambda count: min(
            int(count),
            max(
                int(math.ceil(int(count) * sample_fraction)),
                min(100, int(count)),
            ),
        )
    )
    _require(
        np.array_equal(
            rare_targets["target_image_count"].to_numpy(dtype=int),
            expected_targets.to_numpy(dtype=int),
        ),
        "Protected-class targets do not match the declared coverage formula.",
    )

    rare_coverage = _read_csv_exact(
        sampling_dir / "rare_class_coverage.csv", RARE_COVERAGE_COLUMNS,
        "rare coverage",
    )
    rare_coverage = _numeric(
        rare_coverage,
        [column for column in RARE_COVERAGE_COLUMNS
         if column not in ("dataset_full", "class_name", "dataset_sample")],
        "rare coverage",
        nonnegative=False,
    )
    _require(set(rare_coverage["class_name"].astype(str))
             == set(rare_targets["class_name"].astype(str)),
             "Rare coverage classes differ from target classes.")
    comparison_by_class = comparison.set_index("class_name")
    for row in rare_coverage.itertuples(index=False):
        source = comparison_by_class.loc[str(row.class_name)]
        for column in (
            "class_id",
            "object_count_full",
            "image_count_full",
            "object_share_pct_full",
            "image_share_pct_full",
            "object_count_sample",
            "image_count_sample",
            "object_share_pct_sample",
            "image_share_pct_sample",
            "object_share_diff_pp",
            "image_share_diff_pp",
        ):
            _close(getattr(row, column), source[column],
                   f"rare coverage {row.class_name} {column}")
    rare_join = rare_targets[["class_name", "target_image_count"]].merge(
        rare_coverage[["class_name", "image_count_full", "image_count_sample",
                       "target_image_count", "sample_image_retention_pct"]],
        on="class_name", suffixes=("_target", "_coverage"), validate="one_to_one"
    )
    _require(np.array_equal(rare_join["target_image_count_target"],
                            rare_join["target_image_count_coverage"]),
             "Rare targets differ between target and coverage artifacts.")
    _require((rare_join["image_count_sample"] >= rare_join["target_image_count_target"]).all(),
             "At least one protected-class coverage target was not met.")
    expected_retention = 100.0 * rare_join["image_count_sample"] / rare_join["image_count_full"]
    _require(np.allclose(expected_retention, rare_join["sample_image_retention_pct"],
                         rtol=1e-10, atol=1e-10),
             "Rare-class retention percentages are inconsistent.")

    density_comparison = _read_csv_exact(
        sampling_dir / "density_distribution_comparison.csv",
        DENSITY_COMPARISON_COLUMNS,
        "density comparison",
    )
    _require(density_comparison["density_bucket"].astype(str).tolist()
             == DENSITY_BUCKET_ORDER, "Density comparison bucket order is invalid.")
    density_comparison = _numeric(
        density_comparison,
        ["image_count_full", "image_share_pct_full", "image_count_sample",
         "image_share_pct_sample", "image_share_diff_pp"],
        "density comparison",
        nonnegative=False,
    )
    full_density = index["num_objects"].map(_density_bucket).value_counts()
    selected_density = selected["num_objects"].map(_density_bucket).value_counts()
    for row in density_comparison.itertuples(index=False):
        full_count = int(full_density.get(str(row.density_bucket), 0))
        sample_count = int(selected_density.get(str(row.density_bucket), 0))
        full_share = 100.0 * full_count / len(index)
        sample_share = 100.0 * sample_count / len(selected)
        _close(row.image_count_full, full_count, "density full count")
        _close(row.image_count_sample, sample_count, "density sample count")
        _close(row.image_share_pct_full, full_share, "density full share")
        _close(row.image_share_pct_sample, sample_share, "density sample share")
        _close(row.image_share_diff_pp, sample_share - full_share, "density share difference")

    class_error = comparison["object_share_diff_pp"].abs()
    density_error = density_comparison["image_share_diff_pp"].abs()
    _close(selected_candidate["class_object_share_mae_pp"],
           round(float(class_error.mean()), 4), "selected class-share MAE", atol=5e-5)
    _close(selected_candidate["class_object_share_max_error_pp"],
           round(float(class_error.max()), 4), "selected maximum class-share error", atol=5e-5)
    _close(selected_candidate["density_share_mae_pp"],
           round(float(density_error.mean()), 4), "selected density-share MAE", atol=5e-5)
    _close(selected_candidate["density_share_max_error_pp"],
           round(float(density_error.max()), 4), "selected maximum density-share error", atol=5e-5)
    _close(selected_candidate["min_rare_class_image_retention_pct"],
           round(float(expected_retention.min()), 2), "selected minimum rare retention",
           atol=5e-3)

    return {
        "selected": selected,
        "candidates": candidates,
        "sample_summary": sample_summary,
        "comparison": comparison,
        "rare_targets": rare_targets,
        "rare_coverage": rare_coverage,
        "density_comparison": density_comparison,
    }


def _validate_overlap(overlap_dir, base, sampling):
    index = base["index"]
    selected = sampling["selected"]
    profile = _read_csv_exact(
        overlap_dir / "overlap_profile.csv", OVERLAP_PROFILE_COLUMNS,
        "overlap profile",
    )
    _require(len(profile) == len(index), "Overlap profile must cover the full index.")
    _require(not profile["image_file"].duplicated().any(), "Overlap image files are duplicated.")
    profile = _numeric(
        profile,
        ["num_objects", "pair_count", "max_pairwise_iou", "mean_pairwise_iou",
         "pairs_iou_gt_0_1", "pairs_iou_gt_0_3", "pairs_iou_gt_0_5"],
        "overlap profile",
        nonnegative=True,
    )
    aligned = index.set_index("image_file").loc[profile["image_file"]]
    _require(np.array_equal(profile["num_objects"].astype(int),
                            aligned["num_objects"].astype(int)),
             "Overlap object counts differ from the full index.")
    expected_pairs = profile["num_objects"] * (profile["num_objects"] - 1) // 2
    _require(np.array_equal(profile["pair_count"].astype(int), expected_pairs.astype(int)),
             "Overlap pair counts do not satisfy n(n-1)/2.")
    _require(((profile["max_pairwise_iou"] >= 0) & (profile["max_pairwise_iou"] <= 1)).all(),
             "Maximum pairwise IoU is outside [0, 1].")
    _require(((profile["mean_pairwise_iou"] >= 0) & (profile["mean_pairwise_iou"] <= 1)).all(),
             "Mean pairwise IoU is outside [0, 1].")
    _require((profile["pairs_iou_gt_0_1"] >= profile["pairs_iou_gt_0_3"]).all()
             and (profile["pairs_iou_gt_0_3"] >= profile["pairs_iou_gt_0_5"]).all(),
             "Overlap threshold counts are not nested.")
    _require((profile["pairs_iou_gt_0_1"] <= profile["pair_count"]).all(),
             "Overlap threshold count exceeds total pair count.")
    expected_buckets = profile["pairs_iou_gt_0_1"].map(_crowding_bucket).astype(str)
    _require(profile["crowding_bucket"].astype(str).tolist() == expected_buckets.tolist(),
             "Crowding buckets are inconsistent with overlap counts.")

    selected_profile = profile[profile["image_file"].isin(set(selected["image_file"]))].copy()
    _require(len(selected_profile) == len(selected), "Overlap profile misses selected images.")
    summary = _read_csv_exact(
        overlap_dir / "overlap_summary.csv", OVERLAP_SUMMARY_COLUMNS,
        "overlap summary",
    )
    _require(len(summary) == 2, "Overlap summary must contain full and selected rows.")
    _require(summary["dataset"].astype(str).tolist() == ["full_dataset", SELECTED_NAME],
             "Overlap summary labels are invalid.")
    summary = _numeric(
        summary,
        [column for column in OVERLAP_SUMMARY_COLUMNS if column != "dataset"],
        "overlap summary",
        nonnegative=True,
    )
    for row, table in zip(summary.itertuples(index=False), (profile, selected_profile)):
        _close(row.images, len(table), "overlap summary images")
        _close(row.mean_pair_count, round(float(table["pair_count"].mean()), 3),
               "mean overlap pair count", atol=5e-4)
        _close(row.mean_max_pairwise_iou, round(float(table["max_pairwise_iou"].mean()), 4),
               "mean maximum IoU", atol=5e-5)
        _close(row.mean_pairs_iou_gt_0_1,
               round(float(table["pairs_iou_gt_0_1"].mean()), 3),
               "mean overlapping pairs", atol=5e-4)
        for column, threshold_column in (
            ("images_with_any_iou_gt_0_1", "pairs_iou_gt_0_1"),
            ("images_with_any_iou_gt_0_3", "pairs_iou_gt_0_3"),
            ("images_with_any_iou_gt_0_5", "pairs_iou_gt_0_5"),
        ):
            _close(getattr(row, column), int((table[threshold_column] > 0).sum()), column)
        _close(row.images_with_20plus_iou_gt_0_1_pairs,
               int((table["pairs_iou_gt_0_1"] >= 20).sum()), "20+ crowding count")

    crowding = _read_csv_exact(
        overlap_dir / "crowding_distribution_comparison.csv",
        CROWDING_COMPARISON_COLUMNS,
        "crowding comparison",
    )
    _require(crowding["crowding_bucket"].astype(str).tolist() == CROWDING_BUCKET_ORDER,
             "Crowding comparison bucket order is invalid.")
    crowding = _numeric(
        crowding,
        ["image_count_full", "image_share_pct_full", "image_count_sample",
         "image_share_pct_sample", "image_share_diff_pp"],
        "crowding comparison",
        nonnegative=False,
    )
    full_counts = profile["crowding_bucket"].astype(str).value_counts()
    selected_counts = selected_profile["crowding_bucket"].astype(str).value_counts()
    for row in crowding.itertuples(index=False):
        full_count = int(full_counts.get(str(row.crowding_bucket), 0))
        sample_count = int(selected_counts.get(str(row.crowding_bucket), 0))
        full_share = 100.0 * full_count / len(profile)
        sample_share = 100.0 * sample_count / len(selected_profile)
        _close(row.image_count_full, full_count, "crowding full count")
        _close(row.image_count_sample, sample_count, "crowding sample count")
        _close(row.image_share_pct_full, full_share, "crowding full share")
        _close(row.image_share_pct_sample, sample_share, "crowding sample share")
        _close(row.image_share_diff_pp, sample_share - full_share,
               "crowding share difference")
    return {
        "profile": profile,
        "selected_profile": selected_profile,
        "summary": summary,
        "crowding": crowding,
    }


def load_verified_evidence(
    index_dir=DEFAULT_INDEX_DIR,
    summary_dir=DEFAULT_SUMMARY_DIR,
    sampling_dir=DEFAULT_SAMPLING_DIR,
    overlap_dir=DEFAULT_OVERLAP_DIR,
    *,
    expected_images=EXPECTED_IMAGES,
    expected_labels=EXPECTED_LABELS,
    expected_classes=EXPECTED_CLASSES,
    expected_selected_images=EXPECTED_SELECTED_IMAGES,
    expected_selected_labels=EXPECTED_SELECTED_LABELS,
    expected_rare_classes=EXPECTED_RARE_CLASSES,
    locked_hashes=LOCKED_HASHES,
):
    """Validate every tabular evidence layer and return plotting tables."""

    index_path = _verify_directory(index_dir, INDEX_FILES, "Index evidence")
    summary_path = _verify_directory(summary_dir, SUMMARY_FILES, "Summary evidence")
    sampling_path = _verify_directory(sampling_dir, SAMPLING_FILES, "Sampling evidence")
    overlap_path = _verify_directory(overlap_dir, OVERLAP_FILES, "Overlap evidence")
    directories = {
        "index": (index_path, INDEX_FILES),
        "summary": (summary_path, SUMMARY_FILES),
        "sampling": (sampling_path, SAMPLING_FILES),
        "overlap": (overlap_path, OVERLAP_FILES),
    }
    _verify_locked_hashes(directories, locked_hashes)
    base = _validate_index(index_path, expected_images, expected_labels, expected_classes)
    summary = _validate_summary(summary_path, base)
    sampling = _validate_sampling(
        sampling_path,
        base,
        expected_selected_images,
        expected_selected_labels,
        expected_rare_classes,
    )
    overlap = _validate_overlap(overlap_path, base, sampling)
    return {
        "directories": {name: value[0] for name, value in directories.items()},
        "hashes": {key: _sha256_file(directories[key.split('/', 1)[0]][0]
                                      / key.split('/', 1)[1])
                   for key in (locked_hashes or {
                       f"{group}/{name}": "" for group, (_, names) in directories.items()
                       for name in names
                   })},
        "base": base,
        "summary": summary,
        "sampling": sampling,
        "overlap": overlap,
    }


def _workload_design_figure(evidence):
    full = evidence["summary"]["summary"].iloc[0]
    selected = evidence["sampling"]["sample_summary"].iloc[1]
    selected_overlap = evidence["overlap"]["selected_profile"]
    crowded_images = int(selected_overlap["pairs_iou_gt_0_1"].gt(0).sum())
    panels = [
        {
            "heading": "Input",
            "bullets": [
                f"{int(full['images']):,} indexed .jpg files",
                f"{int(full['total_objects']):,} labeled objects across "
                f"{len(evidence['base']['classes'])} classes",
                "External images, labels, and class vocabulary",
            ],
        },
        {
            "heading": "Controlled",
            "bullets": [
                f"{int(selected['images']):,}-image fixed analysis budget",
                "Three deterministic candidate policies",
                "Six density strata and eight protected classes",
                "Checkpoint outputs excluded",
            ],
        },
        {
            "heading": "Output",
            "bullets": [
                f"{int(selected['images']):,} images / {int(selected['total_objects']):,} labels",
                "All eight class-coverage targets satisfied",
                f"{crowded_images:,}-image overlap slice for NMS analysis",
            ],
        },
    ]
    return three_panel_figure(
        "Experiment 02 design at a glance",
        "The checkpoint selected in Experiment 01 stayed outside sample construction; "
        "this stage fixed a reusable, annotation-driven analysis workload.",
        panels,
    )


def _class_inventory_figure(evidence):
    table = evidence["summary"]["enriched"].sort_values(
        ["object_count", "class_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    fig, (objects_axis, images_axis) = plt.subplots(
        1, 2, figsize=(14.0, 9.7), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.08, top=0.83, wspace=0.22)
    add_header(
        fig,
        "Class inventory exposes two kinds of imbalance",
        "Object volume and image presence answer different workload questions; "
        f"n={int(table['object_count'].sum()):,} labels across "
        f"{len(evidence['base']['index']):,} images.",
    )
    positions = np.arange(len(table))
    objects = table["object_count"].to_numpy(dtype=float)
    objects_axis.barh(positions, objects, color=NAVY, height=0.62)
    objects_axis.set_yticks(positions, table["class_name"].astype(str))
    objects_axis.invert_yaxis()
    objects_axis.set_xlim(0, objects.max() * 1.20)
    objects_axis.set_xlabel("Labeled objects")
    objects_axis.set_title("Annotation volume", loc="left", fontweight="bold")
    for position, value in zip(positions, objects):
        objects_axis.text(value + objects.max() * 0.012, position, f"{int(value):,}",
                          va="center", fontsize=8.8, color=INK)
    clean_axis(objects_axis)

    presence = table["image_share_pct"].to_numpy(dtype=float)
    image_counts = table["image_count"].to_numpy(dtype=int)
    images_axis.hlines(positions, 0, presence, color=GRID, linewidth=2)
    images_axis.scatter(presence, positions, color=TEAL, s=48, zorder=3)
    images_axis.set_yticks(positions, [])
    images_axis.tick_params(axis="y", left=False, labelleft=False)
    images_axis.invert_yaxis()
    images_axis.set_xlim(0, presence.max() * 1.35)
    images_axis.set_xlabel("Images containing class (%)")
    images_axis.set_title("Scene presence", loc="left", fontweight="bold")
    for position, share, count in zip(positions, presence, image_counts):
        images_axis.text(share + presence.max() * 0.018, position,
                         f"{share:.1f}%  ({count:,})", va="center", fontsize=8.8, color=INK)
    clean_axis(images_axis)
    return fig


def _candidate_scorecard_figure(evidence):
    table = evidence["sampling"]["candidates"].set_index("sample_name").loc[CANDIDATE_ORDER]
    labels = ["Seeded random", "Density stratified", "Rare-aware stratified"]
    metrics = [
        ("class_object_share_mae_pp", "Class-share MAE", "percentage points (lower is better)", 4),
        ("density_share_mae_pp", "Density-share MAE", "percentage points (lower is better)", 4),
        ("min_rare_class_image_retention_pct", "Protected-class floor", "% of full-class images (higher is better)", 2),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.2), sharey=True)
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.14, top=0.76, wspace=0.30)
    add_header(
        fig,
        "Candidate sampling trade-off",
        "Each policy contains exactly 5,000 images. Metrics come from one deterministic "
        "seed-42 run; no composite objective or uncertainty interval was defined.",
    )
    positions = np.arange(len(labels))
    colors = [NEUTRAL, NAVY, TEAL]
    for axis, (column, title, unit, decimals) in zip(axes, metrics):
        values = table[column].to_numpy(dtype=float)
        axis.hlines(positions, 0, values, color=GRID, linewidth=2)
        axis.scatter(values, positions, c=colors, s=72, zorder=3)
        axis.set_xlim(0, values.max() * 1.32)
        axis.set_xlabel(unit)
        axis.set_title(title, loc="left", fontweight="bold")
        for position, value in zip(positions, values):
            axis.text(value + values.max() * 0.035, position, f"{value:.{decimals}f}",
                      va="center", fontsize=9, color=INK)
        clean_axis(axis)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    return fig


def _rare_coverage_figure(evidence):
    targets = evidence["sampling"]["rare_targets"].copy()
    coverage = evidence["sampling"]["rare_coverage"].copy()
    table = targets.merge(
        coverage[["class_name", "image_count_full", "image_count_sample",
                  "sample_image_retention_pct"]],
        on="class_name", validate="one_to_one",
    ).sort_values(["object_count", "class_id"], kind="mergesort").reset_index(drop=True)
    table["target_retention_pct"] = (
        100.0 * table["target_image_count"] / table["image_count_full"]
    )
    fig, axis = plt.subplots(figsize=(12.3, 7.2))
    fig.subplots_adjust(left=0.24, right=0.83, bottom=0.13, top=0.80)
    add_header(
        fig,
        "Every protected-class image target was met",
        "Protected classes are the eight lowest object-count classes. Diamonds mark "
        "targets; circles mark achieved image retention in the selected workload.",
    )
    positions = np.arange(len(table))
    target = table["target_retention_pct"].to_numpy(dtype=float)
    actual = table["sample_image_retention_pct"].to_numpy(dtype=float)
    for position, left, right in zip(positions, target, actual):
        axis.plot([left, right], [position, position], color=GRID, linewidth=2.5)
    axis.scatter(target, positions, marker="D", color=NEUTRAL, s=48, zorder=3)
    axis.scatter(actual, positions, marker="o", color=TEAL, s=64, zorder=4)
    axis.set_yticks(positions, table["class_name"].astype(str))
    axis.invert_yaxis()
    lower = min(target.min(), actual.min()) - 0.8
    upper = max(target.max(), actual.max()) + 0.8
    axis.set_xlim(lower, upper)
    axis.set_xlabel("Retained class-containing images (%)")
    for position, row in enumerate(table.itertuples(index=False)):
        axis.text(
            upper + 0.05,
            position,
            f"actual {int(row.image_count_sample):,} / target {int(row.target_image_count):,}",
            va="center",
            fontsize=8.8,
            color=INK,
            clip_on=False,
        )
    clean_axis(axis)
    return fig


def _deviation_panel(axis, labels, values, color, title, xlabel, *, show_labels=True,
                     details=None):
    positions = np.arange(len(labels))
    values = np.asarray(values, dtype=float)
    extent = max(float(np.abs(values).max()), 1e-6)
    axis.axvline(0, color=INK, linewidth=0.9)
    axis.hlines(positions, 0, values, color=GRID, linewidth=2)
    axis.scatter(values, positions, color=color, s=54, zorder=3)
    axis.set_xlim(-1.42 * extent, 1.75 * extent)
    axis.set_yticks(positions)
    if show_labels:
        axis.set_yticklabels(labels)
    else:
        axis.tick_params(axis="y", left=False, labelleft=False)
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    axis.set_title(title, loc="left", fontweight="bold")
    for position, value in zip(positions, values):
        horizontal = 7 if value >= 0 else -7
        alignment = "left" if value >= 0 else "right"
        axis.annotate(f"{value:+.3f}", (value, position), xytext=(horizontal, 0),
                      textcoords="offset points", ha=alignment, va="center",
                      fontsize=8.6, color=INK)
        if details is not None:
            axis.text(1.02, position, details[position], transform=axis.get_yaxis_transform(),
                      va="center", fontsize=8.2, color=MUTED, clip_on=False)
    clean_axis(axis)


def _class_composition_fidelity_figure(evidence):
    table = evidence["sampling"]["comparison"].sort_values(
        ["object_count_full", "class_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.3, 9.7), sharey=True)
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.09, top=0.82, wspace=0.24)
    add_header(
        fig,
        "Selected class composition stays close to the full corpus",
        "Exact finite-population differences: selected share minus full-corpus share. "
        "The selected rows are a subset, so these are descriptive differences without confidence intervals.",
    )
    labels = table["class_name"].astype(str).tolist()
    _deviation_panel(
        axes[0], labels, table["object_share_diff_pp"], NAVY,
        "Annotation-volume share", "Difference (percentage points)", show_labels=True,
    )
    _deviation_panel(
        axes[1], labels, table["image_share_diff_pp"], TEAL,
        "Image-presence share", "Difference (percentage points)", show_labels=False,
    )
    return fig


def _scene_structure_fidelity_figure(evidence):
    density = evidence["sampling"]["density_comparison"]
    crowding = evidence["overlap"]["crowding"]
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.4))
    fig.subplots_adjust(left=0.12, right=0.93, bottom=0.15, top=0.76, wspace=0.47)
    add_header(
        fig,
        "Scene structure is preserved on the measured margins",
        "Selected minus full-corpus image share. Density uses objects per image; "
        "crowding uses the number of ground-truth pairs with IoU > 0.10.",
    )
    density_details = [
        f"full {float(row.image_share_pct_full):.2f}% | selected {float(row.image_share_pct_sample):.2f}%"
        for row in density.itertuples(index=False)
    ]
    crowding_details = [
        f"full {float(row.image_share_pct_full):.2f}% | selected {float(row.image_share_pct_sample):.2f}%"
        for row in crowding.itertuples(index=False)
    ]
    _deviation_panel(
        axes[0], density["density_bucket"].astype(str).tolist(),
        density["image_share_diff_pp"], NAVY,
        "Object-density buckets", "Difference (percentage points)",
        details=density_details,
    )
    _deviation_panel(
        axes[1], crowding["crowding_bucket"].astype(str).tolist(),
        crowding["image_share_diff_pp"], TEAL,
        "Crowding buckets", "Difference (percentage points)",
        details=crowding_details,
    )
    return fig


def build_figure_package(evidence, output_dir):
    """Render all six figures and atomically promote the completed package."""

    builders = (
        (OUTPUT_STEMS[0], lambda: _workload_design_figure(evidence)),
        (OUTPUT_STEMS[1], lambda: _class_inventory_figure(evidence)),
        (OUTPUT_STEMS[2], lambda: _candidate_scorecard_figure(evidence)),
        (OUTPUT_STEMS[3], lambda: _rare_coverage_figure(evidence)),
        (OUTPUT_STEMS[4], lambda: _class_composition_fidelity_figure(evidence)),
        (OUTPUT_STEMS[5], lambda: _scene_structure_fidelity_figure(evidence)),
    )
    try:
        destination = build_atomic_package(
            output_dir,
            builders,
            hash_salt="warehouse-object-detection-experiment-02-v1",
        )
    except FigureBuildError as error:
        if isinstance(error, FigureEvidenceError):
            raise
        raise FigureEvidenceError(str(error)) from error
    print(f"[COMPLETE] Promoted verified figure package: {destination}")
    for path in sorted(destination.iterdir()):
        print(f"[WRITE] {path}")
    return destination


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build verified publication figures for Experiment 02."
    )
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--sampling-dir", type=Path, default=DEFAULT_SAMPLING_DIR)
    parser.add_argument("--overlap-dir", type=Path, default=DEFAULT_OVERLAP_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory to receive the atomic publication-figure package.",
    )
    parser.add_argument("--expected-images", type=positive_int, default=EXPECTED_IMAGES)
    parser.add_argument("--expected-labels", type=positive_int, default=EXPECTED_LABELS)
    parser.add_argument("--expected-classes", type=positive_int, default=EXPECTED_CLASSES)
    parser.add_argument("--expected-selected-images", type=positive_int,
                        default=EXPECTED_SELECTED_IMAGES)
    parser.add_argument("--expected-selected-labels", type=positive_int,
                        default=EXPECTED_SELECTED_LABELS)
    parser.add_argument("--expected-rare-classes", type=positive_int,
                        default=EXPECTED_RARE_CLASSES)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    evidence = load_verified_evidence(
        args.index_dir,
        args.summary_dir,
        args.sampling_dir,
        args.overlap_dir,
        expected_images=args.expected_images,
        expected_labels=args.expected_labels,
        expected_classes=args.expected_classes,
        expected_selected_images=args.expected_selected_images,
        expected_selected_labels=args.expected_selected_labels,
        expected_rare_classes=args.expected_rare_classes,
    )
    return build_figure_package(evidence, args.output_dir)


if __name__ == "__main__":
    main()
