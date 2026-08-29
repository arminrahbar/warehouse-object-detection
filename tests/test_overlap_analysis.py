import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "scripts"
    / "02_dataset_analysis"
    / "04_analyze_overlap.py"
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


overlap = load_module(SCRIPT_PATH, "overlap_analysis_under_test")


class OverlapFixture(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.root = Path(self.temp_directory.name)
        self.labels = self.root / "labels"
        self.labels.mkdir()
        self.output_dir = self.root / "output"
        self.figure_dir = self.root / "figures"

        label_rows = {
            "a.txt": ["0 0.25 0.25 0.20 0.20"],
            "b.txt": [
                "0 0.50 0.50 0.40 0.40",
                "1 0.50 0.50 0.40 0.40",
            ],
            "c.txt": [
                "0 0.20 0.20 0.20 0.20",
                "0 0.25 0.20 0.20 0.20",
                "0 0.90 0.90 0.10 0.10",
            ],
            "d.txt": [],
        }
        for name, rows in label_rows.items():
            (self.labels / name).write_text("\n".join(rows), encoding="utf-8")

        self.index = pd.DataFrame(
            [
                {
                    "image_file": f"{stem}.jpg",
                    "image_path": str(self.root / f"{stem}.jpg"),
                    "label_path": str(self.labels / f"{stem}.txt"),
                    "num_objects": count,
                }
                for stem, count in (("a", 1), ("b", 2), ("c", 3), ("d", 0))
            ]
        )
        self.selected = self.index.iloc[[1, 2]].copy()
        self.index_path = self.root / "dataset_index.csv"
        self.selected_path = self.root / "selected_sample_index.csv"
        self.index.to_csv(self.index_path, index=False)
        self.selected.to_csv(self.selected_path, index=False)

    def load(self):
        return overlap.load_and_validate_indexes(self.index_path, self.selected_path)

    def profile(self):
        index, _ = self.load()
        return overlap.build_overlap_profile(
            index,
            asset_root=self.root,
            progress_interval=None,
        )


class GeometryTests(unittest.TestCase):
    def test_default_paths_follow_the_canonical_stage_hierarchy(self):
        self.assertEqual(
            overlap.DEFAULT_DATASET_INDEX,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "00_dataset_inventory"
            / "dataset_index.csv",
        )
        self.assertEqual(
            overlap.DEFAULT_SELECTED_INDEX,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "02_dataset_analysis"
            / "02_sample_selection"
            / "selected_sample_index.csv",
        )
        self.assertEqual(
            overlap.DEFAULT_OVERLAP_DIR,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "02_dataset_analysis"
            / "03_overlap_analysis",
        )
        self.assertEqual(
            overlap.DEFAULT_FIGURE_DIR,
            PROJECT_ROOT / "scratch" / "diagnostic-figures" / "02_dataset_analysis",
        )

    def test_yolo_center_geometry_converts_to_top_left_xywh(self):
        np.testing.assert_allclose(
            overlap.yolo_to_xywh(0.5, 0.4, 0.2, 0.1),
            [0.4, 0.35, 0.2, 0.1],
        )

    def test_crowding_bucket_boundaries_are_stable(self):
        expected = {
            0: "0",
            1: "1-4",
            4: "1-4",
            5: "5-19",
            19: "5-19",
            20: "20+",
            100: "20+",
        }
        self.assertEqual(
            {count: overlap.crowding_bucket(count) for count in expected},
            expected,
        )

    def test_invalid_crowding_counts_are_rejected(self):
        for value in (-1, 1.5, "1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    overlap.crowding_bucket(value)

    def test_zero_and_one_box_have_no_pairs(self):
        expected = {
            "pair_count": 0,
            "max_pairwise_iou": 0.0,
            "mean_pairwise_iou": 0.0,
            "pairs_iou_gt_0_1": 0,
            "pairs_iou_gt_0_3": 0,
            "pairs_iou_gt_0_5": 0,
        }
        self.assertEqual(overlap.compute_overlap_for_boxes([]), expected)
        self.assertEqual(overlap.compute_overlap_for_boxes([[0, 0, 1, 1]]), expected)

    def test_each_unordered_pair_is_counted_once(self):
        boxes = [[0, 0, 1, 1], [0, 0, 1, 1], [3, 3, 1, 1]]
        result = overlap.compute_overlap_for_boxes(boxes)
        self.assertEqual(result["pair_count"], 3)
        self.assertEqual(result["max_pairwise_iou"], 1.0)
        self.assertAlmostEqual(result["mean_pairwise_iou"], 1.0 / 3.0)
        self.assertEqual(result["pairs_iou_gt_0_1"], 1)
        self.assertEqual(result["pairs_iou_gt_0_5"], 1)

    def test_iou_thresholds_are_strictly_greater_than(self):
        boxes = [[0, 0, 1, 1], [0, 0, 1, 1]]
        with patch.object(overlap, "calculate_iou", return_value=0.1):
            result = overlap.compute_overlap_for_boxes(boxes)
        self.assertEqual(result["max_pairwise_iou"], 0.1)
        self.assertEqual(result["pairs_iou_gt_0_1"], 0)

    def test_invalid_box_geometry_is_rejected(self):
        for boxes in ([[0, 0, 1]], [[0, 0, -1, 1]], [[0, np.nan, 1, 1]]):
            with self.subTest(boxes=boxes):
                with self.assertRaises(ValueError):
                    overlap.compute_overlap_for_boxes(boxes)


class LabelParsingTests(OverlapFixture):
    def test_valid_rows_are_parsed_and_blank_lines_are_ignored(self):
        path = self.labels / "valid.txt"
        path.write_text("\n0 0.5 0.5 0.2 0.4 extra\n\n", encoding="utf-8")
        self.assertEqual(overlap.parse_yolo_boxes(path), [[0.4, 0.3, 0.2, 0.4]])

    def test_missing_label_file_is_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "Label file not found"):
            overlap.parse_yolo_boxes(self.labels / "missing.txt")

    def test_malformed_and_nonfinite_rows_are_rejected(self):
        cases = {
            "short.txt": "0 0.5 0.5 0.2",
            "text.txt": "0 x 0.5 0.2 0.2",
            "nan.txt": "0 nan 0.5 0.2 0.2",
            "class.txt": "0.5 0.5 0.5 0.2 0.2",
            "geometry.txt": "0 1.2 0.5 0.2 0.2",
        }
        for name, content in cases.items():
            with self.subTest(name=name):
                path = self.labels / name
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    overlap.parse_yolo_boxes(path)


class IndexValidationTests(OverlapFixture):
    def test_valid_selected_index_is_proven_to_be_a_subset(self):
        full, selected = self.load()
        self.assertEqual(len(full), 4)
        self.assertEqual(selected["image_file"].tolist(), ["b.jpg", "c.jpg"])

    def test_duplicate_full_image_identifiers_are_rejected(self):
        broken = pd.concat([self.index, self.index.iloc[[0]]], ignore_index=True)
        broken.to_csv(self.index_path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate image_file"):
            self.load()

    def test_selected_image_must_exist_in_full_index(self):
        broken = self.selected.copy()
        broken.loc[broken.index[0], "image_file"] = "unknown.jpg"
        broken.to_csv(self.selected_path, index=False)
        with self.assertRaisesRegex(ValueError, "absent from the full index"):
            self.load()

    def test_selected_metadata_must_match_full_index(self):
        broken = self.selected.copy()
        broken["num_objects"] = broken["num_objects"].astype(float)
        broken.loc[broken.index[0], "num_objects"] = 3
        broken.to_csv(self.selected_path, index=False)
        with self.assertRaisesRegex(ValueError, "num_objects values do not match"):
            self.load()

    def test_counts_must_be_non_negative_integers(self):
        broken = self.index.copy()
        broken["num_objects"] = broken["num_objects"].astype(float)
        broken.loc[0, "num_objects"] = 0.5
        broken.to_csv(self.index_path, index=False)
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            self.load()

    def test_external_asset_root_maps_canonical_namespace(self):
        asset_root = self.root / "assets"
        expected = asset_root / "logistics" / "label.txt"
        value = "detector_service/storage/logistics/label.txt"
        self.assertEqual(
            overlap.resolve_label_path(value, asset_root=asset_root),
            expected,
        )


class EvidenceTests(OverlapFixture):
    def test_profile_reconciles_parsed_box_count_with_index(self):
        profile = self.profile()
        self.assertEqual(list(profile.columns), overlap.PROFILE_COLUMNS)
        self.assertEqual(profile["pair_count"].tolist(), [0, 1, 3, 0])
        self.assertEqual(profile["crowding_bucket"].tolist(), ["0", "1-4", "1-4", "0"])

    def test_profile_rejects_stale_index_object_count(self):
        broken = self.index.copy()
        broken.loc[0, "num_objects"] = 2
        with self.assertRaisesRegex(ValueError, "Label count mismatch"):
            overlap.build_overlap_profile(
                broken,
                asset_root=self.root,
                progress_interval=None,
            )

    def test_summary_and_distribution_have_stable_schemas(self):
        profile = self.profile()
        artifacts = overlap.build_overlap_evidence(
            profile,
            self.selected,
            "rare_aware_density_stratified_2",
        )
        self.assertEqual(
            list(artifacts["overlap_summary.csv"].columns),
            overlap.SUMMARY_COLUMNS,
        )
        comparison = artifacts["crowding_distribution_comparison.csv"]
        self.assertEqual(list(comparison.columns), overlap.CROWDING_COMPARISON_COLUMNS)
        self.assertEqual(comparison["crowding_bucket"].tolist(), list(overlap.CROWDING_BUCKET_ORDER))
        self.assertAlmostEqual(comparison["image_share_pct_full"].sum(), 100.0)
        self.assertAlmostEqual(comparison["image_share_pct_sample"].sum(), 100.0)

    def test_all_three_csv_artifacts_are_written_atomically(self):
        profile = self.profile()
        artifacts = overlap.build_overlap_evidence(profile, self.selected, "selected_2")
        paths = overlap.write_overlap_artifacts(self.output_dir, artifacts)
        self.assertEqual(set(paths), set(artifacts))
        for name, path in paths.items():
            self.assertTrue(path.is_file())
            assert_frame_equal(pd.read_csv(path), artifacts[name], check_dtype=False)
        self.assertEqual(list(self.output_dir.glob(".*.csv")), [])

    def test_main_supports_headless_external_paths(self):
        artifacts, paths, figure_path = overlap.main(
            [
                "--dataset-index",
                str(self.index_path),
                "--selected-index",
                str(self.selected_path),
                "--asset-root",
                str(self.root),
                "--output-dir",
                str(self.output_dir),
                "--selected-name",
                "fixture_sample",
                "--skip-figure",
            ]
        )
        self.assertIsNone(figure_path)
        self.assertEqual(artifacts["overlap_summary.csv"].iloc[1]["dataset"], "fixture_sample")
        self.assertEqual(len(paths), 3)


class ReferenceCompatibilityTests(OverlapFixture):
    def test_valid_geometry_and_aggregates_match_reference(self):
        reference_path = os.environ.get("REFERENCE_OVERLAP_ANALYSIS_SCRIPT")
        if not reference_path:
            self.skipTest("REFERENCE_OVERLAP_ANALYSIS_SCRIPT is not configured")

        pyplot = types.ModuleType("matplotlib.pyplot")
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.pyplot = pyplot
        with patch.dict(
            sys.modules,
            {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot},
        ):
            reference = load_module(Path(reference_path), "reference_overlap_analysis")

        for label_path in sorted(self.labels.glob("*.txt")):
            with self.subTest(label=label_path.name):
                expected = reference.compute_overlap_for_label(label_path)
                observed = overlap.compute_overlap_for_label(label_path)
                self.assertEqual(observed["pair_count"], expected["pair_count"])
                for key in (
                    "max_pairwise_iou",
                    "mean_pairwise_iou",
                    "pairs_iou_gt_0_1",
                    "pairs_iou_gt_0_3",
                    "pairs_iou_gt_0_5",
                ):
                    self.assertAlmostEqual(observed[key], expected[key])

        profile = self.profile()
        for table in (profile, profile.iloc[[1, 2]].copy()):
            expected_summary = reference.summarize_overlap(table, "fixture")
            observed_summary = overlap.summarize_overlap(table, "fixture")
            self.assertEqual(observed_summary, expected_summary)
            assert_frame_equal(
                overlap.crowding_distribution(table, "fixture"),
                reference.crowding_distribution(table, "fixture"),
                check_dtype=False,
            )


if __name__ == "__main__":
    unittest.main()
