import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "scripts" / "03_nms_threshold_sweep.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load_module(SCRIPT_PATH, "nms_threshold_sweep_under_test")


class SweepFixture(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.root = Path(self.temp_directory.name)
        self.output_dir = self.root / "outputs"
        self.figure_dir = self.root / "figures"
        self.classes = ["alpha", "beta", "gamma"]
        self.class_file = self.root / "classes.names"
        self.class_file.write_text("\n".join(self.classes), encoding="utf-8")

        self.index = pd.DataFrame(
            [
                {
                    "image_file": f"{name}.jpg",
                    "image_path": f"detector_service/storage/logistics/{name}.jpg",
                    "label_path": f"detector_service/storage/logistics/{name}.txt",
                    "num_objects": objects,
                }
                for name, objects in (("a", 1), ("b", 1), ("c", 0), ("d", 1))
            ]
        )
        self.index_path = self.root / "selected_sample_index.csv"
        self.index.to_csv(self.index_path, index=False)

        self.overlap = pd.DataFrame(
            {
                "image_file": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
                "pairs_iou_gt_0_1": [1, 0, 0, 2],
            }
        )
        self.overlap_path = self.root / "overlap_profile.csv"
        self.overlap.to_csv(self.overlap_path, index=False)

        self.ground_truth = pd.DataFrame(
            [
                self.gt_row("a.jpg", 0, [0, 0, 10, 10]),
                self.gt_row("b.jpg", 1, [20, 20, 8, 8]),
                self.gt_row("d.jpg", 2, [40, 40, 6, 6]),
            ],
            columns=sweep.GROUND_TRUTH_COLUMNS,
        )
        self.ground_truth_path = self.root / "ground_truth.csv"
        self.ground_truth.to_csv(self.ground_truth_path, index=False)

        self.raw = pd.DataFrame(
            [
                self.raw_row("a.jpg", 0, [0, 0, 10, 10], 0.95, 0.95),
                self.raw_row("a.jpg", 0, [1, 1, 10, 10], 0.90, 0.90),
                self.raw_row("a.jpg", 1, [0, 0, 10, 10], 0.80, 0.80),
                self.raw_row("b.jpg", 1, [20, 20, 8, 8], 0.95, 0.90),
                self.raw_row("d.jpg", 2, [40, 40, 6, 6], 0.90, 0.90),
                self.raw_row("d.jpg", 2, [40, 40, 6, 6], 0.60, 0.60),
            ],
            columns=sweep.RAW_COLUMNS,
        )
        self.raw_path = self.root / "raw_predictions.csv"
        self.raw.to_csv(self.raw_path, index=False)

    def gt_row(self, image_file, class_id, box):
        return {
            "image_file": image_file,
            "image_path": f"detector_service/storage/logistics/{image_file}",
            "class_id": class_id,
            "class_name": self.classes[class_id],
            "bbox_x": box[0],
            "bbox_y": box[1],
            "bbox_w": box[2],
            "bbox_h": box[3],
        }

    def raw_row(self, image_file, class_id, box, objectness, class_probability):
        vector = [0.0] * len(self.classes)
        vector[class_id] = class_probability
        return {
            "model": sweep.MODEL_NAME,
            "image_file": image_file,
            "image_path": f"detector_service/storage/logistics/{image_file}",
            "bbox_x": box[0],
            "bbox_y": box[1],
            "bbox_w": box[2],
            "bbox_h": box[3],
            "class_id": class_id,
            "class_name": self.classes[class_id],
            "object_score": objectness,
            "predicted_class_score": class_probability,
            "combined_confidence": objectness * class_probability,
            "class_scores_json": json.dumps(vector),
        }


class ContractTests(unittest.TestCase):
    def test_fixed_operating_policy_is_explicit(self):
        self.assertEqual(sweep.MODEL_NAME, "model2")
        self.assertEqual(sweep.SCORE_THRESHOLD, 0.5)
        self.assertEqual(sweep.MAP_IOU_THRESHOLD, 0.5)
        self.assertEqual(sweep.EVAL_TYPE, "combined")
        self.assertEqual(
            sweep.NMS_THRESHOLDS,
            (0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7),
        )

    def test_threshold_tags_are_stable(self):
        self.assertEqual(sweep.threshold_tag(0.2), "0_2")
        self.assertEqual(sweep.threshold_tag(0.55), "0_55")

    def test_positive_integer_parser_rejects_zero(self):
        self.assertEqual(sweep.positive_int("3"), 3)
        with self.assertRaisesRegex(Exception, "positive integer"):
            sweep.positive_int("0")

    def test_artifact_schemas_are_explicit(self):
        self.assertEqual(sweep.PREDICTION_COLUMNS, sweep.RAW_COLUMNS + ["nms_threshold"])
        self.assertEqual(len(sweep.SUMMARY_COLUMNS), 10)
        self.assertEqual(len(sweep.PER_CLASS_COLUMNS), 8)
        self.assertEqual(len(sweep.SUBSET_COLUMNS), 10)


class InputValidationTests(SweepFixture):
    def test_sample_index_is_unique_and_supports_a_prefix_bound(self):
        loaded = sweep.load_sample_index(self.index_path, max_images=2)
        self.assertEqual(loaded["image_file"].tolist(), ["a.jpg", "b.jpg"])

    def test_sample_bound_cannot_exceed_index(self):
        with self.assertRaisesRegex(ValueError, "exceeds selected sample size"):
            sweep.load_sample_index(self.index_path, max_images=5)

    def test_duplicate_image_files_are_rejected(self):
        broken = pd.concat([self.index, self.index.iloc[[0]]], ignore_index=True)
        broken.to_csv(self.index_path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate image_file"):
            sweep.load_sample_index(self.index_path)

    def test_classes_load_from_file_or_consistent_cache(self):
        self.assertEqual(sweep.load_classes(self.class_file), self.classes)
        self.assertEqual(
            sweep.load_classes(self.root / "missing.names", (self.ground_truth_path, self.raw_path)),
            self.classes,
        )

    def test_conflicting_cached_class_names_are_rejected(self):
        broken = self.raw.copy()
        broken.loc[0, "class_name"] = "different"
        broken.to_csv(self.raw_path, index=False)
        with self.assertRaisesRegex(ValueError, "Conflicting names"):
            sweep.load_classes(self.root / "missing.names", (self.ground_truth_path, self.raw_path))

    def test_ground_truth_must_reconcile_object_counts(self):
        broken = self.ground_truth.iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "row counts do not match"):
            sweep.validate_ground_truth(broken, self.index, self.classes)

    def test_raw_cache_validates_score_derivations(self):
        validated = sweep.validate_raw_predictions(self.raw, self.index, self.classes)
        self.assertEqual(len(validated), len(self.raw))
        broken = self.raw.copy()
        broken.loc[0, "combined_confidence"] = 0.1
        with self.assertRaisesRegex(ValueError, "does not equal"):
            sweep.validate_raw_predictions(broken, self.index, self.classes)

    def test_raw_cache_rejects_invalid_json_and_vector_length(self):
        for value in ("not-json", json.dumps([1.0])):
            with self.subTest(value=value):
                broken = self.raw.copy()
                broken.loc[0, "class_scores_json"] = value
                with self.assertRaisesRegex(ValueError, r"class[_ ]scores"):
                    sweep.validate_raw_predictions(broken, self.index, self.classes)

    def test_overlap_profile_defines_crowded_selected_subset(self):
        _, crowded = sweep.load_overlap_profile(self.overlap_path, self.index)
        self.assertEqual(crowded["image_file"].tolist(), ["a.jpg", "d.jpg"])

    def test_overlap_profile_must_cover_every_selected_image(self):
        self.overlap.iloc[:-1].to_csv(self.overlap_path, index=False)
        with self.assertRaisesRegex(ValueError, "does not cover"):
            sweep.load_overlap_profile(self.overlap_path, self.index)

    def test_external_storage_namespaces_map_to_asset_root(self):
        asset_root = self.root / "assets"
        expected = asset_root / "logistics" / "a.jpg"
        for value in (
            "detector_service/storage/logistics/a.jpg",
            "techtrack/storage/logistics/a.jpg",
        ):
            self.assertEqual(sweep.resolve_indexed_path(value, asset_root), expected)


class GroundTruthTests(SweepFixture):
    def test_yolo_rows_convert_to_pixel_space(self):
        label = self.root / "label.txt"
        label.write_text("1 0.5 0.5 0.2 0.4", encoding="utf-8")
        rows = sweep.yolo_label_to_xywh(label, 100, 50, self.classes)
        self.assertEqual(rows[0]["class_name"], "beta")
        np.testing.assert_allclose(
            [rows[0][name] for name in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")],
            [40, 15, 20, 20],
        )

    def test_invalid_yolo_geometry_is_rejected(self):
        label = self.root / "label.txt"
        label.write_text("0 1.2 0.5 0.2 0.2", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid normalized"):
            sweep.yolo_label_to_xywh(label, 100, 100, self.classes)

    def test_ground_truth_builder_reconciles_each_label(self):
        asset_root = self.root / "assets"
        logistics = asset_root / "logistics"
        logistics.mkdir(parents=True)
        index = self.index.iloc[[0]].copy()
        (logistics / "a.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
        image_reader = Mock(return_value=np.zeros((50, 100, 3), dtype=np.uint8))
        table = sweep.build_ground_truth(
            index,
            self.classes,
            asset_root=asset_root,
            image_reader=image_reader,
            progress_interval=None,
        )
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]["bbox_w"], 20.0)


class PostprocessingTests(SweepFixture):
    def test_nms_is_class_aware_and_threshold_controlled(self):
        low = sweep.apply_nms_threshold(self.raw, self.index, self.classes, 0.3)
        high = sweep.apply_nms_threshold(self.raw, self.index, self.classes, 0.7)
        self.assertLess(len(low), len(high))
        a_low = low[low["image_file"] == "a.jpg"]
        self.assertEqual(set(a_low["class_id"]), {0, 1})
        self.assertEqual(set(low["nms_threshold"]), {0.3})

    def test_invalid_nms_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            sweep.apply_nms_threshold(self.raw, self.index, self.classes, 1.1)

    def test_metric_lists_retain_empty_images_and_index_order(self):
        predictions = sweep.apply_nms_threshold(self.raw, self.index, self.classes, 0.3)
        values = sweep.build_metric_lists(self.index, predictions, self.ground_truth)
        self.assertEqual(len(values[0]), 4)
        self.assertEqual(values[0][2], [])
        self.assertEqual(values[4][2], [])

    def test_evaluator_produces_all_class_rows_and_summary(self):
        predictions = sweep.apply_nms_threshold(self.raw, self.index, self.classes, 0.3)
        summary, per_class = sweep.evaluate_predictions(
            self.index,
            predictions,
            self.ground_truth,
            self.classes,
            0.3,
        )
        self.assertEqual(list(per_class.columns), sweep.PER_CLASS_COLUMNS)
        self.assertEqual(len(per_class), 3)
        self.assertEqual(summary["evaluation_rows"], len(predictions))
        self.assertGreater(summary["mAP@0.5_11_point"], 0.0)

    def test_duplicate_proxy_counts_same_class_pairs_only(self):
        predictions = pd.DataFrame(
            [
                {**self.raw.iloc[0].to_dict(), "nms_threshold": 0.7},
                {**self.raw.iloc[1].to_dict(), "nms_threshold": 0.7},
                {**self.raw.iloc[2].to_dict(), "nms_threshold": 0.7},
            ],
            columns=sweep.PREDICTION_COLUMNS,
        )
        result = sweep.count_duplicate_like_prediction_pairs(predictions)
        self.assertEqual(result["duplicate_like_pairs_iou_gt_0_5"], 1)
        self.assertEqual(result["images_with_duplicate_like_pairs"], 1)

    def test_duplicate_proxy_rejects_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            sweep.count_duplicate_like_prediction_pairs(pd.DataFrame(), 2.0)


class ArtifactTests(SweepFixture):
    def build(self):
        _, crowded = sweep.load_overlap_profile(self.overlap_path, self.index)
        return sweep.run_threshold_sweep(
            self.index,
            self.raw,
            self.ground_truth,
            self.classes,
            crowded,
            self.output_dir,
            "sample5000",
            refresh_postprocessing=True,
        )

    def test_sweep_writes_seven_prediction_caches_and_four_tables(self):
        artifacts = self.build()
        self.assertEqual(len(artifacts), 4)
        self.assertEqual(
            len(list(self.output_dir.glob("model2_predictions_nms_*_sample5000.csv"))),
            7,
        )
        summary = artifacts["nms_threshold_summary_sample5000.csv"]
        self.assertEqual(summary["nms_threshold"].tolist(), list(sweep.NMS_THRESHOLDS))
        self.assertEqual(list(summary.columns), sweep.SUMMARY_COLUMNS)
        self.assertEqual(len(artifacts["per_class_ap_by_threshold_sample5000.csv"]), 21)
        self.assertEqual(len(artifacts["subset_summary_by_threshold_sample5000.csv"]), 14)

    def test_artifact_writes_are_atomic_and_round_trip(self):
        artifacts = self.build()
        paths = sweep.write_sweep_artifacts(self.output_dir, artifacts)
        for name, path in paths.items():
            self.assertTrue(path.is_file())
            assert_frame_equal(pd.read_csv(path), artifacts[name], check_dtype=False)
        self.assertEqual(list(self.output_dir.glob(".*.csv")), [])

    def test_main_reuses_explicit_caches_without_inference(self):
        artifacts, paths, figures = sweep.main(
            [
                "--sample-index",
                str(self.index_path),
                "--overlap-profile",
                str(self.overlap_path),
                "--class-file",
                str(self.root / "missing.names"),
                "--ground-truth-cache",
                str(self.ground_truth_path),
                "--raw-predictions-cache",
                str(self.raw_path),
                "--output-dir",
                str(self.output_dir),
                "--skip-figures",
                "--refresh-postprocessing",
            ]
        )
        self.assertEqual(len(artifacts), 4)
        self.assertEqual(len(paths), 4)
        self.assertEqual(figures, [])

    def test_force_cannot_overwrite_explicit_external_caches(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            sweep.main(
                [
                    "--force",
                    "--ground-truth-cache",
                    str(self.ground_truth_path),
                ]
            )

    def test_figure_builder_emits_four_canonical_names(self):
        artifacts = self.build()
        summary = artifacts["nms_threshold_summary_sample5000.csv"]
        duplicates = artifacts["duplicate_summary_by_threshold_sample5000.csv"]
        subsets = artifacts["subset_summary_by_threshold_sample5000.csv"]

        pyplot = types.ModuleType("matplotlib.pyplot")
        figures = []

        def subplots(**_kwargs):
            figure = Mock()
            axis = Mock()
            figure.savefig.side_effect = lambda path, **kwargs: Path(path).write_bytes(b"figure")
            figures.append(figure)
            return figure, axis

        pyplot.subplots = Mock(side_effect=subplots)
        pyplot.close = Mock()
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.pyplot = pyplot
        with patch.dict(
            sys.modules,
            {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot},
        ):
            paths = sweep.build_figures(summary, duplicates, subsets, self.figure_dir)
        self.assertEqual(
            [path.name for path in paths],
            [
                "01_map_by_threshold.png",
                "02_prediction_count_by_threshold.png",
                "03_duplicate_pairs_by_threshold.png",
                "04_map_by_threshold_and_subset.png",
            ],
        )
        self.assertTrue(all(path.read_bytes() == b"figure" for path in paths))
        self.assertEqual(pyplot.close.call_count, 4)


class ReferenceCompatibilityTests(SweepFixture):
    def test_valid_postprocessing_and_evaluation_match_reference(self):
        reference_path = os.environ.get("REFERENCE_NMS_SWEEP_SCRIPT")
        if not reference_path:
            self.skipTest("REFERENCE_NMS_SWEEP_SCRIPT is not configured")

        pyplot = types.ModuleType("matplotlib.pyplot")
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.pyplot = pyplot
        with patch.dict(
            sys.modules,
            {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot},
        ):
            reference = load_module(Path(reference_path), "reference_nms_threshold_sweep")
        reference.NMS_OUTPUT_DIR = self.root / "reference-output"
        reference.NMS_OUTPUT_DIR.mkdir()

        for threshold in (0.3, 0.7):
            with self.subTest(threshold=threshold):
                expected = reference.apply_nms_threshold(
                    self.raw,
                    self.index,
                    self.classes,
                    threshold,
                    "fixture",
                    force=True,
                )
                observed = sweep.apply_nms_threshold(
                    self.raw,
                    self.index,
                    self.classes,
                    threshold,
                )
                assert_frame_equal(observed, expected, check_dtype=False)

                expected_summary, expected_classes = reference.evaluate_with_metrics_py(
                    self.index,
                    expected,
                    self.ground_truth,
                    self.classes,
                    threshold,
                )
                observed_summary, observed_classes = sweep.evaluate_predictions(
                    self.index,
                    observed,
                    self.ground_truth,
                    self.classes,
                    threshold,
                )
                self.assertEqual(observed_summary, expected_summary)
                assert_frame_equal(observed_classes, expected_classes, check_dtype=False)
                self.assertEqual(
                    sweep.count_duplicate_like_prediction_pairs(observed),
                    reference.count_duplicate_like_prediction_pairs(expected),
                )


if __name__ == "__main__":
    unittest.main()
