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


if "cv2" not in sys.modules:
    cv2 = types.ModuleType("cv2")
    cv2.COLOR_BGR2RGB = 4
    cv2.INTER_LINEAR = 1
    cv2.flip = lambda image, code: np.flip(image, axis=1 if code == 1 else 0).copy()
    cv2.GaussianBlur = lambda image, kernel, sigma: image.copy()
    cv2.resize = lambda image, size, interpolation=None: np.zeros(
        (size[1], size[0], *image.shape[2:]),
        dtype=image.dtype,
    )
    cv2.imread = Mock(return_value=None)
    cv2.cvtColor = lambda image, code: image[..., ::-1].copy()
    sys.modules["cv2"] = cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "experiments" / "scripts"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demo = load_module(SCRIPT_DIR / "04_augmentation_demo.py", "augmentation_demo_under_test")
robustness = load_module(
    SCRIPT_DIR / "04_augmentation_robustness.py",
    "augmentation_robustness_under_test",
)


class _Axis:
    def imshow(self, *args, **kwargs):
        return None

    def set_title(self, *args, **kwargs):
        return None

    def axis(self, *args, **kwargs):
        return None

    def bar(self, *args, **kwargs):
        return None

    def barh(self, *args, **kwargs):
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def tick_params(self, *args, **kwargs):
        return None

    def set_ylim(self, *args, **kwargs):
        return None


class _Figure:
    def suptitle(self, *args, **kwargs):
        return None

    def tight_layout(self):
        return None

    def savefig(self, path, *args, **kwargs):
        Path(path).write_bytes(b"figure")


def fake_pyplot():
    pyplot = types.ModuleType("matplotlib.pyplot")

    def subplots(rows=1, columns=1, **kwargs):
        figure = _Figure()
        count = rows * columns
        axes = [_Axis() for _ in range(count)]
        return figure, axes if count > 1 else axes[0]

    pyplot.subplots = subplots
    pyplot.close = Mock()
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    return {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}


class RobustnessFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output_dir = self.root / "outputs"
        self.cache_dir = self.root / "cache"
        self.figure_dir = self.root / "figures"
        self.cache_dir.mkdir()
        self.classes = ["alpha", "beta"]
        self.class_file = self.root / "classes.names"
        self.class_file.write_text("alpha\nbeta\n", encoding="utf-8")
        self.index = pd.DataFrame(
            [
                {
                    "image_file": "a.jpg",
                    "image_path": "detector_service/storage/images/a.jpg",
                    "label_path": "detector_service/storage/labels/a.txt",
                    "num_objects": 1,
                },
                {
                    "image_file": "b.jpg",
                    "image_path": "detector_service/storage/images/b.jpg",
                    "label_path": "detector_service/storage/labels/b.txt",
                    "num_objects": 1,
                },
                {
                    "image_file": "c.jpg",
                    "image_path": "detector_service/storage/images/c.jpg",
                    "label_path": "detector_service/storage/labels/c.txt",
                    "num_objects": 0,
                },
            ]
        )
        self.index_path = self.root / "selected_sample_index.csv"
        self.index.to_csv(self.index_path, index=False)

    def ground_truth(self, condition):
        return pd.DataFrame(
            [
                self.gt_row(condition, "a.jpg", 0, [0, 0, 10, 10]),
                self.gt_row(condition, "b.jpg", 1, [20, 20, 8, 8]),
            ],
            columns=robustness.GROUND_TRUTH_COLUMNS,
        )

    def gt_row(self, condition, image_file, class_id, box):
        return {
            "dataset": robustness.DATASET_NAME,
            "augmentation_condition": condition["tag"],
            "augmentation_display": condition["display"],
            "image_file": image_file,
            "image_path": f"techtrack/storage/images/{image_file}",
            "class_id": class_id,
            "class_name": self.classes[class_id],
            "bbox_x": box[0],
            "bbox_y": box[1],
            "bbox_w": box[2],
            "bbox_h": box[3],
        }

    def raw(self, condition):
        return pd.DataFrame(
            [
                self.raw_row(condition, "a.jpg", 0, [0, 0, 10, 10], 0.95, 0.95),
                self.raw_row(condition, "a.jpg", 0, [1, 1, 10, 10], 0.80, 0.80),
                self.raw_row(condition, "a.jpg", 1, [0, 0, 10, 10], 0.75, 0.80),
                self.raw_row(condition, "b.jpg", 1, [20, 20, 8, 8], 0.90, 0.90),
            ],
            columns=robustness.RAW_COLUMNS,
        )

    def raw_row(self, condition, image_file, class_id, box, objectness, probability):
        vector = [0.0] * len(self.classes)
        vector[class_id] = probability
        return {
            "model": robustness.MODEL_NAME,
            "dataset": robustness.DATASET_NAME,
            "augmentation_condition": condition["tag"],
            "augmentation_display": condition["display"],
            "image_file": image_file,
            "image_path": f"techtrack/storage/images/{image_file}",
            "bbox_x": box[0],
            "bbox_y": box[1],
            "bbox_w": box[2],
            "bbox_h": box[3],
            "class_id": class_id,
            "class_name": self.classes[class_id],
            "object_score": objectness,
            "predicted_class_score": probability,
            "combined_confidence": objectness * probability,
            "class_scores_json": json.dumps(vector),
        }

    def write_all_input_caches(self):
        for condition in robustness.CONDITIONS:
            paths = robustness._cache_paths(self.cache_dir, condition, "sample5000")
            self.ground_truth(condition).to_csv(paths["ground_truth"], index=False)
            self.raw(condition).to_csv(paths["raw"], index=False)


class ContractTests(RobustnessFixture):
    def test_conditions_and_operating_policy_are_explicit(self):
        self.assertEqual(
            [condition["tag"] for condition in robustness.CONDITIONS],
            [
                "original",
                "gaussian_blur_k9",
                "vertical_flip",
                "brightness_increase",
                "brightness_decrease",
            ],
        )
        self.assertEqual(robustness.OBJECTNESS_THRESHOLD, 0.5)
        self.assertEqual(robustness.SCORE_THRESHOLD, 0.5)
        self.assertEqual(robustness.NMS_IOU_THRESHOLD, 0.3)
        self.assertEqual(robustness.MAP_IOU_THRESHOLD, 0.5)
        self.assertEqual(robustness.EVAL_TYPE, "combined")

    def test_artifact_schemas_are_explicit_and_unique(self):
        for columns in (
            robustness.RAW_COLUMNS,
            robustness.PREDICTION_COLUMNS,
            robustness.GROUND_TRUTH_COLUMNS,
            robustness.SUMMARY_COLUMNS,
            robustness.PER_CLASS_COLUMNS,
        ):
            self.assertEqual(len(columns), len(set(columns)))
        self.assertEqual(
            robustness.PREDICTION_COLUMNS[-1],
            "nms_threshold",
        )

    def test_positive_int_rejects_zero_and_negative_values(self):
        for value in ("0", "-2"):
            with self.subTest(value=value), self.assertRaises(
                robustness.argparse.ArgumentTypeError
            ):
                robustness.positive_int(value)
        self.assertEqual(robustness.positive_int("3"), 3)

    def test_cache_names_are_condition_specific_and_stable(self):
        paths = robustness._cache_paths(
            self.root,
            robustness._condition("vertical_flip"),
            "first_10",
        )
        self.assertEqual(paths["ground_truth"].name, "ground_truth_vertical_flip_first_10.csv")
        self.assertEqual(paths["raw"].name, "model2_raw_predictions_vertical_flip_first_10.csv")
        self.assertIn("class_aware_nms_0_3", paths["predictions"].name)


class DemonstrationTests(RobustnessFixture):
    def test_sample_loader_validates_and_selects_densest_deterministically(self):
        sample = self.index.copy()
        sample.loc[:, "num_objects"] = [2, 2, 0]
        sample.to_csv(self.index_path, index=False)
        loaded = demo.load_sample_index(self.index_path)
        selected = demo.select_example(loaded)
        self.assertEqual(selected["image_file"], "a.jpg")

    def test_sample_loader_rejects_empty_and_invalid_counts(self):
        pd.DataFrame(columns=self.index.columns).to_csv(self.index_path, index=False)
        with self.assertRaisesRegex(ValueError, "empty"):
            demo.load_sample_index(self.index_path)
        invalid = self.index.copy()
        invalid["num_objects"] = invalid["num_objects"].astype(float)
        invalid.loc[0, "num_objects"] = 1.5
        invalid.to_csv(self.index_path, index=False)
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            demo.load_sample_index(self.index_path)

    def test_path_resolver_maps_both_supported_storage_namespaces(self):
        expected = self.root / "images" / "a.jpg"
        for prefix in ("detector_service/storage", "techtrack/storage"):
            actual = demo.resolve_image_path(
                f"{prefix}/images/a.jpg",
                asset_root=self.root,
            )
            self.assertEqual(actual, expected)

    def test_panel_builder_applies_all_four_fixed_operations(self):
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        augmenter = Mock()
        augmenter.vertical_flip.return_value = image + 1
        augmenter.gaussian_blur.return_value = image + 2
        augmenter.change_brightness.side_effect = [image + 3, image + 4]
        panels = demo.build_panels(image, augmenter=augmenter)
        self.assertEqual([title for title, _ in panels], [item[0] for item in demo.PANEL_DEFINITIONS])
        self.assertEqual(len(panels), 5)
        augmenter.gaussian_blur.assert_called_once_with(image=image, kernel_size=9, sigma=0)
        self.assertEqual(augmenter.change_brightness.call_count, 2)

    def test_figure_renderer_writes_the_canonical_png(self):
        panels = [("Original", np.zeros((2, 2, 3), dtype=np.uint8))]
        output = self.root / "figures" / "01_augmentation_examples.png"
        with patch.dict(sys.modules, fake_pyplot()):
            result = demo.render_figure(panels, output)
        self.assertEqual(result, output)
        self.assertEqual(output.read_bytes(), b"figure")

    def test_main_decodes_bgr_builds_panels_and_reports_output(self):
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        output = self.root / "01_augmentation_examples.png"
        cv2_module = sys.modules["cv2"]
        with patch.object(cv2_module, "imread", return_value=image), patch.object(
            demo,
            "build_panels",
            return_value=[("Original", image)],
        ) as build_panels, patch.object(
            demo,
            "render_figure",
            return_value=output,
        ):
            result, _ = demo.main(
                [
                    "--sample-index",
                    str(self.index_path),
                    "--asset-root",
                    str(self.root),
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(result, output)
        np.testing.assert_array_equal(build_panels.call_args.args[0], image[..., ::-1])


class GeometryAndConditionTests(RobustnessFixture):
    def test_condition_dispatch_is_controlled_and_original_is_copied(self):
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        augmenter = Mock()
        augmenter.gaussian_blur.return_value = image
        augmenter.vertical_flip.return_value = image
        augmenter.change_brightness.return_value = image
        for condition in robustness.CONDITIONS:
            result = robustness.apply_condition(image, condition, augmenter)
            self.assertEqual(result.shape, image.shape)
        self.assertIsNot(
            robustness.apply_condition(image, robustness._condition("original"), augmenter),
            image,
        )
        augmenter.gaussian_blur.assert_called_once_with(image=image, kernel_size=9, sigma=0)
        augmenter.vertical_flip.assert_called_once_with(image=image)
        self.assertEqual(augmenter.change_brightness.call_count, 2)

    def test_unknown_condition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported augmentation type"):
            robustness.apply_condition(
                np.zeros((2, 2, 3)),
                {"type": "rotation"},
                Mock(),
            )

    def test_yolo_conversion_updates_only_vertical_center_geometry(self):
        label = self.root / "label.txt"
        label.write_text("1 0.25 0.20 0.10 0.40\n", encoding="utf-8")
        original = robustness.parse_yolo_ground_truth(
            label,
            200,
            100,
            self.classes,
            robustness._condition("original"),
        )[0]
        flipped = robustness.parse_yolo_ground_truth(
            label,
            200,
            100,
            self.classes,
            robustness._condition("vertical_flip"),
        )[0]
        self.assertEqual(original["bbox_x"], flipped["bbox_x"])
        self.assertEqual(original["bbox_w"], flipped["bbox_w"])
        self.assertEqual(original["bbox_h"], flipped["bbox_h"])
        self.assertAlmostEqual(original["bbox_y"], 0.0)
        self.assertAlmostEqual(flipped["bbox_y"], 60.0)

    def test_label_parser_rejects_malformed_nonfinite_and_invalid_class_rows(self):
        cases = ("0 0.5 0.5 0.2", "0 nan 0.5 0.2 0.2", "2 0.5 0.5 0.2 0.2")
        for content in cases:
            with self.subTest(content=content):
                label = self.root / "bad.txt"
                label.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    robustness.parse_yolo_ground_truth(
                        label,
                        100,
                        100,
                        self.classes,
                        robustness._condition("original"),
                    )


class CacheValidationTests(RobustnessFixture):
    def test_sample_index_is_unique_and_supports_prefix_bound(self):
        loaded = robustness.load_sample_index(self.index_path, max_images=2)
        self.assertEqual(loaded["image_file"].tolist(), ["a.jpg", "b.jpg"])
        duplicate = pd.concat([self.index, self.index.iloc[[0]]], ignore_index=True)
        duplicate.to_csv(self.index_path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate image_file"):
            robustness.load_sample_index(self.index_path)

    def test_ground_truth_reconciles_condition_and_selected_counts(self):
        condition = robustness._condition("original")
        validated = robustness.validate_ground_truth(
            self.ground_truth(condition),
            self.index,
            self.classes,
            condition,
        )
        self.assertEqual(len(validated), 2)
        invalid = self.ground_truth(condition).iloc[:1]
        with self.assertRaisesRegex(ValueError, "counts"):
            robustness.validate_ground_truth(invalid, self.index, self.classes, condition)

    def test_ground_truth_rejects_wrong_condition(self):
        condition = robustness._condition("original")
        invalid = self.ground_truth(condition)
        invalid.loc[:, "augmentation_condition"] = "vertical_flip"
        with self.assertRaisesRegex(ValueError, "condition"):
            robustness.validate_ground_truth(invalid, self.index, self.classes, condition)

    def test_known_brightness_display_alias_is_normalized_but_other_drift_is_rejected(self):
        condition = robustness._condition("brightness_increase")
        legacy = self.ground_truth(condition)
        legacy.loc[:, "augmentation_display"] = "Brightness increase"
        validated = robustness.validate_ground_truth(
            legacy,
            self.index,
            self.classes,
            condition,
        )
        self.assertEqual(
            set(validated["augmentation_display"]),
            {condition["display"]},
        )
        legacy.loc[:, "augmentation_display"] = "Unrelated label"
        with self.assertRaisesRegex(ValueError, "display label"):
            robustness.validate_ground_truth(
                legacy,
                self.index,
                self.classes,
                condition,
            )

    def test_raw_cache_recomputes_class_and_confidence_fields(self):
        condition = robustness._condition("original")
        validated = robustness.validate_raw_predictions(
            self.raw(condition),
            self.index,
            self.classes,
            condition,
        )
        self.assertEqual(len(validated), 4)
        invalid = self.raw(condition)
        invalid.loc[0, "combined_confidence"] = 0.1
        with self.assertRaisesRegex(ValueError, "Combined confidence"):
            robustness.validate_raw_predictions(invalid, self.index, self.classes, condition)

    def test_raw_cache_rejects_predicted_class_that_is_not_argmax(self):
        condition = robustness._condition("original")
        invalid = self.raw(condition)
        invalid.loc[0, "class_scores_json"] = json.dumps([0.4, 0.9])
        invalid.loc[0, "predicted_class_score"] = 0.4
        invalid.loc[0, "combined_confidence"] = 0.95 * 0.4
        with self.assertRaisesRegex(ValueError, "maximum class score"):
            robustness.validate_raw_predictions(invalid, self.index, self.classes, condition)

    def test_class_mapping_can_be_recovered_from_consistent_caches(self):
        condition = robustness._condition("original")
        path = self.root / "ground_truth.csv"
        self.ground_truth(condition).to_csv(path, index=False)
        classes = robustness.load_classes(self.root / "missing.names", [path])
        self.assertEqual(classes, self.classes)

    def test_external_cache_paths_are_read_only_under_main(self):
        self.write_all_input_caches()
        before = {
            path.name: path.read_bytes()
            for path in self.cache_dir.iterdir()
            if path.is_file()
        }
        robustness.main(
            [
                "--sample-index",
                str(self.index_path),
                "--class-file",
                str(self.class_file),
                "--cache-input-dir",
                str(self.cache_dir),
                "--output-dir",
                str(self.output_dir),
                "--skip-figures",
            ]
        )
        after = {path.name: path.read_bytes() for path in self.cache_dir.iterdir() if path.is_file()}
        self.assertEqual(before, after)

    def test_force_is_rejected_with_external_cache_directory(self):
        with self.assertRaisesRegex(ValueError, "read-only"):
            robustness.main(
                [
                    "--sample-index",
                    str(self.index_path),
                    "--cache-input-dir",
                    str(self.cache_dir),
                    "--force",
                    "--skip-figures",
                ]
            )


class PostprocessingTests(RobustnessFixture):
    def test_fixed_nms_is_class_aware_and_uses_combined_confidence(self):
        condition = robustness._condition("original")
        predictions = robustness.apply_fixed_nms(
            self.raw(condition),
            self.index,
            self.classes,
            condition,
        )
        image_a = predictions[predictions["image_file"] == "a.jpg"]
        self.assertEqual(len(image_a), 2)
        self.assertEqual(set(image_a["class_id"]), {0, 1})
        self.assertTrue((predictions["nms_threshold"] == 0.3).all())

    def test_metric_inputs_preserve_empty_images_and_index_order(self):
        condition = robustness._condition("original")
        predictions = robustness.apply_fixed_nms(
            self.raw(condition),
            self.index,
            self.classes,
            condition,
        )
        metric_inputs = robustness.build_metric_lists(
            self.index,
            predictions,
            self.ground_truth(condition),
        )
        self.assertEqual(len(metric_inputs[0]), 3)
        self.assertEqual(metric_inputs[0][2], [])
        self.assertEqual(metric_inputs[4][2], [])

    def test_evaluator_returns_all_classes_and_complete_summary(self):
        condition = robustness._condition("original")
        predictions = robustness.apply_fixed_nms(
            self.raw(condition),
            self.index,
            self.classes,
            condition,
        )
        summary, per_class = robustness.evaluate_condition(
            self.index,
            predictions,
            self.ground_truth(condition),
            self.classes,
            condition,
        )
        self.assertEqual(len(per_class), len(self.classes))
        self.assertEqual(summary["total_ground_truth"], 2)
        self.assertEqual(summary["nms_iou_threshold"], 0.3)

    def test_baseline_change_fields_are_derived_from_original(self):
        summaries = []
        class_tables = []
        for position, condition in enumerate(robustness.CONDITIONS):
            summaries.append(
                {
                    "model": robustness.MODEL_NAME,
                    "dataset": robustness.DATASET_NAME,
                    "augmentation_condition": condition["tag"],
                    "augmentation_display": condition["display"],
                    "mAP@0.5_11_point": 0.8 - position * 0.1,
                    "total_ground_truth": 2,
                    "total_predictions_after_nms": 4 - min(position, 2),
                    "evaluation_rows": 4,
                    "candidate_objectness_threshold": 0.5,
                    "nms_confidence_threshold": 0.5,
                    "nms_iou_threshold": 0.3,
                    "map_iou_threshold": 0.5,
                    "eval_type": "combined",
                }
            )
            class_tables.append(
                pd.DataFrame(
                    [
                        {
                            "model": robustness.MODEL_NAME,
                            "dataset": robustness.DATASET_NAME,
                            "augmentation_condition": condition["tag"],
                            "augmentation_display": condition["display"],
                            "class_id": class_id,
                            "class_name": name,
                            "ground_truth_count": 1,
                            "prediction_count": 1,
                            "ap_11_point": 0.7 - position * 0.05,
                        }
                        for class_id, name in enumerate(self.classes)
                    ]
                )
            )
        summary, per_class = robustness.add_baseline_changes(
            pd.DataFrame(summaries),
            pd.concat(class_tables, ignore_index=True),
        )
        original = summary[summary["augmentation_condition"] == "original"].iloc[0]
        self.assertEqual(original["mAP_change_vs_original"], 0.0)
        changed = summary.iloc[1]
        self.assertAlmostEqual(changed["mAP_change_vs_original"], -0.1)
        self.assertFalse(per_class["original_ap_11_point"].isna().any())


class ArtifactTests(RobustnessFixture):
    def test_complete_cached_run_writes_predictions_and_two_derived_tables(self):
        self.write_all_input_caches()
        summary, per_class, paths, figures = robustness.main(
            [
                "--sample-index",
                str(self.index_path),
                "--class-file",
                str(self.class_file),
                "--cache-input-dir",
                str(self.cache_dir),
                "--output-dir",
                str(self.output_dir),
                "--skip-figures",
            ]
        )
        self.assertEqual(summary.columns.tolist(), robustness.SUMMARY_COLUMNS)
        self.assertEqual(per_class.columns.tolist(), robustness.PER_CLASS_COLUMNS)
        self.assertEqual(len(paths["predictions"]), len(robustness.CONDITIONS))
        self.assertEqual(figures, [])
        self.assertTrue(all(path.is_file() for path in paths["predictions"]))
        self.assertTrue(all(path.is_file() for path in paths["derived"].values()))

    def test_cached_prediction_is_reused_unless_refresh_is_requested(self):
        self.write_all_input_caches()
        arguments = [
            "--sample-index",
            str(self.index_path),
            "--class-file",
            str(self.class_file),
            "--cache-input-dir",
            str(self.cache_dir),
            "--output-dir",
            str(self.output_dir),
            "--skip-figures",
        ]
        robustness.main(arguments)
        with patch.object(robustness, "apply_fixed_nms", wraps=robustness.apply_fixed_nms) as apply:
            robustness.main(arguments)
            apply.assert_not_called()
            robustness.main([*arguments, "--refresh-postprocessing"])
            self.assertEqual(apply.call_count, len(robustness.CONDITIONS))

    def test_figure_builder_emits_four_canonical_names(self):
        summaries = []
        per_class_rows = []
        for position, condition in enumerate(robustness.CONDITIONS):
            summaries.append(
                {
                    **{column: 0 for column in robustness.SUMMARY_COLUMNS},
                    "model": robustness.MODEL_NAME,
                    "dataset": robustness.DATASET_NAME,
                    "augmentation_condition": condition["tag"],
                    "augmentation_display": condition["display"],
                    "mAP@0.5_11_point": 0.8 - position * 0.1,
                    "mAP_change_vs_original": -position * 0.1,
                    "total_predictions_after_nms": 10 - position,
                }
            )
            for class_id, name in enumerate(self.classes):
                per_class_rows.append(
                    {
                        **{column: 0 for column in robustness.PER_CLASS_COLUMNS},
                        "model": robustness.MODEL_NAME,
                        "dataset": robustness.DATASET_NAME,
                        "augmentation_condition": condition["tag"],
                        "augmentation_display": condition["display"],
                        "class_id": class_id,
                        "class_name": name,
                        "ap_change_vs_original": -position * 0.05,
                    }
                )
        with patch.dict(sys.modules, fake_pyplot()):
            paths = robustness.build_figures(
                pd.DataFrame(summaries),
                pd.DataFrame(per_class_rows),
                self.figure_dir,
            )
        self.assertEqual(
            [path.name for path in paths],
            [
                "02_map_by_condition.png",
                "03_map_drop_vs_baseline.png",
                "04_prediction_count_by_condition.png",
                "05_largest_per_class_ap_drops.png",
            ],
        )
        self.assertTrue(all(path.read_bytes() == b"figure" for path in paths))

    def test_figures_only_uses_existing_derived_artifacts(self):
        summary = pd.DataFrame(columns=robustness.SUMMARY_COLUMNS)
        per_class = pd.DataFrame(columns=robustness.PER_CLASS_COLUMNS)
        self.output_dir.mkdir()
        summary.to_csv(self.output_dir / "summary_by_condition_sample5000.csv", index=False)
        per_class.to_csv(
            self.output_dir / "per_class_ap_by_condition_sample5000.csv",
            index=False,
        )
        expected = [self.figure_dir / "figure.png"]
        with patch.object(robustness, "build_figures", return_value=expected) as build:
            result = robustness.main(
                [
                    "--output-dir",
                    str(self.output_dir),
                    "--figure-dir",
                    str(self.figure_dir),
                    "--figures-only",
                ]
            )
        self.assertEqual(result[3], expected)
        build.assert_called_once()


class ReferenceCompatibilityTests(RobustnessFixture):
    def test_valid_geometry_and_evaluation_match_reference(self):
        reference_path = os.environ.get("REFERENCE_AUGMENTATION_ROBUSTNESS_SCRIPT")
        if not reference_path:
            self.skipTest("REFERENCE_AUGMENTATION_ROBUSTNESS_SCRIPT is not configured")
        reference = load_module(Path(reference_path), "reference_augmentation_robustness")
        condition = robustness._condition("vertical_flip")
        label = self.root / "label.txt"
        label.write_text("0 0.25 0.20 0.10 0.40\n", encoding="utf-8")
        ours = pd.DataFrame(
            robustness.parse_yolo_ground_truth(
                label,
                200,
                100,
                self.classes,
                condition,
            )
        )
        theirs = pd.DataFrame(
            reference.yolo_label_to_xywh_for_condition(
                label,
                200,
                100,
                self.classes,
                condition,
            )
        )
        assert_frame_equal(ours, theirs, check_dtype=False, atol=1e-12, rtol=1e-12)

        predictions = robustness.apply_fixed_nms(
            self.raw(condition),
            self.index,
            self.classes,
            condition,
        )
        our_summary, our_per_class = robustness.evaluate_condition(
            self.index,
            predictions,
            self.ground_truth(condition),
            self.classes,
            condition,
        )
        their_summary, their_per_class = reference.evaluate_with_metrics_py(
            self.index,
            predictions,
            self.ground_truth(condition),
            self.classes,
            condition,
        )
        self.assertAlmostEqual(
            our_summary["mAP@0.5_11_point"],
            their_summary["mAP@0.5_11_point"],
        )
        assert_frame_equal(
            our_per_class.reset_index(drop=True),
            their_per_class.reset_index(drop=True),
            check_dtype=False,
            atol=1e-12,
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
