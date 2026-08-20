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


try:
    import cv2
except ModuleNotFoundError:
    cv2 = types.ModuleType("cv2")
    sys.modules["cv2"] = cv2

cv2.dnn = getattr(cv2, "dnn", types.SimpleNamespace())
cv2.imread = getattr(cv2, "imread", Mock(name="imread"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCRIPT = PROJECT_ROOT / "experiments" / "scripts" / "01_model_comparison.py"
SCRIPT_PATH = Path(os.environ.get("MODEL_COMPARISON_SCRIPT", DEFAULT_SCRIPT))

spec = importlib.util.spec_from_file_location("model_comparison_under_test", SCRIPT_PATH)
model_comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_comparison)


class ModelComparisonContractTests(unittest.TestCase):
    def test_fixed_operating_policy_and_model_assets_are_explicit(self):
        self.assertEqual(model_comparison.DETECTOR_OBJECTNESS_THRESHOLD, 0.5)
        self.assertEqual(model_comparison.NMS_CONFIDENCE_THRESHOLD, 0.5)
        self.assertEqual(model_comparison.NMS_THRESHOLD, 0.3)
        self.assertEqual(model_comparison.MAP_IOU_THRESHOLD, 0.5)
        self.assertEqual(model_comparison.EVAL_TYPE, "combined")
        self.assertEqual(list(model_comparison.MODELS), ["model1", "model2"])
        for paths in model_comparison.MODELS.values():
            self.assertEqual(set(paths), {"weights", "cfg", "names"})
            self.assertFalse(any(path.is_absolute() for path in paths.values()))

    def test_evidence_table_schemas_are_stable(self):
        self.assertEqual(
            model_comparison.PRED_COLUMNS,
            model_comparison.RAW_COLUMNS + ["nms_threshold"],
        )
        self.assertEqual(
            model_comparison.GT_COLUMNS,
            [
                "image_file",
                "image_path",
                "class_id",
                "class_name",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
            ],
        )
        for column in (
            "object_score",
            "predicted_class_score",
            "combined_confidence",
            "class_scores_json",
        ):
            self.assertIn(column, model_comparison.RAW_COLUMNS)

    def test_analysis_requirements_include_runtime_and_reporting_stack(self):
        requirements = model_comparison.PROJECT_ROOT / "requirements-analysis.txt"
        active_lines = [
            line.strip()
            for line in requirements.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

        self.assertEqual(
            active_lines,
            [
                "-r detector_service/requirements.txt",
                "pandas",
                "matplotlib",
                "seaborn",
                "scikit-learn",
                "scipy",
                "pyarrow",
            ],
        )


class ClassAndLabelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.root = Path(self.temp_directory.name)
        self.cache_dir = self.root / "cache"
        self.cache_dir.mkdir()

        self.cache_patch = patch.object(
            model_comparison,
            "MODEL_SELECTION_DIR",
            self.cache_dir,
        )
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)

    def _class_file(self):
        path = self.root / model_comparison.MODELS["model1"]["names"]
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def test_class_file_is_preferred_and_blank_lines_are_ignored(self):
        self._class_file().write_text(
            "pallet\n\n forklift \n",
            encoding="utf-8",
        )
        pd.DataFrame(
            {"class_id": [0], "class_name": ["cached-name"]}
        ).to_csv(self.cache_dir / "ground_truth_sample.csv", index=False)

        classes = model_comparison.load_classes(self.root, "sample")

        self.assertEqual(classes, ["pallet", "forklift"])

    def test_contiguous_class_mapping_can_be_recovered_from_caches(self):
        pd.DataFrame(
            {
                "class_id": [0, 1, 1],
                "class_name": ["pallet", "forklift", "forklift"],
            }
        ).to_csv(self.cache_dir / "ground_truth_first_2.csv", index=False)

        classes = model_comparison.load_classes(self.root, "first_2")

        self.assertEqual(classes, ["pallet", "forklift"])

    def test_conflicting_cached_names_are_rejected(self):
        pd.DataFrame(
            {"class_id": [0], "class_name": ["pallet"]}
        ).to_csv(self.cache_dir / "ground_truth_full.csv", index=False)
        pd.DataFrame(
            {"class_id": [0], "class_name": ["crate"]}
        ).to_csv(
            self.cache_dir / "model1_raw_predictions_full.csv",
            index=False,
        )

        with self.assertRaisesRegex(ValueError, "Conflicting names for class 0"):
            model_comparison.load_classes(self.root, "full")

    def test_incomplete_cached_class_ids_are_rejected(self):
        pd.DataFrame(
            {"class_id": [1], "class_name": ["forklift"]}
        ).to_csv(self.cache_dir / "ground_truth_full.csv", index=False)

        with self.assertRaisesRegex(ValueError, "contiguous class IDs"):
            model_comparison.load_classes(self.root, "full")

    def test_missing_asset_and_cache_mapping_is_reported(self):
        with self.assertRaisesRegex(FileNotFoundError, "No raw cache"):
            model_comparison.load_classes(self.root, "full")

    def test_yolo_labels_are_converted_to_pixel_xywh(self):
        label_path = self.root / "image.txt"
        label_path.write_text(
            "1 0.5 0.25 0.2 0.5\n"
            "99 0.5 0.5 0.1 0.1\n"
            "short row\n",
            encoding="utf-8",
        )

        rows = model_comparison.yolo_label_to_xywh(
            label_path,
            image_w=200,
            image_h=100,
            classes=["pallet", "forklift"],
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["class_id"], 1)
        self.assertEqual(rows[0]["class_name"], "forklift")
        self.assertEqual(
            [rows[0][field] for field in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")],
            [80.0, 0.0, 40.0, 50.0],
        )

    def test_empty_label_file_produces_no_rows(self):
        label_path = self.root / "empty.txt"
        label_path.write_text("  \n", encoding="utf-8")

        self.assertEqual(
            model_comparison.yolo_label_to_xywh(label_path, 20, 10, ["item"]),
            [],
        )


class EvidencePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.root = Path(self.temp_directory.name)
        self.cache_dir = self.root / "outputs"
        self.cache_dir.mkdir()
        self.cache_patch = patch.object(
            model_comparison,
            "MODEL_SELECTION_DIR",
            self.cache_dir,
        )
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)

        self.index = pd.DataFrame(
            [
                {
                    "image_file": "a.jpg",
                    "image_path": "images/a.jpg",
                    "label_path": "labels/a.txt",
                },
                {
                    "image_file": "b.jpg",
                    "image_path": "images/b.jpg",
                    "label_path": "labels/b.txt",
                },
            ]
        )

    def test_ground_truth_cache_is_reused_without_decoding_images(self):
        expected = pd.DataFrame(
            [
                {
                    "image_file": "cached.jpg",
                    "image_path": "images/cached.jpg",
                    "class_id": 0,
                    "class_name": "pallet",
                    "bbox_x": 1.0,
                    "bbox_y": 2.0,
                    "bbox_w": 3.0,
                    "bbox_h": 4.0,
                }
            ]
        )
        expected.to_csv(self.cache_dir / "ground_truth_sample.csv", index=False)

        with patch.object(cv2, "imread") as imread:
            result = model_comparison.build_ground_truth(
                self.index,
                ["pallet"],
                self.root,
                "sample",
            )

        imread.assert_not_called()
        pd.testing.assert_frame_equal(result, expected)

    def test_ground_truth_builder_uses_image_dimensions_and_skips_bad_images(self):
        for label_name, contents in (
            ("a.txt", "0 0.5 0.5 0.5 0.5\n"),
            ("b.txt", "0 0.5 0.5 0.5 0.5\n"),
        ):
            label_path = self.root / "labels" / label_name
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(contents, encoding="utf-8")

        with patch.object(
            cv2,
            "imread",
            side_effect=[np.zeros((100, 200, 3), dtype=np.uint8), None],
        ):
            result = model_comparison.build_ground_truth(
                self.index,
                ["pallet"],
                self.root,
                "sample",
                force=True,
            )

        self.assertEqual(result.columns.tolist(), model_comparison.GT_COLUMNS)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result.loc[0, ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].tolist(),
            [50.0, 25.0, 100.0, 50.0],
        )
        self.assertTrue((self.cache_dir / "ground_truth_sample.csv").exists())

    def test_detection_serialization_keeps_full_score_vector(self):
        image = pd.Series(
            {"image_file": "a.jpg", "image_path": "images/a.jpg"}
        )

        record = model_comparison.serialize_detection(
            "model2",
            image,
            [1, 2, 3, 4],
            class_id=1,
            object_score=0.8,
            score_vector=np.asarray([0.1, 0.9]),
            classes=["pallet", "forklift"],
        )

        self.assertEqual(record["class_name"], "forklift")
        self.assertEqual(record["predicted_class_score"], 0.9)
        self.assertAlmostEqual(record["combined_confidence"], 0.72)
        self.assertEqual(json.loads(record["class_scores_json"]), [0.1, 0.9])

    def test_unknown_class_serializes_with_zero_selected_probability(self):
        image = {"image_file": "a.jpg", "image_path": "images/a.jpg"}

        record = model_comparison.serialize_detection(
            "model1",
            image,
            [1, 2, 3, 4],
            class_id=5,
            object_score=0.8,
            score_vector=[0.1, 0.9],
            classes=["pallet", "forklift"],
        )

        self.assertEqual(record["class_name"], "unknown")
        self.assertEqual(record["predicted_class_score"], 0.0)
        self.assertEqual(record["combined_confidence"], 0.0)

    def test_raw_inference_reports_all_missing_model_assets(self):
        paths = {
            "weights": Path("missing.weights"),
            "cfg": Path("missing.cfg"),
            "names": Path("missing.names"),
        }

        with self.assertRaisesRegex(FileNotFoundError, "missing.weights") as error:
            model_comparison.run_raw_inference_for_model(
                "model1",
                paths,
                self.index.iloc[:0],
                ["pallet"],
                self.root,
                "sample",
            )

        self.assertIn("missing.cfg", str(error.exception))
        self.assertIn("missing.names", str(error.exception))

    def test_raw_inference_uses_detector_and_skips_unreadable_images(self):
        paths = {
            "weights": Path("model/model.weights"),
            "cfg": Path("model/model.cfg"),
            "names": Path("model/classes.names"),
        }
        for relative_path in paths.values():
            asset = self.root / relative_path
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.touch()

        detector = Mock()
        detector.predict.return_value = ["raw-output"]
        detector.post_process.return_value = (
            [[1, 2, 3, 4]],
            [1],
            [0.8],
            [[0.1, 0.9]],
        )

        from detector_service.modules.inference import model as detector_module

        with patch.object(
            detector_module,
            "Detector",
            return_value=detector,
        ) as constructor, patch.object(
            cv2,
            "imread",
            side_effect=[np.zeros((10, 20, 3), dtype=np.uint8), None],
        ):
            result = model_comparison.run_raw_inference_for_model(
                "model2",
                paths,
                self.index,
                ["pallet", "forklift"],
                self.root,
                "sample",
                force=True,
            )

        constructor.assert_called_once_with(
            str(self.root / paths["weights"]),
            str(self.root / paths["cfg"]),
            str(self.root / paths["names"]),
            score_threshold=0.5,
        )
        detector.predict.assert_called_once()
        detector.post_process.assert_called_once_with(["raw-output"])
        self.assertEqual(result.columns.tolist(), model_comparison.RAW_COLUMNS)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.loc[0, "combined_confidence"], 0.72)
        self.assertTrue(
            (self.cache_dir / "model2_raw_predictions_sample.csv").exists()
        )

    def test_raw_cache_is_checked_before_model_assets(self):
        cached = pd.DataFrame(
            [{column: 0 for column in model_comparison.RAW_COLUMNS}]
        )
        cached["model"] = "model1"
        cached["image_file"] = "a.jpg"
        cached["image_path"] = "images/a.jpg"
        cached["class_name"] = "pallet"
        cached["class_scores_json"] = "[1.0]"
        cached.to_csv(
            self.cache_dir / "model1_raw_predictions_sample.csv",
            index=False,
        )

        result = model_comparison.run_raw_inference_for_model(
            "model1",
            {"weights": Path("absent.weights")},
            self.index,
            ["pallet"],
            self.root,
            "sample",
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "image_file"], "a.jpg")

    def test_class_aware_nms_is_applied_per_image(self):
        image = {"image_file": "a.jpg", "image_path": "images/a.jpg"}
        raw_records = [
            model_comparison.serialize_detection(
                "model2", image, [0, 0, 10, 10], 0, 0.9, [0.9, 0.1], ["a", "b"]
            ),
            model_comparison.serialize_detection(
                "model2", image, [0, 0, 10, 10], 0, 0.8, [0.9, 0.1], ["a", "b"]
            ),
            model_comparison.serialize_detection(
                "model2", image, [0, 0, 10, 10], 1, 0.85, [0.1, 0.9], ["a", "b"]
            ),
        ]
        raw = pd.DataFrame(raw_records, columns=model_comparison.RAW_COLUMNS)

        result = model_comparison.apply_nms_for_model(
            "model2",
            raw,
            self.index.iloc[[0]],
            ["a", "b"],
            "sample",
            force=True,
        )

        self.assertEqual(result.columns.tolist(), model_comparison.PRED_COLUMNS)
        self.assertEqual(result["class_id"].tolist(), [0, 1])
        np.testing.assert_allclose(result["object_score"], [0.9, 0.85])
        np.testing.assert_allclose(result["nms_threshold"], [0.3, 0.3])

    def test_empty_raw_table_writes_empty_post_nms_schema(self):
        result = model_comparison.apply_nms_for_model(
            "model1",
            pd.DataFrame(columns=model_comparison.RAW_COLUMNS),
            self.index,
            ["pallet"],
            "empty",
            force=True,
        )

        self.assertEqual(len(result), 0)
        self.assertEqual(result.columns.tolist(), model_comparison.PRED_COLUMNS)


class EvaluationAssemblyTests(unittest.TestCase):
    @staticmethod
    def _index():
        return pd.DataFrame(
            {
                "image_file": ["a.jpg", "b.jpg", "c.jpg"],
                "image_path": ["a.jpg", "b.jpg", "c.jpg"],
            }
        )

    def test_metric_lists_follow_index_order_and_represent_empty_images(self):
        predictions = pd.DataFrame(
            [
                {
                    "image_file": "b.jpg",
                    "bbox_x": 1,
                    "bbox_y": 2,
                    "bbox_w": 3,
                    "bbox_h": 4,
                    "class_id": 1,
                    "object_score": 0.8,
                    "class_scores_json": "[0.1, 0.9]",
                }
            ]
        )
        labels = pd.DataFrame(
            [
                {
                    "image_file": "c.jpg",
                    "bbox_x": 5,
                    "bbox_y": 6,
                    "bbox_w": 7,
                    "bbox_h": 8,
                    "class_id": 0,
                }
            ]
        )

        result = model_comparison.build_metric_lists(
            self._index(),
            predictions,
            labels,
        )

        self.assertEqual(result[0], [[], [[1, 2, 3, 4]], []])
        self.assertEqual(result[1], [[], [1], []])
        self.assertEqual(result[2], [[], [0.8], []])
        self.assertEqual(result[3], [[], [[0.1, 0.9]], []])
        self.assertEqual(result[4], [[], [], [[5, 6, 7, 8]]])
        self.assertEqual(result[5], [[], [], [0]])

    def test_perfect_detection_produces_per_class_and_mean_ap(self):
        idx = self._index().iloc[[0]].copy()
        prediction = pd.DataFrame(
            [
                {
                    "image_file": "a.jpg",
                    "bbox_x": 0,
                    "bbox_y": 0,
                    "bbox_w": 10,
                    "bbox_h": 10,
                    "class_id": 0,
                    "object_score": 1.0,
                    "class_scores_json": "[1.0, 0.0]",
                }
            ]
        )
        ground_truth = pd.DataFrame(
            [
                {
                    "image_file": "a.jpg",
                    "bbox_x": 0,
                    "bbox_y": 0,
                    "bbox_w": 10,
                    "bbox_h": 10,
                    "class_id": 0,
                }
            ]
        )

        summary, per_class = model_comparison.evaluate_with_metrics_py(
            "model1",
            idx,
            prediction,
            ground_truth,
            ["pallet", "forklift"],
        )

        self.assertAlmostEqual(summary["mAP@0.5_11_point"], 0.5)
        self.assertEqual(summary["total_ground_truth"], 1)
        self.assertEqual(summary["total_predictions_after_nms"], 1)
        self.assertEqual(summary["evaluation_rows"], 1)
        self.assertEqual(summary["eval_type"], "combined")
        np.testing.assert_allclose(per_class["ap_11_point"], [1.0, 0.0])
        self.assertEqual(per_class["ground_truth_count"].tolist(), [1, 0])
        self.assertEqual(per_class["prediction_count"].tolist(), [1, 0])

    def test_figure_builder_writes_comparison_and_delta_artifacts(self):
        comparison = pd.DataFrame(
            {
                "class_name": ["pallet", "forklift"],
                "model1_ap": [0.2, 0.6],
                "model2_ap": [0.4, 0.5],
                "ap_difference_model2_minus_model1": [0.2, -0.1],
            }
        )
        pyplot = types.ModuleType("matplotlib.pyplot")
        for method_name in (
            "figure",
            "barh",
            "yticks",
            "xlim",
            "xlabel",
            "ylabel",
            "title",
            "grid",
            "legend",
            "tight_layout",
            "savefig",
            "close",
            "axvline",
        ):
            setattr(pyplot, method_name, Mock(name=method_name))
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.__path__ = []
        matplotlib.pyplot = pyplot

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            model_comparison,
            "FIGURE_DIR",
            Path(temp_dir),
        ), patch.dict(
            sys.modules,
            {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot},
        ):
            model_comparison.build_figures(comparison)

        saved_paths = [call.args[0].name for call in pyplot.savefig.call_args_list]
        self.assertEqual(
            saved_paths,
            ["01_per_class_ap.png", "02_per_class_ap_delta.png"],
        )
        self.assertEqual(pyplot.close.call_count, 2)


if __name__ == "__main__":
    unittest.main()
