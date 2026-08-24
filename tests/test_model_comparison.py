import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    os.environ.get(
        "MODEL_COMPARISON_SCRIPT",
        PROJECT_ROOT / "experiments" / "scripts" / "01_model_comparison.py",
    )
)
spec = importlib.util.spec_from_file_location("model_comparison_under_test", SCRIPT_PATH)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class FakeDetector:
    def __init__(self, weights, config, classes, score_threshold):
        self.score_threshold = score_threshold

    def predict(self, frame):
        return [frame]

    def post_process(self, outputs):
        return [[ [0, 0, 10, 10] ][0]], [0], [1.0], [[1.0, 0.0]]


class EmptyDetector(FakeDetector):
    def post_process(self, outputs):
        return [], [], [], []


class WrongVectorDetector(FakeDetector):
    def post_process(self, outputs):
        return [[0, 0, 10, 10]], [0], [1.0], [[1.0]]


class ContractTests(unittest.TestCase):
    def test_default_paths_follow_the_canonical_stage_hierarchy(self):
        self.assertEqual(
            comparison.DEFAULT_DATASET_INDEX,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "00_dataset_inventory"
            / "dataset_index.csv",
        )
        self.assertEqual(
            comparison.DEFAULT_OUTPUT_ROOT,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "01_model_selection"
            / "01_quality_comparison",
        )

    def test_policy_defaults_and_storage_relative_bundles_are_explicit(self):
        self.assertEqual(comparison.DEFAULT_CANDIDATE_FLOOR, 0.001)
        self.assertEqual(comparison.DEFAULT_DEPLOYMENT_CONFIDENCE, 0.50)
        self.assertEqual(comparison.DEFAULT_NMS_IOU, 0.30)
        self.assertEqual(comparison.PRIMARY_AP_POINTS, 101)
        self.assertEqual(comparison.LEGACY_AP_POINTS, 11)
        self.assertEqual(list(comparison.MODELS), ["model1", "model2"])
        for bundle in comparison.MODELS.values():
            self.assertEqual(set(bundle), {"weights", "cfg", "names"})
            self.assertTrue(all(not path.is_absolute() for path in bundle.values()))
            self.assertTrue(all(path.parts[:2] == ("detector_service", "storage") for path in bundle.values()))

    def test_output_schemas_include_low_floor_deployment_and_ledger_evidence(self):
        self.assertIn("mAP50_101pt", comparison.AGGREGATE_COLUMNS)
        self.assertIn("threshold_constrained_mAP50_11pt", comparison.AGGREGATE_COLUMNS)
        self.assertIn("deployment_macro_f1", comparison.AGGREGATE_COLUMNS)
        self.assertIn("combined_confidence", comparison.PREDICTION_COLUMNS)
        self.assertIn("raw_candidate_count", comparison.LEDGER_COLUMNS)
        self.assertIn("post_nms_prediction_count", comparison.LEDGER_COLUMNS)

    def test_cli_requires_asset_root_and_run_id(self):
        parser = comparison.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(["--asset-root", "assets", "--run-id", "pilot-100"])
        self.assertEqual(args.candidate_floor, 0.001)
        self.assertEqual(args.deployment_confidence, 0.50)
        self.assertEqual(args.nms_iou, 0.30)

    def test_full_and_pilot_scopes_are_explicit(self):
        parser = comparison.build_parser()
        base = ["--asset-root", "assets", "--run-id", "comparison"]
        with self.assertRaisesRegex(ValueError, "full mode requires"):
            comparison.validate_run_scope(parser.parse_args(base))
        with self.assertRaisesRegex(ValueError, "only with --pilot"):
            comparison.validate_run_scope(
                parser.parse_args(base + ["--max-images", "100"])
            )
        with self.assertRaisesRegex(ValueError, "requires --max-images"):
            comparison.validate_run_scope(parser.parse_args(base + ["--pilot"]))
        full = parser.parse_args(
            base + ["--expected-images", "9525", "--expected-labels", "36721"]
        )
        pilot = parser.parse_args(base + ["--pilot", "--max-images", "100"])
        self.assertEqual(comparison.validate_run_scope(full), "full")
        self.assertEqual(comparison.validate_run_scope(pilot), "pilot")

    def test_run_id_and_numeric_validators_reject_invalid_values(self):
        for value in ("../escape", ".hidden", "white space"):
            with self.assertRaises(argparse.ArgumentTypeError):
                comparison.validate_run_id(value)
        with self.assertRaises(argparse.ArgumentTypeError):
            comparison.positive_int(0)
        with self.assertRaises(argparse.ArgumentTypeError):
            comparison.probability(1.1)


class IndexAndPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _write_index(self, rows):
        path = self.root / "index.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _long_asset_pair(self):
        directory = self.root / "logistics"
        directory.mkdir(exist_ok=True)
        stem = "x" * 251
        image = directory / f"{stem}.jpg"
        label = directory / f"{stem}.txt"
        image_filesystem = comparison._filesystem_path(image)
        label_filesystem = comparison._filesystem_path(label)
        image_filesystem.write_bytes(b"encoded-long-path-image")
        label_filesystem.write_text(
            "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )
        self.addCleanup(image_filesystem.unlink, missing_ok=True)
        self.addCleanup(label_filesystem.unlink, missing_ok=True)
        return image, label

    @staticmethod
    def _row(name="a", count=1):
        return {
            "image_file": f"{name}.jpg",
            "image_path": f"detector_service/storage/logistics/{name}.jpg",
            "label_path": f"detector_service/storage/logistics/{name}.txt",
            "num_objects": count,
        }

    def test_canonical_and_relative_paths_resolve_through_storage_root(self):
        expected = self.root / "logistics" / "a.jpg"
        for logical in (
            "detector_service/storage/logistics/a.jpg",
            "logistics/a.jpg",
        ):
            self.assertEqual(comparison.resolve_indexed_path(logical, self.root), expected)

    def test_index_enforces_counts_expectations_and_unique_identifiers(self):
        path = self._write_index([self._row()])
        _, index = comparison.load_and_validate_index(
            path, expected_images=1, expected_labels=1
        )
        self.assertEqual(index["num_objects"].tolist(), [1])
        with self.assertRaisesRegex(ValueError, "Expected 2 images"):
            comparison.load_and_validate_index(path, expected_images=2)
        with self.assertRaisesRegex(ValueError, "Expected 2 labels"):
            comparison.load_and_validate_index(path, expected_labels=2)

        duplicate = self._write_index([self._row(), self._row()])
        with self.assertRaisesRegex(ValueError, "duplicate image_file"):
            comparison.load_and_validate_index(duplicate)

    def test_index_rejects_missing_columns_and_fractional_counts(self):
        path = self._write_index([{"image_file": "a.jpg"}])
        with self.assertRaisesRegex(ValueError, "missing columns"):
            comparison.load_and_validate_index(path)
        path = self._write_index([self._row(count=1.5)])
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            comparison.load_and_validate_index(path)

    def test_parent_traversal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "traverse"):
            comparison.resolve_indexed_path(
                "detector_service/storage/../secret", self.root
            )

    def test_absolute_paths_must_remain_inside_asset_root(self):
        inside = self.root / "inside.jpg"
        outside = self.root.parent / "outside.jpg"
        self.assertEqual(
            comparison.resolve_indexed_path(inside, self.root), inside.resolve()
        )
        with self.assertRaisesRegex(ValueError, "inside external storage"):
            comparison.resolve_indexed_path(outside, self.root)

    def test_255_character_assets_resolve_hash_and_parse_canonically(self):
        image, label = self._long_asset_pair()
        logical_image = f"detector_service/storage/logistics/{image.name}"
        logical_label = f"detector_service/storage/logistics/{label.name}"
        resolved = comparison.resolve_indexed_path(logical_image, self.root)
        self.assertEqual(resolved, image.absolute())
        self.assertNotIn("\\\\?\\", str(resolved))
        self.assertEqual(
            comparison.resolve_indexed_path(
                comparison._filesystem_path(image), self.root
            ),
            image.absolute(),
        )
        identity = comparison._file_identity(resolved)
        self.assertEqual(identity["path"], str(image.absolute()))
        self.assertEqual(identity["size_bytes"], len(b"encoded-long-path-image"))
        self.assertEqual(
            identity["sha256"], hashlib.sha256(b"encoded-long-path-image").hexdigest()
        )
        rows = comparison.parse_yolo_labels_strict(
            comparison.resolve_indexed_path(logical_label, self.root),
            20, 10, ["pallet"],
        )
        self.assertEqual(len(rows), 1)

    @unittest.skipUnless(os.name == "nt", "Windows extended-path decode test")
    def test_long_windows_image_uses_byte_decode(self):
        image, _ = self._long_asset_pair()

        class FakeCV2:
            IMREAD_COLOR = 1

            def __init__(self):
                self.encoded = None

            def imread(self, path):
                raise AssertionError(f"cv2.imread received long path: {path}")

            def imdecode(self, encoded, mode):
                self.encoded = bytes(encoded)
                self.mode = mode
                return np.zeros((3, 4, 3), dtype=np.uint8)

        fake = FakeCV2()
        frame = comparison._read_image_cv2(image, cv2_module=fake)
        self.assertEqual(frame.shape, (3, 4, 3))
        self.assertEqual(fake.encoded, b"encoded-long-path-image")
        self.assertEqual(fake.mode, fake.IMREAD_COLOR)


class VocabularyAndLabelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _assets(self, second_names="pallet\nperson\n"):
        for number, names in ((1, "pallet\nperson\n"), (2, second_names)):
            directory = self.root / f"yolo_model_{number}"
            directory.mkdir()
            (directory / f"yolov4-tiny-logistics_size_416_{number}.weights").touch()
            (directory / f"yolov4-tiny-logistics_size_416_{number}.cfg").write_text(
                "[yolo]\nclasses=2\n", encoding="utf-8"
            )
            (directory / "logistics.names").write_text(names, encoding="utf-8")

    def test_identical_ordered_vocabularies_are_required(self):
        self._assets()
        _, bundles, classes = comparison.resolve_and_validate_model_bundles(self.root)
        self.assertEqual(classes, ["pallet", "person"])
        self.assertEqual(set(bundles), {"model1", "model2"})

    def test_vocabulary_order_mismatch_is_rejected(self):
        self._assets("person\npallet\n")
        with self.assertRaisesRegex(ValueError, "vocabularies differ"):
            comparison.resolve_and_validate_model_bundles(self.root)

    def test_cfg_class_count_must_match_vocabulary(self):
        self._assets()
        cfg = (
            self.root / "yolo_model_2" /
            "yolov4-tiny-logistics_size_416_2.cfg"
        )
        cfg.write_text("[yolo]\nclasses=3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "class count"):
            comparison.resolve_and_validate_model_bundles(self.root)

    def test_strict_yolo_parser_converts_valid_rows(self):
        path = self.root / "a.txt"
        path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        rows = comparison.parse_yolo_labels_strict(path, 200, 100, ["pallet"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            [rows[0][field] for field in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")],
            [50.0, 25.0, 100.0, 50.0],
        )

    def test_strict_yolo_parser_rejects_extra_malformed_and_invalid_rows(self):
        cases = (
            ("0 0.5 0.5 0.5 0.5 extra\n", "exactly five"),
            ("zero 0.5 0.5 0.5 0.5\n", "numeric"),
            ("0 nan 0.5 0.5 0.5\n", "finite"),
            ("2 0.5 0.5 0.5 0.5\n", "class identifier"),
            ("0 0.5 0.5 0 0.5\n", "normalized YOLO box"),
        )
        for position, (contents, message) in enumerate(cases):
            path = self.root / f"bad-{position}.txt"
            path.write_text(contents, encoding="utf-8")
            with self.subTest(contents=contents), self.assertRaisesRegex(ValueError, message):
                comparison.parse_yolo_labels_strict(path, 10, 10, ["pallet"])


class EvidenceAndMetricTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "logistics").mkdir()
        (self.root / "logistics" / "a.jpg").touch()
        (self.root / "logistics" / "a.txt").write_text(
            "0 0.25 0.25 0.5 0.5\n", encoding="utf-8"
        )
        self.index = pd.DataFrame([{
            "image_file": "a.jpg",
            "image_path": "detector_service/storage/logistics/a.jpg",
            "label_path": "detector_service/storage/logistics/a.txt",
            "num_objects": 1,
        }])
        self.frame = np.zeros((20, 20, 3), dtype=np.uint8)

    def _prediction_row(self, **overrides):
        row = {
            "model": "model1", "image_index": 1, "image_file": "a.jpg",
            "image_path": self.index.loc[0, "image_path"], "bbox_x": 0,
            "bbox_y": 0, "bbox_w": 10, "bbox_h": 10, "class_id": 0,
            "class_name": "pallet", "object_score": 1.0,
            "predicted_class_score": 1.0, "combined_confidence": 1.0,
            "nms_iou_threshold": 0.3,
        }
        row.update(overrides)
        return row

    def _ledger_row(self, **overrides):
        row = {
            "model": "model1", "image_index": 1, "image_file": "a.jpg",
            "image_path": self.index.loc[0, "image_path"], "status": "processed",
            "raw_candidate_count": 1, "post_nms_prediction_count": 1,
            "read_seconds": 0.1, "predict_seconds": 0.2,
            "postprocess_seconds": 0.3, "nms_seconds": 0.4,
            "total_seconds": 1.0,
        }
        row.update(overrides)
        return row

    def test_ground_truth_fails_on_unreadable_image_and_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Unable to read"):
            comparison.build_ground_truth(
                self.index, ["pallet"], self.root, image_reader=lambda _: None
            )
        bad = self.index.copy()
        bad["num_objects"] = 2
        with self.assertRaisesRegex(ValueError, "Label count mismatch"):
            comparison.build_ground_truth(
                bad, ["pallet"], self.root, image_reader=lambda _: self.frame
            )

    def test_zero_prediction_image_is_preserved_in_ledger(self):
        bundle = {name: self.root / name for name in ("weights", "cfg", "names")}
        for path in bundle.values():
            path.touch()
        prediction_path = self.root / "predictions.csv"
        ledger_path = self.root / "ledger.csv"
        with comparison._atomic_csv_stream(ledger_path, comparison.LEDGER_COLUMNS) as writer:
            result = comparison.run_inference_for_model(
                "model1", bundle, self.index, ["pallet", "person"], self.root,
                0.001, 0.3, prediction_path, writer,
                image_reader=lambda _: self.frame, detector_factory=EmptyDetector,
            )
        predictions = pd.read_csv(prediction_path)
        ledger = pd.read_csv(ledger_path)
        self.assertTrue(predictions.empty)
        self.assertEqual(result["post_nms_predictions"], 0)
        self.assertEqual(ledger["post_nms_prediction_count"].tolist(), [0])
        comparison.validate_ledger(ledger, self.index, ["model1"])

    def test_incomplete_ledger_is_rejected(self):
        empty = pd.DataFrame(columns=comparison.LEDGER_COLUMNS)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            comparison.validate_ledger(empty, self.index, ["model1"])

    def test_prediction_image_keys_and_per_image_counts_are_strict(self):
        policy = {
            "candidate_floor": 0.001, "deployment_confidence": 0.5,
            "nms_iou_threshold": 0.3,
        }
        invalid = pd.DataFrame(
            [self._prediction_row(image_index=2)],
            columns=comparison.PREDICTION_COLUMNS,
        )
        with self.assertRaisesRegex(ValueError, "image keys"):
            comparison.validate_prediction_table(
                invalid, "model1", self.index, ["pallet", "person"], policy
            )

        predictions = pd.DataFrame(
            [self._prediction_row()], columns=comparison.PREDICTION_COLUMNS
        )
        predictions = comparison.validate_prediction_table(
            predictions, "model1", self.index, ["pallet", "person"], policy
        )
        ledger = pd.DataFrame(
            [self._ledger_row(post_nms_prediction_count=0)],
            columns=comparison.LEDGER_COLUMNS,
        )
        ledger = comparison.validate_ledger(ledger, self.index, ["model1"])
        with self.assertRaisesRegex(ValueError, "disagree by image"):
            comparison.validate_prediction_ledger_alignment(
                predictions, ledger, "model1", self.index
            )

    def test_ledger_validates_paths_counts_timings_and_summary(self):
        valid = pd.DataFrame(
            [self._ledger_row()], columns=comparison.LEDGER_COLUMNS
        )
        ledger = comparison.validate_ledger(valid, self.index, ["model1"])
        comparison.validate_inference_summary(
            {
                "images_processed": 1, "raw_candidates": 1,
                "post_nms_predictions": 1,
            },
            ledger, "model1", 1,
        )
        cases = (
            ({"image_path": "wrong.jpg"}, "paths"),
            ({"raw_candidate_count": 0}, "exceeds raw"),
            ({"read_seconds": -0.1, "total_seconds": 0.8}, "read_seconds"),
            ({"total_seconds": 2.0}, "timing totals"),
            ({"image_index": 1.5}, "positions"),
        )
        for overrides, message in cases:
            table = pd.DataFrame(
                [self._ledger_row(**overrides)],
                columns=comparison.LEDGER_COLUMNS,
            )
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ValueError, message
            ):
                comparison.validate_ledger(table, self.index, ["model1"])
        with self.assertRaisesRegex(ValueError, "summary disagrees"):
            comparison.validate_inference_summary(
                {
                    "images_processed": 1, "raw_candidates": 0,
                    "post_nms_predictions": 1,
                },
                ledger, "model1", 1,
            )

    def test_raw_class_score_vector_dimension_is_validated(self):
        bundle = {name: self.root / name for name in ("weights", "cfg", "names")}
        for path in bundle.values():
            path.touch()
        with comparison._atomic_csv_stream(
            self.root / "wrong-vector-ledger.csv", comparison.LEDGER_COLUMNS
        ) as writer:
            with self.assertRaisesRegex(ValueError, "vector length"):
                comparison.run_inference_for_model(
                    "model1", bundle, self.index, ["pallet", "person"], self.root,
                    0.001, 0.3, self.root / "wrong-vector-predictions.csv", writer,
                    image_reader=lambda _: self.frame,
                    detector_factory=WrongVectorDetector,
                )

    def test_dual_metrics_report_perfect_present_class_and_empty_class(self):
        ground_truth = comparison.build_ground_truth(
            self.index, ["pallet", "person"], self.root,
            image_reader=lambda _: self.frame,
        )
        predictions = pd.DataFrame(
            [self._prediction_row()], columns=comparison.PREDICTION_COLUMNS
        )
        policy = {
            "candidate_floor": 0.001, "deployment_confidence": 0.5,
            "nms_iou_threshold": 0.3, "map_iou_threshold": 0.5,
        }
        aggregate, per_class = comparison.evaluate_model(
            "model1", self.index, predictions, ground_truth,
            ["pallet", "person"], policy,
        )
        self.assertAlmostEqual(aggregate["mAP50_101pt"], 0.5)
        self.assertAlmostEqual(aggregate["threshold_constrained_mAP50_11pt"], 0.5)
        self.assertEqual(aggregate["deployment_true_positives"], 1)
        self.assertEqual(per_class["deployment_f1"].tolist(), [1.0, 0.0])


class FullRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / "storage"
        logistics = self.assets / "logistics"
        logistics.mkdir(parents=True)
        (logistics / "a.jpg").touch()
        (logistics / "a.txt").write_text("0 0.25 0.25 0.5 0.5\n", encoding="utf-8")
        for number in (1, 2):
            directory = self.assets / f"yolo_model_{number}"
            directory.mkdir()
            (directory / f"yolov4-tiny-logistics_size_416_{number}.weights").touch()
            (directory / f"yolov4-tiny-logistics_size_416_{number}.cfg").write_text(
                "[yolo]\nclasses=2\n", encoding="utf-8"
            )
            (directory / "logistics.names").write_text("pallet\nperson\n", encoding="utf-8")
        self.index_path = self.root / "index.csv"
        pd.DataFrame([{
            "image_file": "a.jpg",
            "image_path": "detector_service/storage/logistics/a.jpg",
            "label_path": "detector_service/storage/logistics/a.txt",
            "num_objects": 1,
        }]).to_csv(self.index_path, index=False)
        self.output = self.root / "runs"
        self.frame = np.zeros((20, 20, 3), dtype=np.uint8)

    def _args(self, run_id="integration"):
        return argparse.Namespace(
            dataset_index=self.index_path, asset_root=self.assets,
            output_root=self.output, run_id=run_id,
            pilot=False,
            expected_images=1, expected_labels=1, max_images=None,
            candidate_floor=0.001, deployment_confidence=0.5, nms_iou=0.3,
        )

    def test_complete_run_is_verified_and_promoted_without_figures(self):
        final, manifest, aggregate, per_class = comparison.run_experiment(
            self._args(), detector_factory=FakeDetector,
            image_reader=lambda _: self.frame,
        )
        expected = {
            "ground_truth.csv", "model1_predictions.csv", "model2_predictions.csv",
            "inference_ledger.csv", "aggregate_metrics.csv", "per_class_metrics.csv",
            "run_manifest.json",
        }
        self.assertEqual({path.name for path in final.iterdir()}, expected)
        self.assertFalse((self.output / ".integration.incomplete").exists())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["dataset"]["selected_images"], 1)
        self.assertEqual(manifest["dataset"]["selected_labels"], 1)
        self.assertEqual(manifest["run_scope"], "full")
        self.assertEqual(manifest["dataset"]["asset_identity"]["files"], 2)
        self.assertEqual(len(aggregate), 2)
        self.assertEqual(len(per_class), 4)
        self.assertTrue(comparison.verify_run_directory(final, manifest))

    def test_explicit_pilot_scope_is_recorded(self):
        args = self._args("pilot")
        args.pilot = True
        args.max_images = 1
        args.expected_images = None
        args.expected_labels = None
        final, manifest, _, _ = comparison.run_experiment(
            args, detector_factory=FakeDetector,
            image_reader=lambda _: self.frame,
        )
        self.assertEqual(manifest["run_scope"], "pilot")
        self.assertEqual(manifest["dataset"]["selection_max_images"], 1)
        self.assertTrue(comparison.verify_run_directory(final, manifest))

    def test_long_dataset_paths_never_leak_device_prefixes_into_evidence(self):
        logistics = self.assets / "logistics"
        stem = "m" * 251
        image = logistics / f"{stem}.jpg"
        label = logistics / f"{stem}.txt"
        image_filesystem = comparison._filesystem_path(image)
        label_filesystem = comparison._filesystem_path(label)
        image_filesystem.write_bytes(b"long-image")
        label_filesystem.write_text(
            "0 0.25 0.25 0.5 0.5\n", encoding="utf-8"
        )
        self.addCleanup(image_filesystem.unlink, missing_ok=True)
        self.addCleanup(label_filesystem.unlink, missing_ok=True)
        logical_image = f"detector_service/storage/logistics/{image.name}"
        logical_label = f"detector_service/storage/logistics/{label.name}"
        pd.DataFrame([{
            "image_file": image.name, "image_path": logical_image,
            "label_path": logical_label, "num_objects": 1,
        }]).to_csv(self.index_path, index=False)

        final, manifest, _, _ = comparison.run_experiment(
            self._args("long-path"), detector_factory=FakeDetector,
            image_reader=lambda _: self.frame,
        )
        serialized = json.dumps(manifest, sort_keys=True)
        self.assertNotIn("\\\\?\\", serialized)
        self.assertEqual(manifest["external_storage_root"], str(self.assets.resolve()))
        predictions = pd.read_csv(final / "model1_predictions.csv")
        ground_truth = pd.read_csv(final / "ground_truth.csv")
        self.assertEqual(predictions["image_path"].tolist(), [logical_image])
        self.assertEqual(ground_truth["image_path"].tolist(), [logical_image])

    def test_completed_run_is_never_blindly_reused_or_overwritten(self):
        comparison.run_experiment(
            self._args(), detector_factory=FakeDetector,
            image_reader=lambda _: self.frame,
        )
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            comparison.run_experiment(
                self._args(), detector_factory=FakeDetector,
                image_reader=lambda _: self.frame,
            )

    def test_failure_leaves_unpromoted_staging_directory(self):
        with self.assertRaisesRegex(ValueError, "Unable to read"):
            comparison.run_experiment(
                self._args("failed"), detector_factory=FakeDetector,
                image_reader=lambda _: None,
            )
        self.assertTrue((self.output / ".failed.incomplete").is_dir())
        self.assertFalse((self.output / "failed").exists())

    def test_manifest_verification_detects_artifact_tampering(self):
        final, manifest, _, _ = comparison.run_experiment(
            self._args(), detector_factory=FakeDetector,
            image_reader=lambda _: self.frame,
        )
        with (final / "aggregate_metrics.csv").open("a", encoding="utf-8") as handle:
            handle.write("tampered\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            comparison.verify_run_directory(final, manifest)

    def test_manifest_verification_reads_disk_and_rechecks_inputs(self):
        final, manifest, _, _ = comparison.run_experiment(
            self._args(), detector_factory=FakeDetector,
            image_reader=lambda _: self.frame,
        )
        manifest_path = final / "run_manifest.json"
        changed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        changed_manifest["environment"]["platform"] = "tampered"
        manifest_path.write_text(
            json.dumps(changed_manifest, indent=2) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            comparison.verify_run_directory(final, manifest)

        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        label = self.assets / "logistics" / "a.txt"
        label.write_text(
            label.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeError, "dataset_assets"):
            comparison.verify_run_directory(final, manifest)

    def test_input_mutation_prevents_atomic_promotion(self):
        label = self.assets / "logistics" / "a.txt"

        class MutatingDetector(FakeDetector):
            changed = False

            def predict(self, frame):
                if not type(self).changed:
                    label.write_text(
                        label.read_text(encoding="utf-8") + "\n",
                        encoding="utf-8",
                    )
                    type(self).changed = True
                return super().predict(frame)

        with self.assertRaisesRegex(RuntimeError, "dataset_assets"):
            comparison.run_experiment(
                self._args("mutated"), detector_factory=MutatingDetector,
                image_reader=lambda _: self.frame,
            )
        self.assertTrue((self.output / ".mutated.incomplete").is_dir())
        self.assertFalse((self.output / "mutated").exists())


if __name__ == "__main__":
    unittest.main()
