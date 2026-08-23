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
        self.selection_run = self.root / "custom-selection-run"
        self.selection_run.mkdir()
        self.selection = {
            "selection_directory": self.selection_run,
            "selection_run_id": self.selection_run.name,
            "selection_manifest_sha256": "3" * 64,
            "decision_sha256": "4" * 64,
            "selected_checkpoint": "B",
            "selected_model": sweep.MODEL_NAME,
            "model_identity": {
                "weights": "1" * 64,
                "cfg": "2" * 64,
                "names": sweep.sha256_file(self.class_file),
            },
            "asset_paths": {
                "weights": "yolo_model_2/custom.weights",
                "config": "yolo_model_2/custom.cfg",
                "classes": "yolo_model_2/classes.names",
            },
        }

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
        candidate_counts = self.raw.groupby("image_file").size()
        self.ledger = pd.DataFrame(
            [
                {
                    "model": sweep.MODEL_NAME,
                    "image_file": row.image_file,
                    "image_path": row.image_path,
                    "status": "processed",
                    "candidate_count": int(candidate_counts.get(row.image_file, 0)),
                }
                for row in self.index.itertuples(index=False)
            ],
            columns=sweep.LEDGER_COLUMNS,
        )
        self.ledger_path = self.root / "inference_ledger.csv"
        self.ledger.to_csv(self.ledger_path, index=False)
        self.cache_manifest_path = self.root / "inference_cache_manifest.json"
        self.write_cache_manifest()

    def write_cache_manifest(self, path=None):
        destination = Path(path or self.cache_manifest_path)
        payload = sweep.build_inference_cache_manifest(
            sample_index_path=self.index_path,
            index=self.index,
            selection=self.selection,
            classes=self.classes,
            class_file=self.class_file,
            model_name=sweep.MODEL_NAME,
            run_label="sample5000",
            artifact_paths={
                "ground_truth": self.ground_truth_path,
                "raw_predictions": self.raw_path,
                "inference_ledger": self.ledger_path,
            },
            artifact_tables={
                "ground_truth": self.ground_truth,
                "raw_predictions": self.raw,
                "inference_ledger": self.ledger,
            },
        )
        destination.write_text(json.dumps(payload), encoding="utf-8")
        return destination

    def replay_arguments(self, output_dir=None):
        return [
            "--sample-index",
            str(self.index_path),
            "--overlap-profile",
            str(self.overlap_path),
            "--selection-run",
            str(self.selection_run),
            "--class-file",
            str(self.class_file),
            "--ground-truth-cache",
            str(self.ground_truth_path),
            "--raw-predictions-cache",
            str(self.raw_path),
            "--inference-ledger",
            str(self.ledger_path),
            "--inference-cache-manifest",
            str(self.cache_manifest_path),
            "--output-dir",
            str(output_dir or self.output_dir),
            "--skip-figures",
            "--refresh-postprocessing",
        ]

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
    def test_default_paths_follow_numbered_experiment_layout(self):
        output_root = PROJECT_ROOT / "experiments" / "outputs"
        self.assertEqual(
            sweep.DEFAULT_SAMPLE_INDEX,
            output_root
            / "02_dataset_analysis"
            / "02_sample_selection"
            / "selected_sample_index.csv",
        )
        self.assertEqual(
            sweep.DEFAULT_OVERLAP_PROFILE,
            output_root
            / "02_dataset_analysis"
            / "03_overlap_analysis"
            / "overlap_profile.csv",
        )
        self.assertEqual(
            sweep.DEFAULT_OUTPUT_DIR,
            output_root / "03_nms_thresholding" / "01_threshold_sweep",
        )
        self.assertEqual(
            sweep.DEFAULT_FIGURE_DIR,
            PROJECT_ROOT
            / "scratch"
            / "diagnostic-figures"
            / "03_nms_thresholding",
        )

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

    def test_classes_require_a_hash_verified_full_vocabulary(self):
        expected = sweep.sha256_file(self.class_file)
        self.assertEqual(
            sweep.load_classes(self.class_file, expected_sha256=expected),
            self.classes,
        )

        with self.assertRaisesRegex(FileNotFoundError, "trailing unseen classes"):
            sweep.load_classes(
                self.root / "missing.names",
                expected_sha256=expected,
            )

        different = self.root / "different.names"
        different.write_text("alpha\nbeta", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "selected checkpoint identity"):
            sweep.load_classes(different, expected_sha256=expected)

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

    def test_raw_cache_requires_class_id_to_match_score_argmax(self):
        broken = self.raw.copy()
        vector = json.loads(broken.loc[0, "class_scores_json"])
        vector[1] = 0.99
        broken.loc[0, "class_scores_json"] = json.dumps(vector)
        with self.assertRaisesRegex(ValueError, r"argmax\(class_scores_json\)"):
            sweep.validate_raw_predictions(broken, self.index, self.classes)

    def test_live_inference_never_uses_an_implicit_selection_run(self):
        with self.assertRaisesRegex(ValueError, "selection_run is required"):
            sweep.run_raw_inference(
                self.index.iloc[0:0],
                self.classes,
                image_reader=Mock(),
                detector_factory=Mock(),
            )

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

    def test_indexed_paths_reject_parent_traversal_and_external_absolute_paths(self):
        asset_root = self.root / "assets"
        asset_root.mkdir()
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            sweep.resolve_indexed_path(
                "detector_service/storage/../secret.txt",
                asset_root,
            )
        with self.assertRaisesRegex(ValueError, "outside the permitted root"):
            sweep.resolve_indexed_path(self.root / "outside.txt", asset_root)

    def test_inference_ledger_proves_zero_candidate_images_were_processed(self):
        validated = sweep.validate_inference_ledger(
            self.ledger,
            self.index,
            self.raw,
        )
        self.assertEqual(len(validated), len(self.index))
        self.assertEqual(
            int(validated.loc[validated["image_file"] == "c.jpg", "candidate_count"].iloc[0]),
            0,
        )

        incomplete = self.ledger.iloc[:-1]
        with self.assertRaisesRegex(ValueError, "row count"):
            sweep.validate_inference_ledger(incomplete, self.index, self.raw)

    def test_legacy_storage_namespace_is_the_only_path_alias(self):
        legacy_ground_truth = self.ground_truth.copy()
        legacy_ground_truth["image_path"] = legacy_ground_truth["image_path"].str.replace(
            "detector_service/storage/",
            "techtrack/storage/",
            regex=False,
        )
        sweep.validate_ground_truth(legacy_ground_truth, self.index, self.classes)

        legacy_raw = self.raw.copy()
        legacy_raw["image_path"] = legacy_raw["image_path"].str.replace(
            "detector_service/storage/",
            "techtrack/storage/",
            regex=False,
        )
        sweep.validate_raw_predictions(legacy_raw, self.index, self.classes)

        legacy_ledger = self.ledger.copy()
        legacy_ledger["image_path"] = legacy_ledger["image_path"].str.replace(
            "detector_service/storage/",
            "techtrack/storage/",
            regex=False,
        )
        sweep.validate_inference_ledger(
            legacy_ledger,
            self.index,
            legacy_raw,
        )

        for label, table, validator in (
            (
                "ground truth",
                self.ground_truth,
                lambda value: sweep.validate_ground_truth(
                    value,
                    self.index,
                    self.classes,
                ),
            ),
            (
                "raw predictions",
                self.raw,
                lambda value: sweep.validate_raw_predictions(
                    value,
                    self.index,
                    self.classes,
                ),
            ),
            (
                "ledger",
                self.ledger,
                lambda value: sweep.validate_inference_ledger(
                    value,
                    self.index,
                    self.raw,
                ),
            ),
        ):
            with self.subTest(label=label):
                broken = table.copy()
                broken.loc[0, "image_path"] = (
                    "detector_service/storage/unrelated/not-a.jpg"
                )
                with self.assertRaisesRegex(ValueError, "image_path"):
                    validator(broken)


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

    def test_complete_derived_package_validates_before_figure_use(self):
        artifacts = self.build()
        validated = sweep.validate_derived_artifacts(artifacts, "sample5000")
        self.assertEqual(set(validated), set(artifacts))

    def test_derived_package_rejects_schema_policy_threshold_and_cross_table_tampering(self):
        def clone(artifacts):
            return {name: table.copy() for name, table in artifacts.items()}

        valid = self.build()
        summary_name = "nms_threshold_summary_sample5000.csv"
        duplicate_name = "duplicate_summary_by_threshold_sample5000.csv"

        schema = clone(valid)
        schema[summary_name]["unexpected"] = 1

        policy = clone(valid)
        policy[summary_name].loc[0, "score_threshold"] = 0.25

        threshold = clone(valid)
        threshold[summary_name] = threshold[summary_name].iloc[:-1].copy()

        cross_table = clone(valid)
        cross_table[duplicate_name].loc[0, "total_predictions_after_nms"] += 1

        for label, package, message in (
            ("schema", schema, "schema mismatch"),
            ("policy", policy, "policy does not match"),
            ("threshold", threshold, "required rows"),
            ("cross-table", cross_table, "do not reconcile"),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                sweep.validate_derived_artifacts(package, "sample5000")

    def test_figures_only_rejects_invalid_derived_package_before_rendering(self):
        artifacts = self.build()
        sweep.write_sweep_artifacts(self.output_dir, artifacts)
        summary_path = self.output_dir / "nms_threshold_summary_sample5000.csv"
        summary = pd.read_csv(summary_path)
        summary.loc[0, "score_threshold"] = 0.25
        summary.to_csv(summary_path, index=False)

        with (
            patch.object(sweep, "build_figures") as figure_builder,
            self.assertRaisesRegex(ValueError, "policy does not match"),
        ):
            sweep.main(
                [
                    "--figures-only",
                    "--output-dir",
                    str(self.output_dir),
                    "--figure-dir",
                    str(self.figure_dir),
                ]
            )
        figure_builder.assert_not_called()

    def test_main_reuses_explicit_caches_without_inference(self):
        with patch.object(
            sweep,
            "load_verified_checkpoint_selection",
            return_value=self.selection,
        ):
            artifacts, paths, figures = sweep.main(self.replay_arguments())
        self.assertEqual(len(artifacts), 4)
        self.assertEqual(len(paths), 6)
        self.assertEqual(
            paths[self.cache_manifest_path.name],
            self.cache_manifest_path,
        )
        self.assertTrue(paths["operating_point.json"].is_file())
        self.assertEqual(figures, [])

    def test_custom_selection_run_drives_live_inference_asset_resolution(self):
        resolved_paths = {
            "weights": self.root / "custom.weights",
            "config": self.root / "custom.cfg",
            "classes": self.class_file,
        }
        selected_assets = {
            "selected_model": sweep.MODEL_NAME,
            "resolved_paths": resolved_paths,
        }

        def fake_inference(*_args, ledger_rows, **_kwargs):
            ledger_rows.extend(self.ledger.to_dict(orient="records"))
            return self.raw.copy()

        live_output = self.root / "live-output"
        with (
            patch.object(
                sweep,
                "load_verified_checkpoint_selection",
                return_value=self.selection,
            ),
            patch.object(
                sweep,
                "resolve_selected_model_assets",
                return_value=selected_assets,
            ) as resolver,
            patch.object(
                sweep,
                "run_raw_inference",
                side_effect=fake_inference,
            ) as inference,
            patch.object(
                sweep,
                "build_ground_truth",
                return_value=self.ground_truth.copy(),
            ),
        ):
            sweep.main(
                [
                    "--sample-index",
                    str(self.index_path),
                    "--overlap-profile",
                    str(self.overlap_path),
                    "--selection-run",
                    str(self.selection_run),
                    "--class-file",
                    str(self.class_file),
                    "--output-dir",
                    str(live_output),
                    "--skip-figures",
                ]
            )

        resolver.assert_called_once_with(
            sweep.PROJECT_ROOT / "detector_service" / "storage",
            self.selection_run,
        )
        self.assertEqual(inference.call_args.kwargs["model_assets"], resolved_paths)
        self.assertEqual(
            inference.call_args.kwargs["selection_run"],
            self.selection_run,
        )
        self.assertEqual(
            inference.call_args.kwargs["expected_names_sha256"],
            self.selection["model_identity"]["names"],
        )
        manifest_path = live_output / "inference_cache_manifest_sample5000.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["sample_index"]["selected_rows"], len(self.index))
        self.assertEqual(
            manifest["checkpoint_selection"]["run_id"],
            self.selection["selection_run_id"],
        )
        self.assertEqual(
            set(manifest["artifacts"]),
            {"ground_truth", "raw_predictions", "inference_ledger"},
        )

    def test_external_cache_overrides_require_one_manifest_bound_package(self):
        with self.assertRaisesRegex(ValueError, "requires --ground-truth-cache"):
            sweep.main(
                [
                    "--ground-truth-cache",
                    str(self.ground_truth_path),
                    "--raw-predictions-cache",
                    str(self.raw_path),
                    "--inference-ledger",
                    str(self.ledger_path),
                ]
            )

    def test_managed_historical_cache_without_manifest_fails_closed(self):
        managed = self.root / "managed-historical"
        managed.mkdir()
        self.ground_truth.to_csv(
            managed / "ground_truth_sample5000.csv",
            index=False,
        )
        self.raw.to_csv(
            managed / "model2_raw_predictions_sample5000.csv",
            index=False,
        )
        self.ledger.to_csv(
            managed / "model2_inference_ledger_sample5000.csv",
            index=False,
        )
        with (
            patch.object(
                sweep,
                "load_verified_checkpoint_selection",
                return_value=self.selection,
            ),
            self.assertRaisesRegex(FileNotFoundError, "cannot be trusted"),
        ):
            sweep.main(
                [
                    "--sample-index",
                    str(self.index_path),
                    "--selection-run",
                    str(self.selection_run),
                    "--class-file",
                    str(self.class_file),
                    "--output-dir",
                    str(managed),
                    "--skip-figures",
                ]
            )

    def test_replay_rejects_tampered_cache_artifact(self):
        broken = self.raw.copy()
        broken.loc[0, "combined_confidence"] = 0.123
        broken.to_csv(self.raw_path, index=False)
        with (
            patch.object(
                sweep,
                "load_verified_checkpoint_selection",
                return_value=self.selection,
            ),
            self.assertRaisesRegex(ValueError, "manifest is stale"),
        ):
            sweep.main(self.replay_arguments(self.root / "tampered-output"))

    def test_replay_rejects_stale_manifest_policy(self):
        manifest = json.loads(self.cache_manifest_path.read_text(encoding="utf-8"))
        manifest["candidate_policy"]["objectness_threshold"] = 0.25
        self.cache_manifest_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        with (
            patch.object(
                sweep,
                "load_verified_checkpoint_selection",
                return_value=self.selection,
            ),
            self.assertRaisesRegex(ValueError, "manifest is stale"),
        ):
            sweep.main(self.replay_arguments(self.root / "stale-output"))

    def test_invalid_new_cache_never_receives_a_complete_manifest(self):
        invalid = self.raw.copy()
        invalid.loc[0, "combined_confidence"] = 0.123

        def fake_inference(*_args, ledger_rows, **_kwargs):
            ledger_rows.extend(self.ledger.to_dict(orient="records"))
            return invalid

        output = self.root / "invalid-live-output"
        selected_assets = {
            "selected_model": sweep.MODEL_NAME,
            "resolved_paths": {
                "weights": self.root / "custom.weights",
                "config": self.root / "custom.cfg",
                "classes": self.class_file,
            },
        }
        with (
            patch.object(
                sweep,
                "load_verified_checkpoint_selection",
                return_value=self.selection,
            ),
            patch.object(
                sweep,
                "resolve_selected_model_assets",
                return_value=selected_assets,
            ),
            patch.object(sweep, "run_raw_inference", side_effect=fake_inference),
            patch.object(
                sweep,
                "build_ground_truth",
                return_value=self.ground_truth.copy(),
            ),
            self.assertRaisesRegex(ValueError, "does not equal"),
        ):
            sweep.main(
                [
                    "--sample-index",
                    str(self.index_path),
                    "--overlap-profile",
                    str(self.overlap_path),
                    "--selection-run",
                    str(self.selection_run),
                    "--class-file",
                    str(self.class_file),
                    "--output-dir",
                    str(output),
                    "--skip-figures",
                ]
            )
        self.assertFalse(
            (output / "inference_cache_manifest_sample5000.json").exists()
        )

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
