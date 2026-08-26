import importlib.util
import json
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
SCRIPT_DIR = PROJECT_ROOT / "experiments" / "scripts" / "05_hard_negative_mining"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


components_script = load_module(
    SCRIPT_DIR / "05_build_hnm_components.py",
    "hnm_components_under_test",
)
queues_script = load_module(
    SCRIPT_DIR / "05_build_error_review_queues.py",
    "hnm_queues_under_test",
)


class HardNegativeFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output_dir = self.root / "outputs"
        self.figure_dir = self.root / "figures"
        self.sample = pd.DataFrame(
            [
                self.sample_row("a", 1, "1", ["alpha"]),
                self.sample_row("b", 1, "1", ["alpha"]),
                self.sample_row("c", 1, "1", ["beta"]),
                self.sample_row("d", 1, "1", ["alpha"]),
                self.sample_row("e", 1, "1", ["alpha"]),
            ],
            columns=components_script.SAMPLE_COLUMNS,
        )
        self.predictions = pd.DataFrame(
            [
                self.prediction_row("a", 0, [0, 0, 10, 10], 0.90),
                self.prediction_row("b", 0, [0, 0, 10, 10], 0.95),
                self.prediction_row("b", 1, [20, 20, 5, 5], 0.80),
                self.prediction_row("d", 0, [2, 0, 10, 10], 0.55),
                self.prediction_row("e", 1, [0, 0, 10, 10], 0.85),
            ],
            columns=[
                *components_script.PREDICTION_INPUT_COLUMNS,
                "dataset",
                "augmentation_condition",
            ],
        )
        self.ground_truth = pd.DataFrame(
            [
                self.truth_row(name, 0, [0, 0, 10, 10])
                for name in ("a", "b", "c", "d", "e")
            ],
            columns=components_script.GROUND_TRUTH_COLUMNS,
        )
        self.sample_path = self.root / "sample.csv"
        self.prediction_path = self.root / "predictions.csv"
        self.ground_truth_path = self.root / "ground_truth.csv"
        self.sample.to_csv(self.sample_path, index=False)
        self.predictions.to_csv(self.prediction_path, index=False)
        self.ground_truth.to_csv(self.ground_truth_path, index=False)

    def sample_row(self, name, objects, bucket, classes):
        return {
            "image_file": f"{name}.jpg",
            "image_path": f"detector_service/storage/images/{name}.jpg",
            "num_objects": objects,
            "density_bucket": bucket,
            "class_names_present": json.dumps(classes),
        }

    def prediction_row(self, name, class_id, box, confidence):
        return {
            "model": "model2",
            "nms_threshold": 0.3,
            "dataset": components_script.DATASET_NAME,
            "augmentation_condition": components_script.ORIGINAL_CONDITION,
            "image_file": f"{name}.jpg",
            "bbox_x": box[0],
            "bbox_y": box[1],
            "bbox_w": box[2],
            "bbox_h": box[3],
            "class_id": class_id,
            "combined_confidence": confidence,
        }

    def truth_row(self, name, class_id, box):
        return {
            "image_file": f"{name}.jpg",
            "bbox_x": box[0],
            "bbox_y": box[1],
            "bbox_w": box[2],
            "bbox_h": box[3],
            "class_id": class_id,
        }

    def build_components(self):
        return components_script.validate_component_table(
            components_script.build_component_table(
                self.sample,
                self.predictions,
                self.ground_truth,
            )
        )

    def write_components(self):
        path = self.root / "image_error_components_sample5000.csv"
        self.build_components().to_csv(path, index=False)
        return path

    def load_predictions(self, path=None, model="model2", threshold=0.3):
        return components_script.load_predictions(
            path or self.prediction_path,
            self.sample,
            expected_model=model,
            expected_nms_threshold=threshold,
        )

    def resolved_prediction_input(self):
        return {
            "prediction_path": self.prediction_path,
            "selected_model": "model2",
            "nms_threshold": 0.3,
        }


class ComponentContractTests(HardNegativeFixture):
    def test_module_import_does_not_require_ignored_experiment_evidence(self):
        with patch(
            "experiments.scripts.experiment_contracts."
            "load_verified_checkpoint_selection",
            side_effect=AssertionError("selection evidence read during import"),
        ), patch(
            "experiments.scripts.experiment_contracts.load_verified_operating_point",
            side_effect=AssertionError("operating-point evidence read during import"),
        ):
            loaded = load_module(
                SCRIPT_DIR / "05_build_hnm_components.py",
                "hnm_components_clean_checkout_test",
            )

        self.assertEqual(
            loaded.DEFAULT_PREDICTION_PATH,
            loaded.DEFAULT_NMS_OUTPUT_DIR
            / "model2_predictions_nms_0_3_sample5000.csv",
        )

    def test_component_defaults_follow_numbered_experiment_layout(self):
        output_root = PROJECT_ROOT / "experiments" / "outputs"
        self.assertEqual(
            components_script.DEFAULT_SAMPLE_PATH,
            output_root
            / "02_dataset_analysis"
            / "02_sample_selection"
            / "selected_sample_index.csv",
        )
        self.assertEqual(
            components_script.DEFAULT_PREDICTION_PATH,
            output_root
            / "03_nms_thresholding"
            / "01_threshold_sweep"
            / "model2_predictions_nms_0_3_sample5000.csv",
        )
        self.assertEqual(
            components_script.DEFAULT_SELECTION_RUN,
            output_root
            / "01_model_selection"
            / "03_checkpoint_decision"
            / "selection-20260821-v1",
        )
        self.assertEqual(
            components_script.DEFAULT_OPERATING_POINT,
            output_root
            / "03_nms_thresholding"
            / "01_threshold_sweep"
            / "operating_point.json",
        )
        self.assertEqual(
            components_script.DEFAULT_GROUND_TRUTH_PATH,
            output_root
            / "03_nms_thresholding"
            / "01_threshold_sweep"
            / "ground_truth_sample5000.csv",
        )
        self.assertEqual(
            components_script.DEFAULT_OUTPUT_DIR,
            output_root
            / "05_hard_negative_mining"
            / "01_error_components",
        )
        arguments = components_script.build_parser().parse_args([])
        self.assertEqual(arguments.selection_run, components_script.DEFAULT_SELECTION_RUN)
        self.assertEqual(
            arguments.operating_point,
            components_script.DEFAULT_OPERATING_POINT,
        )
        self.assertIsNone(arguments.predictions)

    def test_default_prediction_path_comes_from_verified_decisions(self):
        selection_run = self.root / "selection"
        operating_path = self.root / "thresholds" / "operating_point.json"
        with patch.object(
            components_script,
            "load_verified_checkpoint_selection",
            return_value={"selected_model": "model1"},
        ), patch.object(
            components_script,
            "load_verified_operating_point",
            return_value={
                "path": operating_path,
                "selected_nms_iou_threshold": 0.55,
            },
        ):
            resolved = components_script.resolve_prediction_input(
                None,
                selection_run,
                operating_path,
            )
            explicit = components_script.resolve_prediction_input(
                self.prediction_path,
                selection_run,
                operating_path,
            )

        self.assertEqual(resolved["selected_model"], "model1")
        self.assertEqual(resolved["nms_threshold"], 0.55)
        self.assertEqual(
            resolved["prediction_path"],
            operating_path.parent
            / "model1_predictions_nms_0_55_sample5000.csv",
        )
        self.assertEqual(explicit["prediction_path"], self.prediction_path)

    def test_operating_thresholds_and_schema_are_explicit(self):
        self.assertEqual(components_script.MATCH_IOU_THRESHOLD, 0.5)
        self.assertEqual(components_script.CONFIDENCE_FLOOR, 0.5)
        self.assertEqual(
            components_script.COMPONENT_COLUMNS[:5],
            components_script.SAMPLE_COLUMNS,
        )
        self.assertEqual(
            components_script.COMPONENT_COLUMNS[5:9],
            list(components_script.ERROR_COMPONENT_COLUMNS),
        )
        self.assertEqual(
            len(components_script.COMPONENT_COLUMNS),
            len(set(components_script.COMPONENT_COLUMNS)),
        )
        self.assertEqual(
            components_script.PREDICTION_PROVENANCE_COLUMNS,
            ["model", "nms_threshold"],
        )

    def test_sample_loader_validates_uniqueness_counts_and_buckets(self):
        loaded = components_script.load_sample(self.sample_path, max_images=3)
        self.assertEqual(loaded["image_file"].tolist(), ["a.jpg", "b.jpg", "c.jpg"])
        duplicate = pd.concat([self.sample, self.sample.iloc[[0]]], ignore_index=True)
        duplicate.to_csv(self.sample_path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate image_file"):
            components_script.load_sample(self.sample_path)
        invalid = self.sample.copy()
        invalid.loc[0, "density_bucket"] = "unknown"
        invalid.to_csv(self.sample_path, index=False)
        with self.assertRaisesRegex(ValueError, "density bucket"):
            components_script.load_sample(self.sample_path)

    def test_sample_bound_cannot_exceed_available_rows(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            components_script.load_sample(self.sample_path, max_images=20)

    def test_prediction_loader_filters_to_sample_and_enforces_confidence_floor(self):
        outside = self.prediction_row("outside", 0, [0, 0, 1, 1], 0.9)
        table = pd.concat([self.predictions, pd.DataFrame([outside])], ignore_index=True)
        table.to_csv(self.prediction_path, index=False)
        loaded = self.load_predictions()
        self.assertNotIn("outside.jpg", set(loaded["image_file"]))
        self.assertEqual(loaded.columns.tolist(), components_script.PREDICTION_COLUMNS)
        invalid = self.predictions.copy()
        invalid.loc[0, "combined_confidence"] = 0.49
        invalid.to_csv(self.prediction_path, index=False)
        with self.assertRaisesRegex(ValueError, "confidence floor"):
            self.load_predictions()

    def test_prediction_loader_rejects_missing_or_mismatched_provenance(self):
        missing = self.predictions.drop(columns="model")
        missing.to_csv(self.prediction_path, index=False)
        with self.assertRaisesRegex(ValueError, "missing columns: model"):
            self.load_predictions()

        wrong_model = self.predictions.copy()
        wrong_model["model"] = "model1"
        wrong_model.to_csv(self.prediction_path, index=False)
        with self.assertRaisesRegex(ValueError, "model provenance"):
            self.load_predictions()

        wrong_threshold = self.predictions.copy()
        wrong_threshold["nms_threshold"] = 0.2
        wrong_threshold.to_csv(self.prediction_path, index=False)
        with self.assertRaisesRegex(ValueError, "nms_threshold provenance"):
            self.load_predictions()

    def test_optional_dataset_and_condition_provenance_must_match(self):
        wrong_dataset = self.predictions.copy()
        wrong_dataset["dataset"] = "different_population"
        wrong_dataset.to_csv(self.prediction_path, index=False)
        with self.assertRaisesRegex(ValueError, "dataset provenance"):
            self.load_predictions()

        wrong_condition = self.predictions.copy()
        wrong_condition["augmentation_condition"] = "gaussian_blur_k9"
        wrong_condition.to_csv(self.prediction_path, index=False)
        with self.assertRaisesRegex(ValueError, "augmentation_condition provenance"):
            self.load_predictions()

        ground_truth = self.ground_truth.assign(dataset="different_population")
        ground_truth.to_csv(self.ground_truth_path, index=False)
        with self.assertRaisesRegex(ValueError, "dataset provenance"):
            components_script.load_ground_truth(self.ground_truth_path, self.sample)

    def test_prediction_and_truth_geometry_must_be_finite_and_nonnegative(self):
        invalid_predictions = self.predictions.copy()
        invalid_predictions.loc[0, "bbox_w"] = -1
        invalid_predictions.to_csv(self.prediction_path, index=False)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            self.load_predictions()
        invalid_truth = self.ground_truth.copy()
        invalid_truth["bbox_h"] = invalid_truth["bbox_h"].astype(float)
        invalid_truth.loc[0, "bbox_h"] = np.inf
        invalid_truth.to_csv(self.ground_truth_path, index=False)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            components_script.load_ground_truth(self.ground_truth_path, self.sample)

    def test_ground_truth_loader_reconciles_every_sample_count(self):
        loaded = components_script.load_ground_truth(self.ground_truth_path, self.sample)
        self.assertEqual(len(loaded), len(self.sample))
        self.ground_truth.iloc[:-1].to_csv(self.ground_truth_path, index=False)
        with self.assertRaisesRegex(ValueError, "disagree"):
            components_script.load_ground_truth(self.ground_truth_path, self.sample)


class ComponentConstructionTests(HardNegativeFixture):
    def test_component_builder_preserves_manifest_order_and_all_images(self):
        result = self.build_components()
        self.assertEqual(result["image_file"].tolist(), self.sample["image_file"].tolist())
        self.assertEqual(result.columns.tolist(), components_script.COMPONENT_COLUMNS)
        self.assertEqual(len(result), len(self.sample))

    def test_component_builder_captures_perfect_duplicate_miss_and_wrong_class_cases(self):
        result = self.build_components().set_index("image_file")
        self.assertEqual(result.loc["a.jpg", "false_negative_rate"], 0.0)
        self.assertEqual(result.loc["a.jpg", "false_positive_rate"], 0.0)
        self.assertEqual(result.loc["b.jpg", "false_positive_prediction_count"], 1)
        self.assertEqual(result.loc["b.jpg", "false_positive_rate"], 0.5)
        self.assertEqual(result.loc["c.jpg", "prediction_count"], 0)
        self.assertEqual(result.loc["c.jpg", "false_negative_rate"], 1.0)
        self.assertEqual(result.loc["e.jpg", "false_positive_rate"], 1.0)
        self.assertEqual(result.loc["e.jpg", "false_negative_rate"], 1.0)

    def test_match_dependent_components_use_iou_and_confidence(self):
        result = self.build_components().set_index("image_file")
        self.assertGreater(result.loc["d.jpg", "localization_error"], 0.0)
        self.assertAlmostEqual(result.loc["d.jpg", "confidence_error"], 0.9)
        self.assertEqual(result.loc["c.jpg", "localization_error"], 0.0)
        self.assertEqual(result.loc["c.jpg", "confidence_error"], 0.0)

    def test_component_validator_rejects_broken_count_identity(self):
        invalid = self.build_components()
        invalid.loc[0, "missed_gt_count"] = 2
        with self.assertRaisesRegex(ValueError, "count identities"):
            components_script.validate_component_table(invalid)

    def test_component_main_writes_bounded_artifact(self):
        with patch.object(
            components_script,
            "resolve_prediction_input",
            return_value=self.resolved_prediction_input(),
        ):
            result, output = components_script.main(
                [
                    "--sample-index",
                    str(self.sample_path),
                    "--predictions",
                    str(self.prediction_path),
                    "--ground-truth",
                    str(self.ground_truth_path),
                    "--output-dir",
                    str(self.output_dir),
                ]
            )
        self.assertEqual(output.name, "image_error_components_sample5000.csv")
        self.assertTrue(output.is_file())
        assert_frame_equal(pd.read_csv(output), result, check_dtype=False)

    def test_bounded_main_uses_prefix_and_run_specific_name(self):
        with patch.object(
            components_script,
            "resolve_prediction_input",
            return_value=self.resolved_prediction_input(),
        ):
            result, output = components_script.main(
                [
                    "--sample-index",
                    str(self.sample_path),
                    "--predictions",
                    str(self.prediction_path),
                    "--ground-truth",
                    str(self.ground_truth_path),
                    "--output-dir",
                    str(self.output_dir),
                    "--max-images",
                    "2",
                ]
            )
        self.assertEqual(len(result), 2)
        self.assertEqual(output.name, "image_error_components_first_2.csv")


class QueueContractTests(HardNegativeFixture):
    def test_queue_defaults_separate_components_from_review_tables(self):
        experiment_dir = (
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "05_hard_negative_mining"
        )
        self.assertEqual(
            queues_script.DEFAULT_COMPONENT_PATH,
            experiment_dir
            / "01_error_components"
            / "image_error_components_sample5000.csv",
        )
        self.assertEqual(
            queues_script.DEFAULT_OUTPUT_DIR,
            experiment_dir / "02_review_queues",
        )
        self.assertEqual(
            queues_script.DEFAULT_FIGURE_DIR,
            PROJECT_ROOT
            / "scratch"
            / "diagnostic-figures"
            / "05_hard_negative_mining",
        )

    def test_five_profiles_have_distinct_eligibility_and_weight_contracts(self):
        self.assertEqual(
            [profile["profile_name"] for profile in queues_script.PROFILES],
            [
                "balanced",
                "localization",
                "matched_confidence",
                "false_positive",
                "false_negative",
            ],
        )
        self.assertEqual(
            [profile["eligibility"] for profile in queues_script.PROFILES],
            ["all", "matched", "matched", "predictions", "ground_truth"],
        )
        self.assertEqual(
            queues_script.PROFILES[0]["weights"],
            dict.fromkeys(queues_script.ERROR_COMPONENT_COLUMNS, 1.0),
        )

    def test_parse_class_names_supports_json_python_lists_and_plain_values(self):
        self.assertEqual(queues_script.parse_class_names('["alpha", "beta"]'), ["alpha", "beta"])
        self.assertEqual(queues_script.parse_class_names("['alpha']"), ["alpha"])
        self.assertEqual(queues_script.parse_class_names("alpha"), ["alpha"])
        self.assertEqual(queues_script.parse_class_names(""), [])

    def test_unsupported_eligibility_rule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            queues_script._eligible_rows(self.build_components(), "unknown")


class QueueValidationTests(HardNegativeFixture):
    def test_component_loader_preserves_valid_complete_rows(self):
        path = self.write_components()
        loaded = queues_script.load_component_table(path)
        self.assertEqual(len(loaded), len(self.sample))
        self.assertEqual(loaded.columns.tolist(), queues_script.BASE_COMPONENT_COLUMNS)

    def test_component_loader_rejects_duplicate_images_and_invalid_ranges(self):
        table = self.build_components()
        duplicate = pd.concat([table, table.iloc[[0]]], ignore_index=True)
        path = self.root / "bad.csv"
        duplicate.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "unique"):
            queues_script.load_component_table(path)
        table.loc[0, "false_positive_rate"] = 1.2
        table.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "within"):
            queues_script.load_component_table(path)

    def test_component_loader_rejects_nonzero_match_errors_without_matches(self):
        table = self.build_components()
        miss = table["image_file"] == "c.jpg"
        table.loc[miss, "localization_error"] = 0.5
        path = self.root / "bad.csv"
        table.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "must be zero"):
            queues_script.load_component_table(path)


class QueueConstructionTests(HardNegativeFixture):
    def test_top_samples_enforce_profile_eligibility(self):
        top = queues_script.build_top_samples(self.build_components(), top_n=2)
        self.assertEqual(len(top), 10)
        localization = top[top["profile_name"] == "localization"]
        self.assertTrue((localization["matched_prediction_count"] > 0).all())
        false_positive = top[top["profile_name"] == "false_positive"]
        self.assertTrue((false_positive["prediction_count"] > 0).all())
        false_negative = top[top["profile_name"] == "false_negative"]
        self.assertTrue((false_negative["ground_truth_count"] > 0).all())

    def test_top_samples_are_deterministic_ranked_and_weighted(self):
        first = queues_script.build_top_samples(self.build_components(), top_n=2)
        second = queues_script.build_top_samples(self.build_components(), top_n=2)
        assert_frame_equal(first, second)
        for _, group in first.groupby("profile_name", sort=False):
            self.assertEqual(group["rank"].tolist(), [1, 2])
            self.assertTrue(group["error_score"].is_monotonic_decreasing)

    def test_false_negative_ties_prioritize_zero_prediction_and_missed_counts(self):
        top = queues_script.build_top_samples(self.build_components(), top_n=2)
        queue = top[top["profile_name"] == "false_negative"]
        self.assertEqual(queue.iloc[0]["image_file"], "c.jpg")
        self.assertEqual(queue.iloc[0]["zero_prediction"], 1)

    def test_summary_reports_counts_components_and_profile_weights(self):
        top = queues_script.build_top_samples(self.build_components(), top_n=2)
        summary = queues_script.summarize_profiles(top)
        self.assertEqual(summary.columns.tolist(), queues_script.SUMMARY_COLUMNS)
        self.assertEqual(summary["top_n"].tolist(), [2] * len(queues_script.PROFILES))
        balanced = summary.iloc[0]
        self.assertEqual(balanced["weight_localization_error"], 1.0)

    def test_overlap_is_symmetric_with_unit_diagonal(self):
        top = queues_script.build_top_samples(self.build_components(), top_n=2)
        overlap = queues_script.build_overlap(top)
        matrix = overlap.pivot(index="profile_a", columns="profile_b", values="jaccard_overlap")
        np.testing.assert_allclose(matrix.to_numpy(), matrix.to_numpy().T)
        np.testing.assert_allclose(np.diag(matrix), 1.0)

    def test_density_has_every_bucket_and_normalized_profile_shares(self):
        top = queues_script.build_top_samples(self.build_components(), top_n=2)
        density = queues_script.build_density(top)
        self.assertEqual(len(density), len(queues_script.PROFILES) * len(queues_script.DENSITY_BUCKETS))
        shares = density.groupby("profile_name")["image_share"].sum()
        np.testing.assert_allclose(shares, 1.0)

    def test_class_presence_counts_images_per_queue(self):
        top = queues_script.build_top_samples(self.build_components(), top_n=2)
        presence = queues_script.build_class_presence(top)
        self.assertEqual(presence.columns.tolist(), queues_script.CLASS_PRESENCE_COLUMNS)
        self.assertTrue(presence["image_share"].between(0.0, 1.0).all())

    def test_artifact_builder_returns_six_canonical_tables(self):
        artifacts = queues_script.build_artifacts(self.build_components(), top_n=2)
        self.assertEqual(
            list(artifacts),
            [
                "review_queue_profiles.csv",
                "top_images_by_profile.csv",
                "profile_summary.csv",
                "profile_overlap.csv",
                "density_by_profile.csv",
                "class_presence_by_profile.csv",
            ],
        )


class QueueArtifactTests(HardNegativeFixture):
    def test_queue_main_writes_all_six_artifacts_without_figures(self):
        component_path = self.write_components()
        artifacts, paths, figures = queues_script.main(
            [
                "--components",
                str(component_path),
                "--output-dir",
                str(self.output_dir),
                "--top-n",
                "2",
                "--skip-figures",
            ]
        )
        self.assertEqual(set(paths), set(artifacts))
        self.assertTrue(all(path.is_file() for path in paths.values()))
        self.assertEqual(figures, [])

    def test_repeated_artifact_write_is_byte_deterministic(self):
        artifacts = queues_script.build_artifacts(self.build_components(), top_n=2)
        first = queues_script.write_artifacts(self.output_dir, artifacts)
        before = {name: path.read_bytes() for name, path in first.items()}
        second = queues_script.write_artifacts(self.output_dir, artifacts)
        after = {name: path.read_bytes() for name, path in second.items()}
        self.assertEqual(before, after)

    def test_missing_derived_artifacts_are_reported_together(self):
        with self.assertRaisesRegex(FileNotFoundError, "Missing review-queue artifacts"):
            queues_script.load_artifacts(self.output_dir)


class ReferenceCompatibilityTests(HardNegativeFixture):
    def test_valid_components_and_queue_artifacts_match_reference(self):
        component_reference = os.environ.get("REFERENCE_HNM_COMPONENT_SCRIPT")
        queue_reference = os.environ.get("REFERENCE_HNM_QUEUE_SCRIPT")
        if not component_reference or not queue_reference:
            self.skipTest("Hard-negative reference scripts are not configured")

        reference_components = load_module(Path(component_reference), "reference_hnm_components")
        if "matplotlib" not in sys.modules:
            matplotlib = types.ModuleType("matplotlib")
            pyplot = types.ModuleType("matplotlib.pyplot")
            matplotlib.pyplot = pyplot
            sys.modules["matplotlib"] = matplotlib
            sys.modules["matplotlib.pyplot"] = pyplot
        reference_queues = load_module(Path(queue_reference), "reference_hnm_queues")

        ours = self.build_components()
        theirs = reference_components.build_component_table(
            self.sample,
            self.predictions,
            self.ground_truth,
        )
        assert_frame_equal(ours, theirs, check_dtype=False, atol=1e-12, rtol=1e-12)

        our_top = queues_script.build_top_samples(ours, top_n=2)
        their_top = reference_queues.build_top_samples(theirs, top_n=2)
        assert_frame_equal(our_top, their_top, check_dtype=False, atol=1e-12, rtol=1e-12)
        comparisons = (
            (queues_script.summarize_profiles(our_top), reference_queues.summarize_profiles(their_top)),
            (queues_script.build_overlap(our_top), reference_queues.build_overlap(their_top)),
            (queues_script.build_density(our_top), reference_queues.build_density(their_top)),
            (queues_script.build_class_presence(our_top), reference_queues.build_class_presence(their_top)),
        )
        for left, right in comparisons:
            assert_frame_equal(left, right, check_dtype=False, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
