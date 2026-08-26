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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "scripts"
    / "02_dataset_analysis"
    / "02_dataset_sampling.py"
)

spec = importlib.util.spec_from_file_location("dataset_sampling_under_test", SCRIPT_PATH)
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)


class SamplingFixture(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.root = Path(self.temp_directory.name)
        self.index_path = self.root / "dataset_index.csv"
        self.class_path = self.root / "class_distribution.csv"
        self.output_dir = self.root / "output"
        self.figure_dir = self.root / "figures"
        self.sample_size = 15

        class_names = ["alpha", "beta class", "gamma", "delta"]
        rows = []
        for index in range(30):
            counts = [
                int(index < 10),
                int(index < 10 and index % 2 == 0),
                int(index < 10 and index % 3 == 0),
                int(index < 10 and index % 4 == 0),
            ]
            rows.append(
                {
                    "image_file": f"frame_{index:03d}.jpg",
                    "image_path": f"detector_service/storage/logistics/frame_{index:03d}.jpg",
                    "label_path": f"detector_service/storage/logistics/frame_{index:03d}.txt",
                    "num_objects": sum(counts),
                    **{
                        f"count_{sampling.clean_column_name(name)}": count
                        for name, count in zip(class_names, counts)
                    },
                }
            )
        self.index = pd.DataFrame(rows)
        self.classes = pd.DataFrame(
            [
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "object_count": int(
                        self.index[f"count_{sampling.clean_column_name(class_name)}"].sum()
                    ),
                    "image_count": int(
                        (
                            self.index[
                                f"count_{sampling.clean_column_name(class_name)}"
                            ]
                            > 0
                        ).sum()
                    ),
                }
                for class_id, class_name in enumerate(class_names)
            ]
        )
        self.index.to_csv(self.index_path, index=False)
        self.classes.to_csv(self.class_path, index=False)

    def load(self):
        return sampling.load_and_validate_inputs(
            self.index_path,
            self.class_path,
            self.sample_size,
        )


class SamplingContractTests(unittest.TestCase):
    def test_default_paths_follow_the_canonical_stage_hierarchy(self):
        self.assertEqual(
            sampling.DEFAULT_DATASET_INDEX,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "00_dataset_inventory"
            / "dataset_index.csv",
        )
        self.assertEqual(
            sampling.DEFAULT_OUTPUT_DIR,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "02_dataset_analysis"
            / "02_sample_selection",
        )
        self.assertEqual(
            sampling.DEFAULT_FIGURE_DIR,
            PROJECT_ROOT / "scratch" / "diagnostic-figures" / "02_dataset_analysis",
        )

    def test_constants_and_candidate_policy_are_explicit(self):
        self.assertEqual(sampling.SAMPLE_SIZE, 5000)
        self.assertEqual(sampling.RANDOM_SEED, 42)
        self.assertEqual(sampling.RARE_CLASS_COUNT, 8)
        self.assertEqual(sampling.RARE_CLASS_MINIMUM_IMAGES, 100)
        self.assertEqual(
            sampling.DENSITY_BUCKET_ORDER,
            ["1", "2-4", "5-9", "10-14", "15-19", "20+"],
        )

    def test_density_bucket_boundaries_match_summary_stage(self):
        expected = {
            0: "1",
            1: "1",
            2: "2-4",
            4: "2-4",
            5: "5-9",
            9: "5-9",
            10: "10-14",
            14: "10-14",
            15: "15-19",
            19: "15-19",
            20: "20+",
        }
        self.assertEqual(
            {value: sampling.density_bucket(value) for value in expected},
            expected,
        )

    def test_cli_sample_size_validation(self):
        self.assertEqual(sampling.positive_int("5"), 5)
        with self.assertRaisesRegex(Exception, "positive integer"):
            sampling.positive_int("0")


class InputValidationTests(SamplingFixture):
    def test_valid_inputs_are_reconciled_and_bucketed(self):
        index, classes = self.load()
        self.assertEqual(len(index), 30)
        self.assertEqual(len(classes), 4)
        self.assertIn("density_bucket", index)
        self.assertEqual(set(index["density_bucket"]), {"1", "2-4"})

    def test_sample_size_cannot_exceed_dataset(self):
        with self.assertRaisesRegex(ValueError, "exceeds dataset size"):
            sampling.load_and_validate_inputs(
                self.index_path,
                self.class_path,
                31,
            )

    def test_duplicate_image_paths_are_rejected(self):
        self.index.loc[1, "image_path"] = self.index.loc[0, "image_path"]
        self.index.to_csv(self.index_path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate image_path"):
            self.load()

    def test_num_objects_must_match_class_sum(self):
        self.index.loc[0, "num_objects"] += 1
        self.index.to_csv(self.index_path, index=False)
        with self.assertRaisesRegex(ValueError, "per-class count sum"):
            self.load()

    def test_class_aggregates_must_match_index(self):
        self.classes.loc[0, "object_count"] += 1
        self.classes.to_csv(self.class_path, index=False)
        with self.assertRaisesRegex(ValueError, "Object total mismatch"):
            self.load()

    def test_fractional_counts_are_rejected(self):
        self.index["count_alpha"] = self.index["count_alpha"].astype(float)
        self.index.loc[0, "count_alpha"] = 0.5
        self.index.to_csv(self.index_path, index=False)
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            self.load()


class ProportionalSamplingTests(SamplingFixture):
    def test_largest_remainder_allocation_has_deterministic_ties(self):
        counts = pd.Series([5, 3, 2], index=["a", "b", "c"])
        targets = sampling.proportional_targets(counts, 5)
        self.assertEqual(targets.to_dict(), {"a": 3, "b": 1, "c": 1})
        self.assertEqual(int(targets.sum()), 5)

    def test_proportional_sample_is_repeatable_unique_and_exact_size(self):
        index, _ = self.load()
        first = sampling.proportional_sample(
            index,
            "density_bucket",
            self.sample_size,
            42,
        )
        second = sampling.proportional_sample(
            index,
            "density_bucket",
            self.sample_size,
            42,
        )
        self.assertEqual(len(first), self.sample_size)
        self.assertEqual(first["image_path"].nunique(), self.sample_size)
        self.assertEqual(first["image_path"].tolist(), second["image_path"].tolist())

        full_counts = index["density_bucket"].value_counts().sort_index()
        expected = sampling.proportional_targets(full_counts, self.sample_size)
        actual = first["density_bucket"].value_counts().reindex(expected.index, fill_value=0)
        pd.testing.assert_series_equal(
            actual.astype(np.int64),
            expected,
            check_names=False,
        )

    def test_invalid_group_inputs_are_rejected(self):
        index, _ = self.load()
        with self.assertRaisesRegex(ValueError, "Missing sampling group"):
            sampling.proportional_sample(index, "missing", 5, 42)
        with self.assertRaisesRegex(ValueError, "within the grouped dataset"):
            sampling.proportional_targets(pd.Series([2, 3]), 6)


class RareCoverageTests(SamplingFixture):
    def test_rare_targets_apply_proportional_floor_and_minimum(self):
        classes = pd.DataFrame(
            {
                "class_id": [0, 1, 2],
                "class_name": ["tiny", "medium", "large"],
                "object_count": [5, 200, 500],
                "image_count": [5, 150, 300],
            }
        )
        targets = sampling.rare_class_targets(classes, 0.5)
        self.assertEqual(targets["target_image_count"].tolist(), [5, 100, 150])

    def test_enforcement_meets_every_target_and_preserves_size(self):
        index, classes = self.load()
        base = sampling.proportional_sample(
            index,
            "density_bucket",
            self.sample_size,
            42,
        )
        targets = sampling.rare_class_targets(
            classes,
            self.sample_size / len(index),
        )
        selected = sampling.enforce_rare_class_targets(
            base,
            index,
            targets,
            self.sample_size,
            42,
        )

        self.assertEqual(len(selected), self.sample_size)
        self.assertEqual(selected["image_path"].nunique(), self.sample_size)
        self.assertEqual(
            selected["image_file"].tolist(),
            sorted(selected["image_file"].tolist()),
        )
        for row in targets.itertuples(index=False):
            column = f"count_{sampling.clean_column_name(row.class_name)}"
            self.assertGreaterEqual(
                int((selected[column] > 0).sum()),
                int(row.target_image_count),
            )

    def test_incompatible_targets_cannot_be_silently_sacrificed(self):
        full = pd.DataFrame(
            [
                {
                    "image_file": f"a{i}.jpg",
                    "image_path": f"a{i}.jpg",
                    "count_one": 1,
                    "count_two": 0,
                }
                for i in range(6)
            ]
            + [
                {
                    "image_file": f"b{i}.jpg",
                    "image_path": f"b{i}.jpg",
                    "count_one": 0,
                    "count_two": 1,
                }
                for i in range(6)
            ]
        )
        base = full.iloc[:6].copy()
        targets = pd.DataFrame(
            {
                "class_name": ["one", "two"],
                "target_image_count": [6, 6],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "targets were not preserved"):
            sampling.enforce_rare_class_targets(
                base,
                full,
                targets,
                sample_size=6,
                seed=42,
            )


class EvidenceTests(SamplingFixture):
    def test_distribution_helpers_report_distinct_object_and_image_shares(self):
        index, classes = self.load()
        class_table = sampling.class_distribution(index, classes, "full")
        density_table = sampling.density_distribution(index, "full")
        summary = sampling.dataset_summary(index, "full")

        alpha = class_table[class_table["class_name"] == "alpha"].iloc[0]
        self.assertEqual(alpha["object_count"], 10)
        self.assertEqual(alpha["image_count"], 10)
        self.assertAlmostEqual(class_table["object_share_pct"].sum(), 100.0)
        self.assertAlmostEqual(density_table["image_share_pct"].sum(), 100.0)
        self.assertEqual(summary["images"], 30)

    def test_sampling_evidence_has_stable_artifacts_and_selected_name(self):
        index, classes = self.load()
        artifacts, candidates, selected_name = sampling.build_sampling_evidence(
            index,
            classes,
            self.sample_size,
            42,
        )

        self.assertEqual(selected_name, "rare_aware_density_stratified_15")
        self.assertEqual(
            list(artifacts),
            [
                "rare_class_targets.csv",
                "candidate_sample_quality.csv",
                "selected_sample_index.csv",
                "sample_summary.csv",
                "class_distribution_comparison.csv",
                "rare_class_coverage.csv",
                "density_distribution_comparison.csv",
            ],
        )
        self.assertEqual(
            list(candidates),
            [
                "random_15",
                "density_stratified_15",
                "rare_aware_density_stratified_15",
            ],
        )
        selected = artifacts["selected_sample_index.csv"]
        self.assertEqual(len(selected), self.sample_size)
        self.assertEqual(selected["image_path"].nunique(), self.sample_size)
        self.assertIn("density_bucket", selected.columns)
        self.assertEqual(len(artifacts["candidate_sample_quality.csv"]), 3)

    def test_sampling_evidence_is_repeatable(self):
        index, classes = self.load()
        first, _, _ = sampling.build_sampling_evidence(
            index,
            classes,
            self.sample_size,
            42,
        )
        second, _, _ = sampling.build_sampling_evidence(
            index,
            classes,
            self.sample_size,
            42,
        )
        pd.testing.assert_frame_equal(
            first["selected_sample_index.csv"],
            second["selected_sample_index.csv"],
        )
        pd.testing.assert_frame_equal(
            first["candidate_sample_quality.csv"],
            second["candidate_sample_quality.csv"],
        )

    def test_selected_sample_meets_written_rare_targets(self):
        index, classes = self.load()
        artifacts, _, _ = sampling.build_sampling_evidence(
            index,
            classes,
            self.sample_size,
            42,
        )
        selected = artifacts["selected_sample_index.csv"]
        coverage = artifacts["rare_class_coverage.csv"]
        self.assertTrue(
            (
                coverage["image_count_sample"]
                >= coverage["target_image_count"]
            ).all()
        )
        self.assertEqual(selected["image_path"].nunique(), self.sample_size)


class ArtifactAndCompatibilityTests(SamplingFixture):
    def test_csv_artifacts_are_written_atomically(self):
        index, classes = self.load()
        artifacts, _, _ = sampling.build_sampling_evidence(
            index,
            classes,
            self.sample_size,
            42,
        )
        paths = sampling.write_sampling_artifacts(self.output_dir, artifacts)
        self.assertEqual([path.name for path in paths], list(artifacts))
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertEqual(list(self.output_dir.glob("*.tmp")), [])

    def test_main_can_skip_figures_for_headless_analysis(self):
        result = sampling.main(
            [
                "--dataset-index",
                str(self.index_path),
                "--class-distribution",
                str(self.class_path),
                "--output-dir",
                str(self.output_dir),
                "--sample-size",
                str(self.sample_size),
                "--skip-figures",
            ]
        )
        self.assertEqual(result[2], "rare_aware_density_stratified_15")
        self.assertEqual(len(result[3]), 7)
        self.assertEqual(result[4], [])

    def test_core_algorithms_match_reference_on_valid_fixture(self):
        reference_path = os.environ.get("REFERENCE_DATASET_SAMPLING_SCRIPT")
        if not reference_path:
            self.skipTest("REFERENCE_DATASET_SAMPLING_SCRIPT is not configured")

        pyplot = types.ModuleType("matplotlib.pyplot")
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.__path__ = []
        matplotlib.pyplot = pyplot
        with patch.dict(
            sys.modules,
            {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot},
        ):
            reference_spec = importlib.util.spec_from_file_location(
                "reference_dataset_sampling",
                Path(reference_path),
            )
            reference = importlib.util.module_from_spec(reference_spec)
            reference_spec.loader.exec_module(reference)

        index, classes = self.load()
        reference.SAMPLE_SIZE = self.sample_size
        reference_density = reference.proportional_sample(
            index,
            "density_bucket",
            self.sample_size,
            42,
        )
        new_density = sampling.proportional_sample(
            index,
            "density_bucket",
            self.sample_size,
            42,
        )
        pd.testing.assert_frame_equal(reference_density, new_density)

        fraction = self.sample_size / len(index)
        reference_targets = reference.rare_class_targets(classes, fraction)
        new_targets = sampling.rare_class_targets(classes, fraction)
        pd.testing.assert_frame_equal(
            reference_targets.reset_index(drop=True),
            new_targets.reset_index(drop=True),
        )

        reference_selected = reference.enforce_rare_class_targets(
            reference_density,
            index,
            reference_targets,
            42,
        )
        new_selected = sampling.enforce_rare_class_targets(
            new_density,
            index,
            new_targets,
            self.sample_size,
            42,
        )
        pd.testing.assert_frame_equal(reference_selected, new_selected)
        self.assertEqual(
            reference.compare_sample(
                index,
                reference_selected,
                classes,
                "sample",
                reference_targets,
            ),
            sampling.compare_sample(
                index,
                new_selected,
                classes,
                "sample",
                new_targets,
            ),
        )


if __name__ == "__main__":
    unittest.main()
