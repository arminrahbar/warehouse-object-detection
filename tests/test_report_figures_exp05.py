"""Integrity and rendering tests for Experiment 05 publication figures."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "experiments" / "scripts"
SCRIPT_PATH = SCRIPT_DIR / "05_build_report_figures.py"
QUEUE_SCRIPT_PATH = SCRIPT_DIR / "05_build_error_review_queues.py"
COMPONENT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "outputs"
    / "05_hard_negative_mining"
    / "01_error_components"
)
QUEUE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "outputs"
    / "05_hard_negative_mining"
    / "02_review_queues"
)


def load_module():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    specification = importlib.util.spec_from_file_location("report_figures_exp05", SCRIPT_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


figures = load_module()


def load_queue_module():
    specification = importlib.util.spec_from_file_location(
        "error_review_queues_exp05",
        QUEUE_SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


queues = load_queue_module()


def build_synthetic_components():
    """Create contract-complete evidence without relying on ignored outputs."""

    image_count = 5000
    image_number = np.arange(image_count)

    ground_truth_count = np.empty(image_count, dtype=int)
    ground_truth_count[:839] = 3
    ground_truth_count[839:1677] = 2
    ground_truth_count[1677:3388] = 5
    ground_truth_count[3388:] = 4

    prediction_count = np.zeros(image_count, dtype=int)
    prediction_count[1677:] = 2
    prediction_count[1677 : 1677 + 1081] += 1

    matched_count = np.zeros(image_count, dtype=int)
    matched_count[1677:4809] = 1

    false_positive_count = prediction_count - matched_count
    missed_count = ground_truth_count - matched_count
    has_match = matched_count > 0

    localization_error = np.zeros(image_count, dtype=float)
    confidence_error = np.zeros(image_count, dtype=float)
    false_positive_rate = np.divide(
        false_positive_count,
        prediction_count,
        out=np.zeros(image_count, dtype=float),
        where=prediction_count > 0,
    )
    false_negative_rate = missed_count / ground_truth_count
    confidence_error[has_match] = false_negative_rate[has_match]
    localization_error[has_match] = 1.0 - confidence_error[has_match]

    class_names = np.where(
        image_number % 4 == 0,
        '["person", "forklift"]',
        np.where(image_number % 3 == 0, '["person", "pallet"]', '["person"]'),
    )

    table = pd.DataFrame(
        {
            "image_file": [f"image_{number:05d}.jpg" for number in image_number],
            "image_path": [
                f"detector_service/storage/logistics/image_{number:05d}.jpg"
                for number in image_number
            ],
            "num_objects": ground_truth_count,
            "density_bucket": "2-4",
            "class_names_present": class_names,
            "localization_error": localization_error,
            "confidence_error": confidence_error,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "prediction_count": prediction_count,
            "ground_truth_count": ground_truth_count,
            "matched_prediction_count": matched_count,
            "false_positive_prediction_count": false_positive_count,
            "matched_gt_count": matched_count,
            "missed_gt_count": missed_count,
            "mean_matched_iou": np.where(
                has_match,
                1.0 - 0.5 * localization_error,
                0.0,
            ),
            "mean_matched_confidence": np.where(
                has_match,
                1.0 - 0.5 * confidence_error,
                0.0,
            ),
        }
    )
    return table[queues.BASE_COMPONENT_COLUMNS]


class Experiment05FigureParserTests(unittest.TestCase):
    def test_defaults_separate_components_from_queue_evidence(self):
        args = figures.build_parser().parse_args(["--output-dir", "figures"])
        self.assertEqual(args.component_dir, COMPONENT_DIR)
        self.assertEqual(args.queue_dir, QUEUE_DIR)
        self.assertEqual(args.output_dir, Path("figures"))

    def test_output_directory_is_required(self):
        with self.assertRaises(SystemExit):
            figures.build_parser().parse_args([])


class Experiment05FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(cls.temporary_directory.name)
        component_dir = root / "components"
        queue_dir = root / "queues"
        component_dir.mkdir()
        queue_dir.mkdir()

        component_path = component_dir / "image_error_components_sample5000.csv"
        build_synthetic_components().to_csv(component_path, index=False)
        component_table = queues.load_component_table(component_path)
        queue_artifacts = queues.build_artifacts(component_table, top_n=250)
        queues.write_artifacts(queue_dir, queue_artifacts)
        cls.evidence = figures.load_verified_evidence(component_dir, queue_dir)

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_evidence_reconciles_to_locked_population(self):
        components = self.evidence["components"]
        self.assertEqual(len(components), 5000)
        self.assertEqual(int(components["ground_truth_count"].sum()), 19196)
        self.assertEqual(int(components["prediction_count"].sum()), 7727)
        self.assertEqual(int((components["prediction_count"] == 0).sum()), 1677)
        self.assertEqual(
            int(
                components.loc[
                    components["prediction_count"] == 0, "ground_truth_count"
                ].sum()
            ),
            4193,
        )

    def test_eligible_statistics_and_correlations_are_well_formed(self):
        statistics = figures._eligible_component_statistics(self.evidence["components"])
        statistics = statistics.set_index("component")
        expected_eligible = {
            "localization_error": 3132,
            "confidence_error": 3132,
            "false_positive_rate": 3323,
            "false_negative_rate": 5000,
        }
        for component, eligible in expected_eligible.items():
            row = statistics.loc[component]
            self.assertEqual(int(row["eligible"]), eligible)
            quantiles = row[["p50", "p75", "p95"]].to_numpy(dtype=float)
            self.assertTrue(np.isfinite(quantiles).all())
            self.assertTrue(((0.0 <= quantiles) & (quantiles <= 1.0)).all())
            self.assertTrue(np.all(np.diff(quantiles) >= 0.0))

        correlation = figures._component_correlation(self.evidence)
        values = correlation.to_numpy(dtype=float)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.allclose(values, values.T))
        self.assertTrue(np.allclose(np.diag(values), 1.0))
        self.assertTrue(((values >= -1.0) & (values <= 1.0)).all())

    def test_complete_package_is_exact_and_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = figures.build_figure_package(self.evidence, root / "first")
            second = figures.build_figure_package(self.evidence, root / "second")
            expected = {
                f"{stem}.{extension}"
                for stem in figures.OUTPUT_STEMS
                for extension in ("png", "svg")
            }
            self.assertEqual({path.name for path in first.iterdir()}, expected)
            self.assertEqual({path.name for path in second.iterdir()}, expected)
            for name in expected:
                first_bytes = (first / name).read_bytes()
                second_bytes = (second / name).read_bytes()
                self.assertGreater(len(first_bytes), 0)
                self.assertEqual(
                    hashlib.sha256(first_bytes).hexdigest(),
                    hashlib.sha256(second_bytes).hexdigest(),
                )
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                figures.build_figure_package(self.evidence, first)


if __name__ == "__main__":
    unittest.main()
