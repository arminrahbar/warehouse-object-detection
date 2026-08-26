"""Regression tests for the numbered experiment-output directory contract."""

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "experiments" / "scripts"
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "outputs"
SCRIPT_PATHS = {
    "00_build_dataset_inventory.py": Path(
        "02_dataset_analysis/00_build_dataset_inventory.py"
    ),
    "01_model_comparison.py": Path(
        "01_model_selection/01_model_comparison.py"
    ),
    "01_benchmark_inference.py": Path(
        "01_model_selection/01_benchmark_inference.py"
    ),
    "01_select_checkpoint.py": Path(
        "01_model_selection/01_select_checkpoint.py"
    ),
    "02_summarize_dataset.py": Path(
        "02_dataset_analysis/02_summarize_dataset.py"
    ),
    "02_dataset_sampling.py": Path(
        "02_dataset_analysis/02_dataset_sampling.py"
    ),
    "02_overlap_analysis.py": Path(
        "02_dataset_analysis/02_overlap_analysis.py"
    ),
    "03_nms_threshold_sweep.py": Path(
        "03_nms_thresholding/03_nms_threshold_sweep.py"
    ),
    "04_augmentation_demo.py": Path(
        "04_augmentation_robustness/04_augmentation_demo.py"
    ),
    "04_augmentation_robustness.py": Path(
        "04_augmentation_robustness/04_augmentation_robustness.py"
    ),
    "05_build_hnm_components.py": Path(
        "05_hard_negative_mining/05_build_hnm_components.py"
    ),
    "05_build_error_review_queues.py": Path(
        "05_hard_negative_mining/05_build_error_review_queues.py"
    ),
}


def load_script(filename):
    """Import a numerically prefixed entry point under a stable test name."""

    module_name = "output_contract_" + Path(filename).stem.replace("-", "_")
    specification = importlib.util.spec_from_file_location(
        module_name,
        SCRIPT_DIR / SCRIPT_PATHS[filename],
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Unable to import experiment script: {filename}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


class OutputPathContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filenames = (
            "00_build_dataset_inventory.py",
            "01_model_comparison.py",
            "01_benchmark_inference.py",
            "01_select_checkpoint.py",
            "02_summarize_dataset.py",
            "02_dataset_sampling.py",
            "02_overlap_analysis.py",
            "03_nms_threshold_sweep.py",
            "04_augmentation_demo.py",
            "04_augmentation_robustness.py",
            "05_build_hnm_components.py",
            "05_build_error_review_queues.py",
        )
        cls.modules = {filename: load_script(filename) for filename in filenames}

    def assert_contract_path(self, filename, constant, relative_path):
        module = self.modules[filename]
        actual = Path(getattr(module, constant))
        expected = OUTPUT_ROOT / Path(relative_path)
        self.assertTrue(actual.is_absolute(), f"{filename}.{constant} is not absolute")
        self.assertEqual(actual, expected, f"{filename}.{constant}")

    def test_evidence_producers_use_numbered_stage_directories(self):
        expectations = (
            ("00_build_dataset_inventory.py", "DEFAULT_OUTPUT_DIR", "00_dataset_inventory"),
            (
                "01_model_comparison.py",
                "DEFAULT_OUTPUT_ROOT",
                "01_model_selection/01_quality_comparison",
            ),
            (
                "01_benchmark_inference.py",
                "DEFAULT_OUTPUT_ROOT",
                "01_model_selection/02_runtime_benchmark",
            ),
            (
                "01_select_checkpoint.py",
                "DEFAULT_OUTPUT_ROOT",
                "01_model_selection/03_checkpoint_decision",
            ),
            (
                "02_summarize_dataset.py",
                "DEFAULT_OUTPUT_DIR",
                "02_dataset_analysis/01_dataset_summary",
            ),
            (
                "02_dataset_sampling.py",
                "DEFAULT_OUTPUT_DIR",
                "02_dataset_analysis/02_sample_selection",
            ),
            (
                "02_overlap_analysis.py",
                "DEFAULT_OVERLAP_DIR",
                "02_dataset_analysis/03_overlap_analysis",
            ),
            (
                "03_nms_threshold_sweep.py",
                "DEFAULT_OUTPUT_DIR",
                "03_nms_thresholding/01_threshold_sweep",
            ),
            (
                "04_augmentation_robustness.py",
                "DEFAULT_OUTPUT_DIR",
                "04_augmentation_robustness/01_condition_evaluation",
            ),
            (
                "05_build_hnm_components.py",
                "DEFAULT_OUTPUT_DIR",
                "05_hard_negative_mining/01_error_components",
            ),
            (
                "05_build_error_review_queues.py",
                "DEFAULT_OUTPUT_DIR",
                "05_hard_negative_mining/02_review_queues",
            ),
        )
        for filename, constant, relative_path in expectations:
            with self.subTest(filename=filename, constant=constant):
                self.assert_contract_path(filename, constant, relative_path)

    def test_stage_dependencies_resolve_to_canonical_upstream_artifacts(self):
        expectations = (
            (
                "01_model_comparison.py",
                "DEFAULT_DATASET_INDEX",
                "00_dataset_inventory/dataset_index.csv",
            ),
            (
                "01_benchmark_inference.py",
                "DEFAULT_DATASET_INDEX",
                "00_dataset_inventory/dataset_index.csv",
            ),
            (
                "02_summarize_dataset.py",
                "DEFAULT_DATASET_INDEX",
                "00_dataset_inventory/dataset_index.csv",
            ),
            (
                "02_summarize_dataset.py",
                "DEFAULT_CLASS_DISTRIBUTION",
                "00_dataset_inventory/class_distribution.csv",
            ),
            (
                "02_summarize_dataset.py",
                "DEFAULT_OBJECT_DISTRIBUTION",
                "00_dataset_inventory/object_count_distribution.csv",
            ),
            (
                "02_dataset_sampling.py",
                "DEFAULT_DATASET_INDEX",
                "00_dataset_inventory/dataset_index.csv",
            ),
            (
                "02_dataset_sampling.py",
                "DEFAULT_CLASS_DISTRIBUTION",
                "00_dataset_inventory/class_distribution.csv",
            ),
            (
                "02_overlap_analysis.py",
                "DEFAULT_DATASET_INDEX",
                "00_dataset_inventory/dataset_index.csv",
            ),
            (
                "02_overlap_analysis.py",
                "DEFAULT_SELECTED_INDEX",
                "02_dataset_analysis/02_sample_selection/selected_sample_index.csv",
            ),
            (
                "03_nms_threshold_sweep.py",
                "DEFAULT_SAMPLE_INDEX",
                "02_dataset_analysis/02_sample_selection/selected_sample_index.csv",
            ),
            (
                "03_nms_threshold_sweep.py",
                "DEFAULT_OVERLAP_PROFILE",
                "02_dataset_analysis/03_overlap_analysis/overlap_profile.csv",
            ),
            (
                "04_augmentation_robustness.py",
                "DEFAULT_SAMPLE_INDEX",
                "02_dataset_analysis/02_sample_selection/selected_sample_index.csv",
            ),
            (
                "05_build_hnm_components.py",
                "DEFAULT_SAMPLE_PATH",
                "02_dataset_analysis/02_sample_selection/selected_sample_index.csv",
            ),
            (
                "05_build_hnm_components.py",
                "DEFAULT_PREDICTION_PATH",
                "03_nms_thresholding/01_threshold_sweep/"
                "model2_predictions_nms_0_3_sample5000.csv",
            ),
            (
                "05_build_hnm_components.py",
                "DEFAULT_GROUND_TRUTH_PATH",
                "03_nms_thresholding/01_threshold_sweep/ground_truth_sample5000.csv",
            ),
            (
                "05_build_error_review_queues.py",
                "DEFAULT_COMPONENT_PATH",
                "05_hard_negative_mining/01_error_components/"
                "image_error_components_sample5000.csv",
            ),
        )
        for filename, constant, relative_path in expectations:
            with self.subTest(filename=filename, constant=constant):
                self.assert_contract_path(filename, constant, relative_path)

    def test_contract_document_lists_every_canonical_stage(self):
        text = (PROJECT_ROOT / "experiments" / "OUTPUTS.md").read_text(
            encoding="utf-8"
        )
        directories = (
            "00_dataset_inventory/",
            "01_model_selection/01_quality_comparison",
            "01_model_selection/02_runtime_benchmark",
            "01_model_selection/03_checkpoint_decision",
            "02_dataset_analysis/01_dataset_summary",
            "02_dataset_analysis/02_sample_selection",
            "02_dataset_analysis/03_overlap_analysis",
            "03_nms_thresholding/01_threshold_sweep",
            "04_augmentation_robustness/01_condition_evaluation",
            "05_hard_negative_mining/01_error_components",
            "05_hard_negative_mining/02_review_queues",
        )
        for directory in directories:
            with self.subTest(directory=directory):
                self.assertIn(directory, text)


if __name__ == "__main__":
    unittest.main()
