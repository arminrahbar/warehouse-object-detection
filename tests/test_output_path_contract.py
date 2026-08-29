"""Regression tests for the numbered experiment-output directory contract."""

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "experiments" / "scripts"
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "outputs"
SCRIPT_PATHS = {
    "01_build_dataset_inventory.py": Path(
        "02_dataset_analysis/01_build_dataset_inventory.py"
    ),
    "01_compare_model_quality.py": Path(
        "01_model_selection/01_compare_model_quality.py"
    ),
    "02_benchmark_inference_latency.py": Path(
        "01_model_selection/02_benchmark_inference_latency.py"
    ),
    "03_select_checkpoint.py": Path(
        "01_model_selection/03_select_checkpoint.py"
    ),
    "02_summarize_dataset.py": Path(
        "02_dataset_analysis/02_summarize_dataset.py"
    ),
    "03_select_analysis_workload.py": Path(
        "02_dataset_analysis/03_select_analysis_workload.py"
    ),
    "04_analyze_overlap.py": Path(
        "02_dataset_analysis/04_analyze_overlap.py"
    ),
    "01_sweep_nms_thresholds.py": Path(
        "03_nms_thresholding/01_sweep_nms_thresholds.py"
    ),
    "01_preview_augmentation_conditions.py": Path(
        "04_augmentation_robustness/01_preview_augmentation_conditions.py"
    ),
    "02_measure_augmentation_robustness.py": Path(
        "04_augmentation_robustness/02_measure_augmentation_robustness.py"
    ),
    "01_build_error_components.py": Path(
        "05_hard_negative_mining/01_build_error_components.py"
    ),
    "02_build_error_review_queues.py": Path(
        "05_hard_negative_mining/02_build_error_review_queues.py"
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
            "01_build_dataset_inventory.py",
            "01_compare_model_quality.py",
            "02_benchmark_inference_latency.py",
            "03_select_checkpoint.py",
            "02_summarize_dataset.py",
            "03_select_analysis_workload.py",
            "04_analyze_overlap.py",
            "01_sweep_nms_thresholds.py",
            "01_preview_augmentation_conditions.py",
            "02_measure_augmentation_robustness.py",
            "01_build_error_components.py",
            "02_build_error_review_queues.py",
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
            ("01_build_dataset_inventory.py", "DEFAULT_OUTPUT_DIR", "00_dataset_inventory"),
            (
                "01_compare_model_quality.py",
                "DEFAULT_OUTPUT_ROOT",
                "01_model_selection/01_quality_comparison",
            ),
            (
                "02_benchmark_inference_latency.py",
                "DEFAULT_OUTPUT_ROOT",
                "01_model_selection/02_runtime_benchmark",
            ),
            (
                "03_select_checkpoint.py",
                "DEFAULT_OUTPUT_ROOT",
                "01_model_selection/03_checkpoint_decision",
            ),
            (
                "02_summarize_dataset.py",
                "DEFAULT_OUTPUT_DIR",
                "02_dataset_analysis/01_dataset_summary",
            ),
            (
                "03_select_analysis_workload.py",
                "DEFAULT_OUTPUT_DIR",
                "02_dataset_analysis/02_sample_selection",
            ),
            (
                "04_analyze_overlap.py",
                "DEFAULT_OVERLAP_DIR",
                "02_dataset_analysis/03_overlap_analysis",
            ),
            (
                "01_sweep_nms_thresholds.py",
                "DEFAULT_OUTPUT_DIR",
                "03_nms_thresholding/01_threshold_sweep",
            ),
            (
                "02_measure_augmentation_robustness.py",
                "DEFAULT_OUTPUT_DIR",
                "04_augmentation_robustness/01_condition_evaluation",
            ),
            (
                "01_build_error_components.py",
                "DEFAULT_OUTPUT_DIR",
                "05_hard_negative_mining/01_error_components",
            ),
            (
                "02_build_error_review_queues.py",
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
                "01_compare_model_quality.py",
                "DEFAULT_DATASET_INDEX",
                "00_dataset_inventory/dataset_index.csv",
            ),
            (
                "02_benchmark_inference_latency.py",
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
                "03_select_analysis_workload.py",
                "DEFAULT_DATASET_INDEX",
                "00_dataset_inventory/dataset_index.csv",
            ),
            (
                "03_select_analysis_workload.py",
                "DEFAULT_CLASS_DISTRIBUTION",
                "00_dataset_inventory/class_distribution.csv",
            ),
            (
                "04_analyze_overlap.py",
                "DEFAULT_DATASET_INDEX",
                "00_dataset_inventory/dataset_index.csv",
            ),
            (
                "04_analyze_overlap.py",
                "DEFAULT_SELECTED_INDEX",
                "02_dataset_analysis/02_sample_selection/selected_sample_index.csv",
            ),
            (
                "01_sweep_nms_thresholds.py",
                "DEFAULT_SAMPLE_INDEX",
                "02_dataset_analysis/02_sample_selection/selected_sample_index.csv",
            ),
            (
                "01_sweep_nms_thresholds.py",
                "DEFAULT_OVERLAP_PROFILE",
                "02_dataset_analysis/03_overlap_analysis/overlap_profile.csv",
            ),
            (
                "02_measure_augmentation_robustness.py",
                "DEFAULT_SAMPLE_INDEX",
                "02_dataset_analysis/02_sample_selection/selected_sample_index.csv",
            ),
            (
                "01_build_error_components.py",
                "DEFAULT_SAMPLE_PATH",
                "02_dataset_analysis/02_sample_selection/selected_sample_index.csv",
            ),
            (
                "01_build_error_components.py",
                "DEFAULT_PREDICTION_PATH",
                "03_nms_thresholding/01_threshold_sweep/"
                "model2_predictions_nms_0_3_sample5000.csv",
            ),
            (
                "01_build_error_components.py",
                "DEFAULT_GROUND_TRUTH_PATH",
                "03_nms_thresholding/01_threshold_sweep/ground_truth_sample5000.csv",
            ),
            (
                "02_build_error_review_queues.py",
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
