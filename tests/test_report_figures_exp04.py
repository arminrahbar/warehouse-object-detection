"""Integrity and atomic-output tests for Experiment 04 report figures."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "scripts" / "04_build_report_figures.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


figures = load_module(SCRIPT_PATH, "report_figures_exp04_under_test")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Experiment04ReportFigureTests(unittest.TestCase):
    def test_evidence_defaults_follow_numbered_experiment_layout(self):
        args = figures.build_parser().parse_args(["--output-dir", "figures"])
        output_root = PROJECT_ROOT / "experiments" / "outputs"
        self.assertEqual(
            args.robustness_dir,
            output_root
            / "04_augmentation_robustness"
            / "01_condition_evaluation",
        )
        self.assertEqual(
            args.sample_index,
            output_root
            / "02_dataset_analysis"
            / "02_sample_selection"
            / "selected_sample_index.csv",
        )
        self.assertEqual(
            args.nms_summary,
            output_root
            / "03_nms_thresholding"
            / "01_threshold_sweep"
            / "nms_threshold_summary_sample5000.csv",
        )
        self.assertEqual(args.output_dir, Path("figures"))

    def test_output_directory_is_required(self):
        with self.assertRaises(SystemExit):
            figures.build_parser().parse_args([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.robustness_dir = self.root / "robustness"
        self.robustness_dir.mkdir()
        self.asset_root = self.root / "assets"
        (self.asset_root / "logistics").mkdir(parents=True)
        self.sample_path = self.root / "selected_sample_index.csv"
        self.nms_path = self.root / "nms_summary.csv"
        self.expected_images = 4
        self.expected_labels = 6
        self.expected_classes = 3
        self.expected_source_groups = 4
        self._write_fixture()

    def tearDown(self):
        self.temporary.cleanup()

    def _sample_rows(self):
        rows = []
        object_counts = [1, 2, 1, 2]
        for index, object_count in enumerate(object_counts):
            image_file = f"source-{index}_jpg.rf.synthetic-{index}.jpg"
            image_path = f"detector_service/storage/logistics/{image_file}"
            row = {column: 0 for column in figures.SAMPLE_COLUMNS}
            row.update(
                {
                    "image_path": image_path,
                    "label_path": image_path.replace(".jpg", ".txt"),
                    "image_file": image_file,
                    "label_file": image_file.replace(".jpg", ".txt"),
                    "num_objects": object_count,
                    "class_ids_present": "[0]",
                    "class_names_present": '["alpha"]',
                    "density_bucket": "1-4",
                }
            )
            rows.append(row)
        return rows

    def _write_fixture(self):
        sample = pd.DataFrame(self._sample_rows(), columns=figures.SAMPLE_COLUMNS)
        sample.to_csv(self.sample_path, index=False)
        selected = figures.select_example(sample)
        image_path = self.asset_root / "logistics" / selected["image_file"]
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[:, :, 0] = np.linspace(20, 220, image.shape[1], dtype=np.uint8)
        image[:, :, 1] = 110
        image[:, :, 2] = np.linspace(220, 20, image.shape[1], dtype=np.uint8)
        Image.fromarray(image[:, :, ::-1]).save(image_path)

        condition_values = {
            "original": ([0.40, 0.50, 0.60], 4),
            "gaussian_blur_k9": ([0.20, 0.30, 0.40], 2),
            "vertical_flip": ([0.10, 0.20, 0.30], 1),
            "brightness_increase": ([0.39, 0.48, 0.58], 4),
            "brightness_decrease": ([0.38, 0.47, 0.56], 3),
        }
        baseline_map = float(np.mean(condition_values["original"][0]))
        baseline_predictions = condition_values["original"][1]
        summary_rows = []
        per_class_rows = []
        supports = [2, 2, 2]
        class_names = ["alpha", "beta", "gamma"]
        for condition in figures.robustness.CONDITIONS:
            tag = condition["tag"]
            values, prediction_count = condition_values[tag]
            map_score = float(np.mean(values))
            summary_rows.append(
                {
                    "model": "model2",
                    "dataset": "rare_aware_density_stratified_5000",
                    "augmentation_condition": tag,
                    "augmentation_display": condition["display"],
                    "mAP@0.5_11_point": map_score,
                    "total_ground_truth": self.expected_labels,
                    "total_predictions_after_nms": prediction_count,
                    "evaluation_rows": prediction_count,
                    "candidate_objectness_threshold": 0.5,
                    "nms_confidence_threshold": 0.5,
                    "nms_iou_threshold": 0.3,
                    "map_iou_threshold": 0.5,
                    "eval_type": "combined",
                    "mAP_change_vs_original": map_score - baseline_map,
                    "mAP_percent_change_vs_original": (map_score - baseline_map) / baseline_map * 100.0,
                    "prediction_change_vs_original": prediction_count - baseline_predictions,
                }
            )
            for class_id, (class_name, support, ap) in enumerate(
                zip(class_names, supports, values)
            ):
                original_ap = condition_values["original"][0][class_id]
                per_class_rows.append(
                    {
                        "model": "model2",
                        "dataset": "rare_aware_density_stratified_5000",
                        "augmentation_condition": tag,
                        "augmentation_display": condition["display"],
                        "class_id": class_id,
                        "class_name": class_name,
                        "ground_truth_count": support,
                        "prediction_count": 0,
                        "ap_11_point": ap,
                        "original_ap_11_point": original_ap,
                        "ap_change_vs_original": ap - original_ap,
                    }
                )
        summary = pd.DataFrame(summary_rows, columns=figures.robustness.SUMMARY_COLUMNS)
        per_class = pd.DataFrame(per_class_rows, columns=figures.robustness.PER_CLASS_COLUMNS)
        summary.to_csv(self.robustness_dir / "summary_by_condition_sample5000.csv", index=False)
        per_class.to_csv(self.robustness_dir / "per_class_ap_by_condition_sample5000.csv", index=False)

        sample_files = sample["image_file"].tolist()
        for condition in figures.CONDITION_ORDER:
            row_count = int(
                summary.loc[
                    summary["augmentation_condition"] == condition,
                    "total_predictions_after_nms",
                ].iloc[0]
            )
            rows = []
            for position in range(row_count):
                class_id = position % self.expected_classes
                scores = [0.05] * self.expected_classes
                scores[class_id] = 0.8
                rows.append(
                    {
                        "model": "model2",
                        "dataset": "rare_aware_density_stratified_5000",
                        "augmentation_condition": condition,
                        "augmentation_display": figures.CONDITION_LABELS[condition],
                        "image_file": sample_files[position % len(sample_files)],
                        "image_path": f"detector_service/storage/logistics/{sample_files[position % len(sample_files)]}",
                        "bbox_x": 1.0,
                        "bbox_y": 2.0,
                        "bbox_w": 3.0,
                        "bbox_h": 4.0,
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "object_score": 0.8,
                        "predicted_class_score": 0.8,
                        "combined_confidence": 0.64,
                        "class_scores_json": json.dumps(scores),
                        "nms_threshold": 0.3,
                    }
                )
            pd.DataFrame(rows, columns=figures.robustness.PREDICTION_COLUMNS).to_csv(
                self.robustness_dir / figures._prediction_filename(condition),
                index=False,
            )

        nms_rows = []
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7]:
            nms_rows.append(
                {
                    "model": "model2",
                    "dataset": "rare_aware_density_stratified_5000",
                    "nms_threshold": threshold,
                    "mAP@0.5_11_point": baseline_map if threshold == 0.3 else baseline_map - 0.01,
                    "total_ground_truth": self.expected_labels,
                    "total_predictions_after_nms": baseline_predictions if threshold == 0.3 else baseline_predictions + 1,
                    "evaluation_rows": baseline_predictions if threshold == 0.3 else baseline_predictions + 1,
                    "score_threshold": 0.5,
                    "map_iou_threshold": 0.5,
                    "eval_type": "combined",
                }
            )
        pd.DataFrame(nms_rows, columns=figures.NMS_COLUMNS).to_csv(self.nms_path, index=False)
        self.locked_hashes = {
            "sample_index": sha256(self.sample_path),
            "nms_summary": sha256(self.nms_path),
            "summary": sha256(self.robustness_dir / "summary_by_condition_sample5000.csv"),
            "per_class": sha256(self.robustness_dir / "per_class_ap_by_condition_sample5000.csv"),
        }
        for condition in figures.CONDITION_ORDER:
            self.locked_hashes[f"predictions_{condition}"] = sha256(
                self.robustness_dir / figures._prediction_filename(condition)
            )

    def load_evidence(self):
        return figures.load_verified_evidence(
            self.robustness_dir,
            self.sample_path,
            self.nms_path,
            locked_hashes=self.locked_hashes,
            expected_images=self.expected_images,
            expected_labels=self.expected_labels,
            expected_classes=self.expected_classes,
            expected_source_groups=self.expected_source_groups,
        )

    def test_verified_fixture_locks_schemas_counts_and_baseline(self):
        evidence = self.load_evidence()
        self.assertEqual(len(evidence["summary"]), 5)
        self.assertEqual(len(evidence["per_class"]), 15)
        self.assertEqual(evidence["sample"]["image_file"].nunique(), 4)
        original = evidence["summary"].set_index("augmentation_condition").loc["original"]
        selected = evidence["nms"].loc[np.isclose(evidence["nms"]["nms_threshold"], 0.3)].iloc[0]
        self.assertAlmostEqual(original["mAP@0.5_11_point"], selected["mAP@0.5_11_point"])
        self.assertEqual(original["total_predictions_after_nms"], selected["total_predictions_after_nms"])

    def test_rejects_experiment03_baseline_drift_even_with_updated_hash(self):
        nms = pd.read_csv(self.nms_path)
        nms.loc[np.isclose(nms["nms_threshold"], 0.3), "mAP@0.5_11_point"] += 0.02
        nms.to_csv(self.nms_path, index=False)
        self.locked_hashes["nms_summary"] = sha256(self.nms_path)
        with self.assertRaisesRegex(figures.FigureBuildError, "Experiment 03/04 baseline"):
            self.load_evidence()

    def test_rejects_schema_drift(self):
        summary_path = self.robustness_dir / "summary_by_condition_sample5000.csv"
        summary = pd.read_csv(summary_path)
        summary = summary.rename(columns={"eval_type": "score_type"})
        summary.to_csv(summary_path, index=False)
        self.locked_hashes["summary"] = sha256(summary_path)
        with self.assertRaisesRegex(figures.FigureBuildError, "schema"):
            self.load_evidence()

    def test_class_breadth_uses_declared_one_point_threshold(self):
        evidence = self.load_evidence()
        counts = figures.class_impact_counts(evidence["per_class"])
        lookup = counts.set_index("condition")
        self.assertEqual(int(lookup.loc["brightness_increase", "drop"]), 2)
        self.assertEqual(int(lookup.loc["gaussian_blur_k9", "drop"]), 3)
        self.assertTrue((counts["gain"] == 0).all())

    def test_exact_output_set_and_existing_destination_refusal(self):
        evidence = self.load_evidence()
        output = self.root / "publication-package"
        destination = figures.build_figure_package(evidence, self.asset_root, output)
        expected = {
            f"{stem}.{extension}"
            for stem in figures.OUTPUT_STEMS
            for extension in ("png", "svg")
        }
        self.assertEqual({path.name for path in destination.iterdir()}, expected)
        for path in destination.iterdir():
            self.assertGreater(path.stat().st_size, 0)
        with self.assertRaisesRegex(figures.FigureBuildError, "Refusing to overwrite"):
            figures.build_figure_package(evidence, self.asset_root, output)


if __name__ == "__main__":
    unittest.main()
