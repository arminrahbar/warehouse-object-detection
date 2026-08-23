import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "scripts" / "03_build_report_figures.py"
spec = importlib.util.spec_from_file_location("report_figures_exp03_under_test", SCRIPT_PATH)
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


CLASS_NAMES = [
    "barcode", "car", "cardboard box", "fire", "forklift",
    "freight container", "gloves", "helmet", "ladder", "license plate",
    "person", "qr code", "road sign", "safety vest", "smoke",
    "traffic cone", "traffic light", "truck", "van", "wood pallet",
]
CLASS_COUNTS = [
    151, 737, 2632, 1466, 583, 177, 134, 1159, 153, 206,
    3272, 194, 406, 659, 795, 240, 607, 415, 394, 4816,
]
CROWDED_PREDICTIONS = {
    0.2: 3845,
    0.3: 3883,
    0.4: 3897,
    0.5: 3908,
    0.55: 3922,
    0.6: 3937,
    0.7: 3973,
}


class EvidenceFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True)
        self._write_summary()
        self._write_duplicates()
        self._write_subsets()
        self._write_per_class()

    def _write_summary(self):
        rows = []
        for threshold in figures.THRESHOLDS:
            predictions = figures.CANONICAL_PREDICTIONS[threshold]
            rows.append(
                {
                    "model": figures.MODEL_NAME,
                    "dataset": figures.DATASET_NAME,
                    "nms_threshold": threshold,
                    "mAP@0.5_11_point": figures.CANONICAL_MAP[threshold],
                    "total_ground_truth": figures.EXPECTED_LABELS,
                    "total_predictions_after_nms": predictions,
                    "evaluation_rows": predictions,
                    "score_threshold": 0.5,
                    "map_iou_threshold": 0.5,
                    "eval_type": "combined",
                }
            )
        pd.DataFrame(rows, columns=figures.SUMMARY_COLUMNS).to_csv(
            self.root / figures.SUMMARY_NAME, index=False
        )

    def _write_duplicates(self):
        rows = []
        for threshold in figures.THRESHOLDS:
            pairs = figures.CANONICAL_DUPLICATE_PAIRS[threshold]
            rows.append(
                {
                    "duplicate_like_pairs_iou_gt_0_5": pairs,
                    "images_with_duplicate_like_pairs": figures.CANONICAL_DUPLICATE_IMAGES[threshold],
                    "mean_duplicate_like_pairs_per_image": pairs / figures.EXPECTED_DETECTED_IMAGES,
                    "nms_threshold": threshold,
                    "total_predictions_after_nms": figures.CANONICAL_PREDICTIONS[threshold],
                }
            )
        pd.DataFrame(rows, columns=figures.DUPLICATE_COLUMNS).to_csv(
            self.root / figures.DUPLICATE_NAME, index=False
        )

    def _write_subsets(self):
        rows = []
        for threshold in figures.THRESHOLDS:
            full_predictions = figures.CANONICAL_PREDICTIONS[threshold]
            pairs = figures.CANONICAL_DUPLICATE_PAIRS[threshold]
            rows.append(
                {
                    "subset_name": "all_selected",
                    "image_count": figures.EXPECTED_IMAGES,
                    "ground_truth_count": figures.EXPECTED_LABELS,
                    "nms_threshold": threshold,
                    "mAP@0.5_11_point": figures.CANONICAL_MAP[threshold],
                    "total_predictions_after_nms": full_predictions,
                    "evaluation_rows": full_predictions,
                    "duplicate_like_pairs_iou_gt_0_5": pairs,
                    "images_with_duplicate_like_pairs": figures.CANONICAL_DUPLICATE_IMAGES[threshold],
                    "mean_duplicate_like_pairs_per_image": pairs / figures.EXPECTED_DETECTED_IMAGES,
                }
            )
            crowded_predictions = CROWDED_PREDICTIONS[threshold]
            rows.append(
                {
                    "subset_name": "crowded_any_overlap",
                    "image_count": figures.EXPECTED_CROWDED_IMAGES,
                    "ground_truth_count": figures.EXPECTED_CROWDED_LABELS,
                    "nms_threshold": threshold,
                    "mAP@0.5_11_point": figures.CANONICAL_CROWDED_MAP[threshold],
                    "total_predictions_after_nms": crowded_predictions,
                    "evaluation_rows": crowded_predictions,
                    "duplicate_like_pairs_iou_gt_0_5": 0,
                    "images_with_duplicate_like_pairs": 0,
                    "mean_duplicate_like_pairs_per_image": 0.0,
                }
            )
        pd.DataFrame(rows, columns=figures.SUBSET_COLUMNS).to_csv(
            self.root / figures.SUBSET_NAME, index=False
        )

    def _write_per_class(self):
        rows = []
        for threshold in figures.THRESHOLDS:
            total_predictions = figures.CANONICAL_PREDICTIONS[threshold]
            prediction_counts = [1] * 19 + [total_predictions - 19]
            for class_id, (name, support, predictions) in enumerate(
                zip(CLASS_NAMES, CLASS_COUNTS, prediction_counts)
            ):
                rows.append(
                    {
                        "model": figures.MODEL_NAME,
                        "dataset": figures.DATASET_NAME,
                        "nms_threshold": threshold,
                        "class_id": class_id,
                        "class_name": name,
                        "ground_truth_count": support,
                        "prediction_count": predictions,
                        "ap_11_point": figures.CANONICAL_MAP[threshold],
                    }
                )
        pd.DataFrame(rows, columns=figures.PER_CLASS_COLUMNS).to_csv(
            self.root / figures.PER_CLASS_NAME, index=False
        )


class PublicationFigureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.evidence_dir = self.root / "evidence"
        self.fixture = EvidenceFixture(self.evidence_dir)

    def tearDown(self):
        self.temporary.cleanup()

    def load(self):
        return figures.load_verified_evidence(
            self.evidence_dir,
            locked_hashes=None,
        )

    def test_locked_schemas_and_cross_table_invariants(self):
        evidence = self.load()

        self.assertEqual(len(evidence["summary"]), 7)
        self.assertEqual(len(evidence["duplicates"]), 7)
        self.assertEqual(len(evidence["subsets"]), 14)
        self.assertEqual(len(evidence["per_class"]), 140)
        self.assertEqual(
            evidence["summary"]["nms_threshold"].tolist(),
            list(figures.THRESHOLDS),
        )

    def test_schema_change_is_rejected(self):
        path = self.evidence_dir / figures.SUMMARY_NAME
        table = pd.read_csv(path).rename(columns={"eval_type": "score_type"})
        table.to_csv(path, index=False)

        with self.assertRaisesRegex(Exception, "schema"):
            self.load()

    def test_exact_operating_point_facts_are_hard_gates(self):
        evidence = self.load()
        summary = evidence["summary"]

        selected = summary[summary["nms_threshold"] == 0.3].iloc[0]
        permissive = summary[summary["nms_threshold"] == 0.7].iloc[0]
        self.assertAlmostEqual(selected["mAP@0.5_11_point"], 0.40157294334629645)
        self.assertEqual(int(selected["total_predictions_after_nms"]), 7727)
        self.assertEqual(int(permissive["total_predictions_after_nms"]), 8032)
        self.assertAlmostEqual(
            100 * (permissive["mAP@0.5_11_point"] - selected["mAP@0.5_11_point"]),
            -0.441391040745009,
        )
        self.assertEqual(figures.pareto_thresholds(summary), (0.2, 0.3))

    def test_changed_canonical_value_is_rejected(self):
        path = self.evidence_dir / figures.SUMMARY_NAME
        table = pd.read_csv(path)
        table.loc[table["nms_threshold"] == 0.3, "mAP@0.5_11_point"] = 0.5
        table.to_csv(path, index=False)

        with self.assertRaisesRegex(Exception, "locked evidence|nominal maximum"):
            self.load()

    def test_class_impact_is_deterministic(self):
        table = pd.DataFrame(
            [
                {"class_id": 0, "class_name": "alpha", "ground_truth_count": 10,
                 "prediction_count": 8, "ap_11_point": 0.7, "nms_threshold": 0.3},
                {"class_id": 1, "class_name": "beta", "ground_truth_count": 20,
                 "prediction_count": 9, "ap_11_point": 0.5, "nms_threshold": 0.3},
                {"class_id": 0, "class_name": "alpha", "ground_truth_count": 10,
                 "prediction_count": 9, "ap_11_point": 0.6, "nms_threshold": 0.7},
                {"class_id": 1, "class_name": "beta", "ground_truth_count": 20,
                 "prediction_count": 10, "ap_11_point": 0.5, "nms_threshold": 0.7},
            ]
        )

        impact = figures.derive_class_impact(table)
        self.assertEqual(impact["class_name"].tolist(), ["alpha", "beta"])
        self.assertEqual(impact["delta_pp"].round(8).tolist(), [-10.0, 0.0])

    def test_fixed_png_svg_output_set_and_atomic_refusal(self):
        destination = figures.build_figure_package(self.load(), self.root / "figures")
        expected = {
            f"{stem}.{extension}"
            for stem in figures.OUTPUT_STEMS
            for extension in ("png", "svg")
        }

        self.assertEqual({path.name for path in destination.iterdir()}, expected)
        self.assertFalse((self.root / ".figures.incomplete").exists())
        self.assertTrue(all(path.stat().st_size > 1000 for path in destination.iterdir()))

        with self.assertRaisesRegex(Exception, "Refusing to overwrite"):
            figures.build_figure_package(self.load(), destination)
        self.assertEqual({path.name for path in destination.iterdir()}, expected)


class ParserTests(unittest.TestCase):
    def test_evidence_default_follows_numbered_experiment_layout(self):
        args = figures.build_parser().parse_args(["--output-dir", "figures"])
        self.assertEqual(
            args.evidence_dir,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "03_nms_thresholding"
            / "01_threshold_sweep",
        )
        self.assertEqual(args.output_dir, Path("figures"))

    def test_output_directory_is_required(self):
        with self.assertRaises(SystemExit):
            figures.build_parser().parse_args([])

    def test_required_paths_are_explicit(self):
        args = figures.build_parser().parse_args(
            ["--evidence-dir", "evidence", "--output-dir", "figures"]
        )

        self.assertEqual(args.evidence_dir, Path("evidence"))
        self.assertEqual(args.output_dir, Path("figures"))


if __name__ == "__main__":
    unittest.main()
