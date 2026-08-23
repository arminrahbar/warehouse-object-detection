import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "scripts" / "02_build_report_figures.py"
spec = importlib.util.spec_from_file_location("report_figures_exp02_under_test", SCRIPT_PATH)
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EvidenceFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_dir = self.root / "index"
        self.summary_dir = self.root / "summary"
        self.sampling_dir = self.root / "sampling"
        self.overlap_dir = self.root / "overlap"
        for directory in (
            self.index_dir,
            self.summary_dir,
            self.sampling_dir,
            self.overlap_dir,
        ):
            directory.mkdir()

        self.class_names = ["alpha", "beta", "gamma", "delta"]
        self.count_columns = [f"count_{name}" for name in self.class_names]
        self._write_index()
        self._write_summary()
        self._write_sampling()
        self._write_overlap()
        self.locked_hashes = self._hashes()

    @staticmethod
    def _write(table, path):
        table.to_csv(path, index=False)

    def _write_index(self):
        counts = [
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 5, 0],
            [0, 0, 0, 10],
            [5, 10, 0, 0],
            [0, 0, 8, 12],
        ]
        rows = []
        for number, values in enumerate(counts, start=1):
            present = [index for index, value in enumerate(values) if value]
            row = {
                "image_path": f"logistics/image_{number:02d}.jpg",
                "label_path": f"logistics/image_{number:02d}.txt",
                "image_file": f"image_{number:02d}.jpg",
                "label_file": f"image_{number:02d}.txt",
                "num_objects": sum(values),
                "class_ids_present": str(present),
                "class_names_present": str([self.class_names[index] for index in present]),
            }
            row.update(dict(zip(self.count_columns, values)))
            rows.append(row)
        self.index = pd.DataFrame(rows, columns=figures.BASE_INDEX_COLUMNS + self.count_columns)
        self._write(self.index, self.index_dir / "dataset_index.csv")

        class_rows = []
        for class_id, (class_name, column) in enumerate(
            zip(self.class_names, self.count_columns)
        ):
            class_rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "object_count": int(self.index[column].sum()),
                    "image_count": int((self.index[column] > 0).sum()),
                }
            )
        self.classes = pd.DataFrame(class_rows, columns=figures.CLASS_COLUMNS)
        self._write(self.classes, self.index_dir / "class_distribution.csv")
        distribution = (
            self.index["num_objects"].value_counts().sort_index().rename_axis("num_objects")
            .reset_index(name="image_count")
        )
        self._write(distribution, self.index_dir / "object_count_distribution.csv")

    def _write_summary(self):
        counts = self.index["num_objects"]
        summary = pd.DataFrame(
            [
                {
                    "dataset": "full_dataset",
                    "images": len(self.index),
                    "total_objects": int(counts.sum()),
                    "images_with_zero_objects": int((counts == 0).sum()),
                    "mean_objects_per_image": round(float(counts.mean()), 3),
                    "median_objects_per_image": float(counts.median()),
                    "max_objects_per_image": int(counts.max()),
                    "images_ge_5_objects": int((counts >= 5).sum()),
                    "images_ge_10_objects": int((counts >= 10).sum()),
                    "images_ge_15_objects": int((counts >= 15).sum()),
                    "images_ge_20_objects": int((counts >= 20).sum()),
                }
            ],
            columns=figures.SUMMARY_COLUMNS,
        )
        self._write(summary, self.summary_dir / "full_dataset_summary.csv")

        total = int(counts.sum())
        enriched = self.classes.copy()
        enriched["object_share_pct"] = (
            100.0 * enriched["object_count"] / total
        ).round(4)
        enriched["image_share_pct"] = (
            100.0 * enriched["image_count"] / len(self.index)
        ).round(4)
        self._write(enriched, self.summary_dir / "class_distribution_enriched.csv")
        top = enriched.sort_values(
            ["object_count", "class_id"], ascending=[False, True], kind="mergesort"
        )
        bottom = enriched.sort_values(
            ["object_count", "class_id"], ascending=[True, True], kind="mergesort"
        )
        self._write(top, self.summary_dir / "top10_classes_by_object_count.csv")
        self._write(bottom, self.summary_dir / "bottom10_classes_by_object_count.csv")

        buckets = self.index["num_objects"].map(figures._density_bucket).value_counts()
        density = pd.DataFrame(
            {
                "density_bucket": figures.DENSITY_BUCKET_ORDER,
                "image_count": [int(buckets.get(bucket, 0)) for bucket in figures.DENSITY_BUCKET_ORDER],
            }
        )
        density["image_share_pct"] = (
            100.0 * density["image_count"] / len(self.index)
        ).round(4)
        self._write(density, self.summary_dir / "density_bucket_distribution.csv")
        dense = pd.DataFrame(
            [
                {
                    "threshold": f">={threshold} objects",
                    "image_count": int((counts >= threshold).sum()),
                    "image_share_pct": round(100.0 * int((counts >= threshold).sum()) / len(counts), 4),
                }
                for threshold in (5, 10, 15, 20)
            ],
            columns=figures.DENSE_COUNT_COLUMNS,
        )
        self._write(dense, self.summary_dir / "dense_image_counts.csv")

    def _distribution(self, table, dataset):
        total = int(table["num_objects"].sum())
        rows = []
        for class_id, (class_name, column) in enumerate(
            zip(self.class_names, self.count_columns)
        ):
            objects = int(table[column].sum())
            images = int((table[column] > 0).sum())
            rows.append(
                {
                    "dataset": dataset,
                    "class_id": class_id,
                    "class_name": class_name,
                    "object_count": objects,
                    "image_count": images,
                    "object_share_pct": 100.0 * objects / total,
                    "image_share_pct": 100.0 * images / len(table),
                }
            )
        return pd.DataFrame(rows)

    def _density(self, table, dataset):
        counts = table["num_objects"].map(figures._density_bucket).value_counts()
        return pd.DataFrame(
            {
                "dataset": dataset,
                "density_bucket": figures.DENSITY_BUCKET_ORDER,
                "image_count": [int(counts.get(bucket, 0)) for bucket in figures.DENSITY_BUCKET_ORDER],
                "image_share_pct": [
                    100.0 * int(counts.get(bucket, 0)) / len(table)
                    for bucket in figures.DENSITY_BUCKET_ORDER
                ],
            }
        )

    def _write_sampling(self):
        self.selected = self.index.iloc[[0, 1, 4, 5]].copy().reset_index(drop=True)
        self.selected["density_bucket"] = self.selected["num_objects"].map(
            figures._density_bucket
        )
        self._write(self.selected, self.sampling_dir / "selected_sample_index.csv")

        full_class = self._distribution(self.index, "full_dataset")
        selected_class = self._distribution(self.selected, figures.SELECTED_NAME)
        comparison = full_class.merge(
            selected_class,
            on=["class_id", "class_name"],
            suffixes=("_full", "_sample"),
            validate="one_to_one",
        )
        comparison["object_share_diff_pp"] = (
            comparison["object_share_pct_sample"] - comparison["object_share_pct_full"]
        )
        comparison["image_share_diff_pp"] = (
            comparison["image_share_pct_sample"] - comparison["image_share_pct_full"]
        )
        comparison = comparison[figures.CLASS_COMPARISON_COLUMNS]
        self.class_comparison = comparison
        self._write(comparison, self.sampling_dir / "class_distribution_comparison.csv")

        rare_targets = self.classes.sort_values(
            ["object_count", "class_id"], ascending=[True, True], kind="mergesort"
        ).head(2).copy()
        rare_targets["target_image_count"] = rare_targets["image_count"]
        rare_targets = rare_targets[figures.RARE_TARGET_COLUMNS]
        self._write(rare_targets, self.sampling_dir / "rare_class_targets.csv")
        rare_coverage = comparison[
            comparison["class_name"].isin(set(rare_targets["class_name"]))
        ].copy()
        rare_coverage = rare_coverage.merge(
            rare_targets[["class_name", "target_image_count"]],
            on="class_name",
            validate="one_to_one",
        )
        rare_coverage["sample_image_retention_pct"] = (
            100.0 * rare_coverage["image_count_sample"] / rare_coverage["image_count_full"]
        )
        rare_coverage = rare_coverage[figures.RARE_COVERAGE_COLUMNS]
        self.rare_coverage = rare_coverage
        self._write(rare_coverage, self.sampling_dir / "rare_class_coverage.csv")

        full_density = self._density(self.index, "full_dataset")
        selected_density = self._density(self.selected, figures.SELECTED_NAME)
        density_comparison = full_density.merge(
            selected_density,
            on="density_bucket",
            suffixes=("_full", "_sample"),
            validate="one_to_one",
        )
        density_comparison["image_share_diff_pp"] = (
            density_comparison["image_share_pct_sample"]
            - density_comparison["image_share_pct_full"]
        )
        density_comparison = density_comparison[figures.DENSITY_COMPARISON_COLUMNS]
        self.density_comparison = density_comparison
        self._write(
            density_comparison,
            self.sampling_dir / "density_distribution_comparison.csv",
        )

        full_counts = self.index["num_objects"]
        selected_counts = self.selected["num_objects"]
        sample_summary = pd.DataFrame(
            [
                {
                    "dataset": "full_dataset",
                    "images": len(self.index),
                    "total_objects": int(full_counts.sum()),
                    "mean_objects_per_image": round(float(full_counts.mean()), 3),
                    "median_objects_per_image": float(full_counts.median()),
                    "max_objects_per_image": int(full_counts.max()),
                    "images_ge_5_objects": int((full_counts >= 5).sum()),
                    "images_ge_10_objects": int((full_counts >= 10).sum()),
                    "images_ge_15_objects": int((full_counts >= 15).sum()),
                    "images_ge_20_objects": int((full_counts >= 20).sum()),
                },
                {
                    "dataset": figures.SELECTED_NAME,
                    "images": len(self.selected),
                    "total_objects": int(selected_counts.sum()),
                    "mean_objects_per_image": round(float(selected_counts.mean()), 3),
                    "median_objects_per_image": float(selected_counts.median()),
                    "max_objects_per_image": int(selected_counts.max()),
                    "images_ge_5_objects": int((selected_counts >= 5).sum()),
                    "images_ge_10_objects": int((selected_counts >= 10).sum()),
                    "images_ge_15_objects": int((selected_counts >= 15).sum()),
                    "images_ge_20_objects": int((selected_counts >= 20).sum()),
                },
            ],
            columns=figures.SAMPLE_SUMMARY_COLUMNS,
        )
        self._write(sample_summary, self.sampling_dir / "sample_summary.csv")

        class_error = comparison["object_share_diff_pp"].abs()
        density_error = density_comparison["image_share_diff_pp"].abs()
        min_retention = float(rare_coverage["sample_image_retention_pct"].min())
        candidate_rows = [
            {
                "sample_name": figures.CANDIDATE_ORDER[0],
                "images": 4,
                "total_objects": 34,
                "mean_objects_per_image": 8.5,
                "class_object_share_mae_pp": 0.25,
                "class_object_share_max_error_pp": 0.60,
                "density_share_mae_pp": 0.20,
                "density_share_max_error_pp": 0.50,
                "min_rare_class_image_retention_pct": 50.0,
                "images_ge_10_objects": 2,
                "images_ge_20_objects": 1,
            },
            {
                "sample_name": figures.CANDIDATE_ORDER[1],
                "images": 4,
                "total_objects": 37,
                "mean_objects_per_image": 9.25,
                "class_object_share_mae_pp": 0.10,
                "class_object_share_max_error_pp": 0.30,
                "density_share_mae_pp": 0.01,
                "density_share_max_error_pp": 0.02,
                "min_rare_class_image_retention_pct": 50.0,
                "images_ge_10_objects": 2,
                "images_ge_20_objects": 1,
            },
            {
                "sample_name": figures.CANDIDATE_ORDER[2],
                "images": len(self.selected),
                "total_objects": int(selected_counts.sum()),
                "mean_objects_per_image": round(float(selected_counts.mean()), 3),
                "class_object_share_mae_pp": round(float(class_error.mean()), 4),
                "class_object_share_max_error_pp": round(float(class_error.max()), 4),
                "density_share_mae_pp": round(float(density_error.mean()), 4),
                "density_share_max_error_pp": round(float(density_error.max()), 4),
                "min_rare_class_image_retention_pct": round(min_retention, 2),
                "images_ge_10_objects": int((selected_counts >= 10).sum()),
                "images_ge_20_objects": int((selected_counts >= 20).sum()),
            },
        ]
        self._write(
            pd.DataFrame(candidate_rows, columns=figures.CANDIDATE_COLUMNS),
            self.sampling_dir / "candidate_sample_quality.csv",
        )

    def _write_overlap(self):
        overlap_counts = [
            (0, 0, 0, 0.0, 0.0),
            (1, 0, 0, 0.2, 0.2),
            (3, 2, 1, 0.6, 0.15),
            (6, 3, 1, 0.7, 0.20),
            (20, 10, 5, 0.8, 0.25),
            (30, 15, 8, 0.9, 0.30),
        ]
        rows = []
        for source, overlap in zip(self.index.itertuples(index=False), overlap_counts):
            gt01, gt03, gt05, max_iou, mean_iou = overlap
            pair_count = int(source.num_objects * (source.num_objects - 1) // 2)
            rows.append(
                {
                    "image_file": source.image_file,
                    "image_path": source.image_path,
                    "label_path": source.label_path,
                    "num_objects": source.num_objects,
                    "pair_count": pair_count,
                    "max_pairwise_iou": max_iou,
                    "mean_pairwise_iou": mean_iou,
                    "pairs_iou_gt_0_1": gt01,
                    "pairs_iou_gt_0_3": gt03,
                    "pairs_iou_gt_0_5": gt05,
                    "crowding_bucket": figures._crowding_bucket(gt01),
                }
            )
        profile = pd.DataFrame(rows, columns=figures.OVERLAP_PROFILE_COLUMNS)
        self.profile = profile
        self._write(profile, self.overlap_dir / "overlap_profile.csv")
        selected_profile = profile[
            profile["image_file"].isin(set(self.selected["image_file"]))
        ]

        def summary_row(table, label):
            return {
                "dataset": label,
                "images": len(table),
                "mean_pair_count": round(float(table["pair_count"].mean()), 3),
                "mean_max_pairwise_iou": round(float(table["max_pairwise_iou"].mean()), 4),
                "mean_pairs_iou_gt_0_1": round(float(table["pairs_iou_gt_0_1"].mean()), 3),
                "images_with_any_iou_gt_0_1": int((table["pairs_iou_gt_0_1"] > 0).sum()),
                "images_with_any_iou_gt_0_3": int((table["pairs_iou_gt_0_3"] > 0).sum()),
                "images_with_any_iou_gt_0_5": int((table["pairs_iou_gt_0_5"] > 0).sum()),
                "images_with_20plus_iou_gt_0_1_pairs": int(
                    (table["pairs_iou_gt_0_1"] >= 20).sum()
                ),
            }

        summary = pd.DataFrame(
            [summary_row(profile, "full_dataset"), summary_row(selected_profile, figures.SELECTED_NAME)],
            columns=figures.OVERLAP_SUMMARY_COLUMNS,
        )
        self._write(summary, self.overlap_dir / "overlap_summary.csv")

        def crowding(table, label):
            counts = table["crowding_bucket"].value_counts()
            return pd.DataFrame(
                {
                    "dataset": label,
                    "crowding_bucket": figures.CROWDING_BUCKET_ORDER,
                    "image_count": [int(counts.get(bucket, 0)) for bucket in figures.CROWDING_BUCKET_ORDER],
                    "image_share_pct": [
                        100.0 * int(counts.get(bucket, 0)) / len(table)
                        for bucket in figures.CROWDING_BUCKET_ORDER
                    ],
                }
            )

        full = crowding(profile, "full_dataset")
        sample = crowding(selected_profile, figures.SELECTED_NAME)
        comparison = full.merge(
            sample,
            on="crowding_bucket",
            suffixes=("_full", "_sample"),
            validate="one_to_one",
        )
        comparison["image_share_diff_pp"] = (
            comparison["image_share_pct_sample"] - comparison["image_share_pct_full"]
        )
        comparison = comparison[figures.CROWDING_COMPARISON_COLUMNS]
        self._write(
            comparison,
            self.overlap_dir / "crowding_distribution_comparison.csv",
        )

    def _hashes(self):
        result = {}
        for group, directory, names in (
            ("index", self.index_dir, figures.INDEX_FILES),
            ("summary", self.summary_dir, figures.SUMMARY_FILES),
            ("sampling", self.sampling_dir, figures.SAMPLING_FILES),
            ("overlap", self.overlap_dir, figures.OVERLAP_FILES),
        ):
            for name in names:
                result[f"{group}/{name}"] = sha256(directory / name)
        return result


class Experiment02FigureTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = EvidenceFixture(self.root / "evidence")

    def tearDown(self):
        self.temporary.cleanup()

    def load(self, locked_hashes=None):
        return figures.load_verified_evidence(
            self.fixture.index_dir,
            self.fixture.summary_dir,
            self.fixture.sampling_dir,
            self.fixture.overlap_dir,
            expected_images=6,
            expected_labels=53,
            expected_classes=4,
            expected_selected_images=4,
            expected_selected_labels=38,
            expected_rare_classes=2,
            locked_hashes=(
                self.fixture.locked_hashes if locked_hashes is None else locked_hashes
            ),
        )


class EvidenceValidationTests(Experiment02FigureTestCase):
    def test_all_four_evidence_layers_reconcile(self):
        evidence = self.load()

        self.assertEqual(len(evidence["base"]["index"]), 6)
        self.assertEqual(len(evidence["sampling"]["selected"]), 4)
        self.assertEqual(len(evidence["overlap"]["selected_profile"]), 4)
        self.assertEqual(len(evidence["hashes"]), 19)

    def test_locked_hash_tampering_is_rejected_before_semantic_loading(self):
        path = self.fixture.sampling_dir / "candidate_sample_quality.csv"
        path.write_bytes(path.read_bytes() + b"\n")

        with self.assertRaisesRegex(figures.FigureEvidenceError, "Evidence hash mismatch"):
            self.load()

    def test_ordered_schema_mismatch_is_rejected(self):
        path = self.fixture.sampling_dir / "candidate_sample_quality.csv"
        table = pd.read_csv(path).drop(columns=["density_share_max_error_pp"])
        table.to_csv(path, index=False)

        with self.assertRaisesRegex(figures.FigureEvidenceError, "schema"):
            self.load(locked_hashes={})

    def test_rare_coverage_cannot_drift_from_selected_index(self):
        path = self.fixture.sampling_dir / "rare_class_coverage.csv"
        table = pd.read_csv(path)
        table.loc[0, "image_count_sample"] -= 1
        table.to_csv(path, index=False)

        with self.assertRaisesRegex(figures.FigureEvidenceError, "rare coverage"):
            self.load(locked_hashes={})

    def test_overlap_pair_formula_is_a_hard_gate(self):
        path = self.fixture.overlap_dir / "overlap_profile.csv"
        table = pd.read_csv(path)
        table.loc[5, "pair_count"] -= 1
        table.to_csv(path, index=False)

        with self.assertRaisesRegex(figures.FigureEvidenceError, r"n\(n-1\)/2"):
            self.load(locked_hashes={})


class FigurePackageTests(Experiment02FigureTestCase):
    def test_exact_png_svg_set_is_byte_deterministic(self):
        evidence = self.load()
        first = figures.build_figure_package(evidence, self.root / "figures-one")
        second = figures.build_figure_package(evidence, self.root / "figures-two")
        expected = {
            f"{stem}.{extension}"
            for stem in figures.OUTPUT_STEMS
            for extension in ("png", "svg")
        }

        self.assertEqual({path.name for path in first.iterdir()}, expected)
        self.assertEqual({path.name for path in second.iterdir()}, expected)
        for name in expected:
            self.assertGreater((first / name).stat().st_size, 1000)
            self.assertEqual(sha256(first / name), sha256(second / name), name)

    def test_existing_destination_is_never_overwritten(self):
        destination = self.root / "existing"
        destination.mkdir()

        with self.assertRaisesRegex(figures.FigureEvidenceError, "Refusing to overwrite"):
            figures.build_figure_package(self.load(), destination)
        self.assertEqual(list(destination.iterdir()), [])

    def test_failed_builder_never_promotes_partial_destination(self):
        destination = self.root / "failed-package"
        with mock.patch.object(
            figures,
            "_class_inventory_figure",
            side_effect=RuntimeError("synthetic render failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic render failure"):
                figures.build_figure_package(self.load(), destination)

        self.assertFalse(destination.exists())
        self.assertTrue((self.root / ".failed-package.incomplete").is_dir())


class ParserTests(unittest.TestCase):
    def test_parser_uses_locked_production_scope(self):
        with self.assertRaises(SystemExit):
            figures.build_parser().parse_args([])

        args = figures.build_parser().parse_args(
            ["--output-dir", "new-figure-package"]
        )

        self.assertEqual(args.expected_images, 9525)
        self.assertEqual(args.expected_labels, 36721)
        self.assertEqual(args.expected_classes, 20)
        self.assertEqual(args.expected_selected_images, 5000)
        self.assertEqual(args.expected_selected_labels, 19196)
        self.assertEqual(args.expected_rare_classes, 8)
        self.assertEqual(
            args.index_dir,
            PROJECT_ROOT / "experiments" / "outputs" / "00_dataset_inventory",
        )
        self.assertEqual(
            args.summary_dir,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "02_dataset_analysis"
            / "01_dataset_summary",
        )
        self.assertEqual(
            args.sampling_dir,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "02_dataset_analysis"
            / "02_sample_selection",
        )
        self.assertEqual(
            args.overlap_dir,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "02_dataset_analysis"
            / "03_overlap_analysis",
        )
        self.assertEqual(args.output_dir, Path("new-figure-package"))


if __name__ == "__main__":
    unittest.main()
