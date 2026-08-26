import csv
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "scripts"
    / "02_dataset_analysis"
    / "02_summarize_dataset.py"
)

spec = importlib.util.spec_from_file_location("dataset_summary_under_test", SCRIPT_PATH)
dataset_summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_summary)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        return reader.fieldnames, list(reader)


class SummaryFixture(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.root = Path(self.temp_directory.name)
        self.index_path = self.root / "dataset_index.csv"
        self.class_path = self.root / "class_distribution.csv"
        self.distribution_path = self.root / "object_count_distribution.csv"
        self.output_dir = self.root / "summary"

        self.index_fields = [
            "image_file",
            "num_objects",
            "count_alpha",
            "count_beta_class",
            "count_gamma",
        ]
        self.index_rows = [
            {
                "image_file": "a.jpg",
                "num_objects": 1,
                "count_alpha": 1,
                "count_beta_class": 0,
                "count_gamma": 0,
            },
            {
                "image_file": "b.jpg",
                "num_objects": 3,
                "count_alpha": 0,
                "count_beta_class": 2,
                "count_gamma": 1,
            },
            {
                "image_file": "c.jpg",
                "num_objects": 5,
                "count_alpha": 1,
                "count_beta_class": 1,
                "count_gamma": 3,
            },
        ]
        self.class_rows = [
            {
                "class_id": 0,
                "class_name": "alpha",
                "object_count": 2,
                "image_count": 2,
            },
            {
                "class_id": 1,
                "class_name": "beta class",
                "object_count": 3,
                "image_count": 2,
            },
            {
                "class_id": 2,
                "class_name": "gamma",
                "object_count": 4,
                "image_count": 2,
            },
        ]
        self.distribution_rows = [
            {"num_objects": 1, "image_count": 1},
            {"num_objects": 3, "image_count": 1},
            {"num_objects": 5, "image_count": 1},
        ]
        self.write_sources()

    def write_sources(self):
        write_csv(self.index_path, self.index_fields, self.index_rows)
        write_csv(
            self.class_path,
            dataset_summary.CLASS_COLUMNS,
            self.class_rows,
        )
        write_csv(
            self.distribution_path,
            ["num_objects", "image_count"],
            self.distribution_rows,
        )

    def load(self):
        return dataset_summary.load_and_validate_sources(
            self.index_path,
            self.class_path,
            self.distribution_path,
        )


class SummaryContractTests(unittest.TestCase):
    def test_default_paths_follow_the_canonical_stage_hierarchy(self):
        self.assertEqual(
            dataset_summary.DEFAULT_DATASET_INDEX,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "00_dataset_inventory"
            / "dataset_index.csv",
        )
        self.assertEqual(
            dataset_summary.DEFAULT_OUTPUT_DIR,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "02_dataset_analysis"
            / "01_dataset_summary",
        )

    def test_density_bucket_boundaries_are_stable(self):
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
            224: "20+",
        }
        self.assertEqual(
            {value: dataset_summary.density_bucket(value) for value in expected},
            expected,
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            dataset_summary.density_bucket(-1)

    def test_artifact_schemas_and_orders_are_explicit(self):
        self.assertEqual(
            dataset_summary.DENSITY_BUCKET_ORDER,
            ["1", "2-4", "5-9", "10-14", "15-19", "20+"],
        )
        self.assertEqual(dataset_summary.DENSE_THRESHOLDS, [5, 10, 15, 20])
        self.assertEqual(
            dataset_summary.ENRICHED_CLASS_COLUMNS,
            dataset_summary.CLASS_COLUMNS
            + ["object_share_pct", "image_share_pct"],
        )

    def test_clean_column_name_matches_the_index_builder(self):
        self.assertEqual(dataset_summary.clean_column_name("Safety-Vest"), "safety_vest")
        self.assertEqual(dataset_summary.clean_column_name("QR code"), "qr_code")


class SourceValidationTests(SummaryFixture):
    def test_valid_sources_are_parsed_and_reconciled(self):
        index, classes, counts = self.load()
        self.assertEqual(len(index), 3)
        self.assertEqual([row["class_name"] for row in classes], [
            "alpha",
            "beta class",
            "gamma",
        ])
        self.assertEqual(counts, [1, 3, 5])

    def test_missing_source_has_actionable_message(self):
        with self.assertRaisesRegex(FileNotFoundError, "00_build_dataset_inventory.py"):
            dataset_summary.load_and_validate_sources(
                self.root / "missing.csv",
                self.class_path,
                self.distribution_path,
            )

    def test_duplicate_image_names_are_rejected(self):
        self.index_rows[1]["image_file"] = "a.jpg"
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "duplicate image_file"):
            self.load()

    def test_non_contiguous_class_ids_are_rejected(self):
        self.class_rows[1]["class_id"] = 2
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.load()

    def test_missing_class_count_column_is_rejected(self):
        self.index_fields.remove("count_beta_class")
        for row in self.index_rows:
            row.pop("count_beta_class")
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "count_beta_class"):
            self.load()

    def test_fractional_and_negative_counts_are_rejected(self):
        self.index_rows[0]["count_alpha"] = "0.5"
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            self.load()

        self.index_rows[0]["count_alpha"] = "-1"
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            self.load()

    def test_row_total_must_equal_class_count_sum(self):
        self.index_rows[0]["num_objects"] = 2
        self.distribution_rows[0]["num_objects"] = 2
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "class-count sum"):
            self.load()

    def test_class_object_total_must_match_index(self):
        self.class_rows[0]["object_count"] = 3
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "Object total mismatch"):
            self.load()

    def test_class_image_total_must_match_index(self):
        self.class_rows[0]["image_count"] = 1
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "Image total mismatch"):
            self.load()

    def test_object_distribution_must_match_index(self):
        self.distribution_rows[0]["image_count"] = 2
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.load()

    def test_duplicate_distribution_bucket_is_rejected(self):
        self.distribution_rows.append({"num_objects": 1, "image_count": 0})
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "Duplicate object-count"):
            self.load()


class SummaryConstructionTests(SummaryFixture):
    def test_summary_values_are_derived_from_validated_counts(self):
        index, classes, counts = self.load()
        tables = dataset_summary.build_summary_tables(index, classes, counts)
        summary = tables["full_dataset_summary.csv"][1][0]

        self.assertEqual(
            summary,
            {
                "dataset": "full_dataset",
                "images": 3,
                "total_objects": 9,
                "images_with_zero_objects": 0,
                "mean_objects_per_image": 3.0,
                "median_objects_per_image": 3.0,
                "max_objects_per_image": 5,
                "images_ge_5_objects": 1,
                "images_ge_10_objects": 0,
                "images_ge_15_objects": 0,
                "images_ge_20_objects": 0,
            },
        )

    def test_enriched_class_shares_and_rankings_are_deterministic(self):
        index, classes, counts = self.load()
        tables = dataset_summary.build_summary_tables(index, classes, counts)
        enriched = tables["class_distribution_enriched.csv"][1]
        top = tables["top10_classes_by_object_count.csv"][1]
        bottom = tables["bottom10_classes_by_object_count.csv"][1]

        self.assertEqual([row["class_name"] for row in enriched], [
            "alpha",
            "beta class",
            "gamma",
        ])
        self.assertEqual(enriched[0]["object_share_pct"], 22.2222)
        self.assertEqual(enriched[0]["image_share_pct"], 66.6667)
        self.assertEqual([row["class_name"] for row in top], [
            "gamma",
            "beta class",
            "alpha",
        ])
        self.assertEqual([row["class_name"] for row in bottom], [
            "alpha",
            "beta class",
            "gamma",
        ])

    def test_density_and_dense_scene_tables_cover_all_images(self):
        index, classes, counts = self.load()
        tables = dataset_summary.build_summary_tables(index, classes, counts)
        density = tables["density_bucket_distribution.csv"][1]
        dense = tables["dense_image_counts.csv"][1]

        self.assertEqual([row["image_count"] for row in density], [1, 1, 1, 0, 0, 0])
        self.assertEqual(sum(row["image_count"] for row in density), 3)
        self.assertEqual([row["image_count"] for row in dense], [1, 0, 0, 0])

    def test_zero_object_dataset_produces_finite_zero_class_shares(self):
        self.index_rows = [
            {
                "image_file": "empty.jpg",
                "num_objects": 0,
                "count_alpha": 0,
                "count_beta_class": 0,
                "count_gamma": 0,
            }
        ]
        for row in self.class_rows:
            row["object_count"] = 0
            row["image_count"] = 0
        self.distribution_rows = [{"num_objects": 0, "image_count": 1}]
        self.write_sources()

        index, classes, counts = self.load()
        tables = dataset_summary.build_summary_tables(index, classes, counts)
        enriched = tables["class_distribution_enriched.csv"][1]

        self.assertTrue(all(row["object_share_pct"] == 0.0 for row in enriched))
        self.assertEqual(
            tables["full_dataset_summary.csv"][1][0]["images_with_zero_objects"],
            1,
        )

    def test_tied_class_rankings_use_class_id_as_tiebreaker(self):
        index, classes, counts = self.load()
        for class_info in classes:
            class_info["object_count"] = 3
        tables = dataset_summary.build_summary_tables(index, classes, counts)
        top = tables["top10_classes_by_object_count.csv"][1]
        bottom = tables["bottom10_classes_by_object_count.csv"][1]
        self.assertEqual([row["class_id"] for row in top], [0, 1, 2])
        self.assertEqual([row["class_id"] for row in bottom], [0, 1, 2])


class ArtifactAndCompatibilityTests(SummaryFixture):
    def test_all_six_artifacts_are_written_atomically(self):
        index, classes, counts = self.load()
        tables = dataset_summary.build_summary_tables(index, classes, counts)
        paths = dataset_summary.write_summary_tables(self.output_dir, tables)

        self.assertEqual(
            [path.name for path in paths],
            [
                "full_dataset_summary.csv",
                "class_distribution_enriched.csv",
                "top10_classes_by_object_count.csv",
                "bottom10_classes_by_object_count.csv",
                "density_bucket_distribution.csv",
                "dense_image_counts.csv",
            ],
        )
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertEqual(list(self.output_dir.glob("*.tmp")), [])

    def test_main_accepts_external_source_and_output_paths(self):
        result, paths = dataset_summary.main(
            [
                "--dataset-index",
                str(self.index_path),
                "--class-distribution",
                str(self.class_path),
                "--object-count-distribution",
                str(self.distribution_path),
                "--output-dir",
                str(self.output_dir),
            ]
        )
        self.assertIn("full_dataset_summary.csv", result)
        self.assertEqual(len(paths), 6)

    def test_valid_fixture_matches_reference_summary_artifacts(self):
        reference_path = os.environ.get("REFERENCE_DATASET_SUMMARY_SCRIPT")
        if not reference_path:
            self.skipTest("REFERENCE_DATASET_SUMMARY_SCRIPT is not configured")

        new_output = self.root / "new-output"
        reference_output = self.root / "reference-output"
        index, classes, counts = self.load()
        tables = dataset_summary.build_summary_tables(index, classes, counts)
        dataset_summary.write_summary_tables(new_output, tables)

        reference_spec = importlib.util.spec_from_file_location(
            "reference_dataset_summary",
            Path(reference_path),
        )
        reference = importlib.util.module_from_spec(reference_spec)
        reference_spec.loader.exec_module(reference)
        reference.DATASET_INDEX = self.index_path
        reference.CLASS_DISTRIBUTION = self.class_path
        reference.OBJECT_COUNT_DISTRIBUTION = self.distribution_path
        reference.SUMMARY_OUTPUT_DIR = reference_output
        reference_output.mkdir()
        reference.main()

        for filename in tables:
            new_fields, new_rows = read_csv(new_output / filename)
            reference_fields, reference_rows = read_csv(reference_output / filename)
            self.assertEqual(new_fields, reference_fields)
            self.assertEqual(new_rows, reference_rows)


if __name__ == "__main__":
    unittest.main()
