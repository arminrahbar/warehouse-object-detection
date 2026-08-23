import csv
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "scripts" / "00_build_dataset_inventory.py"

spec = importlib.util.spec_from_file_location("dataset_index_under_test", SCRIPT_PATH)
dataset_index = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dataset_index
spec.loader.exec_module(dataset_index)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


class DatasetFixture(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.root = Path(self.temp_directory.name)
        self.dataset_dir = self.root / "detector_service" / "storage" / "logistics"
        self.classes_path = (
            self.root
            / "detector_service"
            / "storage"
            / "yolo_model_1"
            / "logistics.names"
        )
        self.output_dir = self.root / "outputs"
        self.dataset_dir.mkdir(parents=True)
        self.classes_path.parent.mkdir(parents=True)
        self.classes_path.write_text(
            "pallet\nfork lift\nsafety-vest\n",
            encoding="utf-8",
        )

    def write_pair(self, stem, labels, image_suffix=".jpg"):
        image_path = self.dataset_dir / f"{stem}{image_suffix}"
        image_path.write_bytes(b"synthetic-image-placeholder")
        label_path = self.dataset_dir / f"{stem}.txt"
        label_path.write_text(labels, encoding="utf-8")
        return image_path, label_path

    def build(self, strict=True):
        return dataset_index.build_dataset_index(
            dataset_dir=self.dataset_dir,
            classes_path=self.classes_path,
            output_dir=self.output_dir,
            asset_root=self.root,
            strict=strict,
        )


class DatasetIndexContractTests(DatasetFixture):
    def test_column_contracts_are_explicit(self):
        self.assertEqual(
            dataset_index.BASE_INDEX_COLUMNS,
            [
                "image_path",
                "label_path",
                "image_file",
                "label_file",
                "num_objects",
                "class_ids_present",
                "class_names_present",
            ],
        )
        self.assertEqual(
            dataset_index.CLASS_SUMMARY_COLUMNS,
            ["class_id", "class_name", "object_count", "image_count"],
        )
        self.assertEqual(
            dataset_index.OBJECT_DISTRIBUTION_COLUMNS,
            ["num_objects", "image_count"],
        )

    def test_clean_column_name_normalizes_punctuation(self):
        self.assertEqual(dataset_index.clean_column_name(" Safety Vest "), "safety_vest")
        self.assertEqual(dataset_index.clean_column_name("fork-lift/cart"), "fork_lift_cart")

    def test_class_loader_rejects_blank_lines_that_would_renumber_ids(self):
        self.classes_path.write_text(" pallet \n\nworker\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "blank entries.*lines: 2"):
            dataset_index.load_classes(self.classes_path)

    def test_empty_class_file_is_rejected(self):
        self.classes_path.write_text("\n  \n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty"):
            dataset_index.load_classes(self.classes_path)

    def test_colliding_class_columns_are_rejected(self):
        self.classes_path.write_text("fork-lift\nfork lift\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate count columns"):
            dataset_index.load_classes(self.classes_path)

    def test_default_paths_match_the_runtime_asset_layout(self):
        self.assertEqual(
            dataset_index.DEFAULT_DATASET_RELATIVE.as_posix(),
            "detector_service/storage/logistics",
        )
        self.assertEqual(
            dataset_index.DEFAULT_CLASSES_RELATIVE.as_posix(),
            "detector_service/storage/yolo_model_1/logistics.names",
        )
        self.assertEqual(
            dataset_index.DEFAULT_OUTPUT_DIR,
            PROJECT_ROOT / "experiments" / "outputs" / "00_dataset_inventory",
        )

    def test_portable_path_does_not_dereference_storage_symlink(self):
        external_storage = self.root / "external-storage"
        external_storage.mkdir()
        external_image = external_storage / "frame.jpg"
        external_image.write_bytes(b"image")
        link_parent = self.root / "portable-root" / "detector_service"
        link_parent.mkdir(parents=True)
        storage_link = link_parent / "storage"
        try:
            os.symlink(external_storage, storage_link, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Directory symlinks are unavailable: {error}")

        logical_image = storage_link / "frame.jpg"
        serialized = dataset_index._portable_path(
            logical_image,
            self.root / "portable-root",
        )

        self.assertEqual(serialized, "detector_service/storage/frame.jpg")

    def test_portable_path_rejects_assets_outside_declared_root(self):
        outside = self.root.parent / "outside-frame.jpg"
        with self.assertRaisesRegex(ValueError, "outside the declared asset root"):
            dataset_index._portable_path(outside, self.root)


class LabelParsingTests(DatasetFixture):
    def test_valid_yolo_rows_are_counted_by_class(self):
        _, label_path = self.write_pair(
            "frame",
            "0 0.5 0.5 0.2 0.4\n1 0.2 0.3 0.1 0.1\n1 1 0 1 1\n",
        )

        counts, errors = dataset_index.parse_label_file(label_path, 3, strict=False)

        self.assertEqual(counts, {0: 1, 1: 2})
        self.assertEqual(errors, [])

    def test_invalid_rows_are_excluded_and_reported(self):
        _, label_path = self.write_pair(
            "frame",
            "broken\n"
            "x 0.5 0.5 0.2 0.2\n"
            "1.5 0.5 0.5 0.2 0.2\n"
            "9 0.5 0.5 0.2 0.2\n"
            "1 nan 0.5 0.2 0.2\n"
            "1 1.2 0.5 0.2 0.2\n"
            "2 0.5 0.5 0.2 0.2 extra\n",
        )

        counts, errors = dataset_index.parse_label_file(
            label_path,
            3,
            strict=False,
        )

        self.assertEqual(counts, {})
        self.assertEqual(len(errors), 7)
        self.assertTrue(all("frame.txt:" in message for message in errors))

    def test_strict_parsing_fails_at_first_invalid_row(self):
        _, label_path = self.write_pair("frame", "0 0.5 0.5 -0.2 0.2\n")

        with self.assertRaisesRegex(ValueError, "normalized YOLO values"):
            dataset_index.parse_label_file(label_path, 3, strict=True)

    def test_blank_label_file_is_a_valid_empty_image(self):
        _, label_path = self.write_pair("empty", "\n")
        counts, errors = dataset_index.parse_label_file(label_path, 3)
        self.assertEqual(counts, {})
        self.assertEqual(errors, [])


class ArtifactConstructionTests(DatasetFixture):
    def test_builder_writes_all_compatible_artifacts(self):
        self.write_pair(
            "b",
            "1 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1\n1 0.7 0.7 0.1 0.1\n",
        )
        self.write_pair("a", "")
        (self.dataset_dir / "c.jpg").write_bytes(b"missing-label")
        self.write_pair("ignored", "0 0.5 0.5 0.2 0.2\n", image_suffix=".png")

        result = self.build(strict=False)

        self.assertEqual(result.images_discovered, 3)
        self.assertEqual(result.images_indexed, 2)
        self.assertEqual(result.missing_labels, 1)
        self.assertEqual(result.invalid_annotations, 0)
        self.assertEqual(result.total_objects, 3)

        index_rows = read_rows(result.dataset_index_path)
        self.assertEqual([row["image_file"] for row in index_rows], ["a.jpg", "b.jpg"])
        self.assertEqual(
            list(index_rows[0]),
            dataset_index.BASE_INDEX_COLUMNS
            + ["count_pallet", "count_fork_lift", "count_safety_vest"],
        )
        self.assertEqual(index_rows[0]["num_objects"], "0")
        self.assertEqual(index_rows[0]["class_ids_present"], "[]")
        self.assertEqual(index_rows[1]["num_objects"], "3")
        self.assertEqual(index_rows[1]["class_ids_present"], "[0, 1]")
        self.assertEqual(
            index_rows[1]["class_names_present"],
            '["pallet", "fork lift"]',
        )
        self.assertEqual(index_rows[1]["count_pallet"], "1")
        self.assertEqual(index_rows[1]["count_fork_lift"], "2")
        self.assertEqual(index_rows[1]["count_safety_vest"], "0")

        expected_prefix = "detector_service/storage/logistics/"
        self.assertTrue(index_rows[0]["image_path"].startswith(expected_prefix))
        self.assertNotIn("\\", index_rows[0]["image_path"])

        class_rows = read_rows(result.class_distribution_path)
        self.assertEqual(
            class_rows,
            [
                {
                    "class_id": "0",
                    "class_name": "pallet",
                    "object_count": "1",
                    "image_count": "1",
                },
                {
                    "class_id": "1",
                    "class_name": "fork lift",
                    "object_count": "2",
                    "image_count": "1",
                },
                {
                    "class_id": "2",
                    "class_name": "safety-vest",
                    "object_count": "0",
                    "image_count": "0",
                },
            ],
        )
        self.assertEqual(
            read_rows(result.object_count_distribution_path),
            [
                {"num_objects": "0", "image_count": "1"},
                {"num_objects": "3", "image_count": "1"},
            ],
        )

    def test_invalid_annotation_count_is_preserved_in_result(self):
        self.write_pair(
            "frame",
            "0 0.5 0.5 0.2 0.2\ninvalid row\n2 0.5 0.5 0.2 0.2\n",
        )

        result = self.build(strict=False)

        self.assertEqual(result.invalid_annotations, 1)
        self.assertEqual(result.total_objects, 2)

    def test_strict_mode_rejects_a_missing_label(self):
        (self.dataset_dir / "frame.jpg").write_bytes(b"missing-label")
        with self.assertRaisesRegex(FileNotFoundError, "Label file not found"):
            self.build(strict=True)

    def test_no_image_label_pairs_is_rejected(self):
        (self.dataset_dir / "frame.jpg").write_bytes(b"missing-label")
        with self.assertRaisesRegex(RuntimeError, "No image-label pairs"):
            self.build(strict=False)

    def test_strict_validation_is_the_cli_default(self):
        defaults = dataset_index.build_parser().parse_args([])
        diagnostic = dataset_index.build_parser().parse_args(["--allow-invalid"])

        self.assertFalse(defaults.allow_invalid)
        self.assertTrue(diagnostic.allow_invalid)

    def test_repeated_build_is_byte_for_byte_deterministic(self):
        self.write_pair("b", "1 0.5 0.5 0.2 0.2\n")
        self.write_pair("a", "0 0.5 0.5 0.2 0.2\n")
        first = self.build()
        first_bytes = {
            path.name: path.read_bytes()
            for path in (
                first.dataset_index_path,
                first.class_distribution_path,
                first.object_count_distribution_path,
            )
        }

        second = self.build()
        second_bytes = {
            path.name: path.read_bytes()
            for path in (
                second.dataset_index_path,
                second.class_distribution_path,
                second.object_count_distribution_path,
            )
        }

        self.assertEqual(second_bytes, first_bytes)
        self.assertEqual(list(self.output_dir.glob("*.tmp")), [])

    def test_main_accepts_an_external_asset_root(self):
        self.write_pair("frame", "0 0.5 0.5 0.2 0.2\n")
        cli_output = self.root / "cli-output"

        result = dataset_index.main(
            [
                "--asset-root",
                str(self.root),
                "--output-dir",
                str(cli_output),
            ]
        )

        self.assertEqual(result.images_indexed, 1)
        self.assertTrue((cli_output / "dataset_index.csv").is_file())


class ReferenceCompatibilityTests(DatasetFixture):
    def test_valid_fixture_matches_reference_artifacts(self):
        reference_path = os.environ.get("REFERENCE_DATASET_INDEX_SCRIPT")
        if not reference_path:
            self.skipTest("REFERENCE_DATASET_INDEX_SCRIPT is not configured")

        self.write_pair(
            "b",
            "0 0.5 0.5 0.2 0.2\n1 0.2 0.3 0.1 0.1\n",
        )
        self.write_pair("a", "")
        (self.dataset_dir / "missing.jpg").write_bytes(b"missing-label")

        new_output = self.root / "new-output"
        reference_output = self.root / "reference-output"
        dataset_index.build_dataset_index(
            dataset_dir=self.dataset_dir,
            classes_path=self.classes_path,
            output_dir=new_output,
            asset_root=self.root,
            strict=False,
        )

        reference_spec = importlib.util.spec_from_file_location(
            "reference_dataset_index",
            Path(reference_path),
        )
        reference = importlib.util.module_from_spec(reference_spec)
        reference_spec.loader.exec_module(reference)
        reference.PROJECT_ROOT = self.root
        reference.DATA_DIR = self.dataset_dir
        reference.NAMES_FILE = self.classes_path
        reference.OUTPUT_DIR = reference_output
        reference_output.mkdir()
        reference.main()

        new_index = read_rows(new_output / "dataset_index.csv")
        reference_index = read_rows(reference_output / "dataset_index.csv")
        for rows in (new_index, reference_index):
            for row in rows:
                row["image_path"] = row["image_path"].replace("\\", "/")
                row["label_path"] = row["label_path"].replace("\\", "/")
        self.assertEqual(new_index, reference_index)

        for filename in (
            "class_distribution.csv",
            "object_count_distribution.csv",
        ):
            self.assertEqual(
                (new_output / filename).read_text(encoding="utf-8"),
                (reference_output / filename).read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
