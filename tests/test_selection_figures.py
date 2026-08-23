import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests.test_checkpoint_selection import EvidenceFixture
from tests.test_checkpoint_selection import selection as checkpoint_selection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / "experiments" / "scripts" / "01_build_selection_figures.py"
)
spec = importlib.util.spec_from_file_location("selection_figures_under_test", SCRIPT_PATH)
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


class SelectionFigureFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = EvidenceFixture(
            self.root, bootstrap_samples=20, seed=20260821
        )
        self.selection_dir, _, _ = checkpoint_selection.run_selection(
            self.fixture.quality,
            self.fixture.runtime,
            self.root / "selection-output",
            "verified-selection",
            bootstrap_samples=self.fixture.bootstrap_samples,
            seed=self.fixture.seed,
            expected_images=2,
            expected_labels=2,
            locked_hashes=self.fixture.model_hashes,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def load(self, expected_pairs=2):
        return figures.load_verified_evidence(
            self.fixture.quality,
            self.fixture.runtime,
            self.selection_dir,
            expected_images=2,
            expected_labels=2,
            expected_classes=20,
            expected_pairs=expected_pairs,
            locked_hashes=self.fixture.model_hashes,
        )


class EvidenceValidationTests(SelectionFigureFixture):
    def test_verified_packages_load_with_cross_manifest_integrity(self):
        evidence = self.load()

        self.assertEqual(len(evidence["runtime"]["pair_rows"]), 2)
        self.assertEqual(evidence["selection"]["decision"]["integrity_status"], "passed")
        self.assertEqual(len(evidence["quality"]["classes"]), 20)

    def test_tampered_quality_artifact_is_rejected_before_plotting(self):
        with (self.fixture.quality / "aggregate_metrics.csv").open(
            "a", encoding="utf-8"
        ) as destination:
            destination.write("\n")

        with self.assertRaises((figures.FigureEvidenceError, checkpoint_selection.IntegrityError)):
            self.load()

    def test_selection_cross_link_mismatch_is_rejected(self):
        manifest_path = self.selection_dir / "selection_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["quality_input"]["manifest_sha256"] = "0" * 64
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(
            figures.FigureEvidenceError, "Selection-to-quality manifest hash mismatch"
        ):
            self.load()

    def test_expected_pair_count_is_a_hard_gate(self):
        with self.assertRaisesRegex(
            figures.FigureEvidenceError, "Runtime pair count must be exactly 3"
        ):
            self.load(expected_pairs=3)


class DeterministicSelectionTests(unittest.TestCase):
    def test_delta_labels_are_deterministic_material_extremes_only(self):
        table = pd.DataFrame(
            [
                {"class_id": 0, "class_name": "alpha", "ground_truth_count": 100,
                 "ap_delta": 0.03},
                {"class_id": 1, "class_name": "beta", "ground_truth_count": 20,
                 "ap_delta": -0.05},
                {"class_id": 2, "class_name": "immaterial", "ground_truth_count": 5,
                 "ap_delta": 0.009},
                {"class_id": 3, "class_name": "gamma", "ground_truth_count": 10,
                 "ap_delta": 0.03},
                {"class_id": 4, "class_name": "empty", "ground_truth_count": 0,
                 "ap_delta": 0.90},
            ]
        )

        first = figures.select_delta_labels(table, max_labels=3, materiality=0.01)
        second = figures.select_delta_labels(
            table.sample(frac=1.0, random_state=7), max_labels=3, materiality=0.01
        )

        self.assertEqual(first, ["beta", "gamma", "alpha"])
        self.assertEqual(second, first)


class FigurePackageTests(SelectionFigureFixture):
    def test_fixed_png_svg_package_is_atomic_and_byte_deterministic(self):
        evidence = self.load()
        first = figures.build_figure_package(evidence, self.root / "figures-one")
        second = figures.build_figure_package(evidence, self.root / "figures-two")
        expected = {
            f"{stem}.{suffix}"
            for stem in figures.OUTPUT_STEMS
            for suffix in ("png", "svg")
        }

        self.assertEqual({path.name for path in first.iterdir()}, expected)
        self.assertEqual({path.name for path in second.iterdir()}, expected)
        self.assertFalse((self.root / ".figures-one.incomplete").exists())
        self.assertFalse((self.root / ".figures-two.incomplete").exists())
        for name in expected:
            first_path = first / name
            second_path = second / name
            self.assertGreater(first_path.stat().st_size, 1000)
            self.assertEqual(
                figures._sha256_file(first_path), figures._sha256_file(second_path), name
            )

    def test_existing_destination_is_never_overwritten(self):
        destination = self.root / "existing"
        destination.mkdir()

        with self.assertRaisesRegex(figures.FigureEvidenceError, "Refusing to overwrite"):
            figures.build_figure_package(self.load(), destination)
        self.assertEqual(list(destination.iterdir()), [])


class ParserTests(unittest.TestCase):
    def test_parser_has_production_integrity_defaults(self):
        args = figures.build_parser().parse_args(
            [
                "--quality-run", "quality",
                "--runtime-run", "runtime",
                "--selection-run", "selection",
                "--output-dir", "figures",
            ]
        )

        self.assertEqual(args.expected_images, 9525)
        self.assertEqual(args.expected_labels, 36721)
        self.assertEqual(args.expected_classes, 20)
        self.assertEqual(args.expected_pairs, 1500)


if __name__ == "__main__":
    unittest.main()
