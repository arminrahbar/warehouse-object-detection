"""Regression tests for the shared publication-figure contract."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "scripts" / "report_figure_style.py"


def load_module():
    specification = importlib.util.spec_from_file_location("report_figure_style", SCRIPT_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


style = load_module()


class ReportFigureStyleTests(unittest.TestCase):
    def test_three_panel_contract_rejects_other_panel_counts(self):
        with self.assertRaisesRegex(style.FigureBuildError, "requires three panels"):
            style.three_panel_figure("Title", "Subtitle", [])

    def test_atomic_package_is_complete_deterministic_and_non_overwriting(self):
        def build_card():
            return style.three_panel_figure(
                "Deterministic test",
                "The same verified inputs must produce the same publication files.",
                [
                    {"heading": "Input", "bullets": ["One fixed input"]},
                    {"heading": "Controlled", "bullets": ["One fixed policy"]},
                    {"heading": "Output", "bullets": ["One fixed decision"]},
                ],
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = style.build_atomic_package(
                root / "first",
                (("01_contract", build_card),),
                hash_salt="test-report-figure-style",
            )
            second = style.build_atomic_package(
                root / "second",
                (("01_contract", build_card),),
                hash_salt="test-report-figure-style",
            )
            self.assertEqual(
                {path.name for path in first.iterdir()},
                {"01_contract.png", "01_contract.svg"},
            )
            for name in ("01_contract.png", "01_contract.svg"):
                first_hash = hashlib.sha256((first / name).read_bytes()).hexdigest()
                second_hash = hashlib.sha256((second / name).read_bytes()).hexdigest()
                self.assertEqual(first_hash, second_hash)
            svg_lines = (first / "01_contract.svg").read_bytes().splitlines()
            self.assertTrue(all(line == line.rstrip(b" \t") for line in svg_lines))
            with self.assertRaisesRegex(style.FigureBuildError, "Refusing to overwrite"):
                style.build_atomic_package(
                    first,
                    (("01_contract", build_card),),
                    hash_salt="test-report-figure-style",
                )


if __name__ == "__main__":
    unittest.main()
