"""Tests for the public project-report figure package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "experiments" / "scripts"
SCRIPT_PATH = SCRIPT_DIR / "build_project_report_figures.py"
PUBLIC_FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"
PUBLIC_MANIFEST = PUBLIC_FIGURE_DIR / "FIGURE_MANIFEST.json"
PUBLIC_FIGURE_STEMS = (
    "01_system_scope",
    "02_checkpoint_selection",
    "03_development_workload",
    "04_nms_operating_point",
    "05_input_shift_diagnostics",
    "06_error_review_queues",
)


def load_module():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    specification = importlib.util.spec_from_file_location("project_report_figures", SCRIPT_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


builder = load_module()


class ProjectReportFigureTests(unittest.TestCase):
    def test_cli_requires_a_new_output_directory(self):
        with self.assertRaises(SystemExit):
            builder.build_parser().parse_args([])
        args = builder.build_parser().parse_args(
            ["--output-dir", "new-project-report-figures"]
        )
        self.assertEqual(args.output_dir, Path("new-project-report-figures"))

    def _write_sources(self, root):
        # The builder validates signatures and copies these sources byte-for-byte;
        # full rendering behavior is covered by the shared-style tests.
        png = b"\x89PNG\r\n\x1a\nsynthetic-payload"
        svg = b'<?xml version="1.0" encoding="utf-8"?><svg xmlns="http://www.w3.org/2000/svg"></svg>\n'
        for _, source_stem, _ in builder.CURATED_SOURCES:
            png_path = root / f"{source_stem}.png"
            svg_path = root / f"{source_stem}.svg"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            png_path.write_bytes(png)
            svg_path.write_bytes(svg)

    def test_curated_package_is_exact_deterministic_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = root / "sources"
            self._write_sources(sources)
            first = builder.build_project_figure_package(sources, root / "first")
            second = builder.build_project_figure_package(sources, root / "second")
            expected = {
                f"{stem}.{extension}"
                for stem in (
                    "01_system_scope",
                    *[entry[0] for entry in builder.CURATED_SOURCES],
                )
                for extension in ("png", "svg")
            } | {builder.MANIFEST_NAME}
            self.assertEqual({path.name for path in first.iterdir()}, expected)
            self.assertEqual({path.name for path in second.iterdir()}, expected)
            for name in expected:
                first_hash = hashlib.sha256((first / name).read_bytes()).hexdigest()
                second_hash = hashlib.sha256((second / name).read_bytes()).hexdigest()
                self.assertEqual(first_hash, second_hash)

            manifest = json.loads((first / builder.MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["logical_figure_count"], 6)
            self.assertEqual(manifest["asset_count"], 12)
            self.assertEqual(len(manifest["files"]), 12)
            self.assertEqual(
                {record["target"] for record in manifest["files"]},
                expected - {builder.MANIFEST_NAME},
            )
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                builder.build_project_figure_package(sources, first)

    def test_published_figure_manifest_matches_every_committed_asset(self):
        manifest = json.loads(PUBLIC_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            set(manifest),
            {
                "schema_version",
                "generator",
                "logical_figure_count",
                "asset_count",
                "files",
            },
        )
        expected_targets = {
            f"{stem}.{extension}"
            for stem in PUBLIC_FIGURE_STEMS
            for extension in ("png", "svg")
        }

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(
            manifest["logical_figure_count"],
            len(PUBLIC_FIGURE_STEMS),
        )
        self.assertEqual(manifest["asset_count"], len(expected_targets))
        self.assertEqual(len(manifest["files"]), len(expected_targets))

        records_by_target = {
            record["target"]: record for record in manifest["files"]
        }
        self.assertEqual(set(records_by_target), expected_targets)

        for target, record in records_by_target.items():
            with self.subTest(target=target):
                self.assertEqual(Path(target).name, target)
                asset = PUBLIC_FIGURE_DIR / target
                self.assertTrue(asset.is_file())
                payload = asset.read_bytes()
                self.assertEqual(record["bytes"], len(payload))
                self.assertEqual(
                    record["sha256"],
                    hashlib.sha256(payload).hexdigest(),
                )

    def test_missing_source_pair_fails_without_final_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = root / "sources"
            sources.mkdir()
            destination = root / "final"
            with self.assertRaisesRegex(ValueError, "Missing curated PNG source"):
                builder.build_project_figure_package(sources, destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
