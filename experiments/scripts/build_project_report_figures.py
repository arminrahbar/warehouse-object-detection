"""Assemble the curated, publication-ready figure package for the project report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_figure_style import (  # noqa: E402
    require,
    save_figure_png,
    style_context,
    three_panel_figure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "experiments" / "figures"
MANIFEST_NAME = "FIGURE_MANIFEST.json"

CURATED_SOURCES = (
    (
        "02_checkpoint_selection",
        "01_model_selection/01_decision_summary",
        "Checkpoint comparison and quality-first decision",
    ),
    (
        "03_development_workload",
        "02_dataset_analysis/03_candidate_scorecard",
        "Development-workload candidate comparison",
    ),
    (
        "04_nms_operating_point",
        "03_nms_thresholding/03_quality_output_frontier",
        "NMS quality and output-volume trade-off",
    ),
    (
        "05_input_shift_diagnostics",
        "04_augmentation_robustness/03_condition_quality",
        "Controlled input-shift quality result",
    ),
    (
        "06_error_review_queues",
        "05_hard_negative_mining/05_queue_overlap",
        "Specialized review-queue separation",
    ),
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_png(path):
    require(path.is_file(), f"Missing curated PNG source: {path}")
    require(path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"Invalid PNG signature: {path}")


def _system_scope_figure():
    return three_panel_figure(
        "System responsibilities at a glance",
        "The project separates runtime inference, controlled evaluation, and operational diagnosis.",
        [
            {
                "heading": "Runtime inference",
                "bullets": [
                    "Local-file or UDP video ingestion",
                    "Configurable frame sampling",
                    "YOLOv4-tiny through OpenCV DNN",
                    "Structured detections and optional JPEGs",
                ],
            },
            {
                "heading": "Controlled evaluation",
                "bullets": [
                    "Locked image, label, and checkpoint inputs",
                    "Explicit confidence and NMS policies",
                    "Same-class one-to-one matching",
                    "Quality and operating-point evidence",
                ],
            },
            {
                "heading": "Operational diagnosis",
                "bullets": [
                    "Controlled input-shift diagnostics",
                    "Image-level error components",
                    "Deterministic error-review queues",
                    "Focused human-review priorities",
                ],
            },
        ],
    )


def _copy_curated_png(source_root, staging, target_stem, source_stem, role):
    source = source_root / f"{source_stem}.png"
    _validate_png(source)
    destination = staging / f"{target_stem}.png"
    shutil.copyfile(source, destination)
    require(_sha256(source) == _sha256(destination), f"Copy hash mismatch: {source}")
    return {
        "target": destination.name,
        "source_relative_to_experiment_figure_root": f"{source_stem}.png",
        "role": role,
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def build_project_figure_package(source_root, output_dir):
    """Validate and atomically assemble the public project-report figures."""

    source_root = Path(source_root).expanduser().absolute()
    require(source_root.is_dir(), f"Experiment figure root not found: {source_root}")
    destination = Path(output_dir).expanduser().absolute()
    require(not destination.exists(), f"Refusing to overwrite output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.incomplete"
    require(not staging.exists(), f"Incomplete project figure build already exists: {staging}")

    staging.mkdir()
    records = []
    try:
        with style_context("warehouse-object-detection-project-report-v1"):
            save_figure_png(_system_scope_figure(), staging, "01_system_scope")
        system_path = staging / "01_system_scope.png"
        records.append(
            {
                "target": system_path.name,
                "source_relative_to_experiment_figure_root": None,
                "role": "Project runtime, evaluation, and diagnosis scope",
                "bytes": system_path.stat().st_size,
                "sha256": _sha256(system_path),
            }
        )

        for target_stem, source_stem, role in CURATED_SOURCES:
            records.append(
                _copy_curated_png(
                    source_root,
                    staging,
                    target_stem,
                    source_stem,
                    role,
                )
            )

        manifest = {
            "schema_version": 1,
            "generator": "experiments/scripts/build_project_report_figures.py",
            "logical_figure_count": 6,
            "asset_count": 6,
            "files": sorted(records, key=lambda item: item["target"]),
        }
        manifest_path = staging / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        expected = {
            f"{stem}.png"
            for stem in (
                "01_system_scope",
                *[entry[0] for entry in CURATED_SOURCES],
            )
        } | {MANIFEST_NAME}
        actual = {path.name for path in staging.iterdir() if path.is_file()}
        require(actual == expected, "Project figure package has an unexpected file set.")
        os.replace(staging, destination)
    except Exception:
        raise
    return destination


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory to receive the atomic public-report figure package.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    destination = build_project_figure_package(args.source_root, args.output_dir)
    print(f"[COMPLETE] Promoted project-report figure package: {destination}")
    for path in sorted(destination.iterdir()):
        print(f"[WRITE] {path}")


if __name__ == "__main__":
    main()
