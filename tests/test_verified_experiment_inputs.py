"""Adversarial tests for shared experiment evidence contracts."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.scripts.verified_experiment_inputs import (
    EvidenceContractError,
    OPERATING_POINT_DATASET,
    OPERATING_POINT_EVIDENCE_SCHEMAS,
    OPERATING_POINT_SELECTION_RULE,
    SELECTION_ARTIFACT_SCHEMAS,
    SELECTION_POLICY_RULE,
    load_verified_checkpoint_selection,
    load_verified_operating_point,
)


OPERATING_FILES = (
    "operating_point.json",
    "nms_threshold_summary_sample5000.csv",
    "duplicate_summary_by_threshold_sample5000.csv",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_json(value: dict) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class EvidenceFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.selection = self.root / "synthetic-selection"
        self.selection.mkdir()
        self._write_selection_fixture()
        self.operating_directory = self.root / "operating-point"
        self.operating_directory.mkdir()
        self._write_operating_fixture()
        self.operating = self.operating_directory / "operating_point.json"
        self.original_files = {
            path: path.read_bytes()
            for path in (
                self.manifest_path,
                self.operating,
                self.operating_directory
                / "nms_threshold_summary_sample5000.csv",
                self.operating_directory
                / "duplicate_summary_by_threshold_sample5000.csv",
            )
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _write_selection_fixture(self) -> None:
        decision = {
            "claim_scope": "synthetic contract fixture",
            "decision_rule": "locked_lexicographic_v1",
            "delta_sign": "model2_minus_model1",
            "integrity_status": "passed",
            "quality_intervals": {
                "delta_ap_ci_lower": 0.05,
                "delta_ap_ci_upper": 0.05,
                "delta_f1_ci_lower": 0.05,
                "delta_f1_ci_upper": 0.05,
            },
            "quality_point_estimates": {
                "model1": {
                    "deployment_macro_f1": 0.40,
                    "mAP50_101pt": 0.40,
                },
                "model2": {
                    "deployment_macro_f1": 0.45,
                    "mAP50_101pt": 0.45,
                },
            },
            "reason": "qualifying_low_floor_ap50",
            "runtime_delta_intervals_ms": {
                "mean": [2.0, 2.0],
                "p95": [2.0, 2.0],
            },
            "runtime_point_estimates_ms": {
                "model1_mean": 10.0,
                "model1_p95": 10.0,
                "model2_mean": 12.0,
                "model2_p95": 12.0,
            },
            "selected_checkpoint": "B",
            "selected_model": "model2",
            "status": "selected",
            "step": 1,
            "training_overlap": "unknown",
        }
        summary_rows = [
            {
                "metric": "mAP50_101pt",
                "model1_value": 0.40,
                "model2_value": 0.45,
                "delta_model2_minus_model1": 0.05,
                "ci_lower": 0.05,
                "ci_upper": 0.05,
                "practical_threshold": 0.01,
                "relative_effect_pct": "",
                "selection_role": "step_1_primary_quality",
            },
            {
                "metric": "deployment_macro_f1",
                "model1_value": 0.40,
                "model2_value": 0.45,
                "delta_model2_minus_model1": 0.05,
                "ci_lower": 0.05,
                "ci_upper": 0.05,
                "practical_threshold": 0.01,
                "relative_effect_pct": "",
                "selection_role": "step_2_secondary_quality",
            },
            {
                "metric": "p95_compute_ms",
                "model1_value": 10.0,
                "model2_value": 12.0,
                "delta_model2_minus_model1": 2.0,
                "ci_lower": 2.0,
                "ci_upper": 2.0,
                "practical_threshold": "5%_of_slower_p95",
                "relative_effect_pct": 16.6666666667,
                "selection_role": "step_3_subordinate_runtime",
            },
            {
                "metric": "mean_compute_ms",
                "model1_value": 10.0,
                "model2_value": 12.0,
                "delta_model2_minus_model1": 2.0,
                "ci_lower": 2.0,
                "ci_upper": 2.0,
                "practical_threshold": "none_deterministic_tiebreak",
                "relative_effect_pct": 20.0,
                "selection_role": "step_4_tiebreak_only",
            },
        ]
        bootstrap_rows = [
            {
                "replicate": 0,
                "sampled_source_group_draw_sha256": "a" * 64,
                "delta_mAP50_101pt": 0.05,
                "delta_deployment_macro_f1": 0.05,
            }
        ]
        summary_path = self.selection / "selection_summary.csv"
        bootstrap_path = self.selection / "bootstrap_replicates.csv"
        decision_path = self.selection / "decision.json"
        write_csv(
            summary_path,
            SELECTION_ARTIFACT_SCHEMAS[summary_path.name],
            summary_rows,
        )
        write_csv(
            bootstrap_path,
            SELECTION_ARTIFACT_SCHEMAS[bootstrap_path.name],
            bootstrap_rows,
        )
        write_json(decision_path, decision)
        policy = {
            "lexicographic_rule": SELECTION_POLICY_RULE,
            "delta_sign": "model2_minus_model1",
            "primary_ap_points": 101,
            "map_iou_threshold": 0.5,
            "deployment_confidence": 0.5,
            "bootstrap_samples": 1,
        }
        artifacts = {}
        for path in (summary_path, bootstrap_path):
            columns, rows = read_csv(path)
            artifacts[path.name] = {
                "columns": columns,
                "rows": len(rows),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        write_json(
            self.selection / "selection_manifest.json",
            {
                "schema_version": 1,
                "status": "complete",
                "run_id": self.selection.name,
                "decision": {
                    "sha256": sha256(decision_path),
                    "size_bytes": decision_path.stat().st_size,
                },
                "artifacts": artifacts,
                "selection_policy": policy,
                "selection_policy_sha256": sha256_json(policy),
                "model_identities": {
                    "model1": {
                        "weights": "1" * 64,
                        "cfg": "c" * 64,
                        "names": "e" * 64,
                    },
                    "model2": {
                        "weights": "2" * 64,
                        "cfg": "c" * 64,
                        "names": "e" * 64,
                    },
                },
            },
        )

    def _write_operating_fixture(self) -> None:
        summary_name = "nms_threshold_summary_sample5000.csv"
        duplicate_name = "duplicate_summary_by_threshold_sample5000.csv"
        thresholds = (0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70)
        maps = (0.40, 0.41, 0.405, 0.40, 0.39, 0.38, 0.37)
        predictions = (90, 100, 105, 110, 115, 120, 130)
        duplicate_pairs = (0, 0, 0, 0, 5, 10, 20)
        summary_rows = []
        duplicate_rows = []
        for threshold, score, count, pairs in zip(
            thresholds,
            maps,
            predictions,
            duplicate_pairs,
        ):
            summary_rows.append(
                {
                    "model": "model2",
                    "dataset": OPERATING_POINT_DATASET,
                    "nms_threshold": threshold,
                    "mAP@0.5_11_point": score,
                    "total_ground_truth": 10,
                    "total_predictions_after_nms": count,
                    "evaluation_rows": count,
                    "score_threshold": 0.5,
                    "map_iou_threshold": 0.5,
                    "eval_type": "combined",
                }
            )
            duplicate_rows.append(
                {
                    "duplicate_like_pairs_iou_gt_0_5": pairs,
                    "images_with_duplicate_like_pairs": min(pairs, 4),
                    "mean_duplicate_like_pairs_per_image": pairs / 4.0,
                    "nms_threshold": threshold,
                    "total_predictions_after_nms": count,
                }
            )
        summary_path = self.operating_directory / summary_name
        duplicate_path = self.operating_directory / duplicate_name
        write_csv(
            summary_path,
            OPERATING_POINT_EVIDENCE_SCHEMAS[summary_name],
            summary_rows,
        )
        write_csv(
            duplicate_path,
            OPERATING_POINT_EVIDENCE_SCHEMAS[duplicate_name],
            duplicate_rows,
        )
        selection = load_verified_checkpoint_selection(self.selection)
        write_json(
            self.operating_directory / "operating_point.json",
            {
                "schema_version": 1,
                "status": "complete",
                "selected_model": "model2",
                "selected_nms_iou_threshold": 0.3,
                "selection_rule": OPERATING_POINT_SELECTION_RULE,
                "selected_metrics": {
                    "mAP@0.5_11_point": 0.41,
                    "total_predictions_after_nms": 100,
                    "duplicate_like_pairs_iou_gt_0_5": 0,
                },
                "checkpoint_selection": {
                    "run_id": selection["selection_run_id"],
                    "manifest_sha256": selection["selection_manifest_sha256"],
                    "decision_sha256": selection["decision_sha256"],
                },
                "evidence": {
                    summary_name: {
                        "rows": len(summary_rows),
                        "sha256": sha256(summary_path),
                    },
                    duplicate_name: {
                        "rows": len(duplicate_rows),
                        "sha256": sha256(duplicate_path),
                    },
                },
            },
        )

    def restore(self, path: Path) -> None:
        path.write_bytes(self.original_files[path])

    @property
    def manifest_path(self) -> Path:
        return self.selection / "selection_manifest.json"

    @property
    def decision_path(self) -> Path:
        return self.selection / "decision.json"

    def update_decision_identity(self) -> None:
        manifest = read_json(self.manifest_path)
        manifest["decision"] = {
            "sha256": sha256(self.decision_path),
            "size_bytes": self.decision_path.stat().st_size,
        }
        write_json(self.manifest_path, manifest)

    def update_selection_artifact_identity(self, name: str) -> None:
        path = self.selection / name
        columns, rows = read_csv(path)
        manifest = read_json(self.manifest_path)
        manifest["artifacts"][name] = {
            "columns": columns,
            "rows": len(rows),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        write_json(self.manifest_path, manifest)

    def update_operating_evidence_identity(self, name: str) -> None:
        path = self.operating_directory / name
        _, rows = read_csv(path)
        record = read_json(self.operating)
        record["evidence"][name] = {"rows": len(rows), "sha256": sha256(path)}
        write_json(self.operating, record)


class CheckpointSelectionContractTests(EvidenceFixture):
    def test_canonical_checkpoint_selection_is_accepted(self):
        result = load_verified_checkpoint_selection(self.selection)
        self.assertEqual(result["selection_run_id"], "synthetic-selection")
        self.assertEqual(result["selected_model"], "model2")
        self.assertEqual(result["selected_checkpoint"], "B")

    def test_operational_equivalence_tiebreak_is_a_valid_selection(self):
        decision = read_json(self.decision_path)
        decision.update(
            {
                "status": "operationally_equivalent_under_protocol",
                "step": 4,
                "selected_model": "model1",
                "selected_checkpoint": "A",
                "reason": "deterministic_lower_mean_latency_tiebreak",
                "quality_point_estimates": {
                    "model1": {
                        "mAP50_101pt": 0.5,
                        "deployment_macro_f1": 0.5,
                    },
                    "model2": {
                        "mAP50_101pt": 0.505,
                        "deployment_macro_f1": 0.505,
                    },
                },
                "quality_intervals": {
                    "delta_ap_ci_lower": -0.01,
                    "delta_ap_ci_upper": 0.01,
                    "delta_f1_ci_lower": -0.01,
                    "delta_f1_ci_upper": 0.01,
                },
                "runtime_point_estimates_ms": {
                    "model1_mean": 10.0,
                    "model1_p95": 10.0,
                    "model2_mean": 12.0,
                    "model2_p95": 10.2,
                },
                "runtime_delta_intervals_ms": {
                    "mean": [1.0, 3.0],
                    "p95": [-1.0, 1.0],
                },
            }
        )
        write_json(self.decision_path, decision)

        summary_path = self.selection / "selection_summary.csv"
        columns, rows = read_csv(summary_path)
        values = {
            "mAP50_101pt": (0.5, 0.505, -0.01, 0.01),
            "deployment_macro_f1": (0.5, 0.505, -0.01, 0.01),
            "p95_compute_ms": (10.0, 10.2, -1.0, 1.0),
            "mean_compute_ms": (10.0, 12.0, 1.0, 3.0),
        }
        for row in rows:
            model1, model2, lower, upper = values[row["metric"]]
            row.update(
                {
                    "model1_value": model1,
                    "model2_value": model2,
                    "delta_model2_minus_model1": model2 - model1,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
        write_csv(summary_path, columns, rows)
        self.update_decision_identity()
        self.update_selection_artifact_identity(summary_path.name)

        result = load_verified_checkpoint_selection(self.selection)

        self.assertEqual(result["selected_model"], "model1")
        self.assertEqual(result["selected_checkpoint"], "A")

    def test_schema_and_directory_run_identity_are_enforced(self):
        manifest = read_json(self.manifest_path)
        manifest["run_id"] = "different-run"
        write_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(EvidenceContractError, "run_id"):
            load_verified_checkpoint_selection(self.selection)

    def test_selected_checkpoint_must_correspond_to_recomputed_model(self):
        decision = read_json(self.decision_path)
        decision["selected_checkpoint"] = "A"
        write_json(self.decision_path, decision)
        self.update_decision_identity()
        with self.assertRaisesRegex(EvidenceContractError, "conflicts with recomputed"):
            load_verified_checkpoint_selection(self.selection)

    def test_policy_digest_rejects_undisclosed_modification(self):
        manifest = read_json(self.manifest_path)
        manifest["selection_policy"]["deployment_confidence"] = 0.4
        write_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(EvidenceContractError, "policy hash mismatch"):
            load_verified_checkpoint_selection(self.selection)

    def test_rehashed_policy_still_must_match_locked_semantics(self):
        manifest = read_json(self.manifest_path)
        manifest["selection_policy"]["primary_ap_points"] = 11
        payload = json.dumps(
            manifest["selection_policy"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        manifest["selection_policy_sha256"] = hashlib.sha256(payload).hexdigest()
        write_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(EvidenceContractError, "locked policy"):
            load_verified_checkpoint_selection(self.selection)

    def test_artifact_hash_and_row_count_are_enforced(self):
        for field, value, message in (
            ("sha256", "0" * 64, "hash mismatch"),
            ("rows", 1999, "row-count mismatch"),
        ):
            with self.subTest(field=field):
                self.restore(self.manifest_path)
                manifest = read_json(self.manifest_path)
                manifest["artifacts"]["bootstrap_replicates.csv"][field] = value
                write_json(self.manifest_path, manifest)
                with self.assertRaisesRegex(EvidenceContractError, message):
                    load_verified_checkpoint_selection(self.selection)

    def test_rehashed_summary_cannot_contradict_decision(self):
        path = self.selection / "selection_summary.csv"
        columns, rows = read_csv(path)
        rows[0]["model2_value"] = "0.1"
        write_csv(path, columns, rows)
        self.update_selection_artifact_identity(path.name)
        with self.assertRaisesRegex(EvidenceContractError, "model2 value mismatch"):
            load_verified_checkpoint_selection(self.selection)

    def test_package_rejects_extra_and_traversing_artifact_identities(self):
        manifest = read_json(self.manifest_path)
        manifest["artifacts"]["../outside.csv"] = dict(
            manifest["artifacts"]["selection_summary.csv"]
        )
        write_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(EvidenceContractError, "artifact identities"):
            load_verified_checkpoint_selection(self.selection)

        self.restore(self.manifest_path)
        (self.selection / "untracked.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(EvidenceContractError, "file set"):
            load_verified_checkpoint_selection(self.selection)

    def test_incomplete_model_identity_is_rejected(self):
        manifest = read_json(self.manifest_path)
        del manifest["model_identities"]["model2"]["weights"]
        write_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(EvidenceContractError, "model2 identity"):
            load_verified_checkpoint_selection(self.selection)


class OperatingPointContractTests(EvidenceFixture):
    def test_canonical_operating_point_is_accepted(self):
        result = load_verified_operating_point(self.operating, self.selection)
        self.assertEqual(result["selected_model"], "model2")
        self.assertEqual(result["selected_nms_iou_threshold"], 0.3)
        self.assertIn("sha256", result)

    def test_recorded_threshold_and_metrics_are_recomputed(self):
        modifications = (
            ("threshold", lambda record: record.__setitem__("selected_nms_iou_threshold", 0.2)),
            (
                "map",
                lambda record: record["selected_metrics"].__setitem__(
                    "mAP@0.5_11_point", 0.9
                ),
            ),
            (
                "predictions",
                lambda record: record["selected_metrics"].__setitem__(
                    "total_predictions_after_nms", 1
                ),
            ),
            (
                "duplicates",
                lambda record: record["selected_metrics"].__setitem__(
                    "duplicate_like_pairs_iou_gt_0_5", 1
                ),
            ),
        )
        for name, modify in modifications:
            with self.subTest(name=name):
                self.restore(self.operating)
                record = read_json(self.operating)
                modify(record)
                write_json(self.operating, record)
                with self.assertRaises(EvidenceContractError):
                    load_verified_operating_point(self.operating, self.selection)

    def test_evidence_identity_requires_exact_canonical_basenames(self):
        variants = (
            {},
            {"unexpected.csv": {"rows": 1, "sha256": "0" * 64}},
            {"../nms_threshold_summary_sample5000.csv": {"rows": 7, "sha256": "0" * 64}},
        )
        canonical = read_json(self.operating)["evidence"]
        for replacement in variants:
            with self.subTest(replacement=tuple(replacement)):
                self.restore(self.operating)
                record = read_json(self.operating)
                record["evidence"] = replacement or {}
                if replacement and "unexpected.csv" in replacement:
                    record["evidence"].update(canonical)
                write_json(self.operating, record)
                with self.assertRaisesRegex(EvidenceContractError, "evidence"):
                    load_verified_operating_point(self.operating, self.selection)

    def test_missing_evidence_file_is_rejected(self):
        (self.operating_directory / "nms_threshold_summary_sample5000.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            load_verified_operating_point(self.operating, self.selection)

    def test_evidence_hash_and_row_count_are_enforced(self):
        name = "nms_threshold_summary_sample5000.csv"
        for field, value, message in (
            ("sha256", "0" * 64, "hash mismatch"),
            ("rows", 6, "row-count mismatch"),
        ):
            with self.subTest(field=field):
                self.restore(self.operating)
                record = read_json(self.operating)
                record["evidence"][name][field] = value
                write_json(self.operating, record)
                with self.assertRaisesRegex(EvidenceContractError, message):
                    load_verified_operating_point(self.operating, self.selection)

    def test_empty_evidence_table_is_rejected_even_when_rehashed(self):
        name = "nms_threshold_summary_sample5000.csv"
        path = self.operating_directory / name
        columns, _ = read_csv(path)
        write_csv(path, columns, [])
        self.update_operating_evidence_identity(name)
        with self.assertRaisesRegex(EvidenceContractError, "must be positive"):
            load_verified_operating_point(self.operating, self.selection)

    def test_evidence_columns_are_exact(self):
        name = "nms_threshold_summary_sample5000.csv"
        path = self.operating_directory / name
        columns, rows = read_csv(path)
        columns.remove("eval_type")
        for row in rows:
            row.pop("eval_type")
        write_csv(path, columns, rows)
        self.update_operating_evidence_identity(name)
        with self.assertRaisesRegex(EvidenceContractError, "columns"):
            load_verified_operating_point(self.operating, self.selection)

    def test_rehashed_quality_change_recomputes_the_winner(self):
        name = "nms_threshold_summary_sample5000.csv"
        path = self.operating_directory / name
        columns, rows = read_csv(path)
        rows[0]["mAP@0.5_11_point"] = "0.99"
        write_csv(path, columns, rows)
        self.update_operating_evidence_identity(name)
        with self.assertRaisesRegex(EvidenceContractError, "threshold mismatch"):
            load_verified_operating_point(self.operating, self.selection)

    def test_evidence_tables_require_one_row_per_shared_threshold(self):
        name = "duplicate_summary_by_threshold_sample5000.csv"
        path = self.operating_directory / name
        columns, rows = read_csv(path)
        rows[0]["nms_threshold"] = "0.25"
        write_csv(path, columns, rows)
        self.update_operating_evidence_identity(name)
        with self.assertRaisesRegex(EvidenceContractError, "shared threshold"):
            load_verified_operating_point(self.operating, self.selection)

    def test_policy_constants_and_cross_table_counts_are_enforced(self):
        summary_name = "nms_threshold_summary_sample5000.csv"
        summary_path = self.operating_directory / summary_name
        columns, rows = read_csv(summary_path)
        rows[0]["score_threshold"] = "0.4"
        write_csv(summary_path, columns, rows)
        self.update_operating_evidence_identity(summary_name)
        with self.assertRaisesRegex(EvidenceContractError, "not constant"):
            load_verified_operating_point(self.operating, self.selection)

        self.restore(summary_path)
        self.update_operating_evidence_identity(summary_name)
        duplicate_name = "duplicate_summary_by_threshold_sample5000.csv"
        duplicate_path = self.operating_directory / duplicate_name
        columns, rows = read_csv(duplicate_path)
        rows[0]["total_predictions_after_nms"] = "1"
        write_csv(duplicate_path, columns, rows)
        self.update_operating_evidence_identity(duplicate_name)
        with self.assertRaisesRegex(EvidenceContractError, "prediction totals disagree"):
            load_verified_operating_point(self.operating, self.selection)


if __name__ == "__main__":
    unittest.main()
