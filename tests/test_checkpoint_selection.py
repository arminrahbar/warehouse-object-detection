import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "scripts"
    / "01_model_selection"
    / "01_select_checkpoint.py"
)
spec = importlib.util.spec_from_file_location("checkpoint_selection_under_test", SCRIPT_PATH)
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def file_identity(path, columns):
    return {
        "sha256": selection._sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": selection._csv_row_count(path),
        "columns": list(columns),
    }


def runtime_file_identity(path):
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": selection._sha256_file(path),
        "rows": selection._csv_row_count(path),
    }


class EvidenceFixture:
    def __init__(self, root, bootstrap_samples=20, seed=20260821):
        self.root = Path(root)
        self.quality = self.root / "quality"
        self.runtime = self.root / "runtime"
        self.output = self.root / "selection"
        self.quality.mkdir()
        self.runtime.mkdir()
        self.bootstrap_samples = bootstrap_samples
        self.seed = seed
        self.classes = [f"class_{index}" for index in range(20)]
        self.images = ["source_a_jpg.rf.one.jpg", "source_b_jpg.rf.two.jpg"]
        self.paths = [f"logistics/{name}" for name in self.images]
        self.dataset_hash = "a" * 64
        self.model_hashes = {
            "model1": {"weights": "1" * 64, "cfg": "c" * 64, "names": "e" * 64},
            "model2": {"weights": "2" * 64, "cfg": "c" * 64, "names": "e" * 64},
        }
        self._build_quality()
        self._build_runtime()

    def quality_identity(self, model_name, asset_name):
        return {
            "path": str(self.root / f"{model_name}-{asset_name}"),
            "size_bytes": 10,
            "sha256": self.model_hashes[model_name][asset_name],
        }

    def runtime_identity(self, model_name, asset_name):
        return {
            "path": str(self.root / f"{model_name}-{asset_name}"),
            "bytes": 10,
            "sha256": self.model_hashes[model_name][asset_name],
        }

    def _build_quality(self):
        ground_truth = [
            {
                "image_file": image, "image_path": path, "class_id": 0,
                "class_name": self.classes[0], "bbox_x": 0, "bbox_y": 0,
                "bbox_w": 10, "bbox_h": 10,
            }
            for image, path in zip(self.images, self.paths)
        ]
        model2_predictions = [
            {
                "model": "model2", "image_index": index,
                "image_file": image, "image_path": path,
                "bbox_x": 0, "bbox_y": 0, "bbox_w": 10, "bbox_h": 10,
                "class_id": 0, "class_name": self.classes[0],
                "object_score": 1.0, "predicted_class_score": 1.0,
                "combined_confidence": 1.0, "nms_iou_threshold": 0.3,
            }
            for index, (image, path) in enumerate(zip(self.images, self.paths), start=1)
        ]
        ledger = []
        for model_name, prediction_count in (("model1", 0), ("model2", 1)):
            for index, (image, path) in enumerate(zip(self.images, self.paths), start=1):
                ledger.append({
                    "model": model_name, "image_index": index, "image_file": image,
                    "image_path": path, "status": "processed",
                    "raw_candidate_count": prediction_count,
                    "post_nms_prediction_count": prediction_count,
                    "read_seconds": 0.001, "predict_seconds": 0.01,
                    "postprocess_seconds": 0.001, "nms_seconds": 0.001,
                    "total_seconds": 0.013,
                })
        aggregate = []
        for model_name, quality_value, predictions in (
            ("model1", 0.0, 0), ("model2", 0.05, 2)
        ):
            aggregate.append({
                "model": model_name, "images_evaluated": 2,
                "total_ground_truth": 2, "low_floor_predictions": predictions,
                "deployment_predictions": predictions,
                "mAP50_101pt": quality_value,
                "threshold_constrained_mAP50_11pt": quality_value,
                "deployment_true_positives": predictions,
                "deployment_false_positives": 0,
                "deployment_false_negatives": 2 - predictions,
                "deployment_micro_precision": 1.0 if predictions else 0.0,
                "deployment_micro_recall": 1.0 if predictions else 0.0,
                "deployment_micro_f1": 1.0 if predictions else 0.0,
                "deployment_macro_precision": quality_value,
                "deployment_macro_recall": quality_value,
                "deployment_macro_f1": quality_value,
                "candidate_floor": 0.001, "deployment_confidence": 0.5,
                "nms_iou_threshold": 0.3, "map_iou_threshold": 0.5,
            })
        per_class = []
        for model_name in ("model1", "model2"):
            winning = model_name == "model2"
            for class_id, class_name in enumerate(self.classes):
                supported = class_id == 0
                ground_truth_count = 2 if supported else 0
                prediction_count = 2 if winning and supported else 0
                score = 1.0 if winning and supported else 0.0
                per_class.append({
                    "model": model_name, "class_id": class_id,
                    "class_name": class_name, "ground_truth_count": ground_truth_count,
                    "low_floor_prediction_count": prediction_count,
                    "ap50_101pt": score,
                    "deployment_prediction_count": prediction_count,
                    "deployment_true_positives": prediction_count,
                    "deployment_false_positives": 0,
                    "deployment_false_negatives": ground_truth_count - prediction_count,
                    "deployment_precision": score, "deployment_recall": score,
                    "deployment_f1": score,
                    "threshold_constrained_ap50_11pt": score,
                })
        rows = {
            "ground_truth.csv": ground_truth,
            "model1_predictions.csv": [],
            "model2_predictions.csv": model2_predictions,
            "inference_ledger.csv": ledger,
            "aggregate_metrics.csv": aggregate,
            "per_class_metrics.csv": per_class,
        }
        for name, table_rows in rows.items():
            write_csv(self.quality / name, selection.QUALITY_SCHEMAS[name], table_rows)
        artifacts = {
            name: file_identity(self.quality / name, columns)
            for name, columns in selection.QUALITY_SCHEMAS.items()
        }
        source_files = {
            name: {"path": name, "size_bytes": 100, "sha256": "f" * 64}
            for name in selection.QUALITY_SOURCE_FILES
        }
        models = {
            model_name: {
                asset_name: self.quality_identity(model_name, asset_name)
                for asset_name in ("weights", "cfg", "names")
            } for model_name in ("model1", "model2")
        }
        dataset = {
            "path": str(self.root / "dataset_index.csv"), "size_bytes": 100,
            "sha256": self.dataset_hash, "selected_images": 2,
            "selected_labels": 2, "selection_max_images": None,
            "asset_identity": {
                "images": 2, "labels": 2, "files": 4, "size_bytes": 200,
                "sha256": "d" * 64,
            },
        }
        input_identities = {
            "dataset_index": {
                key: dataset[key] for key in ("path", "size_bytes", "sha256")
            },
            "dataset_assets": dataset["asset_identity"],
            "models": models, "source_files": source_files,
        }
        manifest = {
            "schema_version": 2, "run_id": "synthetic-quality", "run_scope": "full",
            "status": "complete",
            "started_utc": "2026-08-21T00:00:00+00:00",
            "completed_utc": "2026-08-21T00:01:00+00:00",
            "dataset": dataset,
            "external_storage_root": str(self.root / "assets"),
            "models": models,
            "class_vocabulary": self.classes,
            "policy": dict(selection.LOCKED_QUALITY_POLICY),
            "policy_sha256": selection._sha256_json(selection.LOCKED_QUALITY_POLICY),
            "source_files": source_files,
            "source_policy_sha256": selection._sha256_json({
                "policy": selection.LOCKED_QUALITY_POLICY, "source_files": source_files
            }),
            "input_identities_sha256": selection._sha256_json(input_identities),
            "environment": {"python_version": "test"}, "command": ["synthetic"],
            "inference_summaries": {}, "artifacts": artifacts,
        }
        (self.quality / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _build_runtime(self):
        timings = {"model1": 0.010, "model2": 0.012}
        observations, pairs = [], []
        for position, (image, path) in enumerate(zip(self.images, self.paths), start=1):
            first_model = "model1" if position == 1 else "model2"
            order = [first_model, "model2" if first_model == "model1" else "model1"]
            for execution_order, model_name in enumerate(order, start=1):
                value = timings[model_name]
                observations.append({
                    "model": model_name, "repeat_index": 1,
                    "sample_position": position, "image_file": image,
                    "image_path": path, "status": "processed", "detections": 0,
                    "read_seconds": 0.001, "predict_seconds": value,
                    "postprocess_seconds": 0.0, "nms_seconds": 0.0,
                    "total_seconds": value, "benchmark_mode": "paired",
                    "density_bucket": "1", "execution_order": execution_order,
                    "compute_seconds": value,
                })
            pairs.append({
                "record_type": "pair", "repeat_index": 1,
                "sample_position": position, "image_file": image, "image_path": path,
                "source_group": selection.source_group_key(image), "density_bucket": "1",
                "first_model": first_model, "model1_compute_ms": 10.0,
                "model2_compute_ms": 12.0, "delta_model2_minus_model1_ms": 2.0,
                "faster_model": "model1",
            })
        runtime_ci = selection._runtime_bootstrap(
            pairs, self.bootstrap_samples, self.seed
        )
        aggregate = {
            "record_type": "aggregate", "source_groups": 2, "pairs": 2,
            "bootstrap_samples": self.bootstrap_samples, "seed": self.seed,
            "model1_median_ms": 10.0, "model2_median_ms": 12.0,
            "relative_median_difference_pct": 20.0,
            "model1_p95_ms": 10.0, "model2_p95_ms": 12.0,
            "p95_delta_model2_minus_model1_ms": 2.0,
            "relative_p95_difference_pct": 20.0,
            "p95_delta_model2_minus_model1_ci_lower_ms": runtime_ci["p95_lower"],
            "p95_delta_model2_minus_model1_ci_upper_ms": runtime_ci["p95_upper"],
            "relative_p95_difference_ci_lower_pct": 20.0,
            "relative_p95_difference_ci_upper_pct": 20.0,
            "mean_delta_model2_minus_model1_ms": 2.0,
            "relative_mean_difference_pct": 20.0,
            "mean_delta_ci_lower_ms": runtime_ci["mean_lower"],
            "mean_delta_ci_upper_ms": runtime_ci["mean_upper"],
            "relative_mean_difference_ci_lower_pct": 20.0,
            "relative_mean_difference_ci_upper_pct": 20.0,
        }
        comparison_rows = pairs + [aggregate]
        summaries = []
        for model_name, value in timings.items():
            summaries.append({
                "model": model_name, "images_in_dataset": 2, "sample_requested": 2,
                "unique_images_selected": 2, "repeats": 1, "warmup_images": 1,
                "successful_observations": 2, "unreadable_observations": 0,
                "total_detections": 0, "pipeline_setup_seconds": 0.1,
                "measured_wall_seconds": 0.1, "mean_seconds_per_image": value,
                "median_seconds_per_image": value, "p95_seconds_per_image": value,
                "images_per_second": 1 / value,
                "estimated_full_dataset_minutes": value * 2 / 60,
                "candidate_threshold": 0.5, "confidence_threshold": 0.5,
                "nms_iou_threshold": 0.3, "python_version": "test",
                "opencv_version": "test", "platform": "test",
                "benchmark_mode": "paired", "seed": self.seed,
                "sample_selection": "density_stratified",
                "mean_compute_seconds": value, "median_compute_seconds": value,
                "p95_compute_seconds": value,
            })
        write_csv(self.runtime / "inference_benchmark_summary.csv",
                  selection.RUNTIME_SUMMARY_COLUMNS, summaries)
        write_csv(self.runtime / "inference_benchmark_observations.csv",
                  selection.RUNTIME_OBSERVATION_COLUMNS, observations)
        write_csv(self.runtime / "paired_latency_comparison.csv",
                  selection.RUNTIME_COMPARISON_COLUMNS, comparison_rows)
        artifacts = {
            key: runtime_file_identity(self.runtime / filename)
            for key, (filename, _) in selection.RUNTIME_ARTIFACTS.items()
        }
        policy = {
            "candidate_objectness_threshold": 0.5,
            "combined_confidence_threshold": 0.5, "nms_iou_threshold": 0.3,
            "sample_selection": "seeded_density_stratified", "sample_seed": self.seed,
            "repeats": 1, "warmup_images_per_model": 1,
            "timing_scope": "predict + post_process + class-aware NMS",
            "frame_decode_timed_separately": True,
            "execution_order": "seeded alternating first checkpoint",
            "p95_estimator": "linear interpolation",
            "bootstrap_unit": "source group preserving variants and repeats",
            "bootstrap_samples": self.bootstrap_samples, "bootstrap_seed": self.seed,
        }
        core = {
            "schema_version": 1, "status": "complete", "benchmark_mode": "paired",
            "output_directory": str(self.runtime),
            "dataset": {
                "index": {"path": str(self.root / "dataset_index.csv"), "bytes": 100,
                          "sha256": self.dataset_hash},
                "images_in_index": 2, "ordered_sample_images": 2,
                "ordered_sample_sha256": "b" * 64, "source_groups": 2,
                "source_group_policy": "prefix before '_jpg.rf.'; otherwise complete image_file",
            },
            "models": {
                model_name: {
                    asset_name: self.runtime_identity(model_name, asset_name)
                    for asset_name in ("weights", "cfg", "names")
                } for model_name in ("model1", "model2")
            },
            "runtime": {"python_version": "test", "opencv_version": "test",
                        "platform": "test"},
            "policy": policy,
            "completeness": {
                "expected_unique_images": 2, "measured_unique_images": 2,
                "expected_pairs": 2, "measured_pairs": 2,
                "expected_observations": 4, "processed_observations": 4,
                "unreadable_observations": 0,
            },
            "artifacts": artifacts,
        }
        manifest = {
            **core, "run_fingerprint_sha256": selection._sha256_json(core),
            "created_at_utc": "2026-08-21T00:02:00+00:00",
        }
        (self.runtime / "inference_benchmark_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def rewrite_runtime_manifest(self, mutate):
        path = self.runtime / "inference_benchmark_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        mutate(manifest)
        core = {key: value for key, value in manifest.items()
                if key not in {"run_fingerprint_sha256", "created_at_utc"}}
        manifest["run_fingerprint_sha256"] = selection._sha256_json(core)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")


class SourceGroupingTests(unittest.TestCase):
    def test_locked_prefix_and_safe_fallback(self):
        self.assertEqual(selection.source_group_key("capture_jpg.rf.one.jpg"), "capture")
        self.assertEqual(selection.source_group_key("plain.jpg"), "plain.jpg")
        self.assertEqual(
            selection.source_group_key("_jpg.rf.leading.jpg"), "_jpg.rf.leading.jpg"
        )
        with self.assertRaises(selection.IntegrityError):
            selection.source_group_key("   ")


class ContractTests(unittest.TestCase):
    def test_cli_defaults_lock_confirmatory_bootstrap_and_corpus(self):
        args = selection.build_parser().parse_args([
            "--quality-run", "quality", "--runtime-run", "runtime",
            "--run-id", "selection-1",
        ])
        self.assertEqual(args.bootstrap_samples, 2000)
        self.assertEqual(args.seed, 20260821)
        self.assertEqual(args.expected_images, 9525)
        self.assertEqual(args.expected_labels, 36721)
        self.assertEqual(
            args.output_root,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "01_model_selection"
            / "03_checkpoint_decision",
        )

    def test_output_contract_has_four_immutable_artifacts(self):
        self.assertEqual(
            selection.SUMMARY_COLUMNS[0:4],
            ["metric", "model1_value", "model2_value",
             "delta_model2_minus_model1"],
        )
        self.assertIn("sampled_source_group_draw_sha256",
                      selection.BOOTSTRAP_COLUMNS)
        self.assertFalse(selection.RUN_ID_PATTERN.fullmatch("../escape"))
        self.assertFalse(selection.RUN_ID_PATTERN.fullmatch(".hidden"))


class DecisionRuleTests(unittest.TestCase):
    @staticmethod
    def metrics(ap_delta=0.0, f1_delta=0.0):
        return {
            "model1": {"mAP50_101pt": 0.4, "deployment_macro_f1": 0.4},
            "model2": {"mAP50_101pt": 0.4 + ap_delta,
                       "deployment_macro_f1": 0.4 + f1_delta},
        }

    @staticmethod
    def intervals(ap=(-0.001, 0.001), f1=(-0.001, 0.001)):
        return {"delta_ap_ci_lower": ap[0], "delta_ap_ci_upper": ap[1],
                "delta_f1_ci_lower": f1[0], "delta_f1_ci_upper": f1[1]}

    @staticmethod
    def runtime(a_p95=10.0, b_p95=10.0, p95_ci=(-0.1, 0.1),
                a_mean=10.0, b_mean=10.0):
        return {"model1_p95_ms": a_p95, "model2_p95_ms": b_p95,
                "p95_ci_lower": p95_ci[0], "p95_ci_upper": p95_ci[1],
                "model1_mean_ms": a_mean, "model2_mean_ms": b_mean}

    def test_ap_wins_and_losses_at_exact_effect_boundary(self):
        decision = selection.apply_selection_rule(
            self.metrics(ap_delta=0.01), self.intervals(ap=(0.0001, 0.02)),
            self.runtime(),
        )
        self.assertEqual((decision["step"], decision["selected_model"]), (1, "model2"))
        decision = selection.apply_selection_rule(
            self.metrics(ap_delta=-0.01), self.intervals(ap=(-0.02, -0.0001)),
            self.runtime(),
        )
        self.assertEqual((decision["step"], decision["selected_model"]), (1, "model1"))

    def test_zero_touching_interval_does_not_qualify(self):
        decision = selection.apply_selection_rule(
            self.metrics(ap_delta=0.02), self.intervals(ap=(0.0, 0.03)),
            self.runtime(a_mean=9.0, b_mean=10.0),
        )
        self.assertEqual(decision["step"], 4)

    def test_f1_and_p95_branches_are_lexicographic(self):
        decision = selection.apply_selection_rule(
            self.metrics(f1_delta=0.01), self.intervals(f1=(0.001, 0.02)),
            self.runtime(a_p95=10.0, b_p95=8.0, p95_ci=(-3, -1)),
        )
        self.assertEqual((decision["step"], decision["selected_model"]), (2, "model2"))
        decision = selection.apply_selection_rule(
            self.metrics(), self.intervals(),
            self.runtime(a_p95=9.5, b_p95=10.0, p95_ci=(0.1, 0.9)),
        )
        self.assertEqual((decision["step"], decision["selected_model"]), (3, "model1"))

    def test_operational_equivalence_uses_mean_then_stable_model_order(self):
        decision = selection.apply_selection_rule(
            self.metrics(), self.intervals(), self.runtime(a_mean=11, b_mean=10)
        )
        self.assertEqual(decision["status"], "operationally_equivalent_under_protocol")
        self.assertEqual(decision["selected_model"], "model2")
        equal = selection.apply_selection_rule(
            self.metrics(), self.intervals(), self.runtime(a_mean=10, b_mean=10)
        )
        self.assertEqual(equal["selected_model"], "model1")


class MatchingAndBootstrapTests(unittest.TestCase):
    def test_duplicate_detection_is_one_tp_and_one_fp(self):
        predictions = selection.pd.DataFrame([
            {"bbox_x": 0, "bbox_y": 0, "bbox_w": 10, "bbox_h": 10,
             "combined_confidence": 0.9, "_row_order": 0},
            {"bbox_x": 0, "bbox_y": 0, "bbox_w": 10, "bbox_h": 10,
             "combined_confidence": 0.8, "_row_order": 1},
        ])
        labels = selection.pd.DataFrame([
            {"bbox_x": 0, "bbox_y": 0, "bbox_w": 10, "bbox_h": 10}
        ])
        matched, order = selection._match_image_class(predictions, labels)
        self.assertEqual(order.tolist(), [0, 1])
        self.assertEqual(matched.tolist(), [True, False])

    def test_bootstrap_is_deterministic_and_source_group_paired(self):
        events_a = [{"scores": np.array([0.9]), "matches": np.array([1]),
                     "groups": np.array([0])}] + [
            {"scores": np.array([]), "matches": np.array([], dtype=np.int8),
             "groups": np.array([], dtype=np.int64)} for _ in range(19)
        ]
        events_b = [{"scores": np.array([0.9, 0.8]), "matches": np.array([1, 1]),
                     "groups": np.array([0, 1])}] + [
            {"scores": np.array([]), "matches": np.array([], dtype=np.int8),
             "groups": np.array([], dtype=np.int64)} for _ in range(19)
        ]
        base = {
            "source_groups": ["a", "b"],
            "group_gt": np.array([[1] + [0] * 19, [1] + [0] * 19]),
            "group_deployment_tp": np.zeros((2, 20), dtype=int),
            "group_deployment_fp": np.zeros((2, 20), dtype=int),
        }
        first = {**base, "events": events_a}
        second = {**base, "events": events_b}
        rows1, summary1 = selection.paired_quality_bootstrap(first, second, 25, 7)
        rows2, summary2 = selection.paired_quality_bootstrap(first, second, 25, 7)
        self.assertEqual(rows1, rows2)
        self.assertEqual(summary1, summary2)
        self.assertTrue(all(len(row["sampled_source_group_draw_sha256"]) == 64
                            for row in rows1))

    def test_batched_metrics_equal_scalar_weighted_metrics(self):
        events = [{
            "scores": np.array([0.9, 0.7, 0.5]),
            "matches": np.array([1, 0, 1], dtype=np.int8),
            "groups": np.array([0, 1, 1], dtype=np.int64),
        }] + [
            {"scores": np.array([]), "matches": np.array([], dtype=np.int8),
             "groups": np.array([], dtype=np.int64)} for _ in range(19)
        ]
        deployment_tp = np.zeros((2, 20), dtype=int)
        deployment_fp = np.zeros((2, 20), dtype=int)
        deployment_tp[:, 0] = [1, 1]
        deployment_fp[:, 0] = [0, 1]
        evidence = {
            "source_groups": ["a", "b"], "events": events,
            "group_gt": np.array([[1] + [0] * 19, [1] + [0] * 19]),
            "group_deployment_tp": deployment_tp,
            "group_deployment_fp": deployment_fp,
        }
        weights = np.array([[1, 1], [0, 2], [2, 0]], dtype=np.int64)
        batch_ap, batch_f1 = selection._quality_metrics_batch(evidence, weights)
        scalar = [selection.quality_metrics_for_weights(evidence, row) for row in weights]
        np.testing.assert_allclose(
            batch_ap, [row["mAP50_101pt"] for row in scalar], atol=1e-12
        )
        np.testing.assert_allclose(
            batch_f1, [row["deployment_macro_f1"] for row in scalar], atol=1e-12
        )


class EvidenceIntegrityAndEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = EvidenceFixture(self.temp.name)

    def validate_quality(self):
        return selection.validate_quality_run(
            self.fixture.quality, expected_images=2, expected_labels=2,
            locked_hashes=None,
        )

    def test_quality_artifact_tampering_is_rejected(self):
        path = self.fixture.quality / "model2_predictions.csv"
        with path.open("a", encoding="utf-8") as destination:
            destination.write("tampered\n")
        with self.assertRaisesRegex(selection.IntegrityError, "size|hash|row"):
            self.validate_quality()

    def test_cross_stage_model_identity_mismatch_is_rejected(self):
        quality = self.validate_quality()
        self.fixture.rewrite_runtime_manifest(
            lambda manifest: manifest["models"]["model2"]["weights"].update(
                sha256="9" * 64
            )
        )
        with self.assertRaisesRegex(selection.IntegrityError, "identities differ"):
            selection.validate_runtime_run(
                self.fixture.runtime, quality,
                self.fixture.bootstrap_samples, self.fixture.seed,
            )

    def test_end_to_end_selects_b_and_writes_exact_immutable_artifacts(self):
        final, decision, manifest = selection.run_selection(
            self.fixture.quality, self.fixture.runtime, self.fixture.output,
            "synthetic-selection", bootstrap_samples=self.fixture.bootstrap_samples,
            seed=self.fixture.seed, expected_images=2, expected_labels=2,
            locked_hashes=None,
        )
        self.assertEqual((decision["step"], decision["selected_model"]), (1, "model2"))
        self.assertEqual(
            {path.name for path in final.iterdir()},
            {"selection_summary.csv", "bootstrap_replicates.csv", "decision.json",
             "selection_manifest.json"},
        )
        self.assertEqual(manifest["corpus"]["source_groups"], 2)
        with self.assertRaises(FileExistsError):
            selection.run_selection(
                self.fixture.quality, self.fixture.runtime, self.fixture.output,
                "synthetic-selection", bootstrap_samples=self.fixture.bootstrap_samples,
                seed=self.fixture.seed, expected_images=2, expected_labels=2,
                locked_hashes=None,
            )


if __name__ == "__main__":
    unittest.main()
