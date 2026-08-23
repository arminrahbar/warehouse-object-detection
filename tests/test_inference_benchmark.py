import argparse
import csv
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "scripts" / "01_benchmark_inference.py"

spec = importlib.util.spec_from_file_location("inference_benchmark_under_test", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def write_csv(path, fieldnames, rows):
    benchmark._filesystem_path(path.parent).mkdir(parents=True, exist_ok=True)
    with benchmark._filesystem_path(path).open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with benchmark._filesystem_path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        return reader.fieldnames, list(reader)


class IncrementingClock:
    def __init__(self, step=0.01):
        self.value = 0.0
        self.step = step

    def __call__(self):
        current = self.value
        self.value += self.step
        return current


class FakeDetector:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.predict_calls = []
        self.post_process_calls = []

    def predict(self, frame):
        self.predict_calls.append(frame)
        return [f"raw-{frame}"]

    def post_process(self, outputs):
        self.post_process_calls.append(outputs)
        return (
            [[1, 2, 3, 4]],
            [0],
            [0.9],
            [[0.8, 0.2]],
        )


class FakeNMS:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.filter_calls = []

    def filter(self, *decoded):
        self.filter_calls.append(decoded)
        return decoded


class BenchmarkFixture(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.root = Path(self.temp_directory.name)
        self.asset_root = self.root / "assets"
        self.asset_root.mkdir()
        self.output_dir = self.root / "outputs"
        self.instances = {"detectors": [], "nms": []}

        for paths in benchmark.MODELS.values():
            for relative_path in paths.values():
                path = self.asset_root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"placeholder")

        self.sample_rows = [
            {
                "image_file": "a.jpg",
                "image_path": "detector_service/storage/logistics/a.jpg",
            },
            {
                "image_file": "b.jpg",
                "image_path": "detector_service/storage/logistics/b.jpg",
            },
        ]
        for row in self.sample_rows:
            path = self.asset_root / row["image_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"image")

    def detector_factory(self, *args, **kwargs):
        instance = FakeDetector(*args, **kwargs)
        self.instances["detectors"].append(instance)
        return instance

    def nms_factory(self, *args, **kwargs):
        instance = FakeNMS(*args, **kwargs)
        self.instances["nms"].append(instance)
        return instance

    @staticmethod
    def image_reader(path):
        return Path(path).name

    def dependencies(self, clock=None):
        return {
            "detector_factory": self.detector_factory,
            "nms_factory": self.nms_factory,
            "image_reader": self.image_reader,
            "clock": clock or IncrementingClock(),
            "runtime_metadata": {
                "python_version": "test-python",
                "opencv_version": "test-opencv",
                "platform": "test-platform",
            },
        }


class BenchmarkContractTests(BenchmarkFixture):
    def test_default_paths_follow_the_canonical_stage_hierarchy(self):
        self.assertEqual(
            benchmark.DEFAULT_DATASET_INDEX,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "00_dataset_inventory"
            / "dataset_index.csv",
        )
        self.assertEqual(
            benchmark.DEFAULT_OUTPUT_ROOT,
            PROJECT_ROOT
            / "experiments"
            / "outputs"
            / "01_model_selection"
            / "02_runtime_benchmark",
        )

    def test_defaults_and_model_order_are_explicit(self):
        self.assertEqual(benchmark.DEFAULT_SAMPLE_SIZE, 100)
        self.assertEqual(benchmark.DEFAULT_REPEATS, 1)
        self.assertEqual(benchmark.DEFAULT_WARMUP_IMAGES, 1)
        self.assertEqual(benchmark.DEFAULT_SEED, 42)
        self.assertEqual(benchmark.DEFAULT_BOOTSTRAP_SAMPLES, 2000)
        self.assertEqual(list(benchmark.MODELS), ["model1", "model2"])
        self.assertTrue(
            all(
                set(paths) == {"weights", "cfg", "names"}
                for paths in benchmark.MODELS.values()
            )
        )
        self.assertTrue(
            all(
                not path.is_absolute()
                for paths in benchmark.MODELS.values()
                for path in paths.values()
            )
        )

    def test_direct_script_execution_bootstraps_the_project_package_root(self):
        self.assertIn(str(benchmark.PROJECT_ROOT), benchmark.sys.path)

    def test_benchmark_policy_matches_the_inference_pipeline(self):
        comparison_path = (
            PROJECT_ROOT / "experiments" / "scripts" / "01_model_comparison.py"
        )
        comparison_spec = importlib.util.spec_from_file_location(
            "comparison_for_benchmark_contract",
            comparison_path,
        )
        comparison = importlib.util.module_from_spec(comparison_spec)
        comparison_spec.loader.exec_module(comparison)

        self.assertEqual(
            benchmark.CANDIDATE_THRESHOLD,
            comparison.DETECTOR_OBJECTNESS_THRESHOLD,
        )
        self.assertEqual(
            benchmark.CONFIDENCE_THRESHOLD,
            comparison.NMS_CONFIDENCE_THRESHOLD,
        )
        self.assertEqual(benchmark.NMS_IOU_THRESHOLD, comparison.NMS_THRESHOLD)
        self.assertEqual(benchmark.MODELS, comparison.MODELS)

    def test_observation_and_summary_schemas_are_stable(self):
        self.assertEqual(benchmark.OBSERVATION_COLUMNS[0:6], [
            "model",
            "repeat_index",
            "sample_position",
            "image_file",
            "image_path",
            "status",
        ])
        for column in (
            "mean_seconds_per_image",
            "p95_seconds_per_image",
            "images_per_second",
            "estimated_full_dataset_minutes",
            "opencv_version",
            "platform",
            "mean_compute_seconds",
        ):
            self.assertIn(column, benchmark.SUMMARY_COLUMNS)
        for column in (
            "compute_seconds",
            "benchmark_mode",
            "execution_order",
        ):
            self.assertIn(column, benchmark.OBSERVATION_COLUMNS)
        self.assertIn(
            "delta_model2_minus_model1_ms",
            benchmark.PAIRED_COMPARISON_COLUMNS,
        )
        self.assertIn(
            "relative_mean_difference_ci_upper_pct",
            benchmark.PAIRED_COMPARISON_COLUMNS,
        )
        for column in (
            "source_group",
            "p95_delta_model2_minus_model1_ms",
            "p95_delta_model2_minus_model1_ci_lower_ms",
            "p95_delta_model2_minus_model1_ci_upper_ms",
            "relative_p95_difference_ci_lower_pct",
            "relative_p95_difference_ci_upper_pct",
        ):
            self.assertIn(column, benchmark.PAIRED_COMPARISON_COLUMNS)

    def test_cli_integer_validators_reject_invalid_values(self):
        self.assertEqual(benchmark.positive_int("3"), 3)
        self.assertEqual(benchmark.nonnegative_int("0"), 0)
        with self.assertRaisesRegex(Exception, "positive integer"):
            benchmark.positive_int("0")
        with self.assertRaisesRegex(Exception, "non-negative integer"):
            benchmark.nonnegative_int("-1")

    def test_cli_requires_a_safe_run_id(self):
        parser = benchmark.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        self.assertEqual(
            parser.parse_args(["--run-id", "runtime-500x3-v1"]).run_id,
            "runtime-500x3-v1",
        )
        for value in ("../escape", ".hidden", "white space", ".", ".."):
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                benchmark.validate_run_id(value)

    def test_filesystem_adapter_preserves_canonical_evidence_paths(self):
        canonical = (self.root / ("x" * 251 + ".jpg")).absolute()
        filesystem = benchmark._filesystem_path(canonical)
        self.assertEqual(benchmark._normal_path(filesystem), canonical)
        if os.name == "nt":
            self.assertTrue(str(filesystem).startswith("\\\\?\\"))
            self.assertEqual(
                benchmark._filesystem_path(filesystem), filesystem
            )
        else:
            self.assertEqual(filesystem, canonical)


class DatasetLoadingTests(BenchmarkFixture):
    def test_sample_is_the_deterministic_index_prefix(self):
        index_path = self.root / "dataset_index.csv"
        rows = [
            {"image_file": "a.jpg", "image_path": "images/a.jpg", "extra": "1"},
            {"image_file": "b.jpg", "image_path": "images/b.jpg", "extra": "2"},
            {"image_file": "c.jpg", "image_path": "images/c.jpg", "extra": "3"},
        ]
        write_csv(index_path, ["image_file", "image_path", "extra"], rows)

        sample, total = benchmark.load_dataset_sample(index_path, 2)

        self.assertEqual(sample, rows[:2])
        self.assertEqual(total, 3)

    def test_dataset_index_beyond_max_path_loads_normally(self):
        deep = self.root / ("i" * 200)
        index_path = deep / "dataset_index.csv"
        write_csv(
            index_path,
            ["image_file", "image_path"],
            [{"image_file": "a.jpg", "image_path": "logistics/a.jpg"}],
        )
        self.addCleanup(benchmark._filesystem_path(deep).rmdir)
        self.addCleanup(
            benchmark._filesystem_path(index_path).unlink, missing_ok=True
        )
        sample, total = benchmark.load_dataset_sample(index_path, 1)
        self.assertEqual(total, 1)
        self.assertEqual(sample[0]["image_file"], "a.jpg")

    def test_missing_index_has_actionable_message(self):
        with self.assertRaisesRegex(FileNotFoundError, "00_build_dataset_inventory.py"):
            benchmark.load_dataset_sample(self.root / "missing.csv", 2)

    def test_required_columns_are_validated(self):
        path = self.root / "dataset_index.csv"
        write_csv(path, ["image_file"], [{"image_file": "a.jpg"}])
        with self.assertRaisesRegex(ValueError, "image_path"):
            benchmark.load_dataset_sample(path, 1)

    def test_empty_index_is_rejected(self):
        path = self.root / "dataset_index.csv"
        write_csv(path, ["image_file", "image_path"], [])
        with self.assertRaisesRegex(RuntimeError, "empty"):
            benchmark.load_dataset_sample(path, 1)

    def test_duplicate_image_names_are_rejected(self):
        path = self.root / "dataset_index.csv"
        write_csv(
            path,
            ["image_file", "image_path"],
            [
                {"image_file": "a.jpg", "image_path": "one/a.jpg"},
                {"image_file": "a.jpg", "image_path": "two/a.jpg"},
            ],
        )
        with self.assertRaisesRegex(ValueError, "duplicate image_file"):
            benchmark.load_dataset_sample(path, 2)

    def test_paired_sample_requires_valid_num_objects(self):
        path = self.root / "dataset_index.csv"
        write_csv(
            path,
            ["image_file", "image_path"],
            [{"image_file": "a.jpg", "image_path": "images/a.jpg"}],
        )
        with self.assertRaisesRegex(ValueError, "num_objects"):
            benchmark.load_dataset_sample(path, 1, paired=True)

        write_csv(
            path,
            ["image_file", "image_path", "num_objects"],
            [
                {
                    "image_file": "a.jpg",
                    "image_path": "images/a.jpg",
                    "num_objects": "not-an-integer",
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "invalid num_objects"):
            benchmark.load_dataset_sample(path, 1, paired=True)

    def test_paired_sample_is_seeded_exact_and_density_stratified(self):
        index_path = self.root / "dataset_index.csv"
        counts = [0, 1, 2, 5, 10, 15, 20]
        rows = []
        for count in counts:
            for suffix in range(3):
                rows.append(
                    {
                        "image_file": f"density-{count}-{suffix}.jpg",
                        "image_path": f"images/density-{count}-{suffix}.jpg",
                        "num_objects": str(count),
                    }
                )
        write_csv(
            index_path,
            ["image_file", "image_path", "num_objects"],
            rows,
        )

        first, total = benchmark.load_dataset_sample(
            index_path,
            7,
            paired=True,
            seed=19,
        )
        repeated, _ = benchmark.load_dataset_sample(
            index_path,
            7,
            paired=True,
            seed=19,
        )
        different, _ = benchmark.load_dataset_sample(
            index_path,
            7,
            paired=True,
            seed=20,
        )

        self.assertEqual(total, 21)
        self.assertEqual(len(first), 7)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)
        self.assertEqual(
            Counter(benchmark.density_bucket(row["num_objects"]) for row in first),
            Counter({name: 1 for name, _, _ in benchmark.DENSITY_BUCKETS}),
        )

    def test_paired_sample_rejects_request_larger_than_index(self):
        index_path = self.root / "dataset_index.csv"
        write_csv(
            index_path,
            ["image_file", "image_path", "num_objects"],
            [{"image_file": "a.jpg", "image_path": "a.jpg", "num_objects": "0"}],
        )

        with self.assertRaisesRegex(ValueError, "requested 2 images"):
            benchmark.load_dataset_sample(index_path, 2, paired=True, seed=1)


class NumericalUtilityTests(unittest.TestCase):
    def test_linear_percentile_handles_single_and_interpolated_inputs(self):
        self.assertEqual(benchmark.linear_percentile([4.0], 95), 4.0)
        self.assertEqual(benchmark.linear_percentile([0.0, 10.0], 50), 5.0)
        self.assertAlmostEqual(
            benchmark.linear_percentile([1.0, 2.0, 3.0, 4.0], 95),
            3.85,
        )

    def test_linear_percentile_validates_inputs(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            benchmark.linear_percentile([], 95)
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            benchmark.linear_percentile([1], 101)

    def test_source_group_uses_original_capture_prefix(self):
        self.assertEqual(
            benchmark.source_group_key("capture-42_jpg.rf.abc123.jpg"),
            "capture-42",
        )
        self.assertEqual(benchmark.source_group_key("ordinary.jpg"), "ordinary.jpg")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            benchmark.source_group_key("")


class ModelBenchmarkTests(BenchmarkFixture):
    def test_model_assets_are_resolved_beneath_external_root(self):
        resolved = benchmark.resolve_model_assets(
            benchmark.MODELS["model1"],
            self.asset_root,
        )
        self.assertTrue(all(path.is_file() for path in resolved.values()))
        self.assertEqual(
            resolved["weights"],
            (self.asset_root / benchmark.MODELS["model1"]["weights"]).absolute(),
        )

    def test_missing_model_assets_are_reported_together(self):
        missing_root = self.root / "missing-assets"
        missing_root.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "weights"):
            benchmark.resolve_model_assets(
                benchmark.MODELS["model1"],
                missing_root,
            )

    def test_external_storage_root_strips_both_supported_prefixes(self):
        storage_root = self.root / "external-storage"
        storage_root.mkdir()
        for relative_path in benchmark.MODELS["model1"].values():
            target = storage_root / benchmark._storage_relative_path(relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"asset")
        image = storage_root / "logistics" / "a.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")

        resolved = benchmark.resolve_model_assets(
            benchmark.MODELS["model1"],
            storage_root,
        )

        self.assertTrue(all(path.is_file() for path in resolved.values()))
        self.assertEqual(
            benchmark.resolve_indexed_asset_path(
                storage_root,
                "techtrack/storage/logistics/a.jpg",
            ),
            image.absolute(),
        )
        self.assertEqual(
            benchmark.resolve_indexed_asset_path(
                storage_root,
                "detector_service/storage/logistics/a.jpg",
            ),
            image.absolute(),
        )

    def test_255_character_image_resolves_hashes_and_decodes(self):
        storage_root = self.root / "long-storage"
        (storage_root / "logistics").mkdir(parents=True)
        image = storage_root / "logistics" / ("q" * 251 + ".jpg")
        filesystem = benchmark._filesystem_path(image)
        filesystem.write_bytes(b"benchmark-long-image")
        self.addCleanup(filesystem.unlink, missing_ok=True)
        logical = f"techtrack/storage/logistics/{image.name}"

        resolved = benchmark.resolve_indexed_asset_path(storage_root, logical)
        self.assertEqual(resolved, image.absolute())
        self.assertNotIn("\\\\?\\", str(resolved))
        fingerprint = benchmark._fingerprinted_file(resolved)
        self.assertEqual(fingerprint["path"], str(image.absolute()))
        self.assertEqual(fingerprint["bytes"], len(b"benchmark-long-image"))
        self.assertEqual(
            fingerprint["sha256"],
            hashlib.sha256(b"benchmark-long-image").hexdigest(),
        )

        class FakeCV2:
            IMREAD_COLOR = 7

            def imread(self, path):
                self.imread_path = path
                return "normal-decode"

            def imdecode(self, encoded, mode):
                self.encoded = bytes(encoded)
                self.mode = mode
                return "extended-decode"

        fake = FakeCV2()
        decoded = benchmark._read_image_cv2(resolved, cv2_module=fake)
        if os.name == "nt":
            self.assertEqual(decoded, "extended-decode")
            self.assertEqual(fake.encoded, b"benchmark-long-image")
            self.assertEqual(fake.mode, fake.IMREAD_COLOR)
        else:
            self.assertEqual(decoded, "normal-decode")
            self.assertEqual(fake.imread_path, str(image.absolute()))

        outside = self.root / "outside.jpg"
        outside.write_bytes(b"outside")
        with self.assertRaisesRegex(ValueError, "inside external storage"):
            benchmark.resolve_indexed_asset_path(storage_root, outside)

    def test_warmup_is_excluded_and_repeats_are_recorded(self):
        summary, observations = benchmark.benchmark_model(
            model_name="model1",
            paths=benchmark.MODELS["model1"],
            sample_rows=self.sample_rows,
            dataset_size=9525,
            asset_root=self.asset_root,
            repeats=2,
            warmup_images=1,
            **self.dependencies(),
        )

        self.assertEqual(len(observations), 4)
        self.assertEqual(summary["successful_observations"], 4)
        self.assertEqual(summary["unreadable_observations"], 0)
        self.assertEqual(summary["warmup_images"], 1)
        self.assertEqual(summary["total_detections"], 4)
        self.assertEqual(summary["repeats"], 2)
        self.assertEqual([row["repeat_index"] for row in observations], [1, 1, 2, 2])
        self.assertEqual(len(self.instances["detectors"][0].predict_calls), 5)
        self.assertAlmostEqual(summary["mean_seconds_per_image"], 0.04)
        self.assertAlmostEqual(summary["images_per_second"], 25.0)
        self.assertAlmostEqual(
            summary["estimated_full_dataset_minutes"],
            6.35,
        )

    def test_factories_receive_the_validated_operating_policy(self):
        summary, _ = benchmark.benchmark_model(
            model_name="model2",
            paths=benchmark.MODELS["model2"],
            sample_rows=self.sample_rows[:1],
            dataset_size=1,
            asset_root=self.asset_root,
            warmup_images=0,
            **self.dependencies(),
        )

        detector = self.instances["detectors"][0]
        nms = self.instances["nms"][0]
        self.assertEqual(
            detector.init_kwargs,
            {"score_threshold": benchmark.CANDIDATE_THRESHOLD},
        )
        self.assertEqual(
            nms.init_kwargs,
            {
                "score_threshold": benchmark.CONFIDENCE_THRESHOLD,
                "nms_iou_threshold": benchmark.NMS_IOU_THRESHOLD,
            },
        )
        self.assertEqual(summary["opencv_version"], "test-opencv")
        self.assertEqual(summary["platform"], "test-platform")

    def test_unreadable_images_have_explicit_observations(self):
        def reader(path):
            return None if Path(path).name == "b.jpg" else "a-frame"

        dependencies = self.dependencies()
        dependencies["image_reader"] = reader
        summary, observations = benchmark.benchmark_model(
            model_name="model1",
            paths=benchmark.MODELS["model1"],
            sample_rows=self.sample_rows,
            dataset_size=2,
            asset_root=self.asset_root,
            warmup_images=1,
            **dependencies,
        )

        self.assertEqual(summary["successful_observations"], 1)
        self.assertEqual(summary["unreadable_observations"], 1)
        unreadable = observations[1]
        self.assertEqual(unreadable["status"], "unreadable")
        self.assertEqual(unreadable["predict_seconds"], "")
        self.assertEqual(unreadable["detections"], 0)

    def test_all_unreadable_images_are_rejected(self):
        dependencies = self.dependencies()
        dependencies["image_reader"] = lambda path: None
        with self.assertRaisesRegex(RuntimeError, "No images were processed"):
            benchmark.benchmark_model(
                model_name="model1",
                paths=benchmark.MODELS["model1"],
                sample_rows=self.sample_rows,
                dataset_size=2,
                asset_root=self.asset_root,
                warmup_images=1,
                **dependencies,
            )

    def test_invalid_run_parameters_are_rejected(self):
        common = {
            "model_name": "model1",
            "paths": benchmark.MODELS["model1"],
            "sample_rows": self.sample_rows,
            "dataset_size": 2,
            "asset_root": self.asset_root,
            **self.dependencies(),
        }
        with self.assertRaisesRegex(ValueError, "repeats"):
            benchmark.benchmark_model(**common, repeats=0)
        with self.assertRaisesRegex(ValueError, "warmup_images"):
            benchmark.benchmark_model(**common, warmup_images=-1)


class PairedBenchmarkTests(BenchmarkFixture):
    def paired_rows(self):
        return [
            {**self.sample_rows[0], "num_objects": "1"},
            {**self.sample_rows[1], "num_objects": "20"},
        ]

    def test_decode_once_pairing_warmup_and_order_are_balanced(self):
        decoded_paths = []

        def reader(path):
            decoded_paths.append(path)
            return Path(path).name

        dependencies = self.dependencies(clock=IncrementingClock())
        dependencies["image_reader"] = reader
        summaries, observations, comparison = benchmark.benchmark_paired(
            sample_rows=self.paired_rows(),
            dataset_size=9525,
            asset_root=self.asset_root,
            repeats=2,
            warmup_images=1,
            seed=7,
            bootstrap_samples=25,
            **dependencies,
        )

        pair_rows = [row for row in comparison if row["record_type"] == "pair"]
        aggregate = comparison[-1]
        self.assertEqual(len(decoded_paths), 5)
        self.assertEqual(len(observations), 8)
        self.assertEqual(len(pair_rows), 4)
        self.assertEqual(Counter(row["first_model"] for row in pair_rows), {
            "model1": 2,
            "model2": 2,
        })
        self.assertEqual([row["model"] for row in summaries], ["model1", "model2"])
        self.assertTrue(all(row["benchmark_mode"] == "paired" for row in observations))
        for row in observations:
            self.assertAlmostEqual(row["read_seconds"], 0.01)
            self.assertAlmostEqual(row["compute_seconds"], 0.03)
            self.assertAlmostEqual(row["total_seconds"], 0.03)
        for row in pair_rows:
            self.assertAlmostEqual(row["delta_model2_minus_model1_ms"], 0)
        self.assertEqual(aggregate["pairs"], 4)
        self.assertEqual(aggregate["source_groups"], 2)
        self.assertEqual(aggregate["bootstrap_samples"], 25)
        self.assertAlmostEqual(aggregate["mean_delta_ci_lower_ms"], 0)
        self.assertAlmostEqual(aggregate["mean_delta_ci_upper_ms"], 0)
        self.assertEqual(len(self.instances["detectors"]), 2)
        self.assertEqual(
            [len(detector.predict_calls) for detector in self.instances["detectors"]],
            [5, 5],
        )

    def test_paired_schedule_is_deterministic_for_seed(self):
        def run(seed):
            self.instances = {"detectors": [], "nms": []}
            return benchmark.benchmark_paired(
                sample_rows=self.paired_rows(),
                dataset_size=2,
                asset_root=self.asset_root,
                repeats=1,
                warmup_images=0,
                seed=seed,
                bootstrap_samples=10,
                **self.dependencies(),
            )[2]

        first = run(11)
        repeated = run(11)
        self.assertEqual(
            [row.get("first_model") for row in first],
            [row.get("first_model") for row in repeated],
        )

    def test_paired_bootstrap_resamples_image_groups_with_seed(self):
        rows = [
            {
                "image_file": "a.jpg",
                "model1_compute_ms": 10.0,
                "model2_compute_ms": 12.0,
                "delta_model2_minus_model1_ms": 2.0,
            },
            {
                "image_file": "a.jpg",
                "model1_compute_ms": 11.0,
                "model2_compute_ms": 13.0,
                "delta_model2_minus_model1_ms": 2.0,
            },
            {
                "image_file": "b.jpg",
                "model1_compute_ms": 20.0,
                "model2_compute_ms": 18.0,
            },
            {
                "image_file": "b.jpg",
                "model1_compute_ms": 22.0,
                "model2_compute_ms": 20.0,
            },
        ]
        first = benchmark.paired_source_group_bootstrap(rows, 200, seed=5)
        repeated = benchmark.paired_source_group_bootstrap(rows, 200, seed=5)

        self.assertEqual(first, repeated)
        self.assertLessEqual(first["mean_delta_ci_lower_ms"], 0)
        self.assertGreaterEqual(first["mean_delta_ci_upper_ms"], 0)

    def test_paired_bootstrap_reports_p95_uncertainty_by_source_family(self):
        rows = [
            {
                "image_file": "capture-a_jpg.rf.one.jpg",
                "source_group": "capture-a",
                "model1_compute_ms": 10.0,
                "model2_compute_ms": 12.0,
                "delta_model2_minus_model1_ms": 2.0,
            },
            {
                "image_file": "capture-a_jpg.rf.two.jpg",
                "source_group": "capture-a",
                "model1_compute_ms": 11.0,
                "model2_compute_ms": 13.0,
                "delta_model2_minus_model1_ms": 2.0,
            },
            {
                "image_file": "capture-b_jpg.rf.one.jpg",
                "source_group": "capture-b",
                "model1_compute_ms": 20.0,
                "model2_compute_ms": 24.0,
                "delta_model2_minus_model1_ms": 4.0,
            },
            {
                "image_file": "capture-b_jpg.rf.two.jpg",
                "source_group": "capture-b",
                "model1_compute_ms": 21.0,
                "model2_compute_ms": 25.0,
                "delta_model2_minus_model1_ms": 4.0,
            },
        ]

        intervals = benchmark.paired_source_group_bootstrap(rows, 200, seed=9)
        comparison = benchmark.build_paired_comparison_rows(rows, 200, seed=9)
        aggregate = comparison[-1]

        self.assertGreater(
            intervals["p95_delta_model2_minus_model1_ci_lower_ms"],
            0,
        )
        self.assertGreater(
            intervals["relative_p95_difference_ci_lower_pct"],
            0,
        )
        self.assertEqual(aggregate["source_groups"], 2)

    def test_paired_mode_rejects_missing_counts_and_all_unreadable(self):
        with self.assertRaisesRegex(ValueError, "missing num_objects"):
            benchmark.benchmark_paired(
                sample_rows=self.sample_rows,
                dataset_size=2,
                asset_root=self.asset_root,
                bootstrap_samples=10,
                **self.dependencies(),
            )

        dependencies = self.dependencies()
        dependencies["image_reader"] = lambda path: None
        with self.assertRaisesRegex(RuntimeError, "unreadable"):
            benchmark.benchmark_paired(
                sample_rows=self.paired_rows(),
                dataset_size=2,
                asset_root=self.asset_root,
                bootstrap_samples=10,
                **dependencies,
            )

    def test_paired_mode_fails_closed_on_one_unreadable_frame(self):
        dependencies = self.dependencies()
        dependencies["image_reader"] = lambda path: (
            None if Path(path).name == "b.jpg" else Path(path).name
        )

        with self.assertRaisesRegex(RuntimeError, "unreadable"):
            benchmark.benchmark_paired(
                sample_rows=self.paired_rows(),
                dataset_size=2,
                asset_root=self.asset_root,
                repeats=3,
                warmup_images=0,
                bootstrap_samples=10,
                **dependencies,
            )


class ArtifactTests(BenchmarkFixture):
    def test_atomic_artifacts_support_long_directories_and_canonical_payloads(self):
        deep = self.root / ("o" * 200)
        csv_path = deep / "long-evidence.csv"
        json_path = deep / "long-manifest.json"
        benchmark._filesystem_path(deep).mkdir(parents=True)
        self.addCleanup(benchmark._filesystem_path(deep).rmdir)
        self.addCleanup(
            benchmark._filesystem_path(csv_path).unlink, missing_ok=True
        )
        self.addCleanup(
            benchmark._filesystem_path(json_path).unlink, missing_ok=True
        )

        benchmark._write_csv_atomic(csv_path, ["value"], [{"value": "ok"}])
        benchmark._write_json_atomic(
            json_path, {"output_directory": str(deep.absolute())}
        )
        fields, rows = read_csv(csv_path)
        manifest = json.loads(
            benchmark._filesystem_path(json_path).read_text(encoding="utf-8")
        )
        self.assertEqual(fields, ["value"])
        self.assertEqual(rows, [{"value": "ok"}])
        self.assertEqual(manifest["output_directory"], str(deep.absolute()))
        self.assertNotIn("\\\\?\\", json.dumps(manifest))
        self.assertEqual(
            benchmark._fingerprinted_file(csv_path)["path"],
            str(csv_path.absolute()),
        )
        self.assertFalse(any(
            entry.name.endswith(".tmp")
            for entry in benchmark._filesystem_path(deep).iterdir()
        ))

    def test_artifacts_preserve_stable_schemas(self):
        summary, observations = benchmark.benchmark_model(
            model_name="model1",
            paths=benchmark.MODELS["model1"],
            sample_rows=self.sample_rows[:1],
            dataset_size=10,
            asset_root=self.asset_root,
            warmup_images=0,
            **self.dependencies(),
        )

        summary_path, observation_path = benchmark.write_benchmark_artifacts(
            self.output_dir,
            [summary],
            observations,
        )

        summary_fields, summary_rows = read_csv(summary_path)
        observation_fields, observation_rows = read_csv(observation_path)
        self.assertEqual(summary_fields, benchmark.SUMMARY_COLUMNS)
        self.assertEqual(observation_fields, benchmark.OBSERVATION_COLUMNS)
        self.assertEqual(len(summary_rows), 1)
        self.assertEqual(len(observation_rows), 1)
        self.assertEqual(list(self.output_dir.glob("*.tmp")), [])

    def test_two_model_runner_writes_combined_evidence(self):
        summaries, observations, summary_path, observation_path = (
            benchmark.run_benchmark(
                sample_rows=self.sample_rows,
                dataset_size=9525,
                asset_root=self.asset_root,
                output_dir=self.output_dir,
                repeats=1,
                warmup_images=1,
                **self.dependencies(),
            )
        )

        self.assertEqual([row["model"] for row in summaries], ["model1", "model2"])
        self.assertEqual(len(observations), 4)
        self.assertTrue(summary_path.is_file())
        self.assertTrue(observation_path.is_file())
        self.assertEqual(len(self.instances["detectors"]), 2)

    def test_paired_runner_writes_atomic_pair_and_aggregate_evidence(self):
        sample_rows = [
            {**self.sample_rows[0], "num_objects": "1"},
            {**self.sample_rows[1], "num_objects": "20"},
        ]
        index_path = self.root / "dataset_index.csv"
        write_csv(
            index_path,
            ["image_file", "image_path", "num_objects"],
            sample_rows,
        )
        result = benchmark.run_paired_benchmark(
            sample_rows=sample_rows,
            dataset_size=2,
            asset_root=self.asset_root,
            dataset_index=index_path,
            output_dir=self.output_dir,
            repeats=1,
            warmup_images=0,
            seed=23,
            bootstrap_samples=20,
            **self.dependencies(),
        )
        (
            summaries,
            observations,
            comparison,
            summary_path,
            observation_path,
            pair_path,
            manifest_path,
        ) = result

        self.assertEqual(len(summaries), 2)
        self.assertEqual(len(observations), 4)
        self.assertEqual(len(comparison), 3)
        self.assertTrue(summary_path.is_file())
        self.assertTrue(observation_path.is_file())
        self.assertTrue(pair_path.is_file())
        self.assertTrue(manifest_path.is_file())
        fields, rows = read_csv(pair_path)
        self.assertEqual(fields, benchmark.PAIRED_COMPARISON_COLUMNS)
        self.assertEqual([row["record_type"] for row in rows], ["pair", "pair", "aggregate"])
        self.assertEqual(list(self.output_dir.glob("*.tmp")), [])

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertNotIn("\\\\?\\", json.dumps(manifest, sort_keys=True))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completeness"]["expected_pairs"], 2)
        self.assertEqual(manifest["completeness"]["unreadable_observations"], 0)
        self.assertEqual(manifest["policy"]["bootstrap_samples"], 20)
        self.assertEqual(
            manifest["models"]["model1"]["weights"]["sha256"],
            benchmark._sha256_file(
                self.asset_root / benchmark.MODELS["model1"]["weights"]
            ),
        )
        self.assertEqual(
            manifest["artifacts"]["paired_comparison"]["sha256"],
            benchmark._sha256_file(pair_path),
        )
        self.assertEqual(len(manifest["run_fingerprint_sha256"]), 64)

    def test_paired_runner_refuses_to_overwrite_existing_evidence(self):
        manifest_path = self.output_dir / benchmark.PAIRED_MANIFEST_NAME
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("old evidence", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "immutable"):
            benchmark.run_paired_benchmark(
                sample_rows=[{**self.sample_rows[0], "num_objects": "1"}],
                dataset_size=1,
                asset_root=self.asset_root,
                dataset_index=self.root / "unused.csv",
                output_dir=self.output_dir,
                repeats=1,
                warmup_images=0,
                bootstrap_samples=10,
                **self.dependencies(),
            )

    def test_failed_paired_package_is_never_promoted_as_a_completed_run(self):
        sample_rows = [{**self.sample_rows[0], "num_objects": "1"}]
        index_path = self.root / "dataset_index.csv"
        write_csv(
            index_path,
            ["image_file", "image_path", "num_objects"],
            sample_rows,
        )

        with patch.object(
            benchmark,
            "build_paired_manifest",
            side_effect=RuntimeError("manifest gate failed"),
        ), self.assertRaisesRegex(RuntimeError, "manifest gate failed"):
            benchmark.run_paired_benchmark(
                sample_rows=sample_rows,
                dataset_size=1,
                asset_root=self.asset_root,
                dataset_index=index_path,
                output_dir=self.output_dir,
                repeats=1,
                warmup_images=0,
                seed=23,
                bootstrap_samples=20,
                **self.dependencies(),
            )

        self.assertFalse(self.output_dir.exists())
        incomplete = self.output_dir.parent / f".{self.output_dir.name}.incomplete"
        self.assertTrue(incomplete.is_dir())
        self.assertEqual(
            {path.name for path in incomplete.iterdir()},
            {
                "inference_benchmark_summary.csv",
                "inference_benchmark_observations.csv",
                "paired_latency_comparison.csv",
            },
        )

    def test_main_resolves_relative_output_and_delegates(self):
        index_path = self.root / "dataset_index.csv"
        write_csv(
            index_path,
            ["image_file", "image_path"],
            [self.sample_rows[0]],
        )
        expected = ([], [], self.root / "summary.csv", self.root / "observations.csv")

        with patch.object(benchmark, "run_benchmark", return_value=expected) as run:
            result = benchmark.main(
                [
                    "--asset-root",
                    str(self.asset_root),
                    "--dataset-index",
                    str(index_path),
                    "--output-root",
                    "scratch/benchmark-test-output",
                    "--run-id",
                    "runtime-test-v1",
                    "--sample-size",
                    "1",
                    "--warmup-images",
                    "0",
                ]
            )

        self.assertIs(result, expected)
        self.assertEqual(run.call_args.kwargs["dataset_size"], 1)
        self.assertEqual(run.call_args.kwargs["sample_rows"], [self.sample_rows[0]])
        self.assertEqual(
            run.call_args.kwargs["output_dir"],
            benchmark.PROJECT_ROOT
            / "scratch"
            / "benchmark-test-output"
            / "runtime-test-v1",
        )

    def test_main_refuses_an_existing_run_directory(self):
        output_root = self.root / "runtime-runs"
        (output_root / "existing-v1").mkdir(parents=True)

        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            benchmark.main(
                [
                    "--asset-root",
                    str(self.asset_root),
                    "--dataset-index",
                    str(self.root / "unused.csv"),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "existing-v1",
                ]
            )

    def test_main_paired_mode_validates_density_and_delegates_seed(self):
        index_path = self.root / "dataset_index.csv"
        indexed_row = {**self.sample_rows[0], "num_objects": "5"}
        write_csv(
            index_path,
            ["image_file", "image_path", "num_objects"],
            [indexed_row],
        )
        expected = ([], [], [], self.root / "a.csv", self.root / "b.csv", self.root / "c.csv")

        with patch.object(
            benchmark,
            "run_paired_benchmark",
            return_value=expected,
        ) as run:
            result = benchmark.main(
                [
                    "--paired",
                    "--run-id",
                    "paired-test-v1",
                    "--asset-root",
                    str(self.asset_root),
                    "--dataset-index",
                    str(index_path),
                    "--sample-size",
                    "1",
                    "--warmup-images",
                    "0",
                    "--seed",
                    "17",
                    "--bootstrap-samples",
                    "50",
                ]
            )

        self.assertIs(result, expected)
        self.assertEqual(run.call_args.kwargs["sample_rows"], [indexed_row])
        self.assertEqual(run.call_args.kwargs["seed"], 17)
        self.assertEqual(run.call_args.kwargs["bootstrap_samples"], 50)
        self.assertEqual(
            run.call_args.kwargs["output_dir"],
            benchmark.DEFAULT_OUTPUT_ROOT / "paired-test-v1",
        )


if __name__ == "__main__":
    unittest.main()
