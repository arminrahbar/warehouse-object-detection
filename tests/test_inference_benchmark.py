import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "scripts" / "01_benchmark_inference.py"

spec = importlib.util.spec_from_file_location("inference_benchmark_under_test", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


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
    def test_defaults_and_model_order_are_explicit(self):
        self.assertEqual(benchmark.DEFAULT_SAMPLE_SIZE, 100)
        self.assertEqual(benchmark.DEFAULT_REPEATS, 1)
        self.assertEqual(benchmark.DEFAULT_WARMUP_IMAGES, 1)
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
        ):
            self.assertIn(column, benchmark.SUMMARY_COLUMNS)

    def test_cli_integer_validators_reject_invalid_values(self):
        self.assertEqual(benchmark.positive_int("3"), 3)
        self.assertEqual(benchmark.nonnegative_int("0"), 0)
        with self.assertRaisesRegex(Exception, "positive integer"):
            benchmark.positive_int("0")
        with self.assertRaisesRegex(Exception, "non-negative integer"):
            benchmark.nonnegative_int("-1")


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

    def test_missing_index_has_actionable_message(self):
        with self.assertRaisesRegex(FileNotFoundError, "02_build_dataset_index.py"):
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


class ArtifactTests(BenchmarkFixture):
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
                    "--output-dir",
                    "scratch/benchmark-test-output",
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
            benchmark.PROJECT_ROOT / "scratch/benchmark-test-output",
        )


if __name__ == "__main__":
    unittest.main()
