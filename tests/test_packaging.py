import re
import tomllib
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GITATTRIBUTES = PROJECT_ROOT / ".gitattributes"
GITIGNORE = PROJECT_ROOT / ".gitignore"
LICENSE = PROJECT_ROOT / "LICENSE"
README = PROJECT_ROOT / "README.md"
DOCKERFILE = PROJECT_ROOT / "detector_service" / "Dockerfile"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
REQUIREMENTS = PROJECT_ROOT / "detector_service" / "requirements.txt"
ANALYSIS_REQUIREMENTS = PROJECT_ROOT / "requirements-analysis.txt"
DEV_REQUIREMENTS = PROJECT_ROOT / "requirements-dev.txt"
RUFF_CONFIG = PROJECT_ROOT / "ruff.toml"
CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
STORAGE_README = PROJECT_ROOT / "detector_service" / "storage" / "README.md"


def _active_lines(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


class RepositoryLicenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.license_text = LICENSE.read_text(encoding="utf-8")
        cls.readme = README.read_text(encoding="utf-8")

    def test_repository_uses_mit_license(self):
        self.assertTrue(self.license_text.startswith("MIT License\n"))
        self.assertIn("Copyright (c) 2026 Armin Rahbar", self.license_text)
        self.assertIn(
            "Permission is hereby granted, free of charge",
            self.license_text,
        )
        self.assertIn('THE SOFTWARE IS PROVIDED "AS IS"', self.license_text)

    def test_readme_scopes_external_assets_out_of_project_license(self):
        self.assertIn("## License", self.readme)
        self.assertIn("[MIT License](LICENSE)", self.readme)
        for external_asset in (
            "External datasets",
            "model checkpoints",
            "model configuration files",
            "vocabularies",
            "videos",
            "Third-party dependencies",
        ):
            with self.subTest(external_asset=external_asset):
                self.assertIn(external_asset, self.readme)


class ContainerPackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    def test_runtime_dependencies_use_exact_validated_versions(self):
        self.assertEqual(
            _active_lines(REQUIREMENTS),
            [
                "numpy==2.5.2",
                "opencv-python-headless==4.13.0.92",
            ],
        )

    def test_analysis_dependencies_are_minimal_and_exactly_pinned(self):
        self.assertEqual(
            _active_lines(ANALYSIS_REQUIREMENTS),
            [
                "-r detector_service/requirements.txt",
                "pandas==3.0.5",
                "matplotlib==3.11.1",
            ],
        )

    def test_development_dependencies_extend_analysis_with_pinned_ruff(self):
        self.assertEqual(
            _active_lines(DEV_REQUIREMENTS),
            [
                "-r requirements-analysis.txt",
                "ruff==0.16.4",
            ],
        )

    def test_image_uses_python_312_on_debian_bookworm(self):
        self.assertRegex(
            self.dockerfile,
            r"(?m)^FROM python:3\.12-slim-bookworm\s*$",
        )

    def test_runtime_installs_video_libraries_without_recommends(self):
        for required_text in (
            "apt-get install -y --no-install-recommends",
            "ffmpeg",
            "libglib2.0-0",
            "libgl1",
            "rm -rf /var/lib/apt/lists/*",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, self.dockerfile)

    def test_dependencies_are_installed_before_source_is_copied(self):
        requirements_copy = self.dockerfile.index(
            "COPY detector_service/requirements.txt"
        )
        dependency_install = self.dockerfile.index("python -m pip install")
        source_copy = self.dockerfile.index(
            "COPY --chown=app:app detector_service/ /app/detector_service/"
        )

        self.assertLess(requirements_copy, dependency_install)
        self.assertLess(dependency_install, source_copy)

    def test_container_includes_repository_license(self):
        self.assertIn(
            "COPY --chown=app:app LICENSE /app/LICENSE",
            self.dockerfile,
        )

    def test_build_fails_if_opencv_darknet_compatibility_changes(self):
        self.assertIn("import cv2", self.dockerfile)
        self.assertIn("cv2.__version__.startswith('4.13.')", self.dockerfile)

    def test_container_exposes_udp_and_runs_module_entrypoint(self):
        self.assertRegex(self.dockerfile, r"(?m)^EXPOSE 23000/udp\s*$")
        self.assertRegex(
            self.dockerfile,
            re.escape('["python", "-m", "detector_service.app"]'),
        )

    def test_container_runs_as_an_unprivileged_user_with_writable_output(self):
        required_text = (
            "groupadd --system app",
            "useradd --system --gid app --create-home --home-dir /home/app app",
            "COPY --chown=app:app detector_service/ /app/detector_service/",
            "install -d -o app -g app /app/detector_service/storage/detections",
            "USER app",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, self.dockerfile)

        self.assertLess(
            self.dockerfile.index("USER app"),
            self.dockerfile.index("CMD ["),
        )
        self.assertNotRegex(self.dockerfile, r"(?m)^USER\s+(?:0|root)\s*$")

    def test_build_context_excludes_private_and_generated_content(self):
        rules = set(_active_lines(DOCKERIGNORE))
        required_rules = {
            ".git/",
            ".venv/",
            "**/__pycache__/",
            "**/*.pyc",
            "**/storage/",
            "experiments/",
            "tests/",
            "scratch/",
            "**/*.weights",
            "**/*.onnx",
            "**/*.pt",
            "**/*.pth",
            "**/*.engine",
            "**/*.tflite",
            "**/*.mp4",
            "**/*.avi",
            "**/*.mov",
            "**/*.mkv",
            "**/*.webm",
            "**/*.mpeg",
            "**/*.mpg",
            "**/*.m4v",
        }

        self.assertFalse(required_rules - rules)

    def test_dockerfile_copies_only_runtime_package_and_license(self):
        copy_sources = re.findall(
            r"(?m)^COPY(?:\s+--\S+)*\s+(\S+)",
            self.dockerfile,
        )

        self.assertEqual(
            copy_sources,
            [
                "detector_service/requirements.txt",
                "detector_service/",
                "LICENSE",
            ],
        )


class RepositoryPolicyTests(unittest.TestCase):
    def test_git_ignore_excludes_local_secrets_models_and_media(self):
        rules = set(_active_lines(GITIGNORE))
        required_rules = {
            ".env",
            ".env.*",
            "!.env.example",
            "*.weights",
            "*.onnx",
            "*.pt",
            "*.pth",
            "*.engine",
            "*.tflite",
            "*.mp4",
            "*.avi",
            "*.mov",
            "*.mkv",
            "*.webm",
            "*.mpeg",
            "*.mpg",
            "*.m4v",
        }

        self.assertFalse(required_rules - rules)

    def test_storage_contract_is_tracked_while_assets_remain_ignored(self):
        rules = set(_active_lines(GITIGNORE))
        self.assertTrue(STORAGE_README.is_file())
        self.assertTrue(
            {
                "!detector_service/storage/",
                "detector_service/storage/*",
                "!detector_service/storage/README.md",
            }.issubset(rules)
        )

        contract = STORAGE_README.read_text(encoding="utf-8")
        for required_entry in (
            "logistics/",
            "_darknet.labels",
            "yolo_model_1/",
            "yolo_model_2/",
            "test_videos/test_videos/",
            "9,525 paired JPEG and",
            "36,721 labeled objects",
            "does not version the external assets",
            "does not grant rights",
        ):
            with self.subTest(required_entry=required_entry):
                self.assertIn(required_entry, contract)

    def test_git_attributes_enforce_lf_text_and_binary_images(self):
        self.assertEqual(
            _active_lines(GITATTRIBUTES),
            [
                "* text=auto eol=lf",
                "*.svg text eol=lf",
                "*.png binary",
                "*.jpg binary",
                "*.jpeg binary",
                "*.gif binary",
                "*.weights binary",
            ],
        )

    def test_ruff_policy_is_deliberate_and_narrow(self):
        policy = tomllib.loads(RUFF_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(policy["target-version"], "py312")
        self.assertEqual(policy["lint"]["select"], ["E4", "E7", "E9", "F"])

        expected_e402_exceptions = {
            "experiments/scripts/01_model_selection/01_compare_model_quality.py",
            "experiments/scripts/02_dataset_analysis/04_analyze_overlap.py",
            "experiments/scripts/03_nms_thresholding/01_sweep_nms_thresholds.py",
            "experiments/scripts/04_augmentation_robustness/01_preview_augmentation_conditions.py",
            "experiments/scripts/04_augmentation_robustness/02_measure_augmentation_robustness.py",
            "experiments/scripts/05_hard_negative_mining/02_build_error_review_queues.py",
            "experiments/scripts/05_hard_negative_mining/01_build_error_components.py",
            "tests/test_hard_negative_mining.py",
        }
        per_file_ignores = policy["lint"]["per-file-ignores"]
        self.assertEqual(set(per_file_ignores), expected_e402_exceptions)
        self.assertTrue(
            all(codes == ["E402"] for codes in per_file_ignores.values())
        )


class ContinuousIntegrationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    def test_workflow_targets_main_pushes_and_pull_requests(self):
        self.assertIn("on:\n  push:\n", self.workflow)
        self.assertIn("  pull_request:\n", self.workflow)
        self.assertGreaterEqual(self.workflow.count("      - main\n"), 2)
        self.assertNotIn("pull_request_target", self.workflow)

    def test_workflow_uses_read_only_repository_permissions(self):
        self.assertRegex(
            self.workflow,
            r"(?m)^permissions:\n  contents: read\s*$",
        )
        self.assertNotRegex(self.workflow, r"(?m)^\s*\w[\w-]*: write\s*$")

    def test_checkout_credentials_are_not_persisted(self):
        self.assertEqual(self.workflow.count("uses: actions/checkout@v7"), 2)
        self.assertEqual(
            self.workflow.count("persist-credentials: false"),
            2,
        )

    def test_python_job_uses_validated_version_and_dependency_manifest(self):
        self.assertTrue(ANALYSIS_REQUIREMENTS.is_file())
        self.assertTrue(DEV_REQUIREMENTS.is_file())
        self.assertIn("uses: actions/setup-python@v7", self.workflow)
        self.assertIn('python-version: "3.12"', self.workflow)
        self.assertIn("cache: pip", self.workflow)
        self.assertIn("requirements-analysis.txt", self.workflow)
        self.assertIn("requirements-dev.txt", self.workflow)
        self.assertIn(
            "python -m pip install --requirement requirements-dev.txt",
            self.workflow,
        )

    def test_python_job_runs_the_configured_linter(self):
        self.assertIn("python -m ruff check .", self.workflow)

    def test_python_job_checks_dependency_consistency(self):
        self.assertIn("python -m pip check", self.workflow)

    def test_python_job_runs_the_public_test_suite(self):
        self.assertIn(
            "python -m unittest discover -s tests -v",
            self.workflow,
        )

    def test_container_job_builds_without_publishing(self):
        self.assertIn("--file detector_service/Dockerfile", self.workflow)
        self.assertIn("--tag warehouse-object-detection:ci", self.workflow)
        self.assertNotIn("docker push", self.workflow)
        self.assertNotIn("packages: write", self.workflow)
        self.assertNotIn("secrets.", self.workflow)

    def test_container_job_runs_cli_and_dependency_smoke_tests(self):
        image = "warehouse-object-detection:ci"
        self.assertIn(
            f"docker run --rm {image} \\\n"
            "            python -m detector_service.app --help",
            self.workflow,
        )
        self.assertIn(
            f"docker run --rm {image} \\\n"
            '            python -c "import os;',
            self.workflow,
        )
        self.assertIn("cv2.__version__ == '4.13.0'", self.workflow)
        self.assertIn("numpy.__version__ == '2.5.2'", self.workflow)
        self.assertIn("assert os.getuid() != 0", self.workflow)
        self.assertIn("/app/detector_service/storage/detections/.write-check", self.workflow)
        self.assertIn("probe.write_text('ok'", self.workflow)
        self.assertIn("probe.unlink()", self.workflow)


if __name__ == "__main__":
    unittest.main()
