import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = PROJECT_ROOT / "detector_service" / "Dockerfile"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
REQUIREMENTS = PROJECT_ROOT / "detector_service" / "requirements.txt"
ANALYSIS_REQUIREMENTS = PROJECT_ROOT / "requirements-analysis.txt"
CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


def _active_lines(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


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
            "COPY detector_service/ /app/detector_service/"
        )

        self.assertLess(requirements_copy, dependency_install)
        self.assertLess(dependency_install, source_copy)

    def test_build_fails_if_opencv_darknet_compatibility_changes(self):
        self.assertIn("import cv2", self.dockerfile)
        self.assertIn("cv2.__version__.startswith('4.13.')", self.dockerfile)

    def test_container_exposes_udp_and_runs_module_entrypoint(self):
        self.assertRegex(self.dockerfile, r"(?m)^EXPOSE 23000/udp\s*$")
        self.assertRegex(
            self.dockerfile,
            re.escape('["python", "-m", "detector_service.app"]'),
        )

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
        }

        self.assertFalse(required_rules - rules)

    def test_dockerfile_copies_only_the_runtime_package(self):
        copy_sources = re.findall(
            r"(?m)^COPY\s+(\S+)",
            self.dockerfile,
        )

        self.assertEqual(
            copy_sources,
            [
                "detector_service/requirements.txt",
                "detector_service/",
            ],
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
        self.assertEqual(self.workflow.count("uses: actions/checkout@v6"), 2)
        self.assertEqual(
            self.workflow.count("persist-credentials: false"),
            2,
        )

    def test_python_job_uses_validated_version_and_dependency_manifest(self):
        self.assertTrue(ANALYSIS_REQUIREMENTS.is_file())
        self.assertIn("uses: actions/setup-python@v6", self.workflow)
        self.assertIn('python-version: "3.12"', self.workflow)
        self.assertIn("cache: pip", self.workflow)
        self.assertIn("requirements-analysis.txt", self.workflow)
        self.assertIn(
            "python -m pip install --requirement requirements-analysis.txt",
            self.workflow,
        )

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


if __name__ == "__main__":
    unittest.main()
