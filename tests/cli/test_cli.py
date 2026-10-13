"""Exercise the installed CPU CLI from outside the source checkout."""

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


FUTURE_COMMANDS = (
    "train",
    "evaluate",
    "predict",
    "benchmark",
)


class InstalledCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="rptu-cli-")
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name).resolve()
        self.environment = os.environ.copy()
        # An installed entry point must not depend on the test runner's path.
        self.environment.pop("PYTHONPATH", None)
        self.environment["PYTHONNOUSERSITE"] = "1"

    def run_process(self, arguments):
        return subprocess.run(
            arguments,
            cwd=self.directory,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def run_cli(self, *arguments):
        return self.run_process(
            [sys.executable, "-m", "rptu_ai_toolkit", *arguments]
        )

    def write_config(self, content=None):
        config_directory = self.directory / "configuration"
        config_directory.mkdir(exist_ok=True)
        (config_directory / "data" / "input").mkdir(parents=True, exist_ok=True)
        config_file = config_directory / "run.yaml"
        if content is None:
            content = """\
                schema_version: 1
                paths:
                  input_dir: data/input
                  output_dir: runs/output
                runtime:
                  seed: 42
                  num_workers: 0
                """
        config_file.write_text(textwrap.dedent(content), encoding="utf-8")
        return config_file

    def assert_usage_error(self, result):
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("error", result.stderr.lower())
        self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_no_arguments_print_help_without_prompting(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout.lower())
        self.assertIn("doctor", result.stdout)
        self.assertIn("config", result.stdout)
        for command in FUTURE_COMMANDS:
            self.assertIn(command, result.stdout)
        self.assertIn("not implemented", result.stdout.lower())

    def test_module_and_console_entrypoints_report_installed_version(self):
        expected_version = importlib.metadata.version("rptu-ai-toolkit")
        console_script = Path(sys.executable).with_name("rptu")
        self.assertTrue(console_script.is_file(), str(console_script))
        for invocation in (
            [sys.executable, "-m", "rptu_ai_toolkit"],
            [str(console_script)],
        ):
            with self.subTest(invocation=invocation):
                result = self.run_process([*invocation, "--version"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected_version, result.stdout)

    def test_doctor_reports_cpu_foundation_without_claiming_gpu_validation(self):
        result = self.run_cli("doctor", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(
            report["version"], importlib.metadata.version("rptu-ai-toolkit")
        )
        self.assertTrue(report["python"])
        self.assertTrue(report["platform"])
        for package in ("PyYAML", "torch", "warpconvnet"):
            self.assertIn(package, report["packages"])
            self.assertTrue(
                report["packages"][package] is None
                or isinstance(report["packages"][package], str)
            )
        self.assertEqual(
            report["capabilities"],
            {
                "config_validation": True,
                "dataset_preparation": True,
                "ml_pipeline": False,
                "gpu_runtime": "not_checked",
            },
        )

    def test_human_readable_doctor_succeeds(self):
        result = self.run_cli("doctor")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip())

    def test_help_and_doctor_never_import_ml_dependencies(self):
        guard = textwrap.dedent(
            """\
            import importlib.abc
            import runpy
            import sys

            class NoHeavyImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {
                        'torch', 'warpconvnet', 'MinkowskiEngine', 'open3d'
                    }:
                        raise AssertionError('Unexpected ML import: ' + fullname)
                    return None

            sys.meta_path.insert(0, NoHeavyImports())
            sys.argv = ['rptu', *sys.argv[1:]]
            runpy.run_module('rptu_ai_toolkit', run_name='__main__')
            """
        )
        for arguments in (("--help",), ("doctor",), ("doctor", "--json")):
            with self.subTest(arguments=arguments):
                result = self.run_process([sys.executable, "-c", guard, *arguments])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_legacy_source_entrypoint_delegates_without_importing_torch(self):
        source_entrypoint = Path(__file__).resolve().parents[2] / "src" / "main.py"
        guard = textwrap.dedent(
            """\
            import importlib.abc
            import runpy
            import sys

            class NoHeavyImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {
                        'torch', 'warpconvnet', 'MinkowskiEngine', 'open3d'
                    }:
                        raise AssertionError('Unexpected ML import: ' + fullname)
                    return None

            sys.meta_path.insert(0, NoHeavyImports())
            script = sys.argv[1]
            sys.argv = [script, '--help']
            runpy.run_path(script, run_name='__main__')
            """
        )
        result = self.run_process(
            [sys.executable, "-c", guard, str(source_entrypoint)]
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("doctor", result.stdout)

    def test_config_validation_resolves_paths_relative_to_config_file(self):
        config_file = self.write_config()
        result = self.run_cli("config", "validate", "--config", str(config_file), "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        resolved = json.loads(result.stdout)
        self.assertEqual(resolved["schema_version"], 1)
        self.assertEqual(
            resolved["paths"],
            {
                "input_dir": str(config_file.parent / "data" / "input"),
                "output_dir": str(config_file.parent / "runs" / "output"),
            },
        )
        self.assertEqual(resolved["runtime"], {"seed": 42, "num_workers": 0})
        self.assertFalse((config_file.parent / "runs").exists())

    def test_config_validation_populates_runtime_defaults(self):
        config_file = self.write_config(
            """\
            schema_version: 1
            paths:
              input_dir: data/input
              output_dir: runs/output
            """
        )
        result = self.run_cli("config", "validate", "--config", str(config_file), "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["runtime"], {"seed": 0, "num_workers": 0})

    def test_config_validation_has_human_readable_success(self):
        result = self.run_cli("config", "validate", "--config", str(self.write_config()))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Configuration is valid", result.stdout)

    def test_invalid_configuration_is_an_actionable_usage_error(self):
        invalid_documents = {
            "malformed YAML": "schema_version: [\n",
            "unsupported schema": "schema_version: 99\npaths: {input_dir: input, output_dir: output}\n",
            "missing input": "schema_version: 1\npaths: {output_dir: output}\n",
            "invalid seed": (
                "schema_version: 1\npaths: {input_dir: input, output_dir: output}\n"
                "runtime: {seed: wrong, num_workers: 0}\n"
            ),
        }
        for reason, document in invalid_documents.items():
            with self.subTest(reason=reason):
                result = self.run_cli(
                    "config", "validate", "--config", str(self.write_config(document)), "--json"
                )
                self.assert_usage_error(result)

    def test_missing_config_file_and_argument_are_usage_errors(self):
        for arguments in (
            ("config", "validate"),
            ("config", "validate", "--config", str(self.directory / "missing.yaml")),
        ):
            with self.subTest(arguments=arguments):
                self.assert_usage_error(self.run_cli(*arguments))

    def test_future_commands_are_explicitly_unavailable_but_offer_help(self):
        for command in FUTURE_COMMANDS:
            with self.subTest(command=command):
                unavailable = self.run_cli(command)
                self.assertEqual(unavailable.returncode, 2)
                self.assertIn("not implemented", unavailable.stderr.lower())
                self.assertNotIn("Traceback", unavailable.stdout + unavailable.stderr)
                help_result = self.run_cli(command, "--help")
                self.assertEqual(help_result.returncode, 0, help_result.stderr)
                self.assertIn("not implemented", help_result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
