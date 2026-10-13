import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from rptu_ai_toolkit.config import ConfigError, load_config


class TestPackageConfig(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.source = self.root / "config.yaml"
        self.document = {
            "schema_version": 1,
            "paths": {"input_dir": "data", "output_dir": "runs"},
        }

    def write_config(self, document=None):
        self.source.write_text(
            yaml.safe_dump(self.document if document is None else document),
            encoding="utf-8",
        )

    def assert_invalid(self, field):
        with self.assertRaises(ConfigError) as caught:
            load_config(self.source)
        message = str(caught.exception)
        self.assertIn(str(self.source), message)
        self.assertIn(field, message)
        return message

    def test_resolves_paths_against_config_file_and_adds_defaults(self):
        self.write_config()
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            try:
                os.chdir(elsewhere)
                config = load_config(str(self.source))
            finally:
                os.chdir(original_cwd)

        expected = {
            "schema_version": 1,
            "paths": {
                "input_dir": str(self.root / "data"),
                "output_dir": str(self.root / "runs"),
            },
            "runtime": {"seed": 0, "num_workers": 0},
        }
        self.assertEqual(config.to_dict(), expected)
        self.assertEqual(json.loads(json.dumps(config.to_dict())), expected)
        self.assertEqual(config.paths.input_dir, self.root / "data")

    def test_accepts_absolute_paths_and_runtime_boundaries(self):
        self.document["paths"] = {
            "input_dir": str(self.root / "absolute-input"),
            "output_dir": str(self.root / "absolute-output"),
        }
        self.document["runtime"] = {"seed": 2**32 - 1, "num_workers": 4}
        self.write_config()
        config = load_config(self.source)
        self.assertEqual(config.paths.input_dir, self.root / "absolute-input")
        self.assertEqual(config.paths.output_dir, self.root / "absolute-output")
        self.assertEqual(config.runtime.seed, 2**32 - 1)
        self.assertEqual(config.runtime.num_workers, 4)

    def test_paths_expand_home_but_not_environment_variables(self):
        self.document["paths"] = {
            "input_dir": "~/rptu-example-input",
            "output_dir": "$RPTU_TEST_DATA/runs",
        }
        self.write_config()
        with patch.dict(os.environ, {"RPTU_TEST_DATA": "/unexpected"}):
            config = load_config(self.source)
        self.assertEqual(config.paths.input_dir, (Path.home() / "rptu-example-input").resolve())
        self.assertEqual(config.paths.output_dir, self.root / "$RPTU_TEST_DATA/runs")

    def test_validation_does_not_require_or_create_directories(self):
        self.write_config()
        original = self.source.read_bytes()
        before = set(self.root.iterdir())
        load_config(self.source)
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertFalse((self.root / "data").exists())
        self.assertFalse((self.root / "runs").exists())

    def test_same_resolved_input_and_output_are_rejected(self):
        self.document["paths"]["output_dir"] = "data/../data"
        self.write_config()
        self.assertIn("must differ", self.assert_invalid("paths.output_dir"))

    def test_missing_required_fields(self):
        for missing in ("schema_version", "paths", "paths.input_dir", "paths.output_dir"):
            with self.subTest(field=missing):
                document = {
                    "schema_version": 1,
                    "paths": {"input_dir": "data", "output_dir": "runs"},
                }
                if missing.startswith("paths."):
                    del document["paths"][missing.split(".")[1]]
                else:
                    del document[missing]
                self.write_config(document)
                self.assertIn("required", self.assert_invalid(missing))

    def test_unknown_fields_at_every_level(self):
        for location in ("config", "paths", "runtime"):
            with self.subTest(location=location):
                document = {
                    "schema_version": 1,
                    "paths": {"input_dir": "data", "output_dir": "runs"},
                    "runtime": {},
                }
                target = document if location == "config" else document[location]
                target["typo"] = 1
                self.write_config(document)
                field = "typo" if location == "config" else location + ".typo"
                self.assertIn("unknown field", self.assert_invalid(field))

    def test_invalid_schema_versions(self):
        for value in (True, False, 0, 2, "1", 1.0, None, []):
            with self.subTest(value=value):
                self.document["schema_version"] = value
                self.write_config()
                self.assert_invalid("schema_version")

    def test_invalid_runtime_values(self):
        for field in ("seed", "num_workers"):
            for value in (True, False, -1, "0", 0.0, None, [], {}):
                with self.subTest(field=field, value=value):
                    self.document["runtime"] = {field: value}
                    self.write_config()
                    self.assert_invalid("runtime." + field)
        self.document["runtime"] = {"seed": 2**32}
        self.write_config()
        self.assert_invalid("runtime.seed")

    def test_runtime_fields_may_be_omitted_individually(self):
        for values in ({}, {"seed": 9}, {"num_workers": 2}):
            with self.subTest(values=values):
                self.document["runtime"] = values
                self.write_config()
                config = load_config(self.source)
                self.assertEqual(config.runtime.seed, values.get("seed", 0))
                self.assertEqual(config.runtime.num_workers, values.get("num_workers", 0))

    def test_invalid_paths(self):
        for field in ("input_dir", "output_dir"):
            for value in ("", "  ", True, 1, None, [], {}, "bad\0path"):
                with self.subTest(field=field, value=value):
                    self.document["paths"] = {"input_dir": "data", "output_dir": "runs"}
                    self.document["paths"][field] = value
                    self.write_config()
                    self.assert_invalid("paths." + field)

    def test_nonmapping_sections(self):
        for field in ("paths", "runtime"):
            for value in (None, [], "text", 1, True):
                with self.subTest(field=field, value=value):
                    document = {
                        "schema_version": 1,
                        "paths": {"input_dir": "data", "output_dir": "runs"},
                    }
                    document[field] = value
                    self.write_config(document)
                    self.assertIn("expected a mapping", self.assert_invalid(field))

    def test_nonstring_keys_are_rejected(self):
        self.document["runtime"] = {1: 2}
        self.write_config()
        self.assertIn("field names must be strings", self.assert_invalid("runtime"))

    def test_empty_nonmapping_and_malformed_yaml(self):
        for content in ("", "# empty\n", "null\n", "[]\n", "text\n", "1\n"):
            with self.subTest(content=content):
                self.source.write_text(content, encoding="utf-8")
                self.assertIn("expected a mapping", self.assert_invalid("config"))
        self.source.write_text("paths: [\n", encoding="utf-8")
        self.assertIn("column", self.assert_invalid("YAML line"))

    def test_duplicate_yaml_keys_at_every_level_report_location(self):
        for content in (
            "schema_version: 1\nschema_version: 1\n",
            "paths:\n  input_dir: data\n  input_dir: other\n",
            "runtime:\n  seed: 0\n  seed: 1\n",
        ):
            with self.subTest(content=content):
                self.source.write_text(content, encoding="utf-8")
                self.assertIn("duplicate key", self.assert_invalid("YAML line"))

    def test_unsafe_yaml_tags_and_complex_keys_are_rejected(self):
        for content in ("!!python/object:builtins.object {}", "? [a, b]\n: value\n"):
            with self.subTest(content=content):
                self.source.write_text(content, encoding="utf-8")
                self.assert_invalid("YAML line")

    def test_unreadable_and_non_utf8_files_report_the_path(self):
        self.assertIn("cannot read file", self.assert_invalid("config"))
        self.source.write_bytes(b"\xff")
        self.assertIn("cannot read file", self.assert_invalid("config"))


if __name__ == "__main__":
    unittest.main()
