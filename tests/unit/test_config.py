import tempfile
import unittest
from pathlib import Path

import yaml

from config.config import Config


class TestConfig(unittest.TestCase):
    def test_loads_yaml_from_a_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('voxel_size: 0.01\nshuffle: false\nrandom_seed: null\n')

            config = Config(path)

        self.assertEqual(config.voxel_size, 0.01)
        self.assertIs(config.shuffle, False)
        self.assertIsNone(config.random_seed)

    def test_missing_file_raises_with_its_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.yaml'
            with self.assertRaises(FileNotFoundError) as error:
                Config(path)

        self.assertEqual(error.exception.filename, str(path))

    def test_invalid_yaml_raises_a_parser_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.yaml'
            path.write_text('voxel_size: [\n')
            with self.assertRaises(yaml.YAMLError):
                Config(path)

    def test_loads_dictionary_and_reports_missing_setting(self):
        config = Config({'num_workers': 2})

        self.assertEqual(config.num_workers, 2)
        with self.assertRaisesRegex(AttributeError, "Config attribute 'voxel_size' not found"):
            _ = config.voxel_size
