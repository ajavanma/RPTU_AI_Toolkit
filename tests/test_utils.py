import unittest
import numpy as np
import open3d as o3d
from utils import load_label_color_map, mp, out_pcd, visualize_point_cloud_with_labels
from config import Config

cfg = Config("config.yaml")

class TestUtils(unittest.TestCase):
    def test_load_label_color_map(self):
        label_color_map = load_label_color_map(cfg.yaml_file_path)
        self.assertIsInstance(label_color_map, dict)
        self.assertEqual(label_color_map[1], [170, 0, 0])
        self.assertEqual(label_color_map[14], [0, 50, 0])
        self.assertEqual(label_color_map[21], [50, 200, 250])
        self.assertEqual(label_color_map[101], [250, 140, 0])
        
        with self.assertRaises(ValueError):
            load_label_color_map("invalid_file_path")

    def test_mp(self):
        mapper_dict = {
            0: [0, 0, 0],
            1: [170, 0, 0],
            2: [70, 0, 20],
            3: [0, 0, 0],
            4: [0, 0, 170],
            # ...
            100: [250, 200, 0],
            101: [250, 140, 0]
        }
        
        self.assertEqual(mp(mapper_dict, 2), (70, 0, 20))
        self.assertEqual(mp(mapper_dict, 100), (250, 200, 0))
        self.assertEqual(mp(mapper_dict, 101), (250, 140, 0))
        
        # Test with a non-existent key
        self.assertEqual(mp(mapper_dict, 99), (0, 0, 0))

    def test_out_pcd(self):
        coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        colors = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]])
        result = out_pcd(coords, colors)
        self.assertIsInstance(result, o3d.geometry.PointCloud)

    def test_visualize_point_cloud_with_labels(self):
        pass

if __name__ == '__main__':
    unittest.main()
