import unittest
import os
import numpy as np
import pickle
from preprocess_minkowski import Preprocessor

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.voxel_size = 0.1
        self.preprocessor = Preprocessor(self.voxel_size)
        self.base_file_name = None

        self.test_pcd_file = 'data/raw/train/pcd/14NH_A_1st_floor_offices-color_SR_AN_20230309_REV_20230411.pcd'
        self.test_asc_file = 'data/raw/train/asc/14NH_A_1st_floor_offices-color_SR_AN_20230309_REV_20230411.asc'

        self.output_file = os.path.join('data/preprocessed', "14NH_A_1st_floor_offices-color_SR_AN_20230309_REV_20230411_preprocessed.pkl")

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
    # tests everything
    def test_preprocessing(self):
        self.preprocessor.process_files(self.test_pcd_file, self.test_asc_file)

        # Check if output file is created
        self.assertTrue(os.path.exists(self.output_file))

        # Load the preprocessed data
        with open(self.output_file, 'rb') as f:
            data = pickle.load(f)

        # Check if the necessary keys are in the preprocessed data
        self.assertIn('coords', data)
        self.assertIn('features', data)
        self.assertIn('labels', data)

        # Test individual Preprocessor methods
        self.preprocessor.load_pcd(self.test_pcd_file)
        self.assertIsNotNone(self.preprocessor.pcd)

        self.preprocessor.load_asc(self.test_asc_file)
        self.assertIsNotNone(self.preprocessor.labels)

        self.preprocessor.normalize_pcd()
        self.assertTrue(np.all(np.abs(np.asarray(self.preprocessor.pcd.points)) <= 1))

        self.preprocessor.downsample_pcd()
        self.assertIsNotNone(self.preprocessor.down_pcd)

        self.preprocessor.estimate_normals()
        self.assertIsNotNone(self.preprocessor.down_pcd.normals)

        self.preprocessor.assign_label()
        self.assertIsNotNone(self.preprocessor.label_list)

        self.preprocessor.generate_feature_arr()
        self.assertIsNotNone(self.preprocessor.feature_arr)
        self.assertIsNotNone(self.preprocessor.label_arr)

        self.preprocessor.check_and_remove_incomplete_rows()
        self.assertTrue(np.all(np.isfinite(self.preprocessor.feature_arr)))

if __name__ == '__main__':
    unittest.main()
