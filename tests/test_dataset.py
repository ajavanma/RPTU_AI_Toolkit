import unittest
import torch
from pathlib import Path
from dataset import PointCloudDataset
from config import Config

cfg = Config("config.yaml")

class TestPointCloudDataset(unittest.TestCase):

    def test_PointCloudDataset_length(self):
        train_dataset = PointCloudDataset(mode="train")
        val_dataset = PointCloudDataset(mode="val")
        test_dataset = PointCloudDataset(mode="test")
        self.assertEqual(len(train_dataset), len(list(Path(cfg.preprocessed_train_dir).glob("*.ply"))))
        self.assertEqual(len(val_dataset), len(list(Path(cfg.preprocessed_val_dir).glob("*.ply"))))
        self.assertEqual(len(test_dataset), len(list(Path(cfg.preprocessed_test_dir).glob("*.ply"))))

    def test_PointCloudDataset_invalid_mode(self):
        with self.assertRaises(ValueError):
            PointCloudDataset(mode="invalid_mode")
    
    def test_PointCloudDataset_output_types_and_shapes(self):
        train_dataset = PointCloudDataset(mode="train")
        features, labels = train_dataset[0]
        self.assertIsInstance(features, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(features.shape[1], 9)  # Assuming 9 features
        self.assertEqual(labels.shape[1], 1)    # Assuming 1 label
    
    def test_PointCloudDataset_custom_directory(self):
        train_dataset = PointCloudDataset(cfg.preprocessed_train_dir, mode="train")
        self.assertEqual(len(train_dataset), len(list(Path(cfg.preprocessed_train_dir).glob("*.ply"))))
        
    def test_PointCloudDataset_invalid_directory(self):
        with self.assertRaises(FileNotFoundError):
            PointCloudDataset("invalid_directory", mode="train")


if __name__ == "__main__":
    unittest.main()
