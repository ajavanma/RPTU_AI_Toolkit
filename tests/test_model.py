# To ensure the output of the model has the correct number of classes.

import unittest
import torch
from model import MyModel
from torch.utils.data import DataLoader
from dataset import PointCloudDataset
import MinkowskiEngine as ME
from config import Config

cfg = Config("config.yaml")

class TestMyModel(unittest.TestCase):
    def setUp(self):
        self.model = MyModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_forward(self):
        sample_dataset = PointCloudDataset(mode="train")
        sample_dataloader = DataLoader(sample_dataset, batch_size=1, shuffle=False)
        coords, feats, labels = next(iter(sample_dataloader))
        inputs = ME.SparseTensor(
            coordinates=coords,
            features=feats,
            tensor_stride=1,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device=self.device
        )
        outputs = self.model(inputs)
        self.assertEqual(outputs.size(1), cfg.num_classes)

    def test_model_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MyModel().to(device)
        self.assertEqual(next(model.parameters()).device, device)

    def test_model_training_mode(self):
        self.model.train()
        self.assertTrue(self.model.training)
        self.model.eval()
        self.assertFalse(self.model.training)

if __name__ == "__main__":
    unittest.main()
