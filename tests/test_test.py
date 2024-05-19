import unittest
from test import get_test_dataloader, evaluate, calculate_metrics
from model import MyModel
import torch


class TestTest(unittest.TestCase):
    def test_testing_pipeline(self):
        # Set up a small testing pipeline and run it for a single batch
        test_dataloader = get_test_dataloader(num_threads=0)
        
        # Set up the model
        model = MyModel()
        
        # Evaluate the model on a single batch
        single_batch_dataloader = [next(iter(test_dataloader))]
        all_preds, all_labels = evaluate(model, single_batch_dataloader, "cpu")
        
        # Calculate metrics
        metrics = calculate_metrics(all_preds, all_labels)
        
        # Check if metrics are correctly calculated
        for metric_name, metric_value in metrics.items():
            self.assertIsInstance(metric_value, float)

if __name__ == '__main__':
    unittest.main()
