"""
This test checks if the set_seed function works as expected and if the main training pipeline runs without any issues.
This test might take a long time to run, as it's running the training pipeline. 

"""
import unittest
import os
import torch
import torch.multiprocessing as mp
from main import set_seed, parser
from train import train_main
from model import MyModel
from config import Config

cfg = Config("config.yaml")

class TestMain(unittest.TestCase):
    def test_set_seed(self):
        seed = 42
        set_seed(seed)
        self.assertEqual(torch.initial_seed(), seed)

    def test_main(self):
        args = parser.parse_args(args=['--num-processes', '1', '--seed', '1'])

        # Set up a small training pipeline with a single process
        set_seed(args.seed)
        mp.set_start_method('spawn', force=True)
        model = MyModel(lr=cfg.lr, momentum=cfg.momentum)

        processes = []
        for rank in range(args.num_processes):
            p = mp.Process(target=train_main, args=(rank, cfg.model_save_path, args.num_processes))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Check if the model has been saved
        self.assertTrue(os.path.exists(cfg.model_save_path))

if __name__ == '__main__':
    unittest.main()
