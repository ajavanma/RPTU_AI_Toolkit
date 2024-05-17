import os
import pickle
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import torch
import numpy as np
from config import Config
from utils import load_config_file, get_logger


logger = get_logger("logs/log-dataset.txt", __name__)

config_data = load_config_file("config.yaml")
cfg = Config(config_data)


class PointCloudDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing point cloud data in a Minkowskiengine compatible format.

    Attributes:
        data_list (list): A list of dictionaries containing data for each point cloud.
        data_list_files (list): A list of file paths for point cloud data files.
        num_classes (int): The number of unique classes in the dataset.
        Config is instance based.
    """
    def __init__(self, mode, config_file="config.yaml"):
        cfg = Config(config_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        DATA_DIRS = {
            "train": cfg.preprocessed_train_dir,
            "val": cfg.preprocessed_val_dir,
            "test": cfg.preprocessed_test_dir,
        }

        assert mode in DATA_DIRS.keys(), f"Invalid mode. Must be one of {list(DATA_DIRS.keys())}"
        data_list_dir = DATA_DIRS[mode]

        self.data_list = []
        self.data_list_files = [os.path.join(data_list_dir, f) for f in os.listdir(data_list_dir) if f.endswith('.pkl')]

        for data_list_file in self.data_list_files:
            with open(data_list_file, 'rb') as f:
                data_list = pickle.load(f)
                #print(f"Loaded data_list from {data_list_file}: {data_list}")  # for debugging
                self.data_list.append(data_list)

        self.num_classes = cfg.num_classes

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        
        data = self.data_list[i]
        print(f"Data at index {i}: {data}")  # for debugging
        
        '''
        Data at index 1: features
        Data at index 5: labels
        Data at index 4: features
        Data at index 2: labels
        Data at index 3: coords
        Data at index 6: coords
        Data at index 0: coords
        '''
        coords = data['coords']
        
        #stride check
        coords_np = np.array(coords)
        is_aligned = np.all(coords_np % cfg.stride == 0)
        assert is_aligned, "Coordinates are not aligned with the tensor stride: Dataset.py"
        
        feats = data['features']
        labels = data['labels']

        # One-hot encode labels using numpy array
        labels_one_hot = np.eye(self.num_classes)[labels]
        # import torch.nn.functional as F
        # labels = F.one_hot(labels, num_classes=num_classes).float()

        print("Coords shapes:", [c.shape for c in [coords]])
        print("Features shape:", feats.shape)  # Add this line
        print("Labels shape:", labels_one_hot.shape)  # Add this line

        return coords, feats, labels_one_hot
