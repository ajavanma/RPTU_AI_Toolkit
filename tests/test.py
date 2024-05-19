import torch
from torch.utils.data import DataLoader
from dataset import PointCloudDataset
from model import MyModel
import yaml
import numpy as np
from utils import visualize_point_cloud_with_labels, calculate_metrics
import MinkowskiEngine as ME
from config import Config
from colorama import Fore, Style
from utils import load_config_file, get_logger

logger = get_logger('logs/log-test.txt', __name__)
    
config_data = load_config_file("config.yaml")
cfg = Config(config_data)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for coords, feats, labels in dataloader:
            coords, feats, labels = coords.to(device), feats.to(device), labels.to(device)
            stensor = ME.SparseTensor(feats, coords=coords)
            outputs = model(stensor)
            preds = torch.argmax(outputs.F, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return all_preds, all_labels

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = PointCloudDataset("test")
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=ME.utils.batch_sparse_collate,
        shuffle=False,
        num_workers=cfg.num_workers
    )
    
    model = MyModel().to(device) 
    model.load_state_dict(torch.load(cfg.model_save_path))
    logger.info(f"{Fore.BLUE}evaluating... {Style.RESET_ALL}")

    all_preds, all_labels = evaluate(model, test_dataloader, device)

    # Calculate evaluation metrics (e.g., accuracy, precision, recall, F1-score) using all_preds and all_labels
    metrics = calculate_metrics(all_preds, all_labels)
    logger.info("Evaluation Metrics:")
    
    for metric_name, metric_value in metrics.items():
        if metric_name == "iou":
            for idx, iou_value in enumerate(metric_value):
                logger.info(f"IoU for class {idx}: {iou_value}")
        elif metric_name != "per_class_accuracy":
            logger.info(f"{metric_name}: {metric_value}")

    logger.info("Per-class Accuracy:")
    for idx, acc_value in enumerate(metrics["per_class_accuracy"]):
        logger.info(f"Class {idx}: {acc_value}")
    
    # visualize the results
    logger.info('Visualising the first point cloud with predicted labels...')
    test_index = 0  
    # test_index = random.randint(0, len(test_dataset) - 1)
    test_features, test_labels = test_dataset[test_index]
    # TODO: (check) coordinates are not part of the features in miunkowski
    test_coordinates, test_colors, test_normals = test_features[:, :3], test_features[:, 3:6], test_features[:, 6:9]
    test_preds = all_preds[test_index]
   
    with open(cfg.yaml_file_path, 'r') as yaml_file:
        label_colors = yaml.safe_load(yaml_file)

    visualize_point_cloud_with_labels(test_coordinates, test_colors, test_preds, label_colors, ground_truth_labels=test_labels)

