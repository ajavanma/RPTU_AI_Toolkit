import yaml
import numpy as np
import open3d as o3d
import logging
from typing import Tuple
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import random
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import Config
from pathlib import Path

def get_logger(filename: str, name: str = None):
    log_directory = os.path.dirname(filename)
    if log_directory and not os.path.exists(log_directory):
        os.makedirs(log_directory)
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(filename, mode="a")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

def load_config_file(file_path: str):
    try:
        with open(file_path, 'r') as file:
            config = yaml.full_load(file)
        #logger.debug(f"Loaded config file: {file_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Error: Configuration file {file_path} not found.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error: YAML parsing error in {file_path}. Details: {e}")
        return None

logger = get_logger('logs/log-utils.txt', __name__)

config_data = load_config_file("config.yaml")
if config_data is not None:
    cfg = Config(config_data)
else:
    logger.error("Error: Configuration not loaded.")


def load_label_color_map(yaml_file_path) -> dict:
    try:
        with open(yaml_file_path) as yamlfile:
            label_color = yaml.safe_load(yamlfile, Loader=yaml.FullLoader)
            logger.info("Loaded %d labels from %s", len(label_color), yaml_file_path)
    except FileNotFoundError:
        raise ValueError(f"{yaml_file_path} not found.")
    return label_color

def mp(mapper_dict: dict, entry: int) -> tuple:
    """ map class labels to RGB colors
    Args:
        mapper_dict: dictionary loaded from yaml file
        entry: int representing the label
        default color is black
    """
    colors = mapper_dict.get(entry, (0, 0, 0))
    return tuple(colors)

# return a point cloud object
def out_pcd(coords: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Point cloud from numpy arrays of shape (n, 3)
    Args:
        coords: Numpy array of shape (n, 3) with 3D coordinates of points
        colors: Numpy array of shape (n, 3) with colors
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# Visualizes a point cloud with colors based on their respective labels.
def visualize_point_cloud_with_labels(coordinates, colors, labels, label_colors,
                                      ground_truth_labels=None, subplot_idx=1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coordinates)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    label_colors = np.array(label_colors)
    pcd.colors = o3d.utility.Vector3dVector(label_colors[labels])
    o3d.visualization.draw_geometries([pcd])
    if ground_truth_labels is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c=colors)
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title('Ground Truth')
        ax1.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c=[label_colors[l] for l in ground_truth_labels])
        
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title('Predicted')
        ax2.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c=[label_colors[l] for l in labels])

    plt.show()

def calculate_iou(confusion_mtx: np.ndarray) -> Tuple[np.ndarray, float]:
    intersection = np.diag(confusion_mtx)
    union = confusion_mtx.sum(axis=0) + confusion_mtx.sum(axis=1) - intersection
    iou = intersection / union
    mean_iou = np.mean(iou)
    return iou, mean_iou

paths = {
    'pcd_files': cfg.pcd_files_path,
    'asc_files': cfg.asc_files_path,
    'train': cfg.train_path,
    'val': cfg.val_path,
    'test': cfg.test_path
}

# find match based on file names, used for training
def files_match_making(pcd_files, asc_files):
    pcd_files = sorted(list(Path(cfg.pcd_files_path).glob('*.pcd')))
    asc_files = sorted(list(Path(cfg.asc_files_path).glob('*.asc')))

    matched_pairs = []
    for pcd_file in pcd_files:
        pcd_file_name = os.path.splitext(os.path.basename(pcd_file))[0]
        for asc_file in asc_files:
            asc_file_name = os.path.splitext(os.path.basename(asc_file))[0]
            if pcd_file_name == asc_file_name:
                matched_pairs.append((pcd_file, asc_file))
                break
        else:
            print(f"Warning: No corresponding ASC file found for PCD file {pcd_file}.")
    return matched_pairs


def calculate_metrics(all_preds, all_labels):
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    per_class_accuracy = np.diag(confusion_matrix(labels, preds)) / np.bincount(labels)
    overall_accuracy = np.mean(per_class_accuracy)
    confusion_mtx = confusion_matrix(labels, preds)
    iou, mean_iou = calculate_iou(confusion_mtx)
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'overall_accuracy': overall_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'mean_iou': mean_iou
    }


# Split PCD files into training, validation, and test datasets, along with their corresponding ASC files.
def split_data(paths, ratios, shuffle, random_seed):
    # Ensure the ratios add up to 1.0
    assert sum(ratios.values()) == 1.0, "The ratios must add up to 1.0."
    
    pcd_files = [f for f in os.listdir(paths['pcd_files']) if f.endswith(".pcd")]
    
    if shuffle:
        random.seed(random_seed)
        random.shuffle(pcd_files)
    
    num_files = len(pcd_files)
    num_data = {key: int(ratio * num_files) for key, ratio in ratios.items()}
    num_data['test'] = num_files - num_data['train'] - num_data['val']

    for i, pcd_file in enumerate(pcd_files):
        src_pcd_path = os.path.join(paths['pcd_files'], pcd_file)
        base_name = os.path.splitext(pcd_file)[0]
        src_asc_path = os.path.join(paths['asc_files'], base_name + ".asc")
        
        if i < num_data['train']:
            dst_key = 'train'
        elif i < num_data['train'] + num_data['val']:
            dst_key = 'val'
        else:
            dst_key = 'test'
        
        dst_pcd_path = os.path.join(paths[dst_key], pcd_file)
        dst_asc_path = os.path.join(paths[dst_key], base_name + ".asc")
        
        shutil.copy2(src_pcd_path, dst_pcd_path)
        shutil.copy2(src_asc_path, dst_asc_path)

        logger.info("Copied %s %s .pcd files to %s", num_data[dst_key], dst_key, dst_pcd_path)
        logger.info("Copied %s corresponding %s .asc files to %s", num_data[dst_key], dst_key, dst_asc_path)


# minkowski compatible array:
def convert_to_array(self):
        points = np.asarray(self.down_pcd.points, dtype=np.float32)
        colors = np.asarray(self.down_pcd.colors, dtype=np.float32)
        normals = np.asarray(self.down_pcd.normals, dtype=np.float32)
        features = np.concatenate([colors, normals], axis=1)
        labels = np.asarray(self.label_list, dtype=np.int64)

        self.feature_arr = np.concatenate([points, features, labels[:, None]], axis=1)

        logger.info('Colors are normalized: Min: %f, Max: %f', np.min(colors), np.max(colors))
        if points.shape != colors.shape or colors.shape != normals.shape:
            logger.error("Arrays must have the same shape")
            raise ValueError("Arrays must have the same shape")
        else:
            logger.info("Shapes of all arrays are equal: {}".format(points.shape))

        logger.info('points shape: %s', points.shape)
        logger.info('colors shape: %s', colors.shape)
        logger.info('normals shape: %s', normals.shape)
        logger.info('labels shape: %s', labels.shape)
        print(self.feature_arr[:5])


# Disable logger calls during parallel computation, or use a separate logger for each worker to avoid any conflicts.
def main_preprocessing():
   pcd_files = sorted(list(Path(cfg.pcd_files_path).glob('*.pcd')))
   asc_files = sorted(list(Path(cfg.asc_files_path).glob('*.asc')))

   matched_file_pairs = files_match_making(pcd_files, asc_files)

   with ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
       print(f"matched file pairs: {matched_file_pairs}")
       futures = [executor.submit(process_pcd, matched_file_pair, cfg.voxel_size) for matched_file_pair in matched_file_pairs]
       for future in tqdm(as_completed(futures), total=len(futures)):
           try:
               future.result()
           except Exception as e:
               logger.error(f"An error occurred during preprocessing: {e}")
               


# check sparsity of an asc file
# keep in mind that we know for a fact that the first three columns are x,y,z coordinates
# for pcd files it is not this straightforward, we need to check the header
# filename = "construction_small.asc"

# count_x_zeros = 0
# count_y_zeros = 0
# count_z_zeros = 0

# with open(filename, "r") as file:
#     for line in file:
#         # Split the line using semicolon as the delimiter
#         columns = line.strip().split(";")
        
#         # Check if the first column value is zero
#         if float(columns[0]) == 0.0:
#             count_x_zeros += 1

#         # Check if the second column value is zero
#         if float(columns[1]) == 0.0:
#             count_y_zeros += 1

#         # Check if the third column value is zero
#         if float(columns[2]) == 0.0:
#             count_z_zeros += 1

# print("Number of zeros found in x:", count_x_zeros)
# print("Number of zeros found in y:", count_y_zeros)
# print("Number of zeros found in z:", count_z_zeros)

# # check sparsity of a pcd file
# import open3d as o3d
# import numpy as np

# filename = "construction_small.pcd"



# # check the PCD file for sparsity
# from colorama import Fore
# import open3d as o3d
# import numpy as np

# filename = "construction_small.pcd"

# try:
#     point_cloud = o3d.io.read_point_cloud(filename)
# except Exception as e:
#     print(f"Error loading PCD file '{filename}': {e}")
#     exit(1)

# print("PCD file information:")
# print(point_cloud)

# # Define the field names based on the PCD header
# # FIELDS Classification rgb normal_x normal_y normal_z x y z _
# field_names = ["Classification", "rgb", "normal_x", "normal_y", "normal_z", "x", "y", "z", "_"]

# # Initialize counters
# num_fields = len(field_names)
# counter_zeros = [0] * num_fields
# counter_nans = [0] * num_fields
# counter_infs = [0] * num_fields

# try:
#     # Convert the points to a NumPy array
#     points_array = np.asarray(point_cloud.points)
      
#     # Iterate through the points in the point cloud
#     for point in points_array:
#         for i, value in enumerate(point):
#             # Check if the value is zero
#             if value == 0.0:
#                 counter_zeros[i] += 1

#             # Check if the value is NaN
#             if np.isnan(value):
#                 counter_nans[i] += 1

#             # Check if the value is infinite
#             if np.isinf(value):
#                 counter_infs[i] += 1

# except Exception as e:
#     print(f"Error processing points: {e}")
#     exit(1)

# # Print the results for each field
# for i, field_name in enumerate(field_names):
#     print(Fore.GREEN + f"Number of zeros found in {field_name}:" + Fore.RESET , counter_zeros[i])
#     print(f"Number of NaNs found in {field_name}:", counter_nans[i])
#     print(f"Number of infinite values found in {field_name}:", counter_infs[i])
    

# def preprocess_main():
#     pcd_files = sorted(list(Path(cfg.pcd_files_path).glob('*.pcd')))
#     asc_files = sorted(list(Path(cfg.asc_files_path).glob('*.asc')))

#     matched_file_pairs = files_match_making(pcd_files, asc_files)
#     #logger.info(f"matched file pairs: {matched_file_pairs}")


#     results = Parallel(n_jobs=cfg.num_workers)(
#         delayed(process_pcd)(matched_file_pair, cfg.voxel_size)
#         for matched_file_pair in tqdm(matched_file_pairs, desc="Processing files", unit="file")
#     )

#     failed_files_count = results.count(False)
#     if failed_files_count > 0:
#         logger.error(f"{failed_files_count} files failed to process.")
#     else:
#         logger.info("All files processed successfully.")