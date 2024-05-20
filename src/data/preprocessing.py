import logging
import os
from statistics import mode
from pathlib import Path
import open3d as o3d
import numpy as np
from config import Config
import numpy.lib.recfunctions as rfn
from utils import files_match_making
import pickle
from tqdm.contrib.concurrent import process_map
from colorama import Fore
from utils import load_config_file, get_logger
import time 

logger = get_logger("logs/log-preprocess_minkowski.txt", __name__)

config_data = load_config_file("config.yaml")
cfg = Config(config_data)

class Preprocessor:
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
        self.feature_arr = None 
        self.base_file_name = None 
        
    def process_files(self, pcd_file, asc_file):
        self.base_file_name = os.path.splitext(os.path.basename(pcd_file))[0]
        self.load_pcd(pcd_file)
        self.load_asc(asc_file)
        self.normalize_pcd()
        self.downsample_pcd()
        self.estimate_normals()
        self.assign_label()
        self.generate_feature_arr()
        self.check_and_remove_incomplete_rows()
        self.check_and_remove_invalid_rows()
        self.save_processed_pkl()
        # self.save_processed_ply()
        
    def load_pcd(self, pcd_file):
        try:
            self.pcd = o3d.io.read_point_cloud(str(pcd_file), remove_nan_points=True, remove_infinite_points=True)
            if not self.pcd.has_points():
                raise ValueError(f"The PCD file {pcd_file} is empty or corrupted.")
        except Exception as e:
            logging.error(f"Error loading point cloud data file {pcd_file}: {e}")
            raise e
        
    def load_asc(self, asc_file):
        try:
            asc_array = np.loadtxt(asc_file, delimiter=";")
            self.labels = asc_array[:, 6].astype(int)
        except Exception as e:
            logging.error(f"Error loading ASC file {asc_file}: {e}")
    # (0,1)      
    def normalize_pcd(self):
        origin = np.array([0.0, 0.0, 0.0])
        self.pcd.translate(origin)
        logger.info(f"{Fore.CYAN}Normalizing coords...{Fore.RESET}")
        logger.info('Translated to the origin, consider log-scaling or min-max scaling when dealing with scattered values (outliers) or too much detail will be lost')
        logger.info('Translated coords: Min: %f, Max: %f', np.min(self.pcd.points), np.max(self.pcd.points))
        scale_factor = 1.0 / np.max(np.abs(self.pcd.points))
        self.pcd.scale(scale_factor, origin)
        logger.info('Normalized coords with the scalefactor of %f', scale_factor)
        logger.info('Scaled coords: Min: %f, Max: %f', np.min(self.pcd.points), np.max(self.pcd.points))
        #logger.info('Visualizing the normalized point clouds')
        #o3d.visualization.draw_geometries([self.pcd])
        
        # if stride=2 for MinkowskiEngine is required, offset the coords:
        # AssertionError: The minimum coordinates must be divisible by the tensor stride.
        # min_coords = np.min(self.pcd.points, axis=0)
        # new_origin = min_coords - (min_coords % 2)
        # self.pcd.translate(-new_origin)
    
    def downsample_pcd(self):
        self.down_pcd = self.pcd.voxel_down_sample(self.voxel_size)
        
        coords_np = np.asarray(self.down_pcd.points)
        is_aligned = np.all(coords_np % cfg.stride == 0)
        assert is_aligned, f"Coordinates in {self.base_file_name} are not aligned with the tensor stride after downsampling: preprocess_minkowski.py"

        logger.info("Downsampled from {} to {} points".format(len(self.pcd.points), len(self.down_pcd.points)))
        #logger.info('Visualizing the downsampled point clouds')
        #o3d.visualization.draw_geometries([self.down_pcd])
   
    def estimate_normals(self): 
        logger.info(f"{Fore.CYAN}Estimating normals...{Fore.RESET}")
        radius_normal = self.voxel_size * 2
        start_time = time.time()  
        self.down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        elapsed_time = time.time() - start_time 
        logger.info(f"Normals estimated in {elapsed_time:.2f} seconds")
        logger.info('Each point has %d normals', len(self.down_pcd.normals[0]))
        logger.info('Estimated normals with radius %f', radius_normal)
        logger.info('Normals: Min: %f, Max: %f', np.min(self.down_pcd.normals), np.max(self.down_pcd.normals))
        #logger.info('Visualizing the downsampled point clouds with normals')
    
    def assign_label(self):
        logger.info(f"{Fore.CYAN}Assigning Labels...{Fore.RESET}")
        pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        num_points_down_pcd = len(self.down_pcd.points)
        self.label_list = []
        for i in range(num_points_down_pcd):
            num_neighbors = 5
            [num_neighbors, idx, _] = pcd_tree.search_knn_vector_3d(self.down_pcd.points[i], num_neighbors)
            neighbor_labels = [self.labels[index] for index in idx]
            most_frequent_label = mode(neighbor_labels)
            self.label_list.append(most_frequent_label)
        logger.info('Assigned %d labels to %d points', len(self.label_list) , num_points_down_pcd)
        #o3d.visualization.draw_geometries([self.down_pcd])

        # Check data consistency
        #assert points.shape[0] == labels.shape[0], f"Coordinate length {points.shape[0]} != Label length {labels.shape[0]}"
        #for i in range(len(points)):
        #    if points[i].shape != colors[i].shape or colors[i].shape != normals[i].shape:
        #        logger.error("Row %d: Arrays must have the same shape", i)
        #        raise ValueError("Row %d: Arrays must have the same shape" % i)
        #logger.info("Data consistency check: Shapes of all arrays are equal: {}".format(points.shape))
    
    def generate_feature_arr(self):
        self.coords = np.asarray(self.down_pcd.points, dtype=np.float32)
        colors = np.asarray(self.down_pcd.colors, dtype=np.float32)
        normals = np.asarray(self.down_pcd.normals, dtype=np.float32)
        features = np.concatenate([colors, normals], axis=1)
        labels = np.asarray(self.label_list, dtype=np.int64)
        self.feature_arr = features
        self.label_arr = labels

    def check_and_remove_incomplete_rows(self):
        rows_to_remove = []
        for i, row in enumerate(self.feature_arr):
            if not np.all(np.isfinite(row)):
                rows_to_remove.append(i)

        if len(rows_to_remove) > 0:
            logger.warning(f"{len(rows_to_remove)} incomplete rows found and removed.")
            self.coords = np.delete(self.coords, rows_to_remove, axis=0)  # Add this line to remove rows from coords
            self.feature_arr = np.delete(self.feature_arr, rows_to_remove, axis=0)
            self.label_arr = np.delete(self.label_arr, rows_to_remove, axis=0)
        else:
            logger.info("All rows are complete.")
    
    def check_and_remove_invalid_rows(self):
        rows_to_remove = []

        for i, label in enumerate(self.label_arr):
            if label == 0:
                rows_to_remove.append(i)

        if len(rows_to_remove) > 0:
            logger.warning(f"{len(rows_to_remove)} invalid rows found and removed.")
            self.coords = np.delete(self.coords, rows_to_remove, axis=0)
            self.feature_arr = np.delete(self.feature_arr, rows_to_remove, axis=0)
            self.label_arr = np.delete(self.label_arr, rows_to_remove, axis=0)
        else:
            logger.info("No rows with 'Invalid' label found.")

    def save_processed_pkl(self):
        
        coords_np = np.array(self.coords)
        is_aligned = np.all(coords_np % cfg.stride == 0)
        assert is_aligned, f"Coordinates in {self.base_file_name} are not aligned with the tensor stride before saving: preprocess_minkowski.py"

        data = {
            'coords': self.coords,
            'features': self.feature_arr,  # colors and normals concatenated
            'labels': self.label_arr
        }

        processed_pkl_file = os.path.join(cfg.preprocessed_data_dir, f"{self.base_file_name}_preprocessed.pkl")
        with open(processed_pkl_file, 'wb') as f:
            pickle.dump(data, f)

def process_pcd(matched_file_pair, voxel_size: float):
    pcd_file, asc_file = matched_file_pair
    try:
        preprocessor = Preprocessor(voxel_size)
        preprocessor.process_files(pcd_file, asc_file)
    except Exception as e:
        logger.error(f"An error occurred during preprocessing {pcd_file} or its corresponding asc file {asc_file}: {e}")

def process_pcd_with_error_handling(matched_file_pair, voxel_size: float):
    pcd_file, asc_file = matched_file_pair
    try:
        process_pcd(matched_file_pair, voxel_size)
        return True
    except Exception as e:
        logger.error(f"An error occurred during preprocessing {pcd_file} or its corresponding asc file {asc_file}: {e}")
        return False
    
def main():
    pcd_files = sorted(list(Path(cfg.pcd_files_path).glob('*.pcd')))
    asc_files = sorted(list(Path(cfg.asc_files_path).glob('*.asc')))

    matched_file_pairs = files_match_making(pcd_files, asc_files)

    results = process_map(  
        process_pcd_with_error_handling, 
        matched_file_pairs,
        chunksize=1,
        max_workers=cfg.num_workers,
        desc="Processing files",
        unit="file",
        func_args=(cfg.voxel_size,), 
    )

    failed_files_count = results.count(False)
    if failed_files_count > 0:
        logger.error(f"{failed_files_count} files failed to process.")
    else:
        logger.info("All files processed successfully.")

