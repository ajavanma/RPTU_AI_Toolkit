"""Generate raw data Report for pcd files.

The script iterates through all the PCD files in the specified directory, processes them, and writes the results to their respective log files.

First check the field names manually to be able to iterate through all the attributes.
"""

import open3d as o3d
import numpy as np
import os
from colorama import Fore

pcd_files_directory = "../data/raw"
output_logs_path = "../logs"

for filename in os.listdir(pcd_files_directory):
    if filename.endswith(".pcd"):
        file_path = os.path.join(pcd_files_directory, filename)
        log_file_path = os.path.join(output_logs_path, f"{filename}_report.log")

        try:
            point_cloud = o3d.io.read_point_cloud(file_path)
        except Exception as e:
            with open(log_file_path, "w") as log_file:
                log_file.write(f"Error loading PCD file '{file_path}': {e}\n")
            continue

        # Define the field names based on the PCD header
        field_names = ["Classification", "rgb", "normal_x", "normal_y", "normal_z", "x", "y", "z", "_"]

        # Initialize counters
        num_fields = len(field_names)
        counter_zeros = [0] * num_fields
        counter_nans = [0] * num_fields
        counter_infs = [0] * num_fields

        # Convert the points and their attributes to NumPy arrays
        points_array = np.asarray(point_cloud.points)
        colors_array = np.asarray(point_cloud.colors)
        normals_array = np.asarray(point_cloud.normals)

        combined_array = np.column_stack((points_array, colors_array, normals_array))

        # Initialize the min and max arrays
        min_values = np.full(num_fields, np.nan)
        max_values = np.full(num_fields, np.nan)

        for point in combined_array:
            for i, value in enumerate(point):
                if value == 0.0:
                    counter_zeros[i] += 1

                if np.isnan(value):
                    counter_nans[i] += 1
                else:
                    # Update min and max values for the current column
                    min_values[i] = min(min_values[i], value) if not np.isnan(min_values[i]) else value
                    max_values[i] = max(max_values[i], value) if not np.isnan(max_values[i]) else value

                if np.isinf(value):
                    counter_infs[i] += 1

        # Write the results for each field to the log file
        with open(log_file_path, "w") as log_file:
            log_file.write(f"PCD file information for '{file_path}':\n{point_cloud}\n")
            log_file.write("First three relevant rows: xyz, rgb, n:\n")
            log_file.write(str(combined_array[:3]) + "\n")

            for i, field_name in enumerate(field_names):
                log_file.write(f"Number of zeros found in {field_name}: {counter_zeros[i]}\n")
                log_file.write(f"Number of NaNs found in {field_name}: {counter_nans[i]}\n")
                log_file.write(f"Number of infinite values found in {field_name}: {counter_infs[i]}\n")
                if not np.isnan(min_values[i]) and not np.isnan(max_values[i]):
                    log_file.write(f"Min and max values for {field_name}: {min_values[i]}, {max_values[i]}\n")

            log_file.write("Done writing to file\n")

