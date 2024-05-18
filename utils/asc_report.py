import numpy as np
import os

asc_files_path = "../data/raw"
output_logs_path = "../logs"

for file_name in os.listdir(asc_files_path):
    if file_name.endswith(".asc"):
        input_file = os.path.join(asc_files_path, file_name)
        log_file_path = os.path.join(output_logs_path, f"{os.path.splitext(file_name)[0]}_asc_log.log")

        points = []
        colors = []
        labels = []
        normals = []

        with open(input_file, "r") as file:
            for line in file:
                columns = line.strip().split(";")

                x, y, z = map(float, columns[:3])
                r, g, b = map(float, columns[3:6])
                label = float(columns[6])
                nx, ny, nz = map(float, columns[7:])

                points.append([x, y, z])
                colors.append([r, g, b])
                labels.append(label)
                normals.append([nx, ny, nz])

        points_array = np.array(points)
        colors_array = np.array(colors)
        labels_array = np.array(labels, dtype=int)
        normals_array = np.array(normals)

        # Combine all attributes into a single array
        combined_array = np.column_stack((points_array, colors_array, labels_array[:, None], normals_array))

        # Initialize counters and min/max arrays
        num_fields = combined_array.shape[1]
        counter_zeros = [0] * num_fields
        counter_nans = [0] * num_fields
        counter_infs = [0] * num_fields
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

        # Write the report to a log file
        with open(log_file_path, "w") as log_file:
            log_file.write(f"File: {input_file}\n")

            field_names = ["x", "y", "z", "r", "g", "b", "Classification", "normal_x", "normal_y", "normal_z"]

            for i, field_name in enumerate(field_names):
                log_file.write(f"Number of zeros found in {field_name}: {counter_zeros[i]}\n")
                log_file.write(f"Number of NaNs found in {field_name}: {counter_nans[i]}\n")
                log_file.write(f"Number of infinite values found in {field_name}: {counter_infs[i]}\n")
                if not np.isnan(min_values[i]) and not np.isnan(max_values[i]):
                    log_file.write(f"Min and max values for {field_name}: {min_values[i]}, {max_values[i]}\n")

            log_file.write("Done writing to file\n")
