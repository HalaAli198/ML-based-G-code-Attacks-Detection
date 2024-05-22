import csv
import os
# This script is used to  combine all the commands from all the layers (output of Feature_Extraction.py) after labeling them.
# The output csv file used for command-level classification.
def combine_csv_files(source_folder, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]
    all_rows = []  # To store all rows from all files
    header_saved = False  # Flag to check if header has been written to the output file

    for filename in files:
        with open(os.path.join(source_folder, filename), 'r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # Read the header from the file
            if not header_saved:  # Write header once if it hasn't been written yet
                all_rows.append(header)
                header_saved = True
            all_rows.extend(row for row in reader)  # Add all rows from the current file to the list

    # Write all rows to the output CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_rows)

# Usage
source_folder = '  Path to the folder containing the CSV files of your layers'
output_file = '  # Path to the output CSV file that contains all the commands from all the layers'
combine_csv_files(source_folder, output_file)
