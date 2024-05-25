import os
import pandas as pd

def process_csv_files(input_folder, output_folder, columns_to_exclude):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of CSV files in the input folder
    csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]

    # Process each CSV file
    for file_name in csv_files:
        # Construct the input and output file paths
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)

        # Load the data from CSV file
        df = pd.read_csv(input_file_path)

        # Exclude any of the (Layer Indicator, Layer Number, Layer Thickness, Z value) Feature column
        #Drop the specified column from the dataframe to generate Folders of new datasets (DSin, DSn, DSth, SDz).
        df_filtered = df.drop(columns=columns_to_exclude)

        # Write the filtered data to a new CSV file
        df_filtered.to_csv(output_file_path, index=False)

        print(f'Filtered CSV file has been created at {output_file_path}')

# Specify the input and output folder paths
input_folder = 'path to your dataset folder of 12 Features (i.e., DS6)'
output_folder = 'path to your new dataset folder after excluding one feature'

# Specify the column/ columns to exclude
columns_to_exclude = ['Layer_thickness'] # Layer_indicator, layer_number, Layer_thickness, Z_v

# Call the function to process the CSV files
process_csv_files(input_folder, output_folder, columns_to_exclude)