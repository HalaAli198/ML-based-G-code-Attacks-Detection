
import pandas as pd
import os
# This script is used to extract  a subset of commands from each layer after labeling them (Commands_Labeling.py).
# Extract the commands within the same layer that have identical values for the feautrs:
# Nozzle Temperature, Bed Temperature, Fan Speed, Layer Thickness, Z Value, Layer Number, and Layer Indicator.

# Specify the directory containing the CSV files
directory_path = 'path to your dataset of commands'

# Initialize an empty DataFrame for the output
output_df = pd.DataFrame()

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        data = pd.read_csv(file_path)
        output_df = pd.concat([output_df, data])
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        data = pd.read_csv(file_path)
        data_type_0 = data[data['Command_Type'] == 0]
        other_data = data[data['Command_Type'] != 0]
        #columns_to_group_by = [col for col in data_type_0.columns]
        #columns_to_group_by = [col for col in data_type_0.columns if col not in ['Angle', 'Distance', 'Filament']]
        columns_to_group_by = [col for col in data_type_0.columns if col not in ['Angle', 'Distance', 'Filament', 'Command_index', 'Command_Indicator']]
        grouped = data_type_0.groupby(columns_to_group_by)
        for _, group in grouped:
            if len(group) > 1:
                output_df = pd.concat([output_df, group.head(20)])
            else:
                output_df = pd.concat([output_df, group])
       output_df = pd.concat([output_df, other_data])
output_path = 'path to the  output csv files that includes the extrated set from each layer'
output_df.to_csv(output_path, index=False)

print(f"All filtered and grouped rows from multiple files have been saved to {output_path}.")
