import pandas as pd
# This script is run after Commands_Combining.py to verify the labels of the commands to avoid any noise in the dataset. 
# Load the output csv file of your commands dataset 
data = pd.read_csv('path to your commands dataset.csv')

# Define conditions for changing labels
conditions = [
    ((data['Command_number'].isin([104, 109])) & (data['Extruder_temp'] == 205) & (data['Layer_thickness'] == 0.2)),
    ((data['Command_number'].isin([104, 109])) & (data['Extruder_temp'] == 200) & (data['Layer_thickness'].isin([0.15, 0.1]))),
    ((data['Command_number'].isin([104, 109])) & (data['Extruder_temp'] == 195) & (data['Layer_thickness'] == 0.06)),
    ((data['Command_number'].isin([140, 190])) & (data['Bed_Temp'] == 60)),
    ((data['Command_number'] == 106) & (data['Fan_speed'] == 85) & (data['Layer_indicator'] == 1)),
    ((data['Command_number'] == 106) & (data['Fan_speed'] == 170) & (data['Layer_indicator'] == 2)),
    ((data['Command_number'] == 106) & (data['Fan_speed'] == 255) & (data['Layer_indicator'] == 3)),
    (data['Command_number'] == 107),
    ((data['Command_Type']==0) & (data['Extruder_temp'] == 195) & (data['Layer_thickness'] == 0.06)& (data['label'] == 3)),
    ((data['Command_Type']==0) & (data['Extruder_temp'] == 200) & (data['Layer_thickness'].isin([0.15, 0.1])) & (data['label'] == 3)), 
    ((data['Command_Type']==0) & (data['Extruder_temp'] == 205) & (data['Layer_thickness'] == 0.2)& (data['label'] == 3)),
    ((data['Command_Type']==0) & (data['Bed_Temp'] == 60) & (data['label'] == 2)),
    ((data['Command_Type']==0) & (data['Fan_speed'] == 85) &  (data['Layer_indicator'] == 1)&(data['label'] == 1)),
    ((data['Command_Type']==0) & (data['Fan_speed'] == 170) &  (data['Layer_indicator'] == 2)&(data['label'] == 1)),
    ((data['Command_Type']==0) & (data['Fan_speed'] == 255) &  (data['Layer_indicator'] == 3)&(data['label'] == 1)) 
   
]

# Apply label changes based on conditions
for condition in conditions:
    data.loc[condition, 'label'] = 0
data = data[~((data['Command_Type'] == 0) & (data['Layer_thickness'] < 0) & (data['label'] == 0))]
# Remove rows where the first five values are all zeros
data = data[~(data.iloc[:, 0:5] == 0).all(axis=1)]

# Save the modified data to a new CSV file
modified_output_path = 'path to the new csv file after verify the commands labels'
data.to_csv(modified_output_path, index=False)
# Count the number of instances for each label
all_labels = [0, 1, 2, 3, 4]  # Assuming these are the possible labels
label_counts = data['label'].value_counts().reindex(all_labels, fill_value=0)
print(label_counts)
print(f"Modified data has been saved to {modified_output_path}.")
