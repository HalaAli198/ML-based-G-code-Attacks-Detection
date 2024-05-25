import os
import csv
# This script is used to label the commands within the layers to be later used with MLP and RF alogrithms.
# As the command-level classification is designed to detect the thermodynamic and Z-profile attacks, we cnsider 5 classes. 
# 0: benign, 1: fan speed, 2: bed temperature, 3: nozzle temperature, and 4: Z-profile attack
folder_path = "path to the output folder of Feature_Extraction code"
destination_folder_path = "path to the output folder  where you wanna store the csv files after lableing the commands"
count=0
for filename in os.listdir(folder_path):
     if filename.endswith(".csv") and not any(substring in filename for substring in ["filament_cavity", "filament_density", "filament_state"]):
        file_path = os.path.join(folder_path, filename)
        destination_file_path = os.path.join(destination_folder_path, filename)
        #  As all the  commands within the malicious layers are malicious.

        if  "fan_speed" in filename and  "malicious" in filename:
            label_value = 1
        elif "bed_temperature" in filename and "malicious" in filename:
            label_value = 2 
        elif "nozzle_temperature" in filename and  "malicious" in filename:
            label_value = 3
        elif "Z_profile" in filename  and  "malicious" in filename:
            label_value = 4
             
        else:
            label_value = 0
        
        # Read the CSV file
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            rows = list(reader)
       
        if len(rows) > 0 and len(rows[0]) > 0:
            # Assuming the header is in the first row
            header = rows[0]
            
            # Add the "label" column name to the header
            header.append("label")
            
            # Add the "label" column to each row
            for row in rows[1:]:
                row.append(label_value)

        
        # Write the updated rows back to the CSV file
        with open(destination_file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(rows)
        count+=1
        print(f"Processed file: {filename}")
print(count)
print("Processing complete.")
