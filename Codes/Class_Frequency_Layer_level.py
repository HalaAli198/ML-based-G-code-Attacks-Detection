import os
import re
from collections import defaultdict
# This code calculates the class frequency based on  file names of the layers in your dataset folder
# The csv files are named based on the attack class.

# Get the basic G-code file name before spliting into layers csv files.
def extract_common_identifier(filename):
    # Adjust the regular expression based on the mutual part's pattern in the layers filename
    return re.sub(r'(_\d+)?(_malicious)?\.csv', '', filename)

#Count the class frequency
def count_benign_files(directory):
    file_groups = defaultdict(list)
    benign_count = 0
    malicious_count=0
    count_cavity_layer, count_density_layer, count_state_layer, benign_count_layer, count_bed_layer, count_ext_layer,count_fan_layer, count_layer_layer= 0, 0, 0, 0 ,0,0,0,0
    count_layer=0
    for filename in os.listdir(directory):
        count_layer+=1 
        if filename.endswith(".csv"):
            if "filament_cavity"  in filename:
                count_cavity_layer+=1
            elif "filament_density"  in filename:
                count_density_layer+=1
            elif "filament_state" in filename:
                count_state_layer+=1
            elif "fan_speed" in filename:
                count_fan_layer+=1
            elif "bed_temperature" in filename:
                count_bed_layer+=1
            elif "nozzle_temperature" in filename:
                count_ext_layer+=1
            elif "Z_profile" in filename:
                count_layer_layer+=1
            else:
                benign_count_layer+=1
    print(f"Number of total  G-code layer: {count_layer}")
    print(f"Number of Benign  G-code layer: {benign_count_layer}")
    print(f"Number of cavity  G-code layer: {count_cavity_layer}")
    print(f"Number of density  G-code layer: {count_density_layer}")
    print(f"Number of state  G-code layer: {count_state_layer}")
    print(f"Number of fan  G-code layer: {count_fan_layer}")
    print(f"Number of bed  G-code layer: {count_bed_layer}")
    print(f"Number of nozzle  G-code layer: {count_ext_layer}")
    print(f"Number of Z_profile  G-code layer: {count_layer_layer}")
    print("--------------------------------------------------------------------")
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            common_identifier = extract_common_identifier(filename)
            file_groups[common_identifier].append(filename)
    count=0
    count_cavity, count_density, count_state, benign_count2, count_bed, count_ext,count_fan , count_layer=  0, 0, 0, 0 , 0, 0, 0, 0
    for common_id, filenames in file_groups.items():
        if "filament_cavity" in common_id:
            count_cavity+=1
        elif "filament_density" in common_id:
             count_density+=1
        elif "filament_state" in common_id:
             count_state+=1
        elif "fan_speed" in common_id:
             count_fan+=1
        elif "bed_temperature" in common_id:
            count_bed+=1
        elif "nozzle_temperature" in common_id:
             count_ext+=1
        elif "Z_profile" in common_id:
            count_layer+=1
        else:
              benign_count2+=1

    for common_id, filenames in file_groups.items():
        count+=1
        # Check if any of the filenames contain 'cavity', 'density', or 'state'
        if all("filament_cavity"  in f or "filament_density"  in f or "filament_state"  in f  or "fan_speed" in f or "bed_temperature"  in f or "nozzle_temperature" in f or "Z_profile" in f for f in filenames):
            malicious_count += 1
    benign_count=count-malicious_count
    print(f"Number of total  G-code files: {count}")
    print(f"Number of Benign  G-code files: {benign_count2}")
    print(f"Number of cavity  G-code files: {count_cavity}")
    print(f"Number of density  G-code files: {count_density}")
    print(f"Number of state  G-code files: {count_state}")
    print(f"Number of fan  G-code files: {count_fan}")
    print(f"Number of bed  G-code files: {count_bed}")
    print(f"Number of nozzle G-code files: {count_ext}")
    print(f"Number of Z_profile  G-code files: {count_layer}")
    return benign_count, file_groups.keys()
            
if __name__ == '__main__':
    files_dir = 'path to your dataset folder'   # Replace with your directory
    benign_file_count = count_benign_files(files_dir)

     
