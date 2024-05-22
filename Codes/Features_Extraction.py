from sys import argv
import math
import sys
import re
import os
import glob
import csv
import random
from collections import Counter
# This code can run on a CPU. It is designed to extract 14 features representing each command, including the command index and command indicator.
# To extract the main 12 features, exclude the variables (command_num, F_I_L) from the command_vector list.
# For experiments to find the optimal feature set, you can exclude any of the variables (z_v, layer_thickness, layer_number, layer_indicator).
if __name__ == '__main__':   
    # Directory paths for G-code input and output csv layers.
    input = 'path to your G-code files'
    output='Path to your output folder of .csv files of layers'
    ## Compile regular expressions to match relevant G-code lines and commands.
    command_pattern = re.compile(r'^([A-Z]\d*)\s*(.*)$')
    pattern = re.compile(r'^[^;].*$', re.MULTILINE) # for the comments in the file
    layer_pattern = re.compile(r';LAYER:(\d+)', re.IGNORECASE) # start of the layer in the file
    layer_pattern2 = re.compile(r';MESH:NONMESH', re.IGNORECASE)  # start of the layer in the file
    pattern2 = re.compile(r'^(.*?)(?<!:)\s*;.*$', re.MULTILINE)
    for filename in os.listdir(input):       
        if filename.endswith('.gcode'):          
            filepath = os.path.join(input, filename)
            command_vectors_sequence=[]
            layer_number=0
            layer_count=0
            last_layer=0
            with open(filepath, 'r') as f:
                code = f.readlines()
                code = [line for line in code]             
                Previous_Z=0
                Layer_Commands=[]
                layer_number=0
                total_layers=0
                Start_layer=False
                End_layer=False
                
                for line in code:
                    match1 = layer_pattern.search(line)
                    if match1:
                      total_layers+=1

                # default thermodynamic settings.
                Extruder_temperature=210  
                Bed_temperature=60
                Fan_Speed=255
                prev_time_elapsed=0

                # Command Type featue (C_t)
                Command_type=0
                # Layer thickness feature (L_{th})
                layer_thickness=0
                # Z value feature (Z_v)
                z_v = 0.0
                previous_Z = 0.0

                # Iterate through each line in the G-code file
                for line_number, line in enumerate(code):
                    # Find the matched lines to determine the layers
                    match1 = layer_pattern2.search(line)
                    if match1 or line_number == len(code) - 1:
                     Z_there=False 
                     if match1:                    
                        for command_number,command in enumerate(Layer_Commands):
                          match2 = layer_pattern.search(command)
                          for command_inner in Layer_Commands:
                            if "Z" in command_inner:
                              Z_there=True
                              break
                          if match2 and Z_there:
                           #Layer Number Feature (L_{n})
                           layer_number = int(match2.group(1))
                           layer_count+=1

                          # Find the initial nozzle and bed temperatures
                          if ";LAYER:0" in command:
                            layer_index =command_number
                            Start_layer=True
                            Layer_Commands[:] = [line for line in Layer_Commands[:layer_index] if line.startswith("G28") or line.startswith("G92")] + Layer_Commands[layer_index:]                              
                          if ";EXTRUDER_TRAIN.0.INITIAL_TEMPERATURE:" in command:
                             Extruder_temperature= int(command.split(":")[-1])
                          if ";BUILD_PLATE.INITIAL_TEMPERATURE:" in command:
                             Bed_temperature= int(command.split(":")[-1])
                   
                     if line_number == len(code) - 1:
                           for command_inner in Layer_Commands:
                            if "Z" in command_inner:
                              Z_there=True
                              break
                           if Z_there:
                             layer_number+=1
                             End_layer=True

                     # Filter the unnecessary commands in the last layer.
                     if layer_number==total_layers-1:#Last layer
                        last_m205_index = len(Layer_Commands) - 1
                        last_g0_index = len(Layer_Commands) - 1
                        for i in range(len(Layer_Commands) - 1, -1, -1):
                            if Layer_Commands[i].startswith("M205"):
                               last_m205_index = i
                               break
                        for i in range(len(Layer_Commands) - 1, -1, -1):
                            if Layer_Commands[i].startswith("G0"):
                               last_g0_index = i
                               break
                        last_g1_index = len(Layer_Commands) - 1
                        for i in range(len(Layer_Commands) - 1, -1, -1):
                           if Layer_Commands[i].startswith("G1"):
                            last_g1_index = i
                            break
                        if last_g1_index != len(Layer_Commands) - 1:
                            Layer_Commands.pop(last_g1_index)
                        Layer_Commands = Layer_Commands[:last_m205_index + 1] + Layer_Commands[last_g0_index:]
                     found_infill=False    
                     infill_commands=[]
                     for layer_command in Layer_Commands: 
                            infill_commands.append(layer_command)                         

                     layer=[]
                     Previous_X1=Previous_Y1=Previous_E1=0
                     commands_sequence=[]
                     command_vector=[]
                     ######################################################------------------Features---------------############################################################################
                  
                     F_I_L=0
                     Z_v=0
                     command_num=-1 # Command Index Feature (C_{idx})
                     for command_index, layer_command in enumerate( infill_commands):
                           # Command Indicator Feature (C_{in})
                           if command_index==0:
                                     F_I_L=0
                           else:
                                     F_I_L=1
                           # Layer Indicator Feature (L_{in})
                           if layer_number==0:
                              layer_indicator=0
                           elif  layer_number==1: 
                              layer_indicator=1
                           elif  layer_number==2: 
                              layer_indicator=2
                           elif  layer_number==total_layers-1: 
                              layer_indicator=4
                           else:
                              layer_indicator=3

                           match = command_pattern.search(layer_command)
                           if match: 
                            g_full = match.group(1)
                            g_letter = match.group(1)[0]
                            #Command Number Feature (C_n)
                            g_number = match.group(1)[1:]
                            parameters= match.group(2)
                            parameters_list=parameters.split()
                            pc=len(parameters_list) 
                            if g_letter=='M'and g_number=='104'  or g_number=='109':
                              command_num+=1
                              Command_type=1
                              for i in range(pc):
                                pt=parameters_list[i][0]
                                if pt=="S": # Nozzle Temperature Feature (S_n)
                                    Extruder_temperature= parameters_list[i][1:]
                              # no movement with M commands, therefore, dsitance,  direction angle, and extruded filament amount are 0
                              command_vector.append((Command_type,int(g_number),0,0,0,Extruder_temperature, 0,0,0,0,command_num,F_I_L,layer_number,layer_indicator))
                            if g_letter=='M'and g_number=='140'  or g_number=='190':
                              command_num+=1
                              Command_type=1
                              for i in range(pc):
                                pt=parameters_list[i][0]
                                if pt=="S":  # Bed Temperature Feature (S_b)
                                   Bed_temperature= parameters_list[i][1:]
                              # no movement with M commands, therefore, dsitance,  direction angle, and extruded filament amount are 0
                              command_vector.append((Command_type,int(g_number),0,0,0,0, Bed_temperature,0,0,0,command_num, F_I_L,layer_number,layer_indicator))
                            if g_letter=='M'and g_number=='106':
                                command_num+=1
                                Command_type=1
                                for i in range(pc):
                                    pt=parameters_list[i][0]
                                    if pt=="S":      # Fan Speed Feature (S_f)
                                      Fan_Speed= parameters_list[i][1:]
                                # no movement with M commands, therefore, dsitance,  direction angle, and extruded filament amount are 0
                                command_vector.append((Command_type,int(g_number),0,0,0,0, 0,Fan_Speed,0,0,command_num,F_I_L,layer_number,layer_indicator))
                                  

                            if g_letter=='M'and g_number=='107':
                                    command_num+=1
                                    Command_type=1
                                    Fan_Speed= 0 # Fan Speed Feature (S_f)
                                    # no movement with M commands, therefore, dsitance,  direction angle, and extruded filament amount are 0
                                    command_vector.append((Command_type,int(g_number),0,0,0,0, 0,Fan_Speed,0,0,command_num,F_I_L,layer_number,layer_indicator))                                 
                            Z_exist=0
                            if g_letter=='G'and (g_number=='0'  or g_number=='1') and "X" in layer_command and "Y" in layer_command:
                                Command_type=0
                                for i in range(pc):
                                    pt = parameters_list[i][0]
                                    if pt == "Z":
                                        Z_exist=1
                                        z_v = float(parameters_list[i][1:]) # Z value Feature (Z_v)
                                        # Relation between nozzle temperature and ;layer thickness (Ultimaker3 profiles)
                                        # Defaults values
                                        if layer_number == 0 and Extruder_temperature==210: 
                                            layer_thickness = 0.2
                                            previous_Z = z_v
                                        elif layer_number == 0 and Extruder_temperature==205:
                                            layer_thickness = 0.15
                                            previous_Z = z_v
                                        elif layer_number == 0 and Extruder_temperature==200:
                                            layer_thickness = 0.06
                                            previous_Z = z_v
                                        elif layer_number != 0:
                                            layer_thickness = z_v - previous_Z
                                            previous_Z = z_v    # Update Z with the last recent value that is used to calculate the layer thickness
                                if g_letter=='G' and g_number=='0' and Z_exist==1: # Some commands like G0 Z value
                                   command_num+=1
                                   command_vector.append((Command_type,int(g_number),0,0,0,0, 0,0,z_v,layer_thickness,command_num,F_I_L,layer_number,layer_indicator))
 
                            if "E" not in layer_command: 
                               E_Value3=0
                            if g_letter=='G'and g_number=='0'  or g_number=='1' or  g_number=='92' and "E" in layer_command:
                               for i in range(pc):
                                  pt=parameters_list[i][0]
                                  if pt=="E":
                                    E_Value=float(parameters_list[i][1:])
                                    
                            
                            if g_letter=='G'and (g_number=='1' or  g_number=='92'):
                              if "E" in layer_command:
                                for i in range(pc):
                                  pt=parameters_list[i][0]
                                  if pt=="E":
                                    E_Value=float(parameters_list[i][1:])
                                 
                                       
                            if  (g_letter=='G'and g_number=='0') or (g_letter=='G'and g_number=='1'):
                                   for i in range(pc):
                                    pt=parameters_list[i][0]
                                    if pt=="Z":
                                       Z_changed=1
                            
                 
                            if (g_letter=='G'and g_number=='1' and  "X" in layer_command and "Y" in layer_command) or (g_letter=='G'and g_number=='92' and  "E" in layer_command)or (g_letter=='G'and g_number=='1' and  "E" in layer_command and pc==1) or (g_letter=='G'and g_number=='1' and  "E" in layer_command and pc==2):
                               Command_type=0
                              
                               if (g_letter=='G'and g_number=='1' and  "X" in layer_command and "Y" in layer_command):  
                                command_num+=1
                                for i in range(pc):
                                  pt=parameters_list[i][0]
                                  if pt=="X":
                                    X_Value3=float(parameters_list[i][1:])
                                  if pt=="Y":
                                    Y_Value3=float(parameters_list[i][1:])
                                  if pt=="Z":
                                    Z_v=float(parameters_list[i][1:])
                             
                                  if pt=="E":
                                    E_Value3=float(parameters_list[i][1:])
                                delta_x = X_Value3 - Previous_X1
                                delta_y = Y_Value3 - Previous_Y1
                                delta_E = E_Value3 - Previous_E1        
                                # Calcualte the movement direction angle (theta)
                                angle_rad = math.atan2(delta_y, delta_x)
                                angle_deg = round(math.degrees(angle_rad),2)
                                # Clcualte the movement distance (d)
                                distance=round(math.sqrt(delta_x**2 + delta_y**2),7)
                                if distance==0: # avoid dividing by zero
                                   distance=0.00001
                                if angle_deg<=0:
                                   angle_deg=180+angle_deg
                                if layer_number==total_layers-1:
                                   last_layer=1
                                # Clcualte the extruded filament amount Delta E= round(E_Value3-Previous_E1,7)
                                command_vector.append((Command_type,int(g_number),angle_deg,distance,round(E_Value3-Previous_E1,7),Extruder_temperature, Bed_temperature,Fan_Speed, z_v,layer_thickness,command_num,F_I_L,layer_number,layer_indicator))  
                                # update the last coordiantes for calcualting the movement distances, as well as Filament length (E)
                                Previous_X1=X_Value3
                                Previous_Y1=Y_Value3
                                Previous_E1=E_Value3

                               if ((g_number=='92' or g_number=='1') and "E" in layer_command and pc==1) or (g_number=='1' and "E" in layer_command and pc==2):
                                  Command_type=0
                                  command_num+=1
                                  for i in range(pc):
                                    pt=parameters_list[i][0]
                                    if pt=="E":
                                       E_Value3=float(parameters_list[i][1:])
                                    if pt=="Z":
                                       Z_v=float(parameters_list[i][1:])
        
                                  time_elapsed = prev_time_elapsed
                                  angle_deg=0
                                  distance=0
                                  X_Value3=Y_Value3=0
                                  if layer_number==total_layers-1:
                                     last_layer=1
                                  command_vector.append((Command_type,int(g_number),angle_deg,distance,round(E_Value3-Previous_E1,7),Extruder_temperature, Bed_temperature,Fan_Speed, z_v,layer_thickness,command_num,F_I_L,layer_number,layer_indicator))                             
                                  Previous_E1=E_Value3
                                                                   
                     if command_vector: # For test files only we take all the layers not only the attacked
                        last_command = list(command_vector[-1])
                        last_command[-3] = 2
                        command_vector[-1] = tuple(last_command)
                        command_vectors_sequence.append(command_vector)
                          
                     Layer_Commands=[]                    
                    else:
                      Layer_Commands.append(line)
            
        
                features_names = ['Command_Type', 'Command_number','Angle','Distance','Filament','Extruder_temp','Bed_Temp','Fan_speed','Z_v','Layer_thickness','Command_index','Command_Indicator','layer_number','Layer_indicator']
                
                for i, sub_seq in enumerate(command_vectors_sequence):   
                  output_filename = os.path.splitext(filename)[0]+"_"+str(i)+".csv"
                  with open(os.path.join(output.replace("\\", "/"), output_filename), 'w', newline='') as file_ML:
                    writer = csv.writer(file_ML)
                    writer.writerow(features_names)
                    for data_entry in sub_seq:
                        writer.writerow(data_entry)

               