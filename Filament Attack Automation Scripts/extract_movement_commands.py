
# This code represent the G-code file as list of layers. Each layer in turns includes a list of movement commands. 
# All the following process of G-code commands relays on the generated G-code structures by Cura slicer for Ultimaker3 printer. W
def list_movement_commands(gcode_file_name):
    gcode_file = open(gcode_file_name)
    commands = gcode_file.readlines() #  

    # Define the list of layers that each has a list of movement commands.
    movement_commands_list =[[]] 
    
    command_number = layer_number=0;
    #Initialize the movement command parameters (X,Y,Z,E). 
    x_value = Y_value = Z_value=E_value=0

    for command in commands:
        command=command.strip('\n\r')            
        command_number+=1  # Count the number of commands in the G-code file. 
        
        if ";LAYER_COUNT:" in command:  # Check the line that contains this comment to get the number of layers. 
            total_layers = int(command[13:])

        if ";TIME_ELAPSED:" in command and layer_number +1 == total_layers: # Stop when we reach the end of the G-code file.
            break
            
        if ";LAYER:" in command: # Check if the line in the G-code file has this comment which indicated to a new layer starting.
            layer_number = int(command[7:]) # get the layer number.
            
            if layer_number >0:
                movement_commands_list.append([]) # Add an initialized list that represent the layer. 
        if ("G1" in command or "G0" in command): # Check if the command is a movement command.
            if "Z" in command: # Check if the command contains a Z coordinate.
                zIndexStart = command.index("Z")  # Find the start index of the Z value.
                if " " in command[zIndexStart:] : # Check if there is a space after the Z value.
                    zIndexEnd = zIndexStart+ command[zIndexStart+1:].index(" ") # Find the end index of the Z value (first space after the Z).
                    Z_value = command[zIndexStart+1:zIndexEnd+1] # Extract the Z value.
                else: Z_value = command[zIndexStart+1:] # If there is no space after the Z value, extract the Z value to the end of the command.



            if "X" in command: # Check if the command contains a X coordinate.
                xIndexStart = command.index("X")  # Find the start index of the X value.
                if " " in command[xIndexStart:] : # Check if there is a space after the X value.
                    xIndexEnd = xIndexStart+ command[xIndexStart+1:].index(" ") # Find the end index of the X value (first space after the X).
                    X_value = command[xIndexStart+1:xIndexEnd+1] # Extract the X value.
                else: X_value = command[xIndexStart+1:] # If there is no space after the X value, extract the X value to the end of the command.


            if "Y" in command: # Check if the command contains a Y coordinate.
                yIndexStart = command.index("Y") # Find the start index of the Y value.
                if " " in command[yIndexStart:] : # Check if there is a space after the Y value.
                    yIndexEnd = yIndexStart+ command[yIndexStart+1:].index(" ") # Find the end index of the Y value (first space after the Y).
                    Y_value = command[yIndexStart+1:yIndexEnd+1] # Extract the Y value.
                else: Y_value = command[yIndexStart+1:] # If there is no space after the Y value, extract the Y value to the end of the command.



            if "E" in command: # Check if the command contains a E coordinate.
                eIndexStart = command.index("E") # Find the start index of the E value.
                if " " in command[eIndexStart:] : # Check if there is a space after the E value.
                    eIndexEnd = eIndexStart+ command[eIndexStart+1:].index(" ") # Find the end index of the E value (first space after the E).
                    E_value = command[eIndexStart+1:eIndexEnd+1] # Extract the Y value.
                else: E_value = command[eIndexStart+1:] # If there is no space after the E value, extract the E value to the end of the command.


            Command_enrty = [command_number, float(X_value), float(Y_value), float(Z_value), float(E_value)]
            movement_commands_list[layer_number].append(Command_enrty)
    
    
    gcode_file.close()
    return movement_commands_list

#***********************************************************************************************************
# The following function is list_movement_commands, focuing on the infill commands in case the G-code file has an infill section. 
def moves_infill_section(gcode_file_name):
    gcode_file = open(gcode_file_name)
    commands = gcode_file.readlines()

    # Define the list of layers that each has a list of movement commands.
    movement_commands_list =[[]] 
    command_number = layer_number=0;
    #Initialize the movement command parameters (X,Y,Z,E). 
    x_value = Y_value = Z_value=E_value=0
    #Boolean variable to indicate if the G-code file has infill section or not. 
    infill_section=False


    for command in commands:
        command=command.strip('\n\r')            
        command_number+=1
        
        if ";LAYER_COUNT:" in command:   # Check the line that contains this comment to get the number of layers. 
             total_layers = int(command[13:])

        if ";TIME_ELAPSED:" in command and layer_number +1 == total_layers: # Stop when we reach the end of the G-code file.
            break
            
        if ";LAYER:" in command: # Check if the line in the G-code file has this comment which indicated to a new layer starting.
            layer_number = int(command[7:]) # get the layer number.
            if layer_number >0:
                movement_commands_list.append([]) #  Add an initialized list that represent the layer. 
        
        if ";TYPE:FILL" in command: #  Indicate the Iinfill section in case the G-code file has in infill section.
            infill_section=True
            
        if ";TYPE:WALL-INNER" in command: # Indicate the inner walls of the object in case the G-code file has inner walls section. 
            infill_section=False
            
        if ";TYPE:WALL-OUTER" in command: # Indicate the outer walls of the object in case the G-code file has outer walls section. 
            infill_section=False
       
        if ("G1" in command or "G0" in command): # Check if the command is a movement command.
            if "Z" in command: # Check if the command contains a Z coordinate.
                zIndexStart = command.index("Z")  # Find the start index of the Z value.
                if " " in command[zIndexStart:] : # Check if there is a space after the Z value.
                    zIndexEnd = zIndexStart+ command[zIndexStart+1:].index(" ") # Find the end index of the Z value (first space after the Z).
                    Z_value = command[zIndexStart+1:zIndexEnd+1] # Extract the Z value.
                else: Z_value = command[zIndexStart+1:] # If there is no space after the Z value, extract the Z value to the end of the command.



            if "X" in command: # Check if the command contains a X coordinate.
                xIndexStart = command.index("X")  # Find the start index of the X value.
                if " " in command[xIndexStart:] : # Check if there is a space after the X value.
                    xIndexEnd = xIndexStart+ command[xIndexStart+1:].index(" ") # Find the end index of the X value (first space after the X).
                    X_value = command[xIndexStart+1:xIndexEnd+1] # Extract the X value.
                else: X_value = command[xIndexStart+1:] # If there is no space after the X value, extract the X value to the end of the command.


            if "Y" in command: # Check if the command contains a Y coordinate.
                yIndexStart = command.index("Y") # Find the start index of the Y value.
                if " " in command[yIndexStart:] : # Check if there is a space after the Y value.
                    yIndexEnd = yIndexStart+ command[yIndexStart+1:].index(" ") # Find the end index of the Y value (first space after the Y).
                    Y_value = command[yIndexStart+1:yIndexEnd+1] # Extract the Y value.
                else: Y_value = command[yIndexStart+1:] # If there is no space after the Y value, extract the Y value to the end of the command.



            if "E" in command: # Check if the command contains a E coordinate.
                eIndexStart = command.index("E") # Find the start index of the E value.
                if " " in command[eIndexStart:] : # Check if there is a space after the E value.
                    eIndexEnd = eIndexStart+ command[eIndexStart+1:].index(" ") # Find the end index of the E value (first space after the E).
                    E_value = command[eIndexStart+1:eIndexEnd+1] # Extract the Y value.
                else: E_value = command[eIndexStart+1:] # If there is no space after the E value, extract the E value to the end of the command.



            if infill_section: # only add the infill section commands to the layer  list.
                Command_enrty = [command_number, float(X_value), float(Y_value), float(Z_value), float(E_value)]
                movement_commands_list[layer_number].append(Command_enrty)
    
    gcode_file.close()
    return movement_commands_list

#***********************************************************************************************************
#The following function extensd the  moves_infill_section fucntion by considering  multple objects on the built plate, focusing on the infill section of each of them.
# The objects within a layers are split by ";MESH:part_name.stl(number)
    # number may start from none to the count of parts -1  
    # eg: MESH:part.stl, MESH:part.stl(1), MESH:part.stl(2)
    # within each object, the pattern is similar; fill section and walls section
def moves_infill_sec_multi_objects_on_bed(gcode_file_name):
    gcode_file = open(gcode_file_name)
    commands = gcode_file.readlines()


   
    # Define the list of layers that each has a list of movement commands.
    movement_commands_list =[[]] 
    command_number = layer_number=0;
    #Initialize the movement command parameters (X,Y,Z,E). 
    x_value = Y_value = Z_value=E_value=0
    infill_zone=False


    for command in commands:
        command=command.strip('\n\r')            
        command_number+=1
        
        if ";LAYER_COUNT:" in command:   # Check the line that contains this comment to get the number of layers. 
            total_layers = int(command[13:])

        if ";TIME_ELAPSED:" in command and layer_number +1 == total_layers: # Stop when we reach the end of the G-code file.
            break
            
        if ";LAYER:" in command: # Check if the line in the G-code file has this comment which indicated to a new layer starting.
            layer_number = int(command[7:]) # get the layer number.
            if layer_number >0:
                movement_commands_list.append([]) #  Add an initialized list that represent the layer. 
        
        if ";TYPE:SKIN" in command:  #Indicate the Iinfill section in case the G-code file has in infill section.
            infill_section=True

        if ";MESH:" in command: # indicate the start of a new object section in the G-code file. 
            infill_section = False
        if ";MESH:" in command and ".stl" in command:
            if ".stl(" in command:
                object_index = int(command[command.index("(")+1:command.index(")")]) # get the object index >1
                while len(movement_commands_list[layer_number])< object_index+1: # Initialize a movement commands lsit for each object.
                    # Each object has a list of layers, and each layer has list of movement commands. 
                    movement_commands_list[layer_number].append([])
            else: # The first object has index 0. 
                object_index =0
                # The first object with index 0 has also a list of layers, and each layer has list of movement commands. 
                if len(movement_commands_list[layer_number])==0:
                    movement_commands_list[layer_number].append([])
            
        if ";TYPE:WALL-INNER" in command:# Indicate the inner walls of the object in case the G-code file has inner walls section. 
            infill_section=False
            
        if ";TYPE:WALL-OUTER" in command: # Indicate the outer walls of the object in case the G-code file has outer walls section. 
            infill_section=False
        
        if ("G1" in command or "G0" in command): # Check if the command is a movement command.
            if "Z" in command: # Check if the command contains a Z coordinate.
                zIndexStart = command.index("Z")  # Find the start index of the Z value.
                if " " in command[zIndexStart:] : # Check if there is a space after the Z value.
                    zIndexEnd = zIndexStart+ command[zIndexStart+1:].index(" ") # Find the end index of the Z value (first space after the Z).
                    Z_value = command[zIndexStart+1:zIndexEnd+1] # Extract the Z value.
                else: Z_value = command[zIndexStart+1:] # If there is no space after the Z value, extract the Z value to the end of the command.



            if "X" in command: # Check if the command contains a X coordinate.
                xIndexStart = command.index("X")  # Find the start index of the X value.
                if " " in command[xIndexStart:] : # Check if there is a space after the X value.
                    xIndexEnd = xIndexStart+ command[xIndexStart+1:].index(" ") # Find the end index of the X value (first space after the X).
                    X_value = command[xIndexStart+1:xIndexEnd+1] # Extract the X value.
                else: X_value = command[xIndexStart+1:] # If there is no space after the X value, extract the X value to the end of the command.


            if "Y" in command: # Check if the command contains a Y coordinate.
                yIndexStart = command.index("Y") # Find the start index of the Y value.
                if " " in command[yIndexStart:] : # Check if there is a space after the Y value.
                    yIndexEnd = yIndexStart+ command[yIndexStart+1:].index(" ") # Find the end index of the Y value (first space after the Y).
                    Y_value = command[yIndexStart+1:yIndexEnd+1] # Extract the Y value.
                else: Y_value = command[yIndexStart+1:] # If there is no space after the Y value, extract the Y value to the end of the command.



            if "E" in command: # Check if the command contains a E coordinate.
                eIndexStart = command.index("E") # Find the start index of the E value.
                if " " in command[eIndexStart:] : # Check if there is a space after the E value.
                    eIndexEnd = eIndexStart+ command[eIndexStart+1:].index(" ") # Find the end index of the E value (first space after the E).
                    E_value = command[eIndexStart+1:eIndexEnd+1] # Extract the Y value.
                else: E_value = command[eIndexStart+1:] # If there is no space after the E value, extract the E value to the end of the command.

            if infill_zone: # only add the infill section commands to the layer  list.
                # Here, we need to correlate thelayer number with the object index to recognize the movement commands that belong to each object. 
                Command_enrty = [command_number, float(X_value), float(Y_value), float(Z_value), float(E_value)]
                movement_commands_list[layer_number][object_index].append(Command_enrty)

    gcode_file.close()
    return movement_commands_list
