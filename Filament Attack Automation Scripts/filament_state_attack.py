import extract_movement_commands
import copy

#Function to calculate the distance between two points (x1, y1) and (x2, y2). 
def dist(x1,y1,x2,y2):
    return pow((pow((x2-x1),2)+ pow((y2-y1),2)),0.5)
    
 
#Optional: Skipping the first two layers to avoid and visual information. This value can be chosen as the user wants.
attack_starting_layer = 2 

gcode_file_name = input("Enter gcode filname with path: ")
movement_commands_list = extract_movement_commands.moves_infill_section(gcode_file_name)
modified_movement_commands_list = copy.deepcopy(movement_commands_list)

#Get the G-code file commands before manipulate the filament density of the target commands.
gcode_file_before_attack = open(gcode_file_name)
gcode_file_commands_before_attack = gcode_file_before_attack.readlines()

number_of_attacked_commands = int( input("Enter the number of movement commands within the layer to be attacked by the filament state  attack: "))

#delay_in_impact refers to the number of main infill commands after which the impact of the reduced filament density becomes visible.
#The delay is used to ensure that the attack starts early enough so that the effect of reduced filament density peaks at the desired location in the print.
delay_in_impact = int(input("Enter the number of main infill commands after which impact is visible:")) 
if delay_in_impact =='':
    delay_in_impact = 15

                        
first_attacked_layer = 2 # keep it 2 by default, can get user input if uncomment below line
# first_attacked_layer = input(" Please enter the layer no of first attacked layer: 
last_attacked_layer = len(movement_commands_list) - 3  # Should be (-3) here by default to leaving 2 layers from top , can get user input by uncommenting below line
#last_attacked_layer = input(" Please enter the layer no of last attacked layer out of max of "+str(len(movement_commands_list))+": 

#spliting layer infills into zones
zone_count = len(movement_commands_list[attack_starting_layer]) / number_of_attacked_commands
print( "No of movement commands in attacked layer is " +str(len(movement_commands_list[attack_starting_layer])) + "  Number of attacked commands = "+ str(number_of_attacked_commands))
if zone_count < 2:
    print( "Muting attack (or filament state attack) is not feasible as more than 50% infill is muted; suggestion: reduce the number of  of attacked infill lines")
    exit()


# Selecting the commands to be modified; selection is done on the basis of 1st attacked layer; for remaining layers, same cmds chosen
middle_command_number = int(len(movement_commands_list[attack_starting_layer]) / 2)
attack_starting_command = middle_command_number - number_of_attacked_commands # catering for the no of attack lines
attack_starting_command = attack_starting_command - delay_in_impact # Pushing the attack earlier to ensure the impact maximizes near the target zone
############################################################ Muting filament in the target zone lines ######################################################################
manipulated_command_numbers =[]

for i in range (first_attacked_layer,last_attacked_layer+1): # 
 if len(movement_commands_list[i])>0: 
    for j in range(0,number_of_attacked_commands):
      if attack_starting_command+j < len(movement_commands_list[i]): 
        # Calcualte the original Delta E of the compensation command  as E_current-E_previous which is the 4th parameter of the movement command.      
        original_deltaE = movement_commands_list[i][attack_starting_command+j][4] - movement_commands_list[i][attack_starting_command+j-1][4]
        #Mute extrusion the filament amount at the targeted command where the E value of the current command equals to the previous E value, causing new_deltaE = 0.
        modified_movement_commands_list[i][attack_starting_command+j][4] = round ((movement_commands_list[i][attack_starting_command+j-1][4]),5)
        manipulated_command_numbers.append(movement_commands_list[i][attack_starting_command+j][0])
        #Print the E parameter value of the target command and the new value after implementing the filament state attack.
        print(f"Modified command index: {movement_commands_list[i][attack_starting_command+j ][0]}, Old E: {movement_commands_list[i][attack_starting_command+j ][4]}, New E: {modified_movement_commands_list[i][attack_starting_command+j ][4]}")

################################################################# Creating modified Gcode file #################################################    
 #Generate the G-code file after implementing the filament speed attack over multtiple layers and commands. 
#Determine the format of the manipulated G-code file name to include the numbers of the targeted  commands.  
modified_fileName = "filament_state_"+gcode_file_name[gcode_file_name.rfind("\\")+1:gcode_file_name.rfind(".")]  +"_fs_attack_No_of_infills_"+str(number_of_attacked_commands)+".gcode"

gcode_file_after_attack = open(modified_fileName,"w")
command_count=0
found = False
for command in gcode_file_commands_before_attack:
    command_count+=1
    if command_count in manipulated_command_numbers and "E" in command:
        for i in range (0, len(movement_commands_list)):
            for j in range (0,len(movement_commands_list[i])):
                # Ensure that the targeted commands are within the movement commands
                if command_count == movement_commands_list[i][j][0]:
                    found = True        
                    break
            if found == True:
                found = False
                break
        # Replace the value of E parameter of the movement commands by the new manipulated value after the attack.       
        revised_command = command[:command.index("E")+1] + str(modified_movement_commands_list[i][j][4]) +"\n"
        gcode_file_after_attack.write(revised_command)
        # Update the filament length
        next_command= "G92 E"+ str(movement_commands_list[i][j][4])+"\n"
        gcode_file_after_attack.write(next_command)
    else:
        gcode_file_after_attack.write(command)

gcode_file_before_attack.close()
gcode_file_after_attack.close()

print("\n\nAttack Gcode file saved as \"" + modified_fileName+"\"")
 
