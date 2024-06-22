
# This class implement the cavity attacks
# It takes as input the benign gcode file, the number of layers to be attacked, and the  Location of attack.
# This class returns thegcode file impacted by the cavity attcacks. 
# The name of the generated manipulated G-code file includes the attacked layers number. 
import copy
import extract_movement_commands
import movement_distance_split

# Function to calculate the distance between two points (x1, y1) and (x2, y2). 
def dist(x1,y1,x2,y2):
    return pow((pow((x2-x1),2)+ pow((y2-y1),2)),0.5)
    
    
print ("The attack will create cavity by splitting the targeted movement  distnce into segments (e.g., 3 segments), and mute extrusion the filament at specific segments  to create clean cavity \n ")
gcode_filename = input("Enter the entire path and namne of the targeted gcode file: ")

Layers_number=0
with open(gcode_filename, 'r') as file:
        code = file.readlines()
        code = [command for command in code]
        for command_number, command in enumerate(code):  
            # get the number of layers of the targeted G-code file.
            if ";LAYER_COUNT:" in command:        
                Layers_number= int(command.split(":")[-1])

#Extract the list of movement commands (G0, G1) of each layer in the gcode file.
#The movement_commands_list  is a list of layers, and each layer is a lsit of mvoement commands.            
movement_commands_list = extract_movement_commands.list_movement_commands(gcode_filename)



while(True):
    attacked_layer_number = input("Determine the layers that you want to create cavity within them by entering their numbers (non-decreasing order) separated by space Or in range format; eg  3 5 7 OR 3-10: max is "+str(Layers_number)+"\n") # Max here determines the total number of layers in the G-code file. 
   
    if(attacked_layer_number !=""): # Ensure non empty input. 
        break;


attacked_layer_numbers=[]
# If the user enter a range such as 3-10, extract 3, and 10 as boundaries that represent the min and max layer number. 
if '-' in attacked_layer_number:
    attacked_layers_string=attacked_layer_number # to be included in the filename of the generated G-code file. 
    attacked_layers_string+='_'  # to be included in the filename of the generated G-code file.
    attacked_layers_boundaries= attacked_layer_number.split('-')
    for layer_number in range(int(attacked_layers_boundaries[0]),int(attacked_layers_boundaries[1])+1):
        attacked_layer_numbers.append(layer_number)

else: # the numbers of layers are separated by space.  
    attacked_layers_string=''
    attacked_layer_numbers = attacked_layer_number.split(" ")
    for layer_number in range(0,len(attacked_layer_numbers)):
        attacked_layers_string +=attacked_layer_numbers[layer_number]+'_'   # to be included in the filename of the generated G-code file.
        attacked_layer_numbers[layer_number] = int(attacked_layer_numbers[layer_number])
print("The cavity will be created within the layers " + str(attacked_layer_numbers))
#Get the G-code file commands before anjecting the cavity attacks.
gcode_file_before_attack = open(gcode_filename)
gcode_file_commands_before_attack = gcode_file_before_attack.readlines()


#Get the layer number to inject cavity attack within its commans. 
#Selecting the commands to be modified; selection is done on the basis of 1st attacked layer; for remaining layers, same commands are chosen 

attack_layer= attacked_layer_numbers[0];
# Enter the number of movement commands within the layer to be attacked by the cavity attack. 
number_of_attacked_commands = int(input("Enter the number of movement commands within the layer to be attacked by the cavity attack: "))
middle_command_number = int(len(movement_commands_list[attack_layer]) / 2)
# determine the start command to be attacked by the cavity attack. 
attack_starting_command= middle_command_number - int(number_of_attacked_commands)  
command_numbers_to_attack=[]

#Calcualte the minimum and maximum movement distance to each of the targeted  commands to limit the cavity size between these boundaries. 
distances = []
for i in range(number_of_attacked_commands):
    distance = dist(
        movement_commands_list[attack_layer][attack_starting_command + i - 1][1],
        movement_commands_list[attack_layer][attack_starting_command + i - 1][2],
        movement_commands_list[attack_layer][attack_starting_command + i][1],
        movement_commands_list[attack_layer][attack_starting_command + i][2]
    )
    distances.append(distance)

min_distance = min(distances)
max_distance = max(distances)

for i in range(0,number_of_attacked_commands):
    command_numbers_to_attack.append(attack_starting_command+i)
#The cavity size must be less than the maximum distance among the target movement distances and greater than the minimum of them.  
cavity_size= float(input("Enter the portion of infill line in mm to be removed. it should be less than " +str(round(max_distance,3)) +" and greater than "+ str(round(min_distance,3)) +" : "))

#Calcualte the movement distance to the starting command to be attacked using the euclidean distance equation. 
movement_distance_to_starting_command = dist(movement_commands_list[attack_layer][attack_starting_command-1][1], movement_commands_list[attack_layer][attack_starting_command-1][2], movement_commands_list[attack_layer][attack_starting_command][1],movement_commands_list[attack_layer][attack_starting_command][2])


# The cavity size that is determined by the length of the middle segments of the targeted movement ddistance. 
# The cavity size must b less than the  movement distance to the targeted command. 
if (cavity_size > movement_distance_to_starting_command):
        print("cavity width is greater than the infill line length; Not possible !!")
        exit()

manipulated_command_numbers =[]
manipulated_commands = []
#Split the target commands (e.g., 3 segments) and create cavity by calling the function "resultant_commands_of_one_attacked_command" of the class "movement_distance_split"
for i in attacked_layer_numbers:    
    for j in command_numbers_to_attack:
            manipulated_command_numbers.append(movement_commands_list[i-1][j][0])
            manipulated_commands.append(movement_distance_split.resultant_commands_of_one_attacked_command(movement_commands_list[i-1][j-1],movement_commands_list[i-1][j],cavity_size))
print( "Manipulated line numbers are: " + str(manipulated_command_numbers))
print("\n\n manipulated_commands are \n")
print(manipulated_commands)
   

# Generate the G-code file after injecting the cavity attack over multtiple layers and commands. 
#Determine the format of the manipulated G-code file name to include the numbers of the targeted layers, and commands, and the cavity size. 
modified_fileName = "filament_cavity__"+gcode_filename[gcode_filename.rfind("\\")+1:gcode_filename.rfind(".")]  +"__Cavity_Attack_Layers_"+attacked_layers_string+"Lines_"+str(number_of_attacked_commands)+"_length_"+str(cavity_size)+"mm.gcode"

gcode_file_after_attack= open(modified_fileName,"w")
command_count=0
for command in gcode_file_commands_before_attack:
    command_count+=1
    # Replace the targeted movement commands with the commands generated as a result of applying the cavity attack.
    if command_count in manipulated_command_numbers:
        for i in range (0, len(manipulated_commands)):
            if command_count == manipulated_commands[i][0]:
                for j in range (1,len(manipulated_commands[i])):
                    gcode_file_after_attack.write(str(manipulated_commands[i][j]))
                    gcode_file_after_attack.write("\n")

    else:
        gcode_file_after_attack.write(command)

gcode_file_before_attack.close()
gcode_file_after_attack.close()
print("\n\nAttack Gcode file saved as \"" + modified_fileName+"\"")
 
