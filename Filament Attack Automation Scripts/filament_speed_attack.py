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

# This attacks ensures low filament density in the targeted commands in the middle region, while compensating for it at the ends.
#The percentage reduction and no of commands / infill lines can be taken as input

number_of_attacked_commands = int( input("Enter the number of movement commands within the layer to be attacked by the filament speed attack: "))
#infill_with_connecting_lines_count = number_of_attacked_commands*2 -1
percentage_reduction = input("Enter the intended reduction percentage: ")
#delay_in_impact refers to the number of main infill commands after which the impact of the reduced filament density becomes visible.
#The delay is used to ensure that the attack starts early enough so that the effect of reduced filament density peaks at the desired location in the print.
delay_in_impact = int(input("Enter the number of main infill commands after which impact is visible:")) 

if delay_in_impact =='':
   delay_in_impact = 15 #default value

                        
first_attacked_layer = 2 # optional. the user can determine the first layer to be taargeted by the attack.
#")#######################################################################################################
last_attacked_layer = len(movement_commands_list) - 3  # to leaving 2 layers from top.   This value can be chosen as the user wants.

################ spliting layer infills into zones
zone_count = len(movement_commands_list[attack_starting_layer]) / number_of_attacked_commands
print ("Number  of movement commands in attacked layer is " +str(len(movement_commands_list[attack_starting_layer])) + "  No of attacked commands= "+ str(number_of_attacked_commands))
if zone_count < 5:
    print( "Density attack is not feasible over these many lines in this gcode; suggestion: reduce no. of attacked infill lines")
    exit()


#Selecting the commands to be modified; selection is done on the basis of 1st attacked layer; for remaining layers, same commands chosen
middle_command_number = int(len(movement_commands_list[attack_starting_layer]) / 2)
# determine the start command to be attacked by the filemnt speed attack.
attack_starting_command = middle_command_number - number_of_attacked_commands 

attack_starting_command = attack_starting_command - delay_in_impact # Pushing the attack earlier to ensure the impact maximizes near the target zone

#Reducing filament in the target zone lines 

manipulated_command_numbers =[]
Filament_amounts=[]
for i in range (first_attacked_layer,last_attacked_layer+1): # 
    print("---------------------------------------Layer" +str(i)+"------------------------------------")
    reduced_filament_amount=0.0
    if len(movement_commands_list[i])>0:
        for j in range(0,number_of_attacked_commands):
          if attack_starting_command+j < len(movement_commands_list[i]): 
            # Calcualte the original Delta E as E_current-E_previous which is the 4th parameter of the movement command. 
            original_deltaE = movement_commands_list[i][attack_starting_command+j][4] - movement_commands_list[i][attack_starting_command+j-1][4]
            Filament_amounts.append(original_deltaE)
            # Calcualte the new filament amount (new Delta E) after reducing the filament amount of the original Delta E. 
            
            new_deltaE = original_deltaE * (1- float(percentage_reduction) / 100)
            print("Target command previous E_Value "+str(movement_commands_list[i][attack_starting_command+j][4]))
            print("Target command  current E_Value "+str(movement_commands_list[i][attack_starting_command+j-1][4]))
            print("Target command  original DeltaE "+str(original_deltaE))
            print("Target command new DeltaE "+str(new_deltaE))
            reduced_filament_amount=float(new_deltaE-original_deltaE)
            print("reduced_filament_amount "+str(reduced_filament_amount))               
            modified_movement_commands_list[i][attack_starting_command+j][4] = round ((movement_commands_list[i][attack_starting_command+j-1][4] + new_deltaE),5)
            manipulated_command_numbers.append(movement_commands_list[i][attack_starting_command+j][0])


    ############# compensating the attack in two zones: one before and one after the center
    # 1st compensation zone starts after 4 x  'number_of_attacked_commands' instructions are over in the layer
    # 2nd zone start (zone_count -4 )*'number_of_attacked_commands' are over in the layer

    first_compen_start = 4 * number_of_attacked_commands 

    if len(movement_commands_list[i])>0: 
        for j in range(0,number_of_attacked_commands):
           if first_compen_start+j < len(movement_commands_list[i]): 
           # Calcualte the original Delta E of the compensation command  as E_current-E_previous which is the 4th parameter of the movement command. 
            original_deltaE = movement_commands_list[i][first_compen_start+j][4] - movement_commands_list[i][first_compen_start+j-1][4]
            # Calcualte the new filament amount (new Delta E) after compensation the reduced filament amount. 
            new_E=movement_commands_list[i][first_compen_start+j][4] + 0.5 *abs(float(reduced_filament_amount))
            new_deltaE = new_E- movement_commands_list[i][first_compen_start+j-1][4]
            
            print("1st compensating previous E_Value "+str( movement_commands_list[i][first_compen_start+j-1][4]))
            print("1st compensating current E_Value "+str( movement_commands_list[i][first_compen_start+j][4]))
            print("1st compensating new Current E_Value "+str(new_E))
            print("1st compensating original DeltaE "+str(original_deltaE))
            print("1st compensating new DeltaE "+str(new_deltaE))
            compensatin_amount= new_deltaE-original_deltaE
            print("1st compensating  amount "+str(compensatin_amount))
            modified_movement_commands_list[i][first_compen_start+j][4] = round ((movement_commands_list[i][first_compen_start+j-1][4] + new_deltaE),5)
            manipulated_command_numbers.append(movement_commands_list[i][first_compen_start+j][0])

    #Other compensating zone: We need to make sure that the peak impact area should be equidistant from center as with the first compensation
    # 1st compensation starts after 4 blocks (each block is of attack width), and further after 15 infill lines its effect maximizes
    # 2nd compensation should maximize at : Reverse calculate from end: (2 x 15 -1) + 4xblocks
    # implying the start of 2nd compensation should be further 2x15-1) + 1xblock commands earlier { zone is the no. of commands}
    last_compen_start = len(movement_commands_list[attack_starting_layer]) - 4*(number_of_attacked_commands) - (2*delay_in_impact-1) -number_of_attacked_commands # 



    if len(movement_commands_list[i])>0: 
        for j in range(0,number_of_attacked_commands):
          if last_compen_start+j < len(movement_commands_list[i]):
            # Calcualte the original Delta E of the compensation command  as E_current-E_previous which is the 4th parameter of the movement command. 
            original_deltaE = movement_commands_list[i][last_compen_start+j][4] - movement_commands_list[i][last_compen_start+j-1][4]
            # Calcualte the new filament amount (new Delta E) after compensation the reduced filament amount.
            new_E= movement_commands_list[i][last_compen_start+j][4] + 0.5*abs(float(reduced_filament_amount))
            new_deltaE =new_E - movement_commands_list[i][last_compen_start+j-1][4]
           
            print("2nd compensating previous E_Value "+str(movement_commands_list[i][last_compen_start+j-1][4]))
            print("2nd compensating current E_Value "+str(movement_commands_list[i][last_compen_start+j][4]))
            print("2nd compensating new Current E_Value "+str(new_E))
            print("2nd compensating original DeltaE "+str(original_deltaE))
            print("2nd compensating original DeltaE "+str(original_deltaE))
            print("2nd compensating new DeltaE "+str(new_deltaE))
            compensatin_amount= new_deltaE-original_deltaE
            print("2nd compensating  amount "+str(compensatin_amount))
            modified_movement_commands_list[i][last_compen_start+j][4] = round ((movement_commands_list[i][last_compen_start+j-1][4] + new_deltaE),5)
            manipulated_command_numbers.append(movement_commands_list[i][last_compen_start+j][0])
            
   

################################################################# Creating modified Gcode file #################################################    

# Generate the G-code file after implementing the filament speed attack over multtiple layers and commands. 
#Determine the format of the manipulated G-code file name to include the numbers of the targeted  commands, and the reduced filament amount percentage.  
modified_fileName = "filament_density__"+gcode_file_name[gcode_file_name.rfind("\\")+1:gcode_file_name.rfind(".")]+"_fd_attack_No_of_infills_"+str(number_of_attacked_commands)+"_red_percent_"+str(percentage_reduction)+".gcode"
gcode_file_commands_after_attack = open(modified_fileName,"w")
command_count=0
found = False
for command in gcode_file_commands_before_attack:
    command_count+=1
    if command_count in manipulated_command_numbers:
        for i in range (0, len(movement_commands_list)):
            for j in range (0,len(movement_commands_list[i])):
                if command_count == movement_commands_list[i][j][0]:
                    # Ensure that the targeted commands are within the movement commands
                    found = True        
                    break
            if found == True:
                found = False
                break
        if "E" in command:
            # Replace the value of E parameter of the movement commands by the new manipulated value after the attack.
            revised_command = command[:command.index("E")+1] + str(modified_movement_commands_list[i][j][4]) +"\n"
            gcode_file_commands_after_attack.write(revised_command)
            # Update the filament length
            next_command = "G92 E"+ str(movement_commands_list[i][j][4])+"\n"
            gcode_file_commands_after_attack.write(next_command)
        else:
            gcode_file_commands_after_attack.write(command)
    else:
        gcode_file_commands_after_attack.write(command)

gcode_file_before_attack.close()
gcode_file_commands_after_attack.close()
print("\n\n Attack Gcode file saved as \"" + modified_fileName+"\"")

 
