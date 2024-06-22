
import math

# Function to calculate the distance between two points (x1, y1) and (x2, y2). 
def dist(x1,y1,x2,y2):
    return pow((pow((x2-x1),2)+ pow((y2-y1),2)),0.5)

#split function splits a line segment defined by two points (x1,y1) and (x2,y2) into three parts, where the length of the middle part is specified.
#Parameters:
    #    x1, y1: Coordinates of the starting point A.
    #    x2, y2: Coordinates of the ending point B.
    #    middle_seg_length: Length of the middle segment of the line segment.
    #This function retrns a list containing two points  (xa,ya) and (xb,yb), which are the boundaries of the middle segment.

def split(x1,y1,x2,y2, middle_seg_length):
    #Calculate the total length of the line segment.
    len_of_line =  pow((pow((x2-x1),2)+ pow((y2-y1),2)),0.5)
    #Determine the sine and cosine of the angle between the line segment and the horizontal axis.
    sin_theta = (y2-y1) / len_of_line
    cos_theta = (x2-x1) / len_of_line
    #Calculate the length of each of the two exterior portions of the line segment.
    exterior_portions_length = (len_of_line - middle_seg_length) /2
    #Compute the coordinates (xa,ya) and (xb,yb) using trigonometric relationships.
    xa = x1 + exterior_portions_length * cos_theta
    xb = x2 - exterior_portions_length * cos_theta
    ya = y1 + exterior_portions_length * sin_theta
    yb = y2 - exterior_portions_length * sin_theta

    return([(xa,ya),(xb,yb)])


#The following function  generates new G-code commands to replace a single attacked command in a G-code file and returns a list of modified G-code commands..

# Parameters:
#    prev_command: The previous movement command.
#    current_command: The current movement command to be modified.
#    cavity_size: The length of the middle part to be cut out and replaced.
def resultant_commands_of_one_attacked_command(prev_command, current_command, cavity_size):
    # Calculate the full length of the line segment defined by prev_command and current_command.
    full_distance = dist(prev_command[1], prev_command[2], current_command[1], current_command[2])
    # Split the line segment into three parts using the split function, with cavity_size as the length of the middle part.
    middle_segment_points = split(prev_command[1], prev_command[2], current_command[1], current_command[2], cavity_size)
    # Calculate the change in the extrusion amount deltaE.
    deltaE = current_command[4] - prev_command[4]
    # Determine the extrusion amount for the new portion.
    new_deltaE = deltaE * dist(prev_command[1], prev_command[2], middle_segment_points[0][0], middle_segment_points[0][1]) / full_distance
    new_Evalue = round(prev_command[4] + new_deltaE, 5)
    
    # Construct a list of new G-code commands to replace the attacked command.
    resultant_commands = []
    resultant_commands.append(current_command[0])
    resultant_commands.append("G1 X" + str(round(middle_segment_points[0][0], 3)) + " Y" + str(round(middle_segment_points[0][1], 3)) + " E" + str(new_Evalue))
    resultant_commands.append("G1 E" + str(new_Evalue - 4.5))  #  Change the retraction value to ensure a clear cavity.
    resultant_commands.append("G1 X" + str(round(middle_segment_points[1][0], 3)) + " Y" + str(round(middle_segment_points[1][1], 3)))  # Movement without extrusion.
    resultant_commands.append("G1 E" + str(new_Evalue))
    resultant_commands.append("G92 E" + str(round(current_command[4] - new_deltaE, 3)))  # Update the filament length.
    resultant_commands.append("G1 X" + str(round(current_command[1], 3)) + " Y" + str(round(current_command[2], 3)) + " E" + str(current_command[4]))

    return resultant_commands