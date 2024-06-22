# Filament Attack Automation Scripts

This folder contains a collection of Python scripts designed to automate the implementation of various filament attacks on G-code files. Manually performing these attacks can be complex and error-prone, hence these scripts provide a streamlined, automated approach.

## Scripts and Their Functions

### Filament Cavity Attack
- **filament_cavity_attack.py** demonstrates a cavity attack by splitting the target movement distance into three segments and muting the filament extrusion along the middle segment.
- This script can be modified to generate various variants of cavity attacks with different cavity sizes, filament retractions, and number of segments with minimal coding effort.
- The **movement_distance_split.py** class is utilized by the `filament_cavity_attack.py` script to split the target movement distance and generate corresponding commands that are then added to the G-code file.

### Filament Speed Attack
- **filament_speed_attack.py** is designed to adjust the filament density at target commands and compensate accordingly in other areas of the G-code file. This script showcases one method of implementing such attacks.
- Other variants could include increasing the density at specific commands while reducing it elsewhere or varying the methods of compensating for density changes.
- This script can be modified to implement different variants of this attack with minimal coding effort.
- This script includes print statements that provide a numerical illustration of how filament density modifications are applied and compensated, offering clarity and insight into the internal workings and impacts of the attack.

### Filament State Attack
- **filament_state_attack.py** mutes the extrusion of filament at targeted commands. This script currently implements one variant of this attack by making the E parameter of the G92 command equal to the E parameter of the following movement command.
- It can be easily adapted to support other variants, such as ensuring two consecutive movement commands have identical E parameter values or other manipulations of the E parameter.

## Note
All these attacks utilize the **extract_movement_commands.py** class which processes G-code files to represent them as a list of layers. Each layer contains a list of movement commands (G0, G1). Each movement command is represented by a vector of features: [command number, X parameter value, Y parameter value, Z parameter value, E parameter value].

