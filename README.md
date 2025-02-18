# ML-based-G-Code-Attacks-Detection
This repository is associated with our manuscript (under review)

## Description

This project aims to detect various types of G-Code attacks in 3D printing, such as:
- **Filament Attacks**: Issues like cavity creation and filament density variations.
- **Thermodynamic Attacks**: Manipulations involving nozzle temperature, bed temperature, and fan speed.
- **Z-Profile Attacks**: Alterations to the bed level of the print.

The repository includes:
- `Datasets`: Contains raw STL files and G-code files, datasets of benign and malicious layers and commands.
- `Code`:  scripts for extracting features, labeling layers and commands, filtering commands, and training/testing models including LSTM, Bi-GRU, and Bi-LSTM. It also includes code for MLP and RF algorithms for command classification. Moreover, this  repository includes filament attack automation scripts.

## Datasets
You can access all datasets via the following link: https://drive.google.com/drive/folders/1n_4VhXoVnbwdvAuZdiioV_Eq16w2bT9O?usp=sharing

### File Name Structure
**-The main strcuture (Benign File)**: Design Name _X(dimention)_Y(dimention)_Z(height)_T(#top layers)_B(#buttom layers)_W(#walls)_L(layer thickness)_Infill Pattern_DS(infill density)_D(infill direction)_Layer Number within the file.csv

**-Thermodynamic and Z_profile Attacks:** Attack Class_followed by main structure.

**-Cavity Attack:**  Attack class(filament cavity)_followed by main structure_ Cavity_Attack_Layers (#of impacted layers)_ Lines(#of impacted lines)_length(attack magnitude)_Layer Number within the file.csv.

**-Filament Speed Attack:** Attack class(filament density)_followed by main structure_fd_attack_No_of_infills (#of impacted lines)_percent(Attack magnitude)_Layer Number within the file.csv.

**-Filament Speed Attack:** Attack class(filament state)_followed by main structure_fs_attack_No_of_infills (#of impacted lines)_Layer Number within the file.csv.

Note: Some infill patterns don't have D (infill direction).

## Install the Environment
```bash
# Using Pyothn:
python -m venv myenv
source myenv/bin/activate

# Using Miniconda
1. Download the Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

2.Make the installer script executable
chmod +x Miniconda3-latest-Linux-x86_64.sh

3.Run the installer script
./Miniconda3-latest-Linux-x86_64.sh

4. Activate the Miniconda environment
source ~/.bashrc

5.Verify the installation
conda --version

6.Create and activate a new Conda environment
conda create --name myenv python=3.9.18
conda activate myenv

# Install TensorFlow and other necessary packages
pip install tensorflow[cuda]
conda install scikit-learn keras pandas

#Checking GPU Availability (Optional)
# Verify that TensorFlow can access the GPU
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

#Note:  Ensure that both TensorFlow and CUDA versions are compatible.

