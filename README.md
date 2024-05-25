# ML-based-G-Code-Attacks-Detection
This repository is associated with the manuscript:

**Proactive Detection of G-Code Attacks in 3D Printing using Machine Learning: A Comprehensive Analysis**

## Description

This project aims to detect various types of G-Code attacks in 3D printing, such as:
- **Filament Attacks**: Issues like cavity creation and filament density variations.
- **Thermodynamic Attacks**: Manipulations involving nozzle temperature, bed temperature, and fan speed.
- **Z-Profile Attacks**: Alterations to the bed level of the print.

The repository includes:
- `Datasets`: Contains raw STL files and G-code files, datasets of benign and malicious layers and commands.
- `Code`:  scripts for extracting features, labeling layers and commands, filtering commands, and training/testing models including LSTM, Bi-GRU, and Bi-LSTM. It also includes code for MLP and RF algorithms for command classification.

## Datasets
You can access all datasets via the following link : https://drive.google.com/file/d/1V8wD-Ykb8vqT54I8ZtI_hcf70CR3yjXv/view?usp=sharing

### Install the Environment
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

