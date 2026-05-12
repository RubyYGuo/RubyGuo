#!/bin/bash

# Load conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Activate environment
conda activate capuchinyolo26

# Run integrated YOLO + STGT controller
cd /home/capuchin/Desktop/stgt_scripts

# 1. Run the hardware testing phase
python3 stgt_io_test.py

# 2. Proceed to the master script upon test completion
python3 -i stgt_master_script.py --csi 0 1
