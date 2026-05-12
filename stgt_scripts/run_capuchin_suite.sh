#!/bin/bash

# Load conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Activate environment
conda activate capuchinyolo26

# Run integrated YOLO + STGT controller
cd /home/capuchin/Desktop/stgt_scripts

# 1. Run the hardware testing phase
python3 stgt_test.py

# 2. Prompt user to proceed or exit
echo ""
read -p "Testing phase complete. Proceed to run the task? (y/n): " proceed_choice

if [[ "$proceed_choice" == "y" || "$proceed_choice" == "Y" ]]; then
    echo "Proceeding to execution..."
    python3 -i stgt_master_script.py --csi 0 1
else
    echo "Execution cancelled. Exiting."
    exit 0
fi
