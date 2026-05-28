#!/bin/bash

# =========================
# System Time Check
# =========================
echo "Current System Date and Time:"
date "+%Y-%m-%d %H:%M:%S"
echo ""
read -p "Is this date and time correct? (y/n): " time_choice

if [[ "$time_choice" == "n" || "$time_choice" == "N" ]]; then
    echo ""
    echo "Please enter the correct date and time in the following format:"
    echo "YYYY-MM-DD HH:MM (e.g., 2026-05-28 14:30)"
    read -p "New Time: " new_time
    
    echo "Applying new time (you may be prompted for your sudo password)..."
    # Automatically append :00 for the seconds
    sudo date -s "${new_time}:00"
    
    echo ""
    echo "Time updated successfully to:"
    date "+%Y-%m-%d %H:%M:%S"
    echo "----------------------------------------"
else
    echo "Time confirmed. Proceeding..."
    echo "----------------------------------------"
fi

# =========================
# Task Execution
# =========================
# Load conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Activate environment
conda activate capuchinyolo26

# Run integrated YOLO + STGT controller
cd /home/capuchin/Desktop/stgt_scripts

# 1. Run the hardware testing phase
python3 stgt_io_test.py

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
