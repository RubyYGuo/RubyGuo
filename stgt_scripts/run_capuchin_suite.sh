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

# ==============================
# Activate Wifi broadcasting
# ==============================

echo "Activating hotspot for remote desktop"
sudo nmcli connection up platform-hotspot
echo "Hotspot active! You will be able to connect the laptop to 'PlatformRemote.'"
echo "--------------------------------------------"

# Load conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Activate environment
conda activate capuchinyolo26

# Run integrated YOLO + STGT controller
cd /home/capuchin/Desktop/stgt_scripts

# Wrap the testing and master scripts in a loop
while true; do
    echo "Starting Hardware IO Test..."
    # Run the hardware testing phase
    python3 stgt_io_test.py

    echo "Hardware test complete. Launching Master Script..."
    # Run task script
    python3 -i stgt_master_script.py --weights best.pt --img 256 --csi 0 1
    
    # Capture the exit code of stgt_master_script.py
    EXIT_CODE=$?

    # Check if the user selected "Return to Hardware IO Test" (Exit Code 99)
    if [ $EXIT_CODE -eq 99 ]; then
        echo ""
        echo "================================================="
        echo "Returning to Hardware Testing Phase as requested..."
        echo "================================================="
        echo ""
        continue # Restart the loop from the top
    else
        # Break the loop if finished normally or crashed
        echo "Master script finished. Exiting pipeline."
        break
    fi
done
