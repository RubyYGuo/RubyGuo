#!/bin/bash

# Activate conda and run YOLO
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate capuchinyolov5
#python /home/capuchin/yolo5model/yolov5/capuchin_recorder_display_fpsfix_headless.py &
#PID1=$!

# Use system Python for GPIO reward script
# sudo /usr/bin/python3 /home/capuchin/Desktop/stgt_task.py

# Kill YOLO process if still running
#kill $PID1




# USE THIS: 
#!/bin/bash

# Load conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Activate environment 
conda activate capuchinyolov5

# Run integrated YOLO + STGT controller
python3 -i /home/capuchin/yolo5model/yolov5/stgt_scripts/stgt_master_script.py --csi 0 1
