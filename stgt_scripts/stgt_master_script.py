#!/usr/bin/env python3

import os
import glob
import torch
import cv2
import argparse
import time
import csv
import sys
import logging
from pathlib import Path
from datetime import datetime
from collections import deque
import subprocess
import signal
from ultralytics import YOLO
import RPi.GPIO as GPIO

# =========================
# Hardware IO Setup
# =========================
FOODCUP_BEAM_PIN = 21

# Set mode (this is safe to call alongside the subprocess)
GPIO.setmode(GPIO.BCM)
GPIO.setup(FOODCUP_BEAM_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# =========================
# Setup & Initialization
# =========================
execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
T0 = time.time()  # Master Baseline Time

stgt_data_dir = Path("/home/capuchin/stgt_data/task_data")
stgt_data_dir.mkdir(parents=True, exist_ok=True)

# Unified Logging (Terminal + File)
log_file_path = stgt_data_dir / f"system_log_{execution_id}.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Master - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Video tracking CSV
record_dir = Path("/home/capuchin/stgt_data/video_recordings")
record_dir.mkdir(exist_ok=True)
video_csv_path = record_dir / f"sessions_{execution_id}.csv"

# Main Data Log CSV
data_csv_path = stgt_data_dir / f"data_{execution_id}.csv"

# Initialize Header & Event Time 0
with open(data_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Event Time", "Event Name", "Item Name", "Value"])
    writer.writerow(["0.000", "Condition Event", "Execution_Start", execution_id])
    writer.writerow(["0.000", "Timer Event", "Execution_timer", 0.0])
    writer.writerow(["0.000", "Condition Event", "Execution timer activated", ""])
    writer.writerow(["0.000", "Variable Event", "Face_Detection", 0])
    writer.writerow(["0.000", "Variable Event", "Station_Active", 0])
    writer.writerow(["0.000", "Variable Event", "Subject_Present_Flag", 0])
    writer.writerow(["0.000", "Variable Event", "YOLO_Detection_Ratio", "0.00"])

def log_event(ev_name, item_name, value=""):
    sec = time.time() - T0
    with open(data_csv_path, "a", newline="") as f:
        csv.writer(f).writerow([f"{sec:.3f}", ev_name, item_name, value])

# =========================
# Hardware Diagnostics
# =========================
def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read().strip()) / 1000.0
        return round(temp, 1)
    except Exception as e:
        logger.error(f"Failed to read CPU temperature: {e}")
        log_event("Error Event", "CPU_Temperature", "Failed to read")
        return "N/A"

def estimate_fps(time_deque):
    if len(time_deque) < 2:
        return 30.0
    elapsed = time_deque[-1] - time_deque[0]
    return len(time_deque) / elapsed if elapsed > 0 else 30.0

# =========================
# Schedule config
# =========================
def configure_schedule():
    logger.info("========== STGT SCHEDULE CONFIG ==========")

    while True:
        print("\nSelect Operating Mode:")
        print("1) Run ST/GT Task")
        print("2) Run Habituation")
        print("3) Exit")
        choice = input("Enter choice (1-3): ").strip()

        if choice == '1':
            phase = "task"
            break
        elif choice == '2':
            phase = "habituation"
            break
        elif choice == '3':
            logger.info("System closed by user at configuration menu.")
            sys.exit(0)
        else:
            print("Invalid input. Please enter 1, 2, or 3.")

    # Default parameters shared or specific
    max_trial = 12
    buffer_dur = 0
    hab_base = 7
    hab_jitter = 2
    lever_dur = 4
    iti_list = []

    def generate_iti_list(start, end, step):
        vals = []
        val = start
        while val <= end + 1e-9:
            vals.append(round(val, 3))
            val += step
        return vals

    if phase == "task":
        iti_min, iti_max, iti_step = 12.0, 24.0, 3.0
        iti_list = generate_iti_list(iti_min, iti_max, iti_step)

        logger.info(f"Current defaults: phase={phase}, max_trial={max_trial}, lever_dur={lever_dur}, iti_list={iti_list}, buffer={buffer_dur}")
        modify = input("\nModify these default settings? (y/n): ").strip().lower()

        if modify == "y":
            max_trial = int(input(f"Enter max_trial [{max_trial}]: ") or max_trial)
            lever_dur = float(input(f"Enter lever_dur [{lever_dur}]: ") or lever_dur)
            iti_input = input(f"Enter ITI format as 'min, max, s' [{iti_min}, {iti_max}, {iti_step}]: ")
            if iti_input.strip():
                try:
                    parts = [float(x.strip()) for x in iti_input.split(',')]
                    if len(parts) == 3:
                        iti_min, iti_max, iti_step = parts
                        iti_list = generate_iti_list(iti_min, iti_max, iti_step)
                    else:
                        logger.error("Invalid ITI format. Using defaults.")
                except Exception as e:
                    logger.error(f"Error parsing ITI input: {e}. Using defaults.")
            buffer_dur = float(input(f"Enter buffer_dur [{buffer_dur}]: ") or buffer_dur)

    elif phase == "habituation":
        logger.info(f"Current defaults: phase={phase}, max_trial={max_trial}, trial_duration={hab_base}s ±{hab_jitter}s, buffer={buffer_dur}")
        modify = input("\nModify these default settings? (y/n): ").strip().lower()

        if modify == "y":
            max_trial = int(input(f"Enter max_trial [{max_trial}]: ") or max_trial)
            hab_base = float(input(f"Enter base trial duration (s) [{hab_base}]: ") or hab_base)
            hab_jitter = float(input(f"Enter trial duration jitter (s) [{hab_jitter}]: ") or hab_jitter)
            buffer_dur = float(input(f"Enter buffer_dur [{buffer_dur}]: ") or buffer_dur)

    logger.info(f"Final schedule: phase={phase}, max_trial={max_trial}, buffer={buffer_dur}")
    return phase, max_trial, lever_dur, iti_list, buffer_dur, hab_base, hab_jitter

# =========================
# Main run
# =========================
def run(weights='best.pt', img_size=416, conf_thres=0.75, csi_sources=[]):
    phase, max_trial, lever_dur, iti_list, buffer_dur, hab_base, hab_jitter = configure_schedule()

    # Log task parameters at time 0
    log_event("Variable Event", "YOLO_Weights", weights)
    log_event("Variable Event", "YOLO_Img_Size", img_size)
    log_event("Variable Event", "YOLO_Conf_Thres", conf_thres)
    log_event("Variable Event", "Task_Phase", phase)
    log_event("Variable Event", "Task_Max_Trial", max_trial)
    log_event("Variable Event", "Task_Buffer_Dur", buffer_dur)
    if phase == "task":
        log_event("Variable Event", "Task_Lever_Dur", lever_dur)
        log_event("Variable Event", "Task_ITI_List", str(iti_list))
    elif phase == "habituation":
        log_event("Variable Event", "Habituation_Base_Dur", hab_base)
        log_event("Variable Event", "Habituation_Jitter", hab_jitter)

    # PROCESS A: USB Recording State
    recording = False
    out = None
    filename = ""
    usb_last_seen = 0.0

    # PROCESS B: Session & Timeout State
    stgt_process = None
    stgt_started = False
    session_number = 0
    session_subject_present = False
    isb_active = False
    isb_until = 0.0
    presence_timeout_start = 0.0  
    session_absence_start = 0.0  
    last_temp_log_time = time.time()

    # Establish conditional absence threshold
    absence_limit = 90.0

    # CSI state
    csi_outs = [None] * len(csi_sources)
    csi_start_times = [None] * len(csi_sources)
    csi_filenames = [None] * len(csi_sources)
    detection_window = deque(maxlen=10)

    logger.info("Loading YOLO model...")
    model = YOLO(weights)

    # Open USB camera dynamically
    usb_cams = glob.glob("/dev/v4l/by-id/usb-*-video-index0")
    if not usb_cams:
        logger.error("Failed to find any USB camera connected via /dev/v4l/by-id/")
        log_event("Error Event", "USB_Camera", "Failed to find device path")
        return

    device = usb_cams[0]
    real_device = os.path.realpath(device)

    logger.info(f"Using USB camera: {real_device} (from {device})")
    cap = cv2.VideoCapture(real_device, cv2.CAP_V4L2)
    if not cap.isOpened():
        logger.error("Failed to open USB camera")
        log_event("Error Event", "USB_Camera", "Failed to open")
        return

    # STRUCTURAL MODIFICATION: Limit hardware acquisition framerate
    cap.set(cv2.CAP_PROP_FPS, 15)

    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    time_deque = deque(maxlen=30)

    if not video_csv_path.exists():
        with open(video_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "execution_id", "session_number", "camera", "status", "video_filename"])

    logger.info("Starting detection loop...")
    log_event("System Event", "Detection_Loop", "Started")

    # STRUCTURAL MODIFICATION: Frame decimation variables
    frame_counter = 0
    instant_presence = False
    ratio = 0.0
    disable_task_spawning = False  # Allows camera persistence on abort

    try:
        while True:
            try:
                current_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame_counter += 1
                time_deque.append(current_time)
                estimated_fps = estimate_fps(time_deque)

                # ==========================================
                # 10-MINUTE HARDWARE DIAGNOSTICS LOG
                # ==========================================
                if current_time - last_temp_log_time >= 600.0:
                    current_temp = get_cpu_temp()
                    log_event("Diagnostic Event", "CPU_Temperature_Log", f"{current_temp} (10min interval)")
                    last_temp_log_time = current_time

                # ==========================================
                # GLOBAL YOLO DETECTION (Decimated Frame Processing)
                # ==========================================
                if frame_counter % 6 == 0:
                    results = model(frame, imgsz=img_size, conf=conf_thres, iou=0.4, verbose=False)
                    raw_detection = len(results[0].boxes) > 0
                    detection_window.append(1 if raw_detection else 0)
                    ratio = sum(detection_window) / len(detection_window) if len(detection_window) > 0 else 0
                    
                    face_detected = (ratio >= 0.50)
                    beam_broken = (GPIO.input(FOODCUP_BEAM_PIN) == GPIO.LOW)
                    instant_presence = (face_detected or beam_broken)

                # Log official session arrival
                if instant_presence and not session_subject_present:
                    session_subject_present = True
                    log_event("Variable Event", "YOLO_Detection_Ratio", round(ratio, 2))
                    log_event("Variable Event", "Subject_Present_Flag", 1)
                    log_event("Variable Event", "Face_Detection", 1 if face_detected else 0)
                    log_event("Variable Event", "Station_Active", 1)


                # ==========================================
                # PROCESS A: INDEPENDENT USB RECORDING
                # ==========================================
                if instant_presence:
                    usb_last_seen = current_time
                    if not recording:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{timestamp}_FRONT_capuchin.mp4"
                        out_path = record_dir / filename
                        try:
                            out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), estimated_fps, (FRAME_WIDTH, FRAME_HEIGHT))
                            recording = True
                            log_event("Output Event", "USB_Camera_Recording_On", filename)
                            logger.info(f"Started USB recording: {filename}")

                            with open(video_csv_path, "a", newline="") as f:
                                csv.writer(f).writerow([datetime.now().isoformat(), execution_id, session_number, "USB", "started", filename])

                        except Exception as e:
                            logger.error(f"Failed to start USB recording: {e}")
                            log_event("Error Event", "USB_Camera", f"Failed to write video: {e}")

                if recording:
                    if out: out.write(frame)

                    # Strict 10-second absence stops the video immediately
                    if not instant_presence and (current_time - usb_last_seen > 10.0):
                        logger.info(f"Stopping USB recording: {filename}")
                        if out: out.release()
                        recording = False
                        out = None
                        log_event("Output Event", "USB_Camera_Recording_Off", "")

                        with open(video_csv_path, "a", newline="") as f:
                            csv.writer(f).writerow([datetime.now().isoformat(), execution_id, session_number, "USB", "stopped", filename])
                        filename = ""


                # ==========================================
                # PROCESS B: SESSION & CSI TRIGGER LOGIC
                # ==========================================

                # EARLY TERMINATION CHECK 
                if stgt_started:
                    if not instant_presence:
                        if session_absence_start == 0.0:
                            session_absence_start = current_time
                        elif current_time - session_absence_start >= absence_limit:
                            logger.info(f"Subject absent for {absence_limit}s during active session. Terminating early.")
                            log_event("Condition Event", "Session_Terminated_Early", f"{absence_limit}s_Absence")
                            if stgt_process:
                                stgt_process.terminate()
                                stgt_process.wait()
                            session_absence_start = 0.0
                    else:
                        session_absence_start = 0.0

                # SESSION START (Modified to block execution if user aborted)
                if instant_presence and not stgt_started and not isb_active and not disable_task_spawning:
                    session_number += 1
                    current_temp = get_cpu_temp()

                    log_event("Diagnostic Event", "CPU_Temperature_Log", f"{current_temp} (Session Start)")
                    last_temp_log_time = current_time 

                    log_event("Condition Event", "STGT_Subprocess_Start", session_number)
                    log_event("Timer Event", "Session_Timer_Start", 0.0)

                    cmd = [
                        "/usr/bin/python3",
                        "/home/capuchin/Desktop/stgt_scripts/stgt_task.py",
                        "--execution_id", execution_id,
                        "--session_number", str(session_number),
                        "--phase", phase,
                        "--max_trial", str(max_trial),
                        "--lever_dur", str(lever_dur),
                        "--buffer_dur", str(buffer_dur),
                        "--hab_base", str(hab_base),
                        "--hab_jitter", str(hab_jitter),
                        "--data_csv_path", str(data_csv_path),
                        "--t0", str(T0)
                    ]
                    if iti_list:
                        cmd.append("--iti_list")
                        cmd.extend([str(i) for i in iti_list])

                    # Start Task Subprocess
                    stgt_process = subprocess.Popen(cmd)
                    stgt_started = True
                    session_start_time = current_time

                    # Start CSI cameras
                    for i, cam_index in enumerate(csi_sources):
                        timestamp_csi = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename_csi = f"{timestamp_csi}_csi_cam{i}.mp4"
                        out_path_csi = record_dir / filename_csi
                        try:
                            proc = subprocess.Popen([
                                "rpicam-vid", "-t", "0", "-n", "--inline",
                                "--width", "640", "--height", "480", "--framerate", "15",
                                "--camera", str(cam_index), "-o", str(out_path_csi)
                            ])
                            csi_outs[i] = proc
                            csi_filenames[i] = filename_csi
                            log_event("Output Event", f"CSI_Camera_{i}_Recording_On", filename_csi)
                            logger.info(f"Started CSI camera {cam_index}: {filename_csi}")

                            with open(video_csv_path, "a", newline="") as f:
                                csv.writer(f).writerow([datetime.now().isoformat(), execution_id, session_number, f"CSI_{cam_index}", "started", filename_csi])

                        except Exception as e:
                            logger.error(f"Failed to start CSI camera {cam_index}: {e}")
                            log_event("Error Event", f"CSI_Camera_{cam_index}", f"Failed to start: {e}")
                            csi_outs[i] = None


                # ==========================================
                # SESSION COMPLETION & ISB INITIATION
                # ==========================================
                if stgt_started and stgt_process and stgt_process.poll() is not None:
                    ret_code = stgt_process.poll()
                    session_duration = round(current_time - session_start_time, 3)
                    log_event("Timer Event", "Session_Timer_Elapsed", session_duration)

                    current_temp = get_cpu_temp()
                    log_event("Diagnostic Event", "CPU_Temperature_Log", f"{current_temp} (Session End)")
                    last_temp_log_time = current_time 

                    logger.info("STGT finished. Stopping CSI cameras.")
                    for i, proc in enumerate(csi_outs):
                        if proc:
                            proc.send_signal(signal.SIGINT)
                            proc.wait()
                            log_event("Output Event", f"CSI_Camera_{i}_Recording_Off", "")

                            with open(video_csv_path, "a", newline="") as f:
                                csv.writer(f).writerow([datetime.now().isoformat(), execution_id, session_number, f"CSI_{csi_sources[i]}", "stopped", str(csi_filenames[i])])

                    stgt_started = False
                    csi_outs = [None] * len(csi_sources)
                    session_absence_start = 0.0

                    if ret_code == 0:
                        # FULL COMPLETION: Start 4-Minute ISB
                        log_event("Condition Event", "STGT_Subprocess_End", "Complete")
                        logger.info("Session complete. Starting 4-minute Inter-Session Break.")
                        isb_active = True
                        isb_until = current_time + 240.0
                        presence_timeout_start = 0 
                        log_event("Condition Event", "Phase_Transition", "Inter_Session_Break")
                        log_event("Timer Event", "ISB_Timer_Start", 240.0)
                    else:
                        # EARLY TERMINATION 
                        log_event("Condition Event", "STGT_Subprocess_End", "Early_Termination")
                        logger.info("Monkey might have left. Skipping ISB. System primed for new arrival.")
                        isb_active = False
                        session_subject_present = False
                        presence_timeout_start = 0


                # ==========================================
                # ISB 30-SECOND DEPARTURE TRACKING
                # ==========================================
                if isb_active:
                    if instant_presence:
                        if presence_timeout_start != 0:
                            logger.info("Subject returned before timeout elapsed. Cancelling timeout.")
                            log_event("Condition Event", "Presence_Timeout_Cancelled", "Subject Returned")
                            presence_timeout_start = 0 
                    else:
                        if presence_timeout_start == 0:
                            presence_timeout_start = current_time
                            log_event("Variable Event", "YOLO_Detection_Ratio", round(ratio, 2))
                            log_event("Condition Event", "Presence_Timeout_Begin", 30.0)
                        elif current_time - presence_timeout_start >= 30.0:
                            logger.info("Subject departed. 30-second presence timeout elapsed.")
                            log_event("Timer Event", "Presence_Timeout_Elapsed", 30.0)
                            log_event("Variable Event", "Subject_Present_Flag", 0)
                            log_event("Variable Event", "Face_Detection", 0)
                            log_event("Variable Event", "Station_Active", 0)

                            logger.info("Aborting Inter-Session Break due to departure.")
                            log_event("Condition Event", "ISB_Timer_Aborted", "Subject Departed")
                            log_event("Condition Event", "System_Ready_Next_Subject", "")

                            # Full state reset for next monkey
                            isb_active = False
                            isb_until = 0
                            presence_timeout_start = 0
                            session_subject_present = False

                    if isb_active and current_time >= isb_until:
                        logger.info("Inter-Session Break elapsed. Ready for next session.")
                        log_event("Timer Event", "ISB_Timer_Elapsed", 240.0)
                        isb_active = False

                time.sleep(0.005)

            except KeyboardInterrupt:
                print("\n")
                logger.info("Task interruption requested by user.")
                choice = input("Do you wish to keep the webcam face detection and recording process running? (y/n): ").strip().lower()
                if choice == 'y':
                    logger.info("Deactivating future STGT/habituation tasks. Webcam tracking will remain active.")
                    log_event("System Event", "Execution_State", "Task aborted, webcam continued")
                    disable_task_spawning = True
                    if stgt_process:
                        stgt_process.terminate()
                    continue
                else:
                    raise  # Propagates the interrupt up to shut down the system

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down entirely.")
        log_event("System Event", "Execution", "Interrupted by user")
    finally:
        if out: out.release()
        cap.release()
        # Clean up only the pin initialized by this master script
        GPIO.cleanup(FOODCUP_BEAM_PIN)
        for proc in csi_outs:
            if proc:
                proc.send_signal(signal.SIGINT)
                proc.wait()
        if stgt_process:
            stgt_process.terminate()
            stgt_process.wait()
        logger.info("All resources released. System closed.")
        log_event("System Event", "Execution", "Resources released, system closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt')
    parser.add_argument('--img', type=int, default=416)
    parser.add_argument('--conf', type=float, default=0.75)
    parser.add_argument('--csi', type=int, nargs='+', default=[])
    args = parser.parse_args()

    run(weights=args.weights, img_size=args.img, conf_thres=args.conf, csi_sources=args.csi)
