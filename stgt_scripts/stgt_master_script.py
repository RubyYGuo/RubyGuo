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

GPIO.setmode(GPIO.BCM)
GPIO.setup(FOODCUP_BEAM_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# =========================
# Setup & Initialization
# =========================
execution_id = datetime.now().strftime("%y%m%d_%H%M%S")
T0 = time.time()  # Master Baseline Time

stgt_data_dir = Path("/home/capuchin/stgt_data/task_data")
stgt_data_dir.mkdir(parents=True, exist_ok=True)

# Unified Logging
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
video_csv_path = record_dir / f"videolog_{execution_id}.csv"

with open(video_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Event Time", "Camera Type", "Event", "Filename"])

# Main Data Log CSV Path (Default placeholder, updated dynamically in run())
data_csv_path = stgt_data_dir / f"data_{execution_id}.csv"

def log_event(ev_name, item_name, value=""):
    sec = time.time() - T0
    try:
        with open(data_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{sec:.3f}", ev_name, item_name, value])
    except Exception as e:
        logger.error(f"Failed to write to data CSV ({ev_name} / {item_name}): {e}")

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read().strip()) / 1000.0
        return round(temp, 1)
    except Exception as e:
        logger.error(f"Failed to read CPU temperature: {e}")
        return "N/A"

def estimate_fps(time_deque):
    if len(time_deque) < 2: return 30.0
    elapsed = time_deque[-1] - time_deque[0]
    return len(time_deque) / elapsed if elapsed > 0 else 30.0

def configure_schedule():
    logger.info("========== STGT SCHEDULE CONFIG ==========")
    while True:
        print("\nSelect Operating Mode:")
        print("1) Run ST/GT Task")
        print("2) Run Pre-training")
        print("3) Run Habituation (Camera Only)")
        print("4) Return to Hardware IO Test")
        print("5) Exit")
        choice = input("Enter choice (1-5): ").strip()

        if choice == '1':
            phase = "task"; break
        elif choice == '2':
            phase = "pretrain"; break
        elif choice == '3':
            phase = "habituation"; break
        elif choice == '4':
            logger.info("Returning to Hardware IO Test phase...")
            os._exit(99)
        elif choice == '5':
            logger.info("System closed by user at configuration menu.")
            sys.exit(0)
        else:
            print("Invalid input. Please enter 1-5.")

    max_trial = 12; buffer_dur = 0; pretrain_base = 7; pretrain_jitter = 2; lever_dur = 4; iti_list = []
    enable_beam_trigger = False  # Disabled by default

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
        logger.info(f"Current defaults: phase={phase}, max_trial={max_trial}, lever_dur={lever_dur}, iti_list={iti_list}, buffer={buffer_dur}, beam_trigger={enable_beam_trigger}")
        if input("\nModify these default settings? (y/n): ").strip().lower() == "y":
            max_trial = int(input(f"Enter max_trial [{max_trial}]: ") or max_trial)
            lever_dur = float(input(f"Enter lever_dur [{lever_dur}]: ") or lever_dur)
            iti_input = input(f"Enter ITI format as 'min, max, s' [{iti_min}, {iti_max}, {iti_step}]: ")
            if iti_input.strip():
                try:
                    parts = [float(x.strip()) for x in iti_input.split(',')]
                    if len(parts) == 3:
                        iti_list = generate_iti_list(*parts)
                except Exception as e:
                    logger.error(f"Failed to parse custom ITI input: {e}")
            buffer_dur = float(input(f"Enter buffer_dur [{buffer_dur}]: ") or buffer_dur)
            beam_input = input(f"Enable task trigger via beam break? (y/n) [n]: ").strip().lower()
            if beam_input == 'y':
                enable_beam_trigger = True

    elif phase == "pretrain":
        logger.info(f"Current defaults: phase={phase}, max_trial={max_trial}, trial_duration={pretrain_base}s ±{pretrain_jitter}s, buffer={buffer_dur}, beam_trigger={enable_beam_trigger}")
        if input("\nModify these default settings? (y/n): ").strip().lower() == "y":
            max_trial = int(input(f"Enter max_trial [{max_trial}]: ") or max_trial)
            pretrain_base = float(input(f"Enter base trial duration (s) [{pretrain_base}]: ") or pretrain_base)
            pretrain_jitter = float(input(f"Enter trial duration jitter (s) [{pretrain_jitter}]: ") or pretrain_jitter)
            buffer_dur = float(input(f"Enter buffer_dur [{buffer_dur}]: ") or buffer_dur)
            beam_input = input(f"Enable task trigger via beam break? (y/n) [n]: ").strip().lower()
            if beam_input == 'y':
                enable_beam_trigger = True

    elif phase == "habituation":
        logger.info("Habituation selected. Subprocess task spawning is disabled. FRONT camera will record autonomously upon face detection.")

    return phase, max_trial, lever_dur, iti_list, buffer_dur, pretrain_base, pretrain_jitter, enable_beam_trigger

# =========================
# Main run
# =========================
def run(weights='best.pt', img_size=416, conf_thres=0.75, csi_sources=[]):
    global data_csv_path  # Declare global so we can update it before file creation
    
    phase, max_trial, lever_dur, iti_list, buffer_dur, pretrain_base, pretrain_jitter, enable_beam_trigger = configure_schedule()

    # Dynamically update the CSV name based on the chosen phase
    data_csv_path = stgt_data_dir / f"{phase}_{execution_id}.csv"

    # Create the Main Data Log CSV with the selected phase recorded at T0
    with open(data_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Event Time", "Event Name", "Item Name", "Value"])
        writer.writerow(["0.000", "Condition Event", "Execution_Start", execution_id])
        writer.writerow(["0.000", "Condition Event", "Operating_Mode", phase])
        writer.writerow(["0.000", "Timer Event", "Execution_timer", 0.0])
        writer.writerow(["0.000", "Condition Event", "Execution timer activated", ""])
        writer.writerow(["0.000", "Variable Event", "Face_Detection", 0])
        writer.writerow(["0.000", "Variable Event", "Station_Active", 0])
        writer.writerow(["0.000", "Variable Event", "Subject_Present_Flag", 0])
        writer.writerow(["0.000", "Variable Event", "YOLO_Detection_Ratio", "0.00"])

    recording = False; out = None; filename = ""; usb_last_seen = 0.0
    stgt_process = None; stgt_started = False; session_number = 0; session_subject_present = False
    isb_active = False; isb_until = 0.0; presence_timeout_start = 0.0  
    face_absence_start = 0.0; last_temp_log_time = time.time()

    csi_outs = [None] * len(csi_sources)
    csi_filenames = [None] * len(csi_sources)
    detection_window = deque(maxlen=10)

    logger.info("Loading YOLO model...")
    model = YOLO(weights)

    usb_cams = glob.glob("/dev/v4l/by-id/usb-*-video-index0")
    if not usb_cams:
        logger.error("Failed to find any USB camera connected via /dev/v4l/by-id/")
        return

    cap = cv2.VideoCapture(os.path.realpath(usb_cams[0]), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FPS, 15)
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    time_deque = deque(maxlen=30)

    logger.info("Starting detection loop...")

    frame_counter = 0
    instant_presence = False
    face_detected = False
    beam_broken = False
    disable_task_spawning = False

    try:
        while True:
            try:
                current_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05); continue

                frame_counter += 1
                time_deque.append(current_time)
                estimated_fps = estimate_fps(time_deque)

                if current_time - last_temp_log_time >= 600.0:
                    last_temp_log_time = current_time

                # ==========================================
                # DETECTION & DYNAMIC HANDOFF LOGIC
                # ==========================================
                if frame_counter % 6 == 0:
                    results = model(frame, imgsz=img_size, conf=conf_thres, iou=0.4, verbose=False)
                    detection_window.append(1 if len(results[0].boxes) > 0 else 0)
                    ratio = sum(detection_window) / len(detection_window) if detection_window else 0
                    face_detected = (ratio >= 0.50)
                    
                    # Only read the GPIO pin if the subprocess isn't actively holding it AND the trigger is enabled
                    if not stgt_started and enable_beam_trigger:
                        beam_broken = (GPIO.input(FOODCUP_BEAM_PIN) == GPIO.LOW)
                    else:
                        beam_broken = False
                        
                    instant_presence = (face_detected or beam_broken)

                if instant_presence and not session_subject_present:
                    session_subject_present = True

                # ==========================================
                # USB FRONT CAMERA RECORDING (WITH LOGGING)
                # ==========================================
                if face_detected:
                    usb_last_seen = current_time
                    if not recording:
                        filename = f"{datetime.now().strftime('%y%m%d_%H%M%S')}_FRONT_capuchin.mp4"
                        try:
                            out = cv2.VideoWriter(str(record_dir / filename), cv2.VideoWriter_fourcc(*'mp4v'), estimated_fps, (FRAME_WIDTH, FRAME_HEIGHT))
                            if not out.isOpened():
                                raise RuntimeError("cv2.VideoWriter failed to open file stream.")
                            recording = True
                            
                            # --- FRAME DROP INITIALIZATION ---
                            last_frame_write_time = current_time
                            session_dropped_frames = 0
                            max_drop_duration = 0.0
                            
                            logger.info(f"Started USB video recording (Face detected): {filename}")
                            with open(video_csv_path, "a", newline="") as f:
                                csv.writer(f).writerow([f"{current_time - T0:.3f}", "USB_FRONT", "START", filename])
                        except Exception as e:
                            logger.error(f"Failed to start USB_FRONT video recording ({filename}): {e}")
                            log_event("Error Event", "USB_FRONT_VideoWriter", str(e))
                            recording = False
                            out = None
                            filename = ""

                if recording:
                    if out:
                        try:
                            # --- SILENT FRAME DROP TRACKING ---
                            frame_interval = current_time - last_frame_write_time
                            if frame_interval > 0.14:  # >140ms indicates at least 1 skipped frame at ~15 FPS
                                dropped_count = int(frame_interval / 0.0667) - 1
                                session_dropped_frames += dropped_count
                                if frame_interval > max_drop_duration:
                                    max_drop_duration = frame_interval
                                
                            out.write(frame)
                            last_frame_write_time = current_time
                        except Exception as e:
                            logger.error(f"Error writing frame to USB_FRONT video ({filename}): {e}")
                            log_event("Error Event", "USB_FRONT_FrameWrite", str(e))
                            out.release()
                            recording = False
                            out = None
                            filename = ""

                    if not face_detected and (current_time - usb_last_seen > 10.0):
                        # --- PRINT FRAME DROP SUMMARY STAMP ---
                        if session_dropped_frames > 0:
                            drop_summary = f"{session_dropped_frames}_dropped_max_lag_{max_drop_duration*1000:.0f}ms"
                            logger.warning(f"USB_FRONT ({filename}) Summary: Missed ~{session_dropped_frames} frames. Longest lag: {max_drop_duration*1000:.1f}ms")
                            log_event("Warning Event", "USB_FRONT_FrameDrop_Summary", drop_summary)
                            
                        logger.info(f"Stopping USB video recording (No face detected for 10s): {filename}")
                        try:
                            with open(video_csv_path, "a", newline="") as f:
                                csv.writer(f).writerow([f"{current_time - T0:.3f}", "USB_FRONT", "STOP", filename])
                        except Exception as e:
                            logger.error(f"Failed to write USB_FRONT STOP to video CSV: {e}")
                        
                        if out:
                            out.release()
                        recording = False; out = None; filename = ""

                # ==========================================
                # SESSION MANAGEMENT
                # ==========================================
                if stgt_started:
                    # Strict 30-second face absence limit
                    if not face_detected:
                        if face_absence_start == 0.0:
                            face_absence_start = current_time
                        elif current_time - face_absence_start >= 30.0:
                            logger.info("Terminating STGT session early: 30s_No_Face_Verified")
                            if stgt_process:
                                stgt_process.terminate()
                                stgt_process.wait()
                            face_absence_start = 0.0
                    else:
                        face_absence_start = 0.0

                # Note: Subprocess spawn block is bypassed if phase == "habituation"
                if instant_presence and not stgt_started and not isb_active and not disable_task_spawning and phase != "habituation":
                    session_number += 1
                    
                    # *** THE HANDOFF: Release the pin to the OS so the subprocess can claim it ***
                    GPIO.cleanup(FOODCUP_BEAM_PIN)

                    cmd = [
                        "/usr/bin/python3", "/home/capuchin/Desktop/stgt_scripts/stgt_task.py",
                        "--execution_id", execution_id, "--session_number", str(session_number),
                        "--phase", phase, "--max_trial", str(max_trial),
                        "--lever_dur", str(lever_dur), "--buffer_dur", str(buffer_dur),
                        "--pretrain_base", str(pretrain_base), "--pretrain_jitter", str(pretrain_jitter),
                        "--data_csv_path", str(data_csv_path), "--t0", str(T0)
                    ]
                    if iti_list:
                        cmd.append("--iti_list")
                        cmd.extend([str(i) for i in iti_list])

                    logger.info(f"Trigger condition met. Spawning STGT Subprocess (Session {session_number}).")
                    stgt_process = subprocess.Popen(cmd)
                    stgt_started = True
                    session_start_time = current_time
                    face_absence_start = 0.0

                    # ==========================================
                    # CSI CAMERAS SPAWNING (WITH LOGGING)
                    # ==========================================
                    for i, cam_index in enumerate(csi_sources):
                        cam_name = "TOP" if i == 0 else "SIDE" if i == 1 else f"CAM{i}"
                        filename_csi = f"{datetime.now().strftime('%y%m%d_%H%M%S')}_{cam_name}.mp4"
                        csi_filenames[i] = filename_csi
                        try:
                            time.sleep(0.5)
                            proc = subprocess.Popen([
                                "rpicam-vid", "-t", "0", "-n", "--inline",
                                "--width", "640", "--height", "480", "--framerate", "15",
                                "--camera", str(cam_index), "-o", str(record_dir / filename_csi)
                            ], preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            csi_outs[i] = proc
                            logger.info(f"Started CSI camera {cam_name} recording: {filename_csi}")
                            with open(video_csv_path, "a", newline="") as f:
                                csv.writer(f).writerow([f"{current_time - T0:.3f}", f"CSI_{cam_name}", "START", filename_csi])
                        except Exception as e:
                            logger.error(f"Failed to start CSI camera {cam_name} ({filename_csi}): {e}")
                            log_event("Error Event", f"CSI_{cam_name}_Start", str(e))
                            csi_outs[i] = None

                if stgt_started and stgt_process and stgt_process.poll() is not None:
                    ret_code = stgt_process.poll()
                    logger.info("STGT subprocess concluded. Stopping CSI cameras.")
                    for i, proc in enumerate(csi_outs):
                        if proc:
                            try:
                                proc.send_signal(signal.SIGINT)
                                proc.wait()
                                cam_name = "TOP" if i == 0 else "SIDE" if i == 1 else f"CAM{i}"
                                stop_time = time.time()
                                with open(video_csv_path, "a", newline="") as f:
                                    csv.writer(f).writerow([f"{stop_time - T0:.3f}", f"CSI_{cam_name}", "STOP", csi_filenames[i]])
                            except Exception as e:
                                cam_name = "TOP" if i == 0 else "SIDE" if i == 1 else f"CAM{i}"
                                logger.error(f"Error stopping CSI camera {cam_name}: {e}")
                                log_event("Error Event", f"CSI_{cam_name}_Stop", str(e))

                    stgt_started = False
                    csi_outs = [None] * len(csi_sources)
                    csi_filenames = [None] * len(csi_sources)
                    face_absence_start = 0.0
                    
                    # *** THE RECLAIM: Take the pin back so the master script can watch for the next session ***
                    GPIO.setup(FOODCUP_BEAM_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

                    if ret_code == 0:
                        logger.info("STGT Session fully completed. Starting 4-minute Inter-Session Break.")
                        isb_active = True
                        isb_until = current_time + 240.0
                        presence_timeout_start = 0 
                    else:
                        logger.info("Session concluded prematurely. Skipping ISB. System primed for new arrival.")
                        isb_active = False
                        session_subject_present = False
                        presence_timeout_start = 0

                if isb_active:
                    if instant_presence:
                        if presence_timeout_start != 0:
                            presence_timeout_start = 0 
                    else:
                        if presence_timeout_start == 0:
                            presence_timeout_start = current_time
                        elif current_time - presence_timeout_start >= 30.0:
                            logger.info("Aborting Inter-Session Break due to complete subject departure.")
                            isb_active = False; isb_until = 0; presence_timeout_start = 0; session_subject_present = False

                    if isb_active and current_time >= isb_until:
                        logger.info("4-Minute Inter-Session Break elapsed. System ready for next session.")
                        isb_active = False

                time.sleep(0.005)

            except KeyboardInterrupt:
                print("\n")
                if input("Do you wish to keep the webcam face detection and recording process running? (y/n): ").strip().lower() == 'y':
                    logger.info("Keeping webcam face detection and recording process active. Task spawning disabled.")
                    disable_task_spawning = True
                    if stgt_process: stgt_process.terminate()
                    continue
                else: raise  

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down entirely.")
    finally:
        if out:
            out.release()
        cap.release()
        try:
            GPIO.cleanup(FOODCUP_BEAM_PIN)
        except Exception:
            pass
        for proc in csi_outs:
            if proc:
                try:
                    proc.send_signal(signal.SIGINT)
                    proc.wait()
                except Exception as e:
                    logger.error(f"Error during final cleanup of CSI process: {e}")
        if stgt_process:
            stgt_process.terminate(); stgt_process.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt')
    parser.add_argument('--img', type=int, default=416)
    parser.add_argument('--conf', type=float, default=0.75)
    parser.add_argument('--csi', type=int, nargs='+', default=[])
    args = parser.parse_args()
    run(weights=args.weights, img_size=args.img, conf_thres=args.conf, csi_sources=args.csi)
