#!/usr/bin/env python3

import torch
import cv2
import argparse
import time
import csv
from pathlib import Path
from datetime import datetime
from collections import deque
import subprocess
import signal
from ultralytics import YOLO

# =========================
# Setup & Initialization
# =========================
execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
T0 = time.time()  # Master Baseline Time

stgt_data_dir = Path("/home/capuchin/stgt_data/task_data")
stgt_data_dir.mkdir(parents=True, exist_ok=True)

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
    writer.writerow([0, "Condition Event", "Execution_Start", execution_id])
    writer.writerow([0, "Timer Event", "Execution_timer", 0])
    writer.writerow([0, "Condition Event", "Execution timer activated", ""])
    writer.writerow([0, "Variable Event", "Face_Detection", 0])
    writer.writerow([0, "Variable Event", "Station_Active", 0])
    writer.writerow([0, "Variable Event", "Subject_Present_Flag", 0])
    writer.writerow([0, "Variable Event", "YOLO_Detection_Ratio", "0.00"])

def log_event(ev_name, item_name, value=""):
    ms = int((time.time() - T0) * 1000)
    with open(data_csv_path, "a", newline="") as f:
        csv.writer(f).writerow([ms, ev_name, item_name, value])

# =========================
# Hardware Diagnostics
# =========================
def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read().strip()) / 1000.0
        return round(temp, 1)
    except Exception as e:
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
    print("\n========== STGT SCHEDULE CONFIG ==========")
    max_trial, lever_dur, iti_base, iti_jitter, buffer_dur = 5, 2, 2, 1, 5
    print(f"Current settings: max_trial={max_trial}, lever_dur={lever_dur}, iti={iti_base} ± {iti_jitter}, buffer={buffer_dur}")
    choice = input("\nModify settings? (y/n): ").strip().lower()
    if choice == "y":
        max_trial = int(input("Enter max_trial: ") or max_trial)
        lever_dur = float(input("Enter lever_dur: ") or lever_dur)
        iti_base = float(input("Enter iti_base: ") or iti_base)
        iti_jitter = float(input("Enter iti_jitter: ") or iti_jitter)
        buffer_dur = float(input("Enter buffer_dur: ") or buffer_dur)
    print(f"Final schedule: max_trial={max_trial}, lever_dur={lever_dur}, iti={iti_base} ± {iti_jitter}, buffer={buffer_dur}\n")
    return max_trial, lever_dur, iti_base, iti_jitter, buffer_dur

# =========================
# Main run
# =========================
def run(weights='best.pt', img_size=416, conf_thres=0.75, csi_sources=[]):
    max_trial, lever_dur, iti_base, iti_jitter, buffer_dur = configure_schedule()

    # Log task parameters at time 0
    log_event("Variable Event", "YOLO_Weights", weights)
    log_event("Variable Event", "YOLO_Img_Size", img_size)
    log_event("Variable Event", "YOLO_Conf_Thres", conf_thres)
    log_event("Variable Event", "Task_Max_Trial", max_trial)
    log_event("Variable Event", "Task_Lever_Dur", lever_dur)
    log_event("Variable Event", "Task_ITI_Base", iti_base)
    log_event("Variable Event", "Task_ITI_Jitter", iti_jitter)
    log_event("Variable Event", "Task_Buffer_Dur", buffer_dur)

    # State variables
    recording = False
    out = None
    last_detection_time = 0
    filename = ""

    stgt_process = None
    stgt_started = False
    session_number = 0

    # Presence & Wait Logic
    detection_window = deque(maxlen=30)
    is_present = False
    presence_timeout_start = 0  # Starts 30s countdown when monkey leaves
    isb_until = 0               # Holds the 4 min wait (240s) for consecutive sessions
    isb_active = False

    # CSI state
    csi_outs = [None] * len(csi_sources)
    csi_start_times = [None] * len(csi_sources)
    csi_filenames = [None] * len(csi_sources)

    print("[INFO] Loading YOLO model...")
    model = YOLO(weights)

    device = "/dev/video0"
    print(f"[INFO] Using USB camera: {device}")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[ERROR] Failed to open USB camera")
        log_event("Error Event", "USB_Camera", "Failed to open")
        return

    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    time_deque = deque(maxlen=30)

    # Setup video log CSV
    if not video_csv_path.exists():
        with open(video_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "execution_id", "session_number", "status", "reason", "video"])

    print("[INFO] Starting detection loop...")
    log_event("System Event", "Detection_Loop", "Started")

    try:
        while True:
            current_time = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            time_deque.append(current_time)
            estimated_fps = estimate_fps(time_deque)

            # =========================
            # YOLO & Presence Logic
            # =========================
            results = model(frame, imgsz=img_size, conf=conf_thres, iou=0.4, verbose=False)
            raw_detection = len(results[0].boxes) > 0
            detection_window.append(1 if raw_detection else 0)
            detection_ratio = sum(detection_window) / len(detection_window) if len(detection_window) > 0 else 0

            if detection_ratio >= 0.50:
                if not is_present:
                    is_present = True
                    log_event("Variable Event", "YOLO_Detection_Ratio", round(detection_ratio, 2))
                    log_event("Variable Event", "Subject_Present_Flag", 1)
                    log_event("Variable Event", "Face_Detection", 1)
                    log_event("Variable Event", "Station_Active", 1)
                presence_timeout_start = 0 
                last_detection_time = current_time
            else:
                if is_present:
                    if presence_timeout_start == 0:
                        presence_timeout_start = current_time
                        log_event("Variable Event", "YOLO_Detection_Ratio", round(detection_ratio, 2))
                        log_event("Condition Event", "Presence_Timeout_Begin", 30000)
                    elif current_time - presence_timeout_start >= 30.0:
                        is_present = False
                        log_event("Timer Event", "Presence_Timeout_Elapsed", 30000)
                        log_event("Variable Event", "Subject_Present_Flag", 0)
                        log_event("Variable Event", "Face_Detection", 0)
                        log_event("Variable Event", "Station_Active", 0)
                        log_event("Condition Event", "System_Ready_Next_Subject", "")
                        isb_until = 0  # Reset ISB so next arrival triggers instantly
                        isb_active = False
                        presence_timeout_start = 0

            # =========================
            # USB Recording
            # =========================
            if is_present and not recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capuchin_{timestamp}.mp4"
                out_path = record_dir / filename
                
                try:
                    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), estimated_fps, (FRAME_WIDTH, FRAME_HEIGHT))
                    recording = True
                    log_event("Output Event", "USB_Camera_Recording_On", filename)
                    print(f"[INFO] Started USB recording: {filename}")
                except Exception as e:
                    log_event("Error Event", "USB_Camera", f"Failed to write video: {e}")

            if recording and out:
                out.write(frame)

            if recording and (current_time - last_detection_time > 10) and not is_present:
                print(f"[INFO] Stopping USB recording: {filename}")
                if out:
                    out.release()
                recording = False
                out = None
                log_event("Output Event", "USB_Camera_Recording_Off", "")
                filename = ""

            # =========================
            # Inter-Session Break (ISB) Check
            # =========================
            if isb_active and current_time >= isb_until:
                log_event("Timer Event", "ISB_Timer_Elapsed", 240000)
                isb_active = False

            # =========================
            # Trigger STGT Subprocess
            # =========================
            if is_present and not stgt_started and not isb_active:
                session_number += 1
                current_temp = get_cpu_temp()
                
                log_event("Diagnostic Event", "CPU_Temperature_Log", current_temp)
                log_event("Condition Event", "STGT_Subprocess_Start", session_number)
                log_event("Timer Event", "Session_Timer_Start", 0)

                with open(video_csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([datetime.now().isoformat(), execution_id, session_number, "started", "", filename if recording else ""])

                # Start Task Subprocess
                stgt_process = subprocess.Popen([
                    "/usr/bin/python3",
                    "/home/capuchin/Desktop/stgt_scripts/stgt_task.py",
                    "--execution_id", execution_id,
                    "--session_number", str(session_number),
                    "--max_trial", str(max_trial),
                    "--lever_dur", str(lever_dur),
                    "--iti_base", str(iti_base),
                    "--iti_jitter", str(iti_jitter),
                    "--buffer_dur", str(buffer_dur),
                    "--data_csv_path", str(data_csv_path),
                    "--t0", str(T0)
                ])
                stgt_started = True
                session_start_time = current_time

                # Start CSI cameras
                for i, cam_index in enumerate(csi_sources):
                    timestamp_csi = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename_csi = f"csi_cam{i}_{timestamp_csi}.mp4"
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
                        print(f"[INFO] Started CSI camera {cam_index}: {filename_csi}")
                    except Exception as e:
                        print(f"[ERROR] Failed to start CSI camera {cam_index}: {e}")
                        log_event("Error Event", f"CSI_Camera_{cam_index}", f"Failed to start: {e}")
                        csi_outs[i] = None

            # =========================
            # STGT Completion & Cleanup
            # =========================
            if stgt_started and stgt_process and stgt_process.poll() is not None:
                session_duration = int((current_time - session_start_time) * 1000)
                log_event("Timer Event", "Session_Timer_Elapsed", session_duration)
                log_event("Condition Event", "STGT_Subprocess_End", "Complete")
                
                print("[INFO] STGT finished, stopping CSI cameras")
                for i, proc in enumerate(csi_outs):
                    if proc:
                        proc.send_signal(signal.SIGINT)
                        proc.wait()
                        log_event("Output Event", f"CSI_Camera_{i}_Recording_Off", "")
                        with open(video_csv_path, "a", newline="") as f:
                            csv.writer(f).writerow([datetime.now().isoformat(), execution_id, session_number, "stopped", "", str(csi_filenames[i])])
                
                stgt_started = False
                csi_outs = [None] * len(csi_sources)
                
                # Apply the 4-minute continuous presence wait (Inter-Session Break)
                if is_present:
                    isb_active = True
                    isb_until = current_time + 240.0
                    log_event("Condition Event", "Phase_Transition", "Inter_Session_Break")
                    log_event("Timer Event", "ISB_Timer_Start", 240000)

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        log_event("System Event", "Execution", "Interrupted by user")
    finally:
        if out: out.release()
        cap.release()
        for proc in csi_outs:
            if proc:
                proc.send_signal(signal.SIGINT)
                proc.wait()
        if stgt_process:
            stgt_process.terminate()
            stgt_process.wait()
        print("[INFO] All resources released")
        log_event("System Event", "Execution", "Resources released, system closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt')
    parser.add_argument('--img', type=int, default=416)
    parser.add_argument('--conf', type=float, default=0.75)
    parser.add_argument('--csi', type=int, nargs='+', default=[])
    args = parser.parse_args()

    run(weights=args.weights, img_size=args.img, conf_thres=args.conf, csi_sources=args.csi)
