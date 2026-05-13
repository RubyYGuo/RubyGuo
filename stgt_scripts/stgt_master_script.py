# Stopping CSI camera recording when stgt session ends. Now figure out:
### If an animal is at the box for a long time (I'll call it "interaction bout" for now), do we want to trigger only one session, or keep triggering the next?
### right now only new face detection triggers the task; if the animal is always in camera then the next session won't be triggered.
### If we don't want the session to be triggered constantly, need to prevent the case when facial detection briefly fails and comes back ("new detection" same interaction bout)
### waiting time between sessions e.g. min 5min?

#### ideally: 12 trials per session, 2 sesseions in a row max per individual
#### current: we can't yet tell apart individuals; 2 sessions in a row (within 10 sec apart) at most and then wait for 5min to trigger the next"
# left off March20: made session triggering based on detection status (if animal is always there, possible to trigger a 2nd session) instead of recording status
# todo: 3rd session still gets triggered, with error mossages from the usb camera.


# video storage (e.g., hard drive)
#!/usr/bin/env python3

import torch
import cv2
import argparse
import time
import csv
import logging
from pathlib import Path
from datetime import datetime
from collections import deque
import subprocess
import re
import signal
from ultralytics import YOLO

# =========================
# Logging setup
# =========================
execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")

stgt_data_dir = Path("/home/capuchin/stgt_data/task_data")
stgt_data_dir.mkdir(parents=True, exist_ok=True)

log_file_path = stgt_data_dir / f"session_log_{execution_id}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

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
        return "N/A"

# =========================
# FPS Helper
# =========================
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
    max_trial, lever_dur, itt_base, itt_jitter = 5, 2, 2, 1
    logger.info(f"Current settings: max_trial={max_trial}, lever_dur={lever_dur}, itt={itt_base} ± {itt_jitter}")
    choice = input("\nModify settings? (y/n): ").strip().lower()
    if choice == "y":
        max_trial = int(input("Enter max_trial: ") or max_trial)
        lever_dur = float(input("Enter lever_dur: ") or lever_dur)
        itt_base = float(input("Enter itt_base: ") or itt_base)
        itt_jitter = float(input("Enter itt_jitter: ") or itt_jitter)
    logger.info(f"Final schedule: max_trial={max_trial}, lever_dur={lever_dur}, itt={itt_base} ± {itt_jitter}")
    return max_trial, lever_dur, itt_base, itt_jitter

# =========================
# Session number helper
# =========================
def get_next_session_number(csv_path):
    if not csv_path.exists():
        return 1
    
    max_session = 0
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            # Skip the header row
            next(reader, None) 
            
            for row in reader:
                # Ensure row has enough columns and isn't metadata
                if len(row) >= 3:
                    session_str = row[2].strip()
                    if session_str.isdigit():
                        current_val = int(session_str)
                        if current_val > max_session:
                            max_session = current_val
                            
        return max_session + 1
    except Exception as e:
        logger.error(f"Error reading session number: {e}")
        return 1

# =========================
# Main run
# =========================
def run(weights='best.pt', img_size=416, conf_thres=0.75, csi_sources=[]):

    # Load schedule
    max_trial, lever_dur, itt_base, itt_jitter = configure_schedule()

    # New session control parameters
    max_sessions_in_burst = 2      # max sessions in a row
    burst_interval = 10            # seconds between consecutive sessions
    cooldown_time = 300            # 5 min cooldown after burst

    # =========================
    # State variables
    # =========================
    recording = False
    out = None
    last_detection_time = 0
    filename = ""

    stgt_process = None
    stgt_started = False

    session_count_in_burst = 0
    last_stgt_end_time = 0
    cooldown_until = 0

    last_blocked_log_time = 0
    BLOCK_LOG_INTERVAL = 2  # sec
    session_number = ""     # Initialized to ensure continuous scope

    # New states for time-window buffer & hysteresis
    detection_window = deque(maxlen=30)
    is_present = False
    last_verified_presence_time = 0

    # CSI state
    csi_outs = [None] * len(csi_sources)
    csi_start_times = [None] * len(csi_sources)
    csi_filenames = [None] * len(csi_sources)

    # Load YOLO
    logger.info("Loading YOLO model")
    model = YOLO(weights)

    # Open USB camera
    device = "/dev/video0"
    logger.info(f"Using USB camera: {device}")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        logger.error("Failed to open USB camera")
        return

    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    time_deque = deque(maxlen=30)

    # Recording directory and CSV
    record_dir = Path("/home/capuchin/stgt_data/video_recordings")
    record_dir.mkdir(exist_ok=True)
    log_csv_path = record_dir / f"sessions_{execution_id}.csv"
    csv_file = open(log_csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "execution_id", "session_number", "status", "reason", "video"])

    logger.info("Starting detection loop")

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
            # YOLO Detection & Time-Window Buffer
            # =========================
            results = model(frame, imgsz=img_size, conf=conf_thres, iou=0.4, verbose=False)
            raw_detection = len(results[0].boxes) > 0
            
            # Add to rolling window buffer
            detection_window.append(1 if raw_detection else 0)
            
            # Calculate detection ratio over the last ~1 second
            detection_ratio = sum(detection_window) / len(detection_window) if len(detection_window) > 0 else 0

            # Evaluate Hysteresis / Interaction Bout
            if detection_ratio >= 0.50:
                is_present = True
                last_verified_presence_time = current_time
            elif is_present and (current_time - last_verified_presence_time > 3.0):
                # If 3 continuous seconds pass without verified presence, bout ends
                is_present = False

            # =========================
            # USB Recording (independent)
            # =========================
            if is_present:
                last_detection_time = current_time
                if not recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"capuchin_{timestamp}.mp4"
                    out_path = record_dir / filename
                    out = cv2.VideoWriter(
                        str(out_path),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        estimated_fps,
                        (FRAME_WIDTH, FRAME_HEIGHT)
                    )
                    recording = True
                    logger.info(f"Started USB recording: {filename}")

            if recording and out:
                out.write(frame)

            if recording and (current_time - last_detection_time > 10):
                recording_end_time = datetime.now().isoformat()
                logger.info(f"Stopping USB recording: {filename}")
                if out:
                    out.release()
                recording = False
                out = None
                filename = ""

            # =========================
            # Detect STGT completion & stop CSI
            # =========================
            if stgt_started and stgt_process and stgt_process.poll() is not None:
                logger.info("STGT finished, stopping CSI cameras")
                for i, proc in enumerate(csi_outs):
                    if proc:
                        proc.send_signal(signal.SIGINT)
                        proc.wait()
                        csv_writer.writerow([
                            datetime.now().isoformat(),
                            execution_id,
                            session_number, 
                            "stopped",
                            "",
                            str(csi_filenames[i])
                        ])
                        csv_file.flush()
                stgt_started = False
                last_stgt_end_time = current_time
                csi_outs = [None] * len(csi_sources)
                csi_start_times = [None] * len(csi_sources)
                csi_filenames = [None] * len(csi_sources)
                
                # Start cooldown timer ONLY AFTER the final session in a burst ends
                if session_count_in_burst >= max_sessions_in_burst:
                    cooldown_until = current_time + cooldown_time
                    logger.info(f"Burst limit reached. Cooldown applied for {cooldown_time} seconds")

            # =========================
            # Session state-trigger logic
            # =========================
            session_allowed = True
            reason = ""

            # Check if cooldown has expired
            if current_time >= cooldown_until and cooldown_until > 0:
                session_count_in_burst = 0
                cooldown_until = 0

            if current_time < cooldown_until:
                session_allowed = False
                reason = "cooldown"
            elif session_count_in_burst >= max_sessions_in_burst:
                session_allowed = False
                reason = "burst_limit"
            elif not stgt_started and session_count_in_burst > 0 and (current_time - last_stgt_end_time < burst_interval):
                # Enforce the 10-second wait interval between the 1st and 2nd session of a burst
                session_allowed = False
                reason = "burst_interval_wait"

            if is_present and session_allowed and not stgt_started:
                # Start new session
                session_number = get_next_session_number(log_csv_path)
                session_count_in_burst += 1
                current_temp = get_cpu_temp()

                logger.info(f"Starting STGT session {session_number} | CPU Temp: {current_temp}°C")

                csv_writer.writerow([
                    datetime.now().isoformat(),
                    execution_id,
                    session_number,
                    "started",
                    "",
                    filename if recording else ""
                ])
                csv_file.flush()

                # Start STGT task
                stgt_process = subprocess.Popen([
                    "/usr/bin/python3",
                    "/home/capuchin/Desktop/stgt_scripts/stgt_task.py",
                    "--execution_id", execution_id,
                    "--session_number", str(session_number),
                    "--max_trial", str(max_trial),
                    "--lever_dur", str(lever_dur),
                    "--itt_base", str(itt_base),
                    "--itt_jitter", str(itt_jitter)
                ])
                stgt_started = True

                # Start CSI cameras
                for i, cam_index in enumerate(csi_sources):
                    timestamp_csi = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename_csi = f"csi_cam{i}_{timestamp_csi}.mp4"
                    out_path_csi = record_dir / filename_csi
                    try:
                        proc = subprocess.Popen([
                            "rpicam-vid",
                            "-t", "0",
                            "-n",
                            "--inline",
                            "--width", "640",
                            "--height", "480",
                            "--framerate", "15",
                            "--camera", str(cam_index),
                            "-o", str(out_path_csi)
                        ])
                        csi_outs[i] = proc
                        csi_start_times[i] = datetime.now().isoformat()
                        csi_filenames[i] = out_path_csi
                        logger.info(f"Started CSI camera {cam_index}: {filename_csi}")
                    except Exception as e:
                        logger.error(f"Failed to start CSI camera {cam_index}: {e}")
                        csi_outs[i] = None

            # Blocked session logging
            elif is_present and not session_allowed:
                if current_time - last_blocked_log_time > 2:
                    csv_writer.writerow([
                        datetime.now().isoformat(),
                        execution_id,
                        session_number,
                        "blocked",
                        reason,
                        filename if recording else ""
                    ])
                    csv_file.flush()
                    logger.info(f"Session blocked: {reason}")
                    last_blocked_log_time = current_time

            time.sleep(0.005)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        if out:
            out.release()
        cap.release()
        for proc in csi_outs:
            if proc:
                proc.send_signal(signal.SIGINT)
                proc.wait()
        if stgt_process:
            stgt_process.terminate()
            stgt_process.wait()
        csv_file.close()
        logger.info("All resources released")


# =========================
# Argument parser
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt')
    parser.add_argument('--img', type=int, default=416)
    parser.add_argument('--conf', type=float, default=0.75)
    parser.add_argument('--csi', type=int, nargs='+', default=[])
    args = parser.parse_args()

    run(
        weights=args.weights,
        img_size=args.img,
        conf_thres=args.conf,
        csi_sources=args.csi
    )
