# RPi pins that might have been burnt: (BCM) 21,16,12,25,24,(23?)

#!/usr/bin/env python3

import sys
import time
import csv
from pathlib import Path
from datetime import datetime
import RPi.GPIO as GPIO
import random
import argparse


# =========================
# STGT task settings
# =========================
# max_trial = 10 
# lever_dur = 7
# itt_base = 25
# itt_jitter = 5

parser = argparse.ArgumentParser()

parser.add_argument("--execution_id", type=str)
parser.add_argument("--session_number", type=int, default=1)
parser.add_argument("--max_trial", type=int, default=10)
parser.add_argument("--lever_dur", type=float, default=1)
parser.add_argument("--itt_base", type=float, default=5)
parser.add_argument("--itt_jitter", type=float, default=2)

args = parser.parse_args()

execution_id = args.execution_id or datetime.now().strftime("%Y%m%d_%H%M%S")
session_number = args.session_number
max_trial = args.max_trial
lever_dur = args.lever_dur
itt_base = args.itt_base
itt_jitter = args.itt_jitter

# =========================
# STGT DATA DIRECTORY
# =========================
stgt_data_dir = Path("/home/capuchin/SSD/stgt_data/task_data")
stgt_data_dir.mkdir(parents=True, exist_ok=True)  # ensure folder exists


# =========================
# CSV FILE NAMING
# =========================
csv_filename = f"stgt_{execution_id}.csv"  # one CSV per execution
csv_path = stgt_data_dir / csv_filename
print(f"[INFO] STGT CSV file will be saved to: {csv_path.resolve()}")

# =========================
# GPIO PINS
# =========================
relay_lv_out = 19
lv_press_pin = 13
relay_mgz = 22
mgz_beam_pin = 17

# =========================
# GPIO SETUP
# =========================
GPIO.setmode(GPIO.BCM)

try:
    GPIO.setup(relay_lv_out, GPIO.OUT)
    GPIO.setup(relay_mgz, GPIO.OUT)
    GPIO.setup(lv_press_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(mgz_beam_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # Initial relay state
    GPIO.output(relay_lv_out, True)
    GPIO.output(relay_mgz, True)

except Exception as e:
    print(f"[ERROR] GPIO setup failed: {e}")
    sys.exit(1)

# =========================
# CALLBACK AND GLOBALS
# =========================
mgz_reach_trial = 0
mgz_reach_itt = 0
phase = "idle"

def magazine_callback(channel):
    global mgz_reach_trial, mgz_reach_itt, phase
    if phase == "lever":
        mgz_reach_trial += 1
        print(f"[{datetime.now().isoformat()}] magazine reach (trial)")
    elif phase == "itt":
        mgz_reach_itt += 1
        print(f"[{datetime.now().isoformat()}] magazine reach (ITT)")

GPIO.add_event_detect(mgz_beam_pin, GPIO.FALLING, callback=magazine_callback, bouncetime=200)

# =========================
# CSV SETUP
# =========================
csv_exists = csv_path.exists()
csv_file = open(csv_path, "a", newline="")
csv_writer = csv.writer(csv_file)

if not csv_exists:

    # ---- Write Schedule Metadata ----
    csv_writer.writerow(["# SCHEDULE"])
    csv_writer.writerow(["execution_id", execution_id])
    csv_writer.writerow(["session", session_number])
    csv_writer.writerow(["max_trial", max_trial])
    csv_writer.writerow(["lever_dur", lever_dur])
    csv_writer.writerow(["itt_base", itt_base])
    csv_writer.writerow(["itt_jitter", itt_jitter])
    csv_writer.writerow([])  # blank line

    # ---- Write Trial Header ----
    csv_writer.writerow([
        "execution_id",
        "session",
        "trial",
        "lever_presses",
        "magazine_reaches_trial",
        "magazine_reaches_itt",
        "itt_duration",
        "timestamp"
    ])

# =========================
# MAIN TRIAL LOOP
# =========================
try:

    for trial_n in range(max_trial):
        lv_press_count = 0
        mgz_reach_trial = 0
        mgz_reach_itt = 0

        # ----- LEVER PHASE -----
        phase = "lever"
        GPIO.output(relay_lv_out, False)
        start_time = time.time()

        # Edge detection: track previous lever state; holding = 1 press
        last_state = GPIO.input(lv_press_pin)

        while time.time() - start_time < lever_dur:
            current_state = GPIO.input(lv_press_pin)
            # Detect HIGH -> LOW transition
            if last_state == GPIO.HIGH and current_state == GPIO.LOW:
                lv_press_count += 1
                print(f"[{datetime.now().isoformat()}] lever pressed")
            last_state = current_state
            time.sleep(0.05) 

        GPIO.output(relay_lv_out, True)

        # ----- REWARD -----
        GPIO.output(relay_mgz, False)
        print(f"[{datetime.now().isoformat()}] dispensing reward")
        time.sleep(0.01)
        GPIO.output(relay_mgz, True)

        # ----- ITT -----
        phase = "itt"
        print(f"[{datetime.now().isoformat()}] ITT started")
        
        trial_itt = random.uniform(itt_base - itt_jitter,
                             itt_base + itt_jitter)

        print(f"[{datetime.now().isoformat()}] ITT started. Scheduled duration: {trial_itt:.2f}s")
        
        time.sleep(trial_itt)
        
        phase = "idle"

        # ----- SAVE TRIAL -----
        csv_writer.writerow([
            execution_id,
            session_number,
            trial_n + 1,
            lv_press_count,
            mgz_reach_trial,
            mgz_reach_itt,
            trial_itt,
            datetime.now().isoformat()
        ])

        csv_file.flush()

        print(f"[{datetime.now().isoformat()}] Trial {trial_n + 1} completed")
        print(f"    lever presses: {lv_press_count}")
        print(f"    mag reaches (trial): {mgz_reach_trial}")
        print(f"    mag reaches (ITT): {mgz_reach_itt}")

    print(f"[INFO] STGT session completed")

except KeyboardInterrupt:
    print("[INFO] STGT session interrupted")

finally:
    GPIO.cleanup()  # only cleanup once at the very end
    csv_file.close()
    print("[INFO] GPIO cleaned up, CSV saved")
