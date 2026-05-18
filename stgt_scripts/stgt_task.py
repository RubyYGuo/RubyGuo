#!/usr/bin/env python3

import sys
import time
import csv
import logging
from datetime import datetime
from pathlib import Path
import RPi.GPIO as GPIO
import random
import argparse

# =========================
# Arguments & Parameters
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--execution_id", type=str)
parser.add_argument("--session_number", type=int, default=1)
parser.add_argument("--max_trial", type=int, default=12)
parser.add_argument("--lever_dur", type=float, default=4.0)
parser.add_argument("--iti_base", type=float, default=18.0)
parser.add_argument("--iti_jitter", type=float, default=6.0)
parser.add_argument("--buffer_dur", type=float, default=5.0)
parser.add_argument("--data_csv_path", type=str, required=True)
parser.add_argument("--t0", type=float, required=True)
args = parser.parse_args()

execution_id = args.execution_id
session_number = args.session_number
max_trial = args.max_trial
lever_dur = args.lever_dur
iti_base = args.iti_base
iti_jitter = args.iti_jitter
buffer_dur = args.buffer_dur
data_csv_path = args.data_csv_path
t0 = args.t0

# =========================
# Logging Setup
# =========================
log_file_path = Path("/home/capuchin/stgt_data/task_data") / f"task_log_{execution_id}.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Subprocess - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =========================
# Data Logging Helper
# =========================
def log_event(ev_name, item_name, value=""):
    sec = time.time() - t0
    try:
        with open(data_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{sec:.3f}", ev_name, item_name, value])
    except Exception as e:
        logger.error(f"Failed to write to CSV: {e}")

# =========================
# GPIO PINS & SETUP
# =========================
relay_lv_out = 19
lv_press_pin = 13
relay_cue_light = 12
relay_dispenser = 22
foodcup_beam_pin = 21

GPIO.setmode(GPIO.BCM)
try:
    GPIO.setup(relay_lv_out, GPIO.OUT)
    GPIO.setup(relay_cue_light, GPIO.OUT)
    GPIO.setup(relay_dispenser, GPIO.OUT)
    # Using BOTH to capture Foodcup On and Off
    GPIO.setup(lv_press_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(foodcup_beam_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    GPIO.output(relay_lv_out, True)
    GPIO.output(relay_dispenser, True)
    GPIO.output(relay_cue_light, True)
except Exception as e:
    logger.error(f"GPIO setup failed: {e}")
    log_event("Error Event", "GPIO_Setup", f"Failed: {e}")
    sys.exit(1)

# =========================
# GLOBALS & CALLBACKS
# =========================
phase = "idle"
session_lever_counts = 0
session_foodcup_cs_entries = 0
session_foodcup_iti_entries = 0

def foodcup_callback(channel):
    global session_foodcup_cs_entries, session_foodcup_iti_entries
    state = GPIO.input(foodcup_beam_pin)
    
    if state == GPIO.LOW:  # Beam Broken
        log_event("Input Event", "Foodcup_Entry_On")
        logger.info("Foodcup beam broken (Entry)")
    else:  # Beam Restored
        log_event("Input Event", "Foodcup_Entry_Off")
        log_event("Condition Event", "Foodcup_Activate")
        
        if phase == "lever":
            session_foodcup_cs_entries += 1
            log_event("Variable Event", "Session_Foodcup_CS_Entries", session_foodcup_cs_entries)
            logger.info(f"Foodcup entry completed (CS Phase). Total: {session_foodcup_cs_entries}")
        elif phase == "iti":
            session_foodcup_iti_entries += 1
            log_event("Variable Event", "Session_Foodcup_ITI_Entries", session_foodcup_iti_entries)
            logger.info(f"Foodcup entry completed (ITI Phase). Total: {session_foodcup_iti_entries}")

GPIO.add_event_detect(foodcup_beam_pin, GPIO.BOTH, callback=foodcup_callback, bouncetime=100)

# =========================
# MAIN SESSION LOGIC
# =========================
try:
    logger.info(f"STGT Subprocess {session_number} Initialized")
    
    # Session Resets
    log_event("Variable Event", "Session_Counter", session_number)
    log_event("Variable Event", "Trial_Counter", 0)
    log_event("Variable Event", "Session_Lever_Counts", 0)
    log_event("Variable Event", "Session_Foodcup_CS_Entries", 0)
    log_event("Variable Event", "Session_Foodcup_ITI_Entries", 0)

    # ----- PRE-TRIAL BUFFER -----
    log_event("Condition Event", "Phase_Transition", "Pre_Trial_Buffer")
    phase = "buffer"
    log_event("Variable Event", "Task_Phase_State", "buffer")
    
    logger.info(f"Buffer phase started ({buffer_dur}s)")
    time.sleep(buffer_dur)
    log_event("Timer Event", "Buffer_Timer", round(buffer_dur, 3))

    # ----- TRIAL LOOP -----
    for trial_n in range(max_trial):
        logger.info(f"Starting Trial {trial_n + 1}")
        log_event("Condition Event", "Phase_Transition", "Trial_Start")
        log_event("Variable Event", "Trial_Counter", trial_n + 1)
        
        trial_iti = random.uniform(iti_base - iti_jitter, iti_base + iti_jitter)
        log_event("Variable Event", "ITI_Value", round(trial_iti, 3))

        # ----- CS (LEVER) PHASE -----
        log_event("Condition Event", "Phase_Transition", "CS_Active")
        phase = "lever"
        log_event("Variable Event", "Task_Phase_State", "lever")
        
        GPIO.output(relay_lv_out, False)
        GPIO.output(relay_cue_light, False)
        log_event("Output Event", "Lever_Extend_On")
        log_event("Output Event", "Cue_Light_On")
        logger.info("Lever extended, Cue Light on")
        
        start_time = time.time()
        last_state = GPIO.input(lv_press_pin)

        # Polling for Lever (Allows On & Off capture during CS window)
        while time.time() - start_time < lever_dur:
            current_state = GPIO.input(lv_press_pin)
            if last_state == GPIO.HIGH and current_state == GPIO.LOW:
                log_event("Input Event", "Lever_Press_On")
                logger.info("Lever pressed down")
            elif last_state == GPIO.LOW and current_state == GPIO.HIGH:
                # Count logic is now triggered upon release (Off transition)
                log_event("Input Event", "Lever_Press_Off")
                log_event("Condition Event", "Lever_Activate")
                session_lever_counts += 1
                log_event("Variable Event", "Session_Lever_Counts", session_lever_counts)
                logger.info(f"Lever released and counted. Total: {session_lever_counts}")
            
            last_state = current_state
            time.sleep(0.01) 

        log_event("Timer Event", "CS_Timer", round(lever_dur, 3))

        # ----- REWARD PHASE -----
        log_event("Condition Event", "Phase_Transition", "Reward_Dispense")
        
        GPIO.output(relay_lv_out, True)
        GPIO.output(relay_cue_light, True)
        log_event("Output Event", "Lever_Extend_Off")
        log_event("Output Event", "Cue_Light_Off")

        # Pulse Dispenser
        GPIO.output(relay_dispenser, False)
        log_event("Pulse Output Event", "Dispenser", 0.01)
        time.sleep(0.01)
        GPIO.output(relay_dispenser, True)
        logger.info("Dispensing reward (Dispenser pulsed)")

        # ----- ITI PHASE -----
        log_event("Condition Event", "Phase_Transition", "ITI_Active")
        phase = "iti"
        log_event("Variable Event", "Task_Phase_State", "iti")
        
        logger.info(f"ITI phase started. Scheduled duration: {trial_iti:.2f}s")
        time.sleep(trial_iti)
        log_event("Timer Event", "ITI_Timer", round(trial_iti, 3))
        
        logger.info(f"Trial {trial_n + 1} completed")

    log_event("Condition Event", "Session_End", "Complete")

except KeyboardInterrupt:
    logger.info("STGT subprocess interrupted by user")
    log_event("System Event", "Task_Subprocess", "Interrupted by user")
finally:
    phase = "idle"
    log_event("Variable Event", "Task_Phase_State", "idle")
    GPIO.cleanup()
    logger.info("Subprocess complete, GPIO cleaned up")
