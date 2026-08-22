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
import signal

# =========================
# Arguments & Parameters
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--execution_id", type=str)
parser.add_argument("--session_number", type=int, default=1)
parser.add_argument("--phase", type=str, default="task", choices=["task", "pretrain"])
parser.add_argument("--max_trial", type=int, default=12)
parser.add_argument("--lever_dur", type=float, default=4.0)
parser.add_argument("--iti_list", type=float, nargs='*', default=[])
parser.add_argument("--buffer_dur", type=float, default=5.0)
parser.add_argument("--pretrain_base", type=float, default=5.0)
parser.add_argument("--pretrain_jitter", type=float, default=1.0)
parser.add_argument("--data_csv_path", type=str, required=True)
parser.add_argument("--t0", type=float, required=True)
args = parser.parse_args()

execution_id = args.execution_id
session_number = args.session_number
phase_arg = args.phase
max_trial = args.max_trial
lever_dur = args.lever_dur
iti_list = args.iti_list
buffer_dur = args.buffer_dur
pretrain_base = args.pretrain_base
pretrain_jitter = args.pretrain_jitter
data_csv_path = args.data_csv_path
t0 = args.t0

# =========================
# Unified Logging Setup
# =========================
log_file_path = Path("/home/capuchin/stgt_data/task_data") / f"system_log_{execution_id}.txt"
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
# Signal Handler
# =========================
def sigterm_handler(signum, frame):
    logger.info("Received termination signal from Master. Aborting session.")
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, signal.SIG_IGN)

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
foodcup_beam_pin = 21  # RESTORED

GPIO.setmode(GPIO.BCM)
try:
    GPIO.setup(relay_lv_out, GPIO.OUT)
    GPIO.setup(relay_cue_light, GPIO.OUT)
    GPIO.setup(relay_dispenser, GPIO.OUT)
    GPIO.setup(lv_press_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(foodcup_beam_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # RESTORED

    GPIO.output(relay_lv_out, True)
    GPIO.output(relay_dispenser, True)
    GPIO.output(relay_cue_light, True)
except Exception as e:
    logger.error(f"GPIO setup failed: {e}")
    log_event("Error Event", "GPIO_Setup", f"Failed: {e}")
    sys.exit(1)

# =========================
# GLOBALS & POLLING LOGIC
# =========================
phase = "idle"
session_lever_counts = 0
session_foodcup_cs_entries = 0
session_foodcup_iti_entries = 0

last_interaction_time = time.time()
inactivity_limit = 30.0 if phase_arg == "pretrain" else 90.0
last_foodcup_state = GPIO.HIGH

def poll_foodcup():
    """Restored software polling for Pin 21 inside the subprocess trials"""
    global last_foodcup_state, session_foodcup_cs_entries, session_foodcup_iti_entries, last_interaction_time
    try:
        current_state = GPIO.input(foodcup_beam_pin)
        
        if last_foodcup_state == GPIO.HIGH and current_state == GPIO.LOW:  
            last_interaction_time = time.time() 
            log_event("Input Event", "Foodcup_Beam_Broken")
            logger.info("Foodcup beam broken (Entry)")
            log_event("Condition Event", "Foodcup_Entry")
            
            if phase == "lever":
                session_foodcup_cs_entries += 1
                log_event("Variable Event", "Session_Foodcup_CS_Entries", session_foodcup_cs_entries)
                logger.info(f"Foodcup entry completed (CS Phase). Total: {session_foodcup_cs_entries}")
            elif phase in ["iti", "pretrain", "buffer"]:
                session_foodcup_iti_entries += 1
                log_event("Variable Event", "Session_Foodcup_ITI_Entries", session_foodcup_iti_entries)
                logger.info(f"Foodcup entry completed ({phase} Phase). Total: {session_foodcup_iti_entries}")
                
        elif last_foodcup_state == GPIO.LOW and current_state == GPIO.HIGH:  
            log_event("Input Event", "Foodcup_Beam_Restored")

        last_foodcup_state = current_state
    except Exception:
        pass

def check_inactivity():
    return (time.time() - last_interaction_time) > inactivity_limit

def wait_with_inactivity_check(duration):
    t_start = time.time()
    while time.time() - t_start < duration:
        poll_foodcup() # Poll the food cup continuously
        if check_inactivity():
            return True
        time.sleep(0.01)
    return False

# =========================
# MAIN SESSION LOGIC
# =========================
try:
    logger.info(f"STGT Subprocess {session_number} Initialized. Mode: {phase_arg}")
    
    last_interaction_time = time.time()
    
    log_event("Variable Event", "Session_Counter", session_number)
    log_event("Variable Event", "Trial_Counter", 0)
    log_event("Variable Event", "Session_Lever_Counts", 0)
    log_event("Variable Event", "Session_Foodcup_CS_Entries", 0)
    log_event("Variable Event", "Session_Foodcup_ITI_Entries", 0)

    log_event("Condition Event", "Phase_Transition", "Pre_Trial_Buffer")
    phase = "buffer"
    log_event("Variable Event", "Task_Phase_State", "buffer")
    
    logger.info(f"Buffer phase started ({buffer_dur}s)")
    
    early_exit = False
    
    if wait_with_inactivity_check(buffer_dur):
        early_exit = True
    else:
        log_event("Timer Event", "Buffer_Timer", round(buffer_dur, 3))

    if not early_exit:
        for trial_n in range(max_trial):
            logger.info(f"Starting Trial {trial_n + 1}")
            log_event("Condition Event", "Phase_Transition", "Trial_Start")
            log_event("Variable Event", "Trial_Counter", trial_n + 1)
            
            if phase_arg == "pretrain":
                phase = "pretrain"
                log_event("Variable Event", "Task_Phase_State", "pretrain")
                
                log_event("Condition Event", "Phase_Transition", "Reward_Dispense")
                GPIO.output(relay_dispenser, False)
                log_event("Pulse Output Event", "Dispenser", 0.1)
                time.sleep(0.1)
                GPIO.output(relay_dispenser, True)
                logger.info("Dispensing reward (Pre-training)")
                
                trial_interval = random.uniform(pretrain_base - pretrain_jitter, pretrain_base + pretrain_jitter)
                log_event("Variable Event", "Pretrain_Interval", round(trial_interval, 3))
                logger.info(f"Pre-training interval started. Scheduled duration: {trial_interval:.2f}s")
                
                if wait_with_inactivity_check(trial_interval):
                    early_exit = True
                    break
                    
                log_event("Timer Event", "Pretrain_Timer", round(trial_interval, 3))
                
            elif phase_arg == "task":
                if not iti_list:
                    iti_list = [12.0] 
                trial_iti = random.choice(iti_list)
                log_event("Variable Event", "ITI_Value", round(trial_iti, 3))

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

                while time.time() - start_time < lever_dur:
                    poll_foodcup() # Poll the food cup during lever phase
                    if check_inactivity():
                        early_exit = True
                        break
                    
                    current_state = GPIO.input(lv_press_pin)
                    if last_state == GPIO.HIGH and current_state == GPIO.LOW:
                        last_interaction_time = time.time() 
                        log_event("Input Event", "Lever_Press_On")
                        logger.info("Lever pressed down")
                        
                        log_event("Condition Event", "Lever_Pressed")
                        session_lever_counts += 1
                        log_event("Variable Event", "Session_Lever_Counts", session_lever_counts)
                        logger.info(f"Lever released and counted. Total: {session_lever_counts}")
                        
                    elif last_state == GPIO.LOW and current_state == GPIO.HIGH:
                        log_event("Input Event", "Lever_Press_Off")
                                        
                    last_state = current_state
                    time.sleep(0.01) 

                if early_exit:
                    break
                    
                log_event("Timer Event", "CS_Timer", round(lever_dur, 3))

                log_event("Condition Event", "Phase_Transition", "Reward_Dispense")
                
                GPIO.output(relay_lv_out, True)
                GPIO.output(relay_cue_light, True)
                log_event("Output Event", "Lever_Extend_Off")
                log_event("Output Event", "Cue_Light_Off")

                GPIO.output(relay_dispenser, False)
                log_event("Pulse Output Event", "Dispenser", 0.1)
                time.sleep(0.1)
                GPIO.output(relay_dispenser, True)
                logger.info("Dispensing reward (Dispenser pulsed)")

                log_event("Condition Event", "Phase_Transition", "ITI_Active")
                phase = "iti"
                log_event("Variable Event", "Task_Phase_State", "iti")
                
                logger.info(f"ITI phase started. Scheduled duration: {trial_iti:.2f}s")
                
                if wait_with_inactivity_check(trial_iti):
                    early_exit = True
                    break
                    
                log_event("Timer Event", "ITI_Timer", round(trial_iti, 3))
                
            logger.info(f"Trial {trial_n + 1} completed")

    if early_exit:
        log_event("Condition Event", "Session_End", f"Early_Termination_{inactivity_limit}s_Inactivity")
        logger.info(f"Session terminating early due to {inactivity_limit}s of inactivity.")
        sys.exit(2) 
    else:
        log_event("Condition Event", "Session_End", "Complete")
        sys.exit(0) 

except KeyboardInterrupt:
    logger.info("STGT subprocess interrupted by user")
    log_event("System Event", "Task_Subprocess", "Interrupted by user")
    sys.exit(1)
finally:
    phase = "idle"
    log_event("Variable Event", "Task_Phase_State", "idle")
    
    try:
        GPIO.output(relay_lv_out, True)
        GPIO.output(relay_cue_light, True)
        GPIO.output(relay_dispenser, True)
        logger.info("Hardware explicitly reset to safe state.")
    except Exception:
        pass

    GPIO.cleanup()
    logger.info("Subprocess complete, GPIO cleaned up")
