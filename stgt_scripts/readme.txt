stgt_scripts: folder; files needed to run the tasks in python (requires YOLO environment)
  best.pt: current best model for facial recognition; has to be in the same directory as stgt_master_script.py
  run_capuchin_suite.sh: shell; set environment and run stgt_master_script.py
  stgt_master_script.py: python; the main script (FR + record videos + calls stgt_task.py + save videos)
  stgt_task.py: python; called by stgt_master_script (relay control + save session data)
