## setup instructions

Before starting each experiment:

1. Setup robot
   1. Connect leader and follower via USB.
   2. Setup and start virtual camera (iPhone DroidCam) on OBS.
   3. Update KochRobotConfig in configs.py
2. Calibrate leader and follower (`01_configure_motor.ipynb`)
   1. Make sure leader and follower are about the same range. Otherwise, recalibrate by removing `calibration/koch/main_follower.json`
