#!/usr/bin/env python

import logging
import time
from os.path import dirname, abspath

from nicomotion import Motion

# Set logging level
logging.basicConfig(level=logging.WARNING)

vrep = False
nico_root = dirname(abspath(__file__)) + "/../../.."

if vrep:
    vrepConfig = Motion.Motion.vrepRemoteConfig()
    vrepConfig["vrep_scene"] = nico_root + "/v-rep/NICO-seated.ttt"
    robot = Motion.Motion(
        nico_root + "/json/nico_humanoid_vrep.json", vrep=True, vrepConfig=vrepConfig
    )
else:
    robot = Motion.Motion(nico_root + "/json/nico_humanoid_upper.json", vrep=False)

# Manually patch _rightHand if not initialized
if not hasattr(robot, "_rightHand"):
    try:
        # Directly assign right hand motors based on your JSON config
        motor_names = ["r_wrist_z", "r_wrist_x", "r_indexfingers_x", "r_virtualhand_x"]
        robot._rightHand = [robot._motors[name] for name in motor_names if name in robot._motors]
        if not robot._rightHand:
            raise RuntimeError("Right hand motors not found in robot.")
    except Exception as e:
        print("Error patching _rightHand:", e)

# Perform movement
position = 20
for i in range(10):
    # Right arm movement
    robot.setAngle("r_arm_x", -80 + position, 0.05)
    robot.setAngle("r_elbow_y", -40 + position, 0.05)

    # Head movement
    if i % 2 == 0:
        robot.setAngle("head_z", -position if i % 4 == 0 else position, 0.05)
    else:
        robot.setAngle("head_y", position if i % 4 == 1 else -position, 0.05)

    # Right hand open/close
    if position > 0:
        print("Closing hand")
        robot.closeHand("RHand")
    else:
        print("Opening hand")
        robot.openHand("RHand")

    position *= -1
    time.sleep(2)

# Move to safe position
print("Moving to safe position...")
robot.toSafePosition()
time.sleep(7)

# Cleanup
if vrep:
    robot.stopSimulation()
else:
    robot.disableTorqueAll()

del robot

