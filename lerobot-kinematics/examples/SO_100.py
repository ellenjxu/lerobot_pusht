from lerobot_kinematics import lerobot_FK, get_robot
import numpy as np

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

def deg_to_rad(degrees):
    return np.radians(degrees)

init_qpos_deg = np.array([-45.3515625, 63.896484375, 92.548828125, 59.326171875, -148.359375, -6.943359375])

init_qpos_rad = deg_to_rad(init_qpos_deg)

pose = lerobot_FK(init_qpos_rad[3:], robot=get_robot())

x, y, z = pose

print(f"Position: X={x:.5f}, Y={y:.5f}, Z={z:.5f}")