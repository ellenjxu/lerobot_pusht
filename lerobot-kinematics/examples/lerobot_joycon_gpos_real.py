import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import threading

# for joycon
from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot2, feetech_arm
from joyconrobotics import JoyconRobotics
import math

np.set_printoptions(linewidth=200)
os.environ["MUJOCO_GL"] = "egl"

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

xml_path = "./examples/scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

robot = get_robot2()

qlimit = [[-2.1, -3.14, -0.1, -2.0, -3.1, -0.1], 
          [2.1,   0.2,   3.14, 1.8,  3.1, 1.0]]
glimit = [[0.125, -0.4,  0.046, -1.5, -1.57, -1.5], 
          [0.380,  0.4,  0.23,  3.1,  1.57,  1.5]]

init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
target_qpos = init_qpos.copy() 
init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)

# init_gpos = np.concatenate((lerobot_FK(init_qpos[1:5], robot=robot),))

target_gpos = init_gpos.copy() 

lock = threading.Lock()
target_gpos_last = init_gpos.copy()

joyconrobotics_right = JoyconRobotics(device="right", 
                                      horizontal_stick_mode='yaw_diff', 
                                      close_y=True, 
                                      limit_dof=True, 
                                      init_gpos=init_gpos, 
                                    #   dof_speed=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
                                      common_rad=False,
                                      lerobot = True)

follower_arm = feetech_arm(driver_port="/dev/lerobot_tty1", calibration_file="examples/main_follower.json" )

t = 0
try:
    # while 1:
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()
            if t ==0 :
                mjdata.qpos[qpos_indices] = init_qpos
                mujoco.mj_step(mjmodel, mjdata)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()
                mjdata.qpos[qpos_indices] = init_qpos   
                mujoco.mj_step(mjmodel, mjdata)    
            t = t + 1
    
            target_pose, gripper_state_r = joyconrobotics_right.get_control()
            
            for i in range(6):
                target_pose[i] = glimit[0][i] if target_pose[i] < glimit[0][i] else (glimit[1][i] if target_pose[i] > glimit[1][i] else target_pose[i])
    
            x_r = target_pose[0] # init_gpos[0] + 
            z_r = target_pose[2] # init_gpos[2] + 
            _, _, _, roll_r, pitch_r, yaw_r = target_pose
            y_r = 0.01
            pitch_r = -pitch_r 
            roll_r = roll_r - math.pi/2 # lerobo末端旋转90度

            right_target_gpos = np.array([x_r, y_r, z_r, roll_r, pitch_r, 0.0])
            fd_qpos_mucojo = mjdata.qpos[qpos_indices][1:5]

            qpos_inv_mujoco, IK_success = lerobot_IK(fd_qpos_mucojo, right_target_gpos, robot=robot)
            
            if IK_success:
                target_qpos = np.concatenate(([yaw_r,], qpos_inv_mujoco[:4], [gripper_state_r,])) # 使用陀螺仪控制yaw

                print("target_qpos:", [f"{x:.3f}" for x in target_qpos])
                
                mjdata.qpos[qpos_indices] = target_qpos
                
                follower_arm.action(target_qpos)
                
                mujoco.mj_step(mjmodel, mjdata)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()
                
                target_gpos_last = right_target_gpos.copy() 
            else:
                right_target_gpos = target_gpos_last.copy()
                joyconrobotics_right.set_position = right_target_gpos[0:3]
            
            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            time.sleep(0.01)
except KeyboardInterrupt:
    print("用户中断了模拟。")
    follower_arm.disconnect()
