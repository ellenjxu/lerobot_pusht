
# LeRobot-Kinematics: Simple and Accurate Forward and Inverse Kinematics Examples for the Lerobot SO100 ARM

## Declaration

This repository is a fork of the following projects [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python).

## A. Installation (Ubuntu 20.04)

```bash
  git clone https://github.com/box2ai-robotics/lerobot-kinematics.git
  cd lerobot-kinematics
  pip install -e .
```

## B. Examples in simulation

We recommended to click on the terminal window with the mouse after startup and then enter the keys, to avoid that the keys in the departure [mujoco](https://github.com/google-deepmind/mujoco) change the configuration of the scene.

```shell
pip install mujoco==3.2.5
```

#### (1) qpos control

Example of joint angle control, when opened the Mucojo visualization will appear and you can use the keyboard to control the corresponding angle change of the robot arm.

```shell
python examples/lerobot_keycon_qpos.py
```

- ``1, 2, 3, 4, 5, 6`` Increase the angle.

- ``q, w, e, r, t, y`` Decrease Angle.

Press and hold '0' to return to position


#### (2) gpos Control

Example of Gripper Posture (gpos) control, where you can use the keyboard to control the end-posture changes of Lerobot in mucojo.

```shell
python examples/lerobot_keycon_gpos.py
```

| Key | Action +            | Key | Action -            |
|-----|---------------------|-----|---------------------|
| `w` | Move Forward        | `s` | Move Backward       |
| `a` | Move Right          | `d` | Move Left           |
| `r` | Move Up             | `f` | Move Down           |
| `e` | Roll +              | `q` | Roll -              |
| `t` | Pitch +             | `g` | Pitch -             |
| `z` | Gripper Open        | `c` | Gripper Close       |

Press and hold '0' to return to position


#### (3) Joycon Control

This is an example of using joycon to control Lerobot in mucojo, if you want to use it, please install [joycon-robotics
](https://github.com/box2ai-robotics/joycon-robotics) repository first!

```shell
python examples/lerobot_joycon_gpos.py
```

<!-- #### (4) Genesis IK Control

Example of Gripper Posture (gpos) control based on the Genesis positive inverse kinematics library.

First, if you want try this, you need install the genesis repo:
```shell
pip install genesis-world
```

```shell
python examples/lerobot_genesis.py
``` -->

If this repository was helpful to you, please give us a little star and have a great time! ⭐ ⭐ ⭐ ⭐ ⭐

## C. Examples in Real

#### (1) Keyboard Control in Real

Example of gripper posture (gpos) control, where you can use the keyboard to control the Lerobot's end posture changes in mucojo while going from simulation to physical control of a real Lerobot arm.

```shell
python examples/lerobot_keycon_gpos_real.py
```

If you're interested in this, you can try using the keyboard to collect data.

#### (2) Joycon Control in Real

```shell
python examples/lerobot_joycon_gpos_real.py
```
