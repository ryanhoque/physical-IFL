# Fleet-DAgger Physical Experiment Code (ROS2)
------------
Code for physical experiments in the following papers: 

R. Hoque, L.Y. Chen, S. Sharma, K. Dharmarajan, B. Thananjeyan, P. Abbeel, K. Goldberg. Fleet-DAgger: Interactive Robot Fleet Learning with Scalable Human Supervision. Conference on Robot Learning (CoRL), 2022.

K. Chen, R. Hoque, K. Dharmarajan, E. LLontop, S. Adebola, J. Ichnowski, J. Kubiatowicz, K. Goldberg. FogROS2-SCG: A ROS2 Cloud Robotics Platform for Secure Global Connectivity. In submission, 2023.

The task is an image-based block pushing task on 4 robot arms belonging to 2 ABB YuMis with simultaneous execution of all arms. While the original code was implemented with SSH and SFTP, this version is re-implemented in ROS2 for the experiments in [FogROS2-SCG](https://sites.google.com/view/fogros2-sgc). FogROS2-SCG can connect disjoint ROS2 networks (e.g., local networks for robot 1, robot 2, and centralized compute node).

## Quickstart

Install requried Python packages:
```
pip install gym==0.21.0 torch==1.11.0 dotmap==1.3.30 imgaug==0.4.0
git clone --recurse-submodules https://github.com/BerkeleyAutomation/yumirws.git
pip install -e yumirws
```

Assuming you have installed ROS2 (tested on the Humble distribution but may work with others), move this folder to the `src` directory of your ROS2 workspace and build:
```
source /opt/ros/humble/setup.bash
cd [your ROS2 workspace]
colcon build
source install/setup.bash
```

If this is the Robot 1 node ("Etch"), run:
```
ros2 run block_pusher robot_executor
```

If this is the Robot 2 node ("BWW"), run:
```
ros2 run block_pusher robot_executor --robot BWW
```

If this is the Server node, run:
```
ros2 run block_pusher cloud_executor @src/physical-IFL/scripts/args_demo.txt
```

Note that you will likely need to calibrate the workspace parameters and image crop parameters in `block_pusher/robot_executor.py`.