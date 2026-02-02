# Autonomous Robot Navigation and Control in ROS 2

## Project Overview

This repository contains the complete software solution for a final robotics course project. The primary objective is to implement a full autonomy stack for a mobile robot within a simulated environment. The project leverages the Robot Operating System (ROS 2 Jazzy) and the Gazebo simulator to achieve localization, path planning, and motion control.

The system is broken down into three core modules, each addressing a fundamental challenge in mobile robotics:
1.  **Localization:** Determining the robot's position and orientation within a known map using the AMCL algorithm.
2.  **Path Planning:** Generating an optimal, collision-free path from the robot's current location to a specified goal using the A* search algorithm.
3.  **Path Following:** Executing the planned path by training a Reinforcement Learning agent (DDPG) to generate continuous velocity commands.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Core Features](#core-features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Technical Implementation Details](#technical-implementation-details)
  - [Module 1: Localization (AMCL)](#module-1-localization-amcl)
  - [Module 2: Path Planning (A*)](#module-2-path-planning-a)
  - [Module 3: Path Following (DDPG)](#module-3-path-following-ddpg)
- [Execution Guide](#execution-guide)

---

## System Architecture

The project follows a modular, data-driven architecture where ROS 2 nodes communicate via topics and services. The data flows as follows:

1.  **Gazebo Simulator:** Provides the robot model, the world, and simulates raw sensor data (Lidar scans, odometry).
2.  **Localization Module:** The `amcl` node consumes sensor data (`/scan`, `/odom`, `/tf`) and the static map (`/map`) to compute the robot's most likely pose. It publishes this refined pose to the `/amcl_pose` topic and provides the crucial `map` -> `odom` transform.
3.  **Path Planning Module:** The `astar_planner` service server waits for a goal. Upon receiving one, it uses the current pose from `/amcl_pose` and the static map from `/map` to compute a path.
4.  **Path Publication:** The computed series of waypoints is published as a `nav_msgs/Path` message on the `/plan` topic.
5.  **Control Module:** The `rl_line_follower` node subscribes to the `/plan` topic. It uses the path and the robot's current odometry (`/odom`) to calculate state information (e.g., errors) and feeds it to the DDPG agent. The agent's action (a velocity command) is then published to the `/cmd_vel` topic, moving the robot in the simulation.

---

## Core Features

-   **Adaptive Monte Carlo Localization (AMCL):** Robust, real-time robot pose tracking in a static map.
-   **A\* Global Path Planner:** An efficient, grid-based service that generates optimal, collision-free paths.
-   **Deep Deterministic Policy Gradient (DDPG):** A Reinforcement Learning agent for continuous control, enabling smooth path-following behavior.
-   **Modular ROS 2 Packages:** A clean and organized workspace with separate packages for localization, planning, and control, promoting code reusability and maintainability.
-   **Virtual Environment for Python:** Utilizes a Python `venv` to manage dependencies like `torch`, ensuring no conflicts with system-level packages.

---

## Project Structure

The workspace is organized into several ROS 2 packages within the `src` directory.

```
exam_ws/
├── src/
│   ├── robotic_course/         # Provided simulation assets (robot, world)
│   ├── exam_localization/      # Package for AMCL and map server
│   │   ├── launch/
│   │   ├── maps/
│   │   └── config/
│   ├── exam_path_planning/     # Package for the A* planner service
│   │   └── exam_path_planning/
│   └── exam_rl/                # Package for the DDPG controller
│       └── exam_rl/
├── build/
├── install/
├── log/
├── README.md
└── requirements.txt
```

---

## Setup and Installation

Follow these steps to set up the project environment.

**1. Prerequisites:**
-   **OS:** Ubuntu 24.04 LTS
-   **ROS:** ROS 2 Jazzy Jalisco

**2. System Dependencies:**
Install the necessary ROS 2 packages for navigation and simulation.
```bash
sudo apt update
sudo apt install ros-2-jazzy-navigation2 ros-2-jazzy-nav2-bringup ros-2-jazzy-gazebo-ros-pkgs
```

**3. Python Environment:**
This project uses a Python virtual environment to manage dependencies.
```bash
# Navigate to the project root
cd ~/exam_ws

# Create and activate the virtual environment
python3 -m venv my_ros_venv
source my_ros_venv/bin/activate

# Install required Python packages
pip install -r requirements.txt
```

**4. Build the ROS 2 Workspace:**
Compile all custom packages. Ensure the virtual environment is active before building.
```bash
# Ensure venv is active
source my_ros_venv/bin/activate

# Build the workspace
cd ~/exam_ws
colcon build
```

---

## Technical Implementation Details

### Module 1: Localization (AMCL)

-   **Objective:** To accurately estimate the robot's pose (`x, y, θ`) within the provided `depot.pgm` map.
-   **Algorithm:** Adaptive Monte Carlo Localization (AMCL) is a probabilistic algorithm that uses a particle filter. It distributes a cloud of particles (potential poses) across the map. Each particle is weighted based on how well its hypothetical sensor readings match the actual sensor readings from the robot's Lidar. Over time, particles that are inconsistent with the sensor data are eliminated, and the cloud converges on the true pose of the robot.
-   **Implementation:**
    -   The `exam_localization` package contains the launch file `localization.launch.py`.
    -   This file starts the `map_server` node to load the static map and the `amcl` node.
    -   All key parameters for AMCL (e.g., number of particles, sensor model parameters) are configured in `config/nav2_params.yaml`. This allows for easy tuning without modifying code.
    -   A `lifecycle_manager` is used to start and manage the nodes in the correct order.

### Module 2: Path Planning (A*)

-   **Objective:** To create a ROS 2 service that finds the shortest collision-free path from the robot's current position to a given goal.
-   **Algorithm:** The A* (A-Star) algorithm is used. It operates on a grid representation of the map, where each cell is a node in a graph. A* is optimal because it balances the actual cost from the start node (`g(n)`) with an estimated cost to the goal node (`h(n)`, the heuristic). The search prioritizes nodes with the lowest `f(n) = g(n) + h(n)`. In this implementation, the grid is the map's pixel grid, `g(n)` is the number of steps from the start, and `h(n)` is the Euclidean distance to the goal.
-   **Implementation:**
    -   The `astar_planner.py` script in the `exam_path_planning` package implements the service server.
    -   It listens on the `/plan_path` service name, using the standard `nav_msgs/srv/GetPlan` message type.
    -   Upon receiving a request, it fetches the latest robot pose from `/amcl_pose` and converts both start and goal world coordinates to grid cell indices.
    -   The A* algorithm traverses the grid, avoiding cells marked as obstacles (pixel value > 65) in the `OccupancyGrid` message.
    -   The resulting path (a sequence of grid cells) is converted back to world coordinates and published as a `nav_msgs/msg/Path` on the `/plan` topic for visualization and use by the controller.

### Module 3: Path Following (DDPG)

-   **Objective:** To develop a controller that generates continuous velocity commands (`Twist` messages) to make the robot accurately follow the path generated by the A* planner.
-   **Algorithm:** Deep Deterministic Policy Gradient (DDPG) is an advanced Reinforcement Learning algorithm ideal for this task. As an Actor-Critic, off-policy algorithm, it is data-efficient and designed for continuous action spaces.
    -   **Actor Network:** Learns a policy that maps states directly to actions (i.e., given the current errors, it outputs the best linear and angular velocities).
    -   **Critic Network:** Learns to evaluate the action proposed by the Actor by estimating the expected future reward (the Q-value). The Critic's feedback is used to train the Actor.
-   **Custom RL Environment:** The core of this module is the custom environment defined within `rl_line_follower.py`:
    -   **State Space:** The "state" observed by the agent is a vector containing four critical pieces of information:
        1.  `Cross-Track Error (CTE)`: The perpendicular distance from the robot to the closest segment of the path.
        2.  `Heading Error (HE)`: The angular difference between the robot's current orientation and the path's direction.
        3.  Current linear velocity.
        4.  Current angular velocity.
    -   **Action Space:** The agent's output is a continuous 2D vector representing `[linear_velocity, angular_velocity]`.
    -   **Reward Function:** The reward function is carefully engineered to encourage desired behavior: `reward = 1.0 - (5.0 * |CTE| + 2.0 * |HE|)`. This heavily penalizes deviation from the path. A large penalty is applied if the robot strays too far, and a large positive reward is granted upon successfully reaching the goal.
    -   **Training Loop:** The node subscribes to the `/plan` and `/odom` topics. In a loop, it calculates the current state, passes it to the DDPG agent, receives an action, publishes it as a `/cmd_vel` command, and then uses the resulting new state and reward to update the agent's networks via the `Replay Buffer` and `optimize` methods.

---

## Execution Guide

To run the full project, open **5 separate terminals**. In **each terminal**, first source the necessary setup files:

```bash
# Activate the Python virtual environment
source ~/my_ros_venv/bin/activate

# Source the ROS 2 workspace
source ~/exam_ws/install/setup.bash
```

Then, execute the following commands in their respective terminals.

**Terminal 1: Launch Gazebo Simulation**
```bash
# This command spawns the robot in the Gazebo world.
# Note: The exact launch file name may vary based on the provided course repository.
ros2 launch robotic_course_description spawn_robot.launch.py
```

**Terminal 2: Launch Localization and RViz**
```bash
ros2 launch exam_localization localization.launch.py
```
> **Action Required:** Once RViz opens, use the "2D Pose Estimate" tool from the top toolbar to provide an initial pose estimate for the robot on the map. This helps the AMCL particles to converge quickly.

**Terminal 3: Run the A* Path Planner Service**
```bash
ros2 run exam_path_planning astar_planner
```

**Terminal 4: Call the Planner Service to Generate a Path**
```bash
# Choose a valid, obstacle-free destination on the map.
ros2 service call /plan_path nav_msgs/srv/GetPlan "{goal: {header: {frame_id: 'map'}, pose: {position: {x: 3.5, y: -1.0, z: 0.0}, orientation: {w: 1.0}}}}"
```
> **Observation:** A green line representing the computed path should appear in RViz, connecting the robot to the goal.

**Terminal 5: Run the RL Path Following Controller**
```bash
ros2 run exam_rl rl_line_follower
```
> **Observation:** The robot will start moving in Gazebo, attempting to follow the green path. The terminal will display training logs, including episode numbers and accumulated rewards.
```

---
