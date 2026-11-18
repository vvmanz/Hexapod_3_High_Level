# Hexapod_3_High_Level

**High-level control of a six-legged RHex-style robot in ROS 2 and Gazebo: five gaits, CPG and phase-oscillator controllers, autonomous waypoint navigation, and tools to analyse reinforcement-learning episodes.**

English | [Русский](README.ru.md)

![a logged episode: the real path against the waypoints of a clover-shaped reference](rl/analyze/output_plots/episode_000002.png)

*A logged episode: the real path (blue) follows the waypoints of a clover-shaped reference (orange); 28 of 28 waypoints
reached in 2 326 steps.*

The robot walks on six rotating legs, one motor per leg, so a gait is a rule for when each motor turns and how fast. This
repository is the top level of the Hexapod stack: Python ROS 2 nodes that turn a keyboard command, a phase oscillator or a
waypoint list into target angles for the six motors, log what the robot did, and a set of scripts that evaluate episodes
of a learned policy. The lower levels are the [middle-level coordinator](https://github.com/<your-account>/Hexapod_2_Middle_Level)
and the [motor boards](https://github.com/<your-account>/Hexapod_1_Low_Level); here the robot is the Gazebo model
`hexapod_0`.

## Features

* **Five gaits**: tripod (legs `0, 3, 4` against `1, 2, 5`), diagonals, four-two, waves and quadro, each as a keyboard
  controller: `W/A/S/D` to walk and turn, `Space` or `X` to stop. A step is a small state machine that waits until the
  active leg group has turned far enough before it releases the next one.
* **CPG controller** (`cpg.py`): six coupled phase oscillators with the two tripods locked in anti-phase, a heading
  regulator that steers the robot along a circle, 50 Hz loop.
* **Phase-oscillator controller** (`phase_oscillator.py`): the same idea with selectable reference paths (circle,
  ellipse, figure-eight, Lissajous, zigzag, square, rectangle) and a PD heading regulator.
* **Differential-drive controller** (`diff_drive.py`): the legs on each side are driven like the wheels of a
  differential drive to follow a circle, with a speed limit of 13 rad/s, 20 Hz loop.
* **Autonomous navigator** (`autonomous_controller.py`): builds a square, circle, polygon or figure-eight waypoint
  trajectory, reads the odometry and publishes one of four discrete commands (`w`, `a`, `s`, `d`) on
  `/hexapod_0/command`; it knows nothing about the gait that executes them.
* **Logging** (`hexapod_logger.py`): CSV of angle, speed and current of every motor plus `x, y, φ` and their increments,
  at a fixed rate; the logs of the gait experiments are in `hexapod_movement/logs/`.
* **RL analysis** (`rl/analyze/`): 498 logged episodes with their metrics, plots of training curves and trajectories, and
  a trajectory animation.

### ROS 2 interface

| topic | type | direction | what |
|---|---|---|---|
| `/hexapod_0/set_position_targets` | `std_msgs/Float64MultiArray` | controller → robot | target for the six motors |
| `/hexapod_0/joint_states` | `sensor_msgs/JointState` | robot → controller | joint positions and velocities |
| `/hexapod_0/dataset` | `std_msgs/Float64MultiArray` | robot → controller | motor telemetry: angle, speed, current |
| `/hexapod_0/odom` | `nav_msgs/Odometry` | robot → controller | pose for navigation and logging |
| `/hexapod_0/command` | `std_msgs/String` | navigator → gait | `w`, `a`, `s`, `d` |

### What the logged RL episodes show

Each episode JSON holds the reward, the success flag, the stopping reason, RMS distance to the goal and to the path,
heading error, roll and pitch, path length and how many waypoints were reached. Over the 498 episodes in the repository,
235 reached the end of their trajectory and 263 timed out; the mean completion ratio is 0.80.

| path | episodes | completed | | path | episodes | completed |
|---|---|---|---|---|---|---|
| line | 25 | 25 | | straight | 33 | 32 |
| curve | 31 | 31 | | ellipse | 18 | 18 |
| square | 25 | 24 | | figure8 | 19 | 17 |
| circle | 31 | 27 | | clover | 77 | 30 |
| spiral | 81 | 29 | | zigzag | 74 | 2 |
| lissajous | 84 | 0 | | | | |

> The episodes come from several training runs of a policy whose training environment is not part of this repository, so
> read the table as a picture of which paths were hard, not as a benchmark.

## Requirements

* Ubuntu 24.04, ROS 2 Jazzy (`rclpy`, `std_msgs`, `sensor_msgs`, `nav_msgs`);
* a running simulation of the hexapod under the namespace `hexapod_0` (Gazebo model plus the ROS bridge) that publishes
  the topics above and accepts `set_position_targets`;
* for the analysis scripts: Python ≥ 3.10 with pandas and matplotlib.

## Quick start

```bash
git clone https://github.com/<your-account>/Hexapod_3_High_Level.git
cd Hexapod_3_High_Level/hexapod_movement
source /opt/ros/jazzy/setup.bash

python3 0_keyboard_tripod.py          # walk with W/A/S/D, stop with Space or X
python3 0_keyboard_waves.py           # the same with another gait
python3 cpg.py                        # follow a circle with coupled oscillators
python3 phase_oscillator.py           # follow the path chosen by `self.mode` in the file
python3 autonomous_controller.py      # waypoint navigation (square by default)
```

The scripts are run from `hexapod_movement/` because they import `hexapod_logger`. The gain, step and threshold
constants are at the top of each controller.

### Analysing episodes

```bash
cd rl/analyze
python3 plot_successful_trajectories.py --help
python3 plot_partial_trajectories.py --help
python3 plot_training_metrics.py <training_metrics.csv>      # training curves, split into sessions
python3 traj_anim.py --help                                  # trajectory animation
```

## Repository layout

```
hexapod_movement/
  0_keyboard_<gait>.py     keyboard controllers with logging: tripod, diagonals, four_two, waves, quadro (+ tripod_z_fall)
  <gait>.py, new_<gait>.py other versions of the same gaits
  cpg.py                   coupled phase oscillators, circle following
  phase_oscillator.py      gait generator with selectable reference paths
  diff_drive.py            differential-drive style circle following
  autonomous_controller.py waypoint navigator → w/a/s/d commands
  hexapod_logger.py        CSV logging of motors and pose
  logs/                    recorded runs per gait
rl/analyze/
  episodes/                498 episode summaries (JSON) and one step log (CSV)
  output_plots/            trajectory plots of selected episodes
  plot_*.py, traj_anim.py  metrics, trajectories, animation
```

## Part of the Hexapod stack

| level | repository | runs on |
|---|---|---|
| 1 — motors | [Hexapod_1_Low_Level](https://github.com/<your-account>/Hexapod_1_Low_Level) | STM32F103C8T6 × 3 |
| 2 — coordination | [Hexapod_2_Middle_Level](https://github.com/<your-account>/Hexapod_2_Middle_Level) | STM32F446RE |
| 3 — gaits and navigation | **Hexapod_3_High_Level** | ROS 2 / Gazebo |
