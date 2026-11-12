# ROS 2 Humble Integration

This module provides ROS 2 telemetry, task control, full Panda robot visualization, and rosbag2 recording for the existing SAC-based Franka Panda pick-and-place simulation.

## System Requirements

- Windows host with the project's Python simulation environment
- WSL2 Ubuntu with ROS 2 Humble
- RViz2 and `robot_state_publisher`
- `ros-humble-moveit-resources-panda-description`
- Built `panda_ros2_bridge` package
- Original SAC checkpoint (not included in this repository)

## ROS 2 Interfaces

| Interface | Type | Purpose |
|---|---|---|
| `/joint_states` | `sensor_msgs/msg/JointState` | Seven Panda arm joints |
| `/panda/urdf_joint_states` | `sensor_msgs/msg/JointState` | URDF-compatible arm and gripper joints |
| `/panda/ee_pose` | Pose telemetry | End-effector state |
| `/panda/cube_pose` | Pose telemetry | Cube state |
| `/panda/target_pose` | Pose telemetry | Placement target |
| `/panda/markers` | `visualization_msgs/msg/MarkerArray` | Cube, target, and trajectory |
| `/panda/task_status` | Task telemetry | Episode status |
| `/panda/sac_action` | Action telemetry | SAC action |
| `/clock` | `rosgraph_msgs/msg/Clock` | Simulation time |
| `/tf`, `/tf_static` | `tf2_msgs/msg/TFMessage` | Robot transforms |
| `/panda/reset_episode` | `std_srvs/srv/Trigger` | Trigger an episode |

## Live Visualization

### Terminal A — Panda Robot Model

```bash
source /opt/ros/humble/setup.bash
source ~/panda_ros2_ws/install/setup.bash
ros2 launch panda_ros2_bridge panda_robot_model.launch.py
```

### Terminal B — Windows Simulation

Run from the repository root in PowerShell:

```powershell
$WslIP = (((wsl.exe hostname -I) | Out-String).Trim() -split '\s+')[0]

python ros2_demo_runner.py `
  --model "PATH_TO_ORIGINAL_SAC_CHECKPOINT.zip" `
  --controller dls `
  --seed 42000 `
  --wsl-host $WslIP `
  --port 15555 `
  --viewer `
  --realtime
```

Replace `dls` with `osc` to visualize the OSC controller.

### Terminal F — ROS 2 Telemetry Bridge

```bash
source /opt/ros/humble/setup.bash
source ~/panda_ros2_ws/install/setup.bash
ros2 launch panda_ros2_bridge panda_monitor.launch.py
```

### Terminal D — RViz2

```bash
source /opt/ros/humble/setup.bash
source ~/panda_ros2_ws/install/setup.bash
rviz2
```

Configure:

- Fixed Frame: `world`
- RobotModel Description Topic: `/robot_description`
- MarkerArray Topic: `/panda/markers`
- Enable TF display

Set simulation time:

```bash
ros2 param set /rviz use_sim_time true
```

### Terminal C — Trigger an Episode

```bash
ros2 service call /panda/reset_episode std_srvs/srv/Trigger '{}'
```

## Recording

Start recording before triggering the episode:

```bash
ros2 bag record --use-sim-time \
  -o ~/panda_ros_bags/panda_dls_full_robot_01 \
  /clock \
  /joint_states \
  /panda/gripper_opening \
  /panda/ee_pose \
  /panda/cube_pose \
  /panda/target_pose \
  /panda/markers \
  /panda/task_status \
  /panda/sac_action
```

Trigger the episode from Terminal C. After completion, stop recording with `Ctrl+C` and wait for rosbag2 to exit.

Verify:

```bash
ros2 bag info ~/panda_ros_bags/panda_dls_full_robot_01
```

## Offline Playback

Stop the Windows simulation and live telemetry bridge before playback. Keep the Panda robot model and RViz2 running.

Make sure RViz2 uses simulation time:

```bash
ros2 param set /rviz use_sim_time true
```

Replay DLS:

```bash
ros2 bag play ~/panda_ros_bags/panda_dls_full_robot_01 -r 0.5
```

Replay OSC:

```bash
ros2 bag play ~/panda_ros_bags/panda_osc_full_robot_01 -r 0.5
```

The `-r 0.5` argument replays the recording at half speed.

## Troubleshooting

**Cube and trajectory move, but the Panda arm does not**

1. Verify `/joint_states` messages are present.
2. Verify `/panda/urdf_joint_states` contains nine joints.
3. Verify `/tf` is being published.
4. Check that RViz2 and `robot_state_publisher` both use `use_sim_time=true`.
5. Restart playback after correcting simulation-time settings.

**Robot model is missing**

Verify that `panda_robot_model.launch.py` is running and RobotModel is using `/robot_description`.

## Experimental Scope

The ROS 2 bridge does not modify the frozen SAC policy, the original 46D observation and 4D action spaces, or the DLS/OSC benchmark methodology. The original controllers remain inside the simulation process. ROS 2 is used for telemetry, episode triggering, visualization, and recording.