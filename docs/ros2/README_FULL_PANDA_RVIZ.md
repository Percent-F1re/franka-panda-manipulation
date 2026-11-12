# ROS 2 Humble：RViz2 完整 Panda 机器人模型（增量补丁）

目标：**不改变 Windows robosuite/SAC/DLS/OSC 控制和原始 Benchmark**，只在 WSL2 ROS 2 中增加 Panda URDF 的机器人模型显示，以及夹爪状态支持。适用于前一个 `ros2_wsl_panda_bridge_patch.zip` 已构建并成功录制 rosbag2 的场景。

## 结构与内容

- Windows `ros2_demo_runner.py` 不修改。它在原有 JSON/UDP 快照中已经发送 `gripper_opening_m`。
- WSL `bridge_node.py` 新增 `/panda/gripper_opening`，类型 `std_msgs/msg/Float32`；旧 ROS2 topic 全部保留。
- 新增 `joint_adapter_node`，把当前 `/joint_states`（MuJoCo 名字 `joint1`..`joint7`）映射到 `/panda/urdf_joint_states`（URDF 名字 `panda_joint1`..`panda_joint7` + 两个夹爪关节）；不覆盖原始 Topic。
- 新增 `panda_robot_model.launch.py`：运行 `robot_state_publisher`、关节适配器、`world -> viz_panda_link0` 固定 TF。URDF、DAE mesh 从 `moveit_resources_panda_description` 安装包读取，无需拷贝上游资产。
- URDF 关节名称不变，连杆名（TF）增加 `viz_` 前缀。避免与旧 bridge 中的 `world -> panda_hand` TF 冲突。RViz2 根据 `/robot_description` 显示 `viz_panda_*` 连杆。

注意：使用的 Panda URDF 与 MuJoCo Menagerie Panda 来源相近，但不是原 MJCF 的直接导出。需验证视觉连杆与 MuJoCo 手部位姿的一致性；此功能仅作监控和展示，不参与物理计算。

## 一、Windows PowerShell：把 ZIP 解压到原仓库

假设已把 `panda_rviz_full_robot_patch.zip` 放在 Windows 下载文件夹。

```powershell
$Repo = 'D:\github\franka-panda-manipulation-v3-publish'
$Zip  = "$env:USERPROFILE\Downloads\panda_rviz_full_robot_patch.zip"
Expand-Archive -LiteralPath $Zip -DestinationPath $Repo -Force
Test-Path "$Repo\ros2_ws\src\panda_ros2_bridge\launch\panda_robot_model.launch.py"
```

这里只覆盖之前 ROS2 功能包的 `setup.py`、`package.xml` 和 `bridge_node.py`，并新增文件；**不修改** Windows Demo Runner 或环境/模型/基准评估源码。如果原先修改过这三个文件，先备份或用 git diff 比对后合并。

## 二、Ubuntu / WSL2：安装标准 Panda URDF 包并重新编译

在 WSL2 Ubuntu 新开终端，使用你已经安装的 Humble：

```bash
source /opt/ros/humble/setup.bash
sudo apt update
sudo apt install -y \
  ros-humble-moveit-resources-panda-description \
  ros-humble-robot-state-publisher \
  ros-humble-tf2-ros

ros2 pkg prefix moveit_resources_panda_description
ls "$(ros2 pkg prefix moveit_resources_panda_description)/share/moveit_resources_panda_description/urdf/panda.urdf"
```

如果 `ls` 找不到 URDF，查找：

```bash
find "$(ros2 pkg prefix moveit_resources_panda_description)/share" -name panda.urdf
```

同步你在 Windows 项目中的 ROS2 包到 WSL 工作区：

```bash
cp -a /mnt/d/github/franka-panda-manipulation-v3-publish/ros2_ws/. ~/panda_ros2_ws/
cd ~/panda_ros2_ws
colcon build --symlink-install --packages-select panda_ros2_bridge
source install/setup.bash
ros2 pkg executables panda_ros2_bridge
```

除旧的 `bridge_node` 和 `monitor_node` 外，还应显示：

```text
panda_ros2_bridge joint_adapter_node
```

**切勿安装或启动 `joint_state_publisher_gui` 作为真实关节状态来源**：已有 MuJoCo Bridge 提供实际关节状态，启动第二个 `/joint_states` 发布者会制造数据冲突。

## 三、最快验证：用旧 DLS rosbag2 回放完整 Panda

此步骤不需要启动 Windows Demo Runner，也不需要原 ROS2 Bridge。

**Ubuntu 终端 A**，先确保旧 Bridge 已经 Ctrl+C 停止且旧 bag 播放也已经停止：

```bash
source /opt/ros/humble/setup.bash
source ~/panda_ros2_ws/install/setup.bash
ros2 launch panda_ros2_bridge panda_robot_model.launch.py
```

这一步只启动关节适配器、URDF 的 robot_state_publisher 和一个固定 TF；不启动 UDP/Service。

**Ubuntu 终端 D**，保留原本 RViz2：

- `Fixed Frame = world`
- 在终端中执行 `ros2 param set /rviz2 use_sim_time true`（如果 RViz2 节点不叫 `/rviz2`，先 `ros2 node list` 查看）
- 点击 `Add` -> `RobotModel`。在 RobotModel 的显示设置中选择 **Description Source = Topic**、**Description Topic = `/robot_description`**（Humble 的版本与界面选项可能略有差异）
- 保留 `/panda/markers` 的 MarkerArray；可将 TF 中的 `Show Names` 关闭以减少遮挡

**Ubuntu 终端 E**，播放之前已经成功记录的 DLS bag：

```bash
source /opt/ros/humble/setup.bash
ros2 bag play ~/panda_ros_bags/panda_dls_seed42000
```

旧 bag 已录制 `/clock`，因此本例不额外加 `--clock`。回放时应看到 Panda 七关节和原来的方块/末端轨迹。**旧包未录制夹爪开口，所以两根手指将保持默认的总开口 0.04 米**。这不代表夹爪状态被成功回放；下一次录制增加 `/panda/gripper_opening` 才可同步。

回放期间，另一 Ubuntu 终端确认：

```bash
ros2 topic echo /panda/urdf_joint_states --once
ros2 topic info /robot_description
ros2 run tf2_ros tf2_echo world viz_panda_hand
```

`/panda/urdf_joint_states` 应有 9 个关节名称（七个臂关节 + 两根手指）。

## 四、实时显示完整 Panda：DLS/OSC 都可

1. 先停止 bag 回放（终端 E Ctrl+C）。
2. 原来的终端 A 或新终端启动原 Bridge：

```bash
source /opt/ros/humble/setup.bash
source ~/panda_ros2_ws/install/setup.bash
ros2 launch panda_ros2_bridge panda_monitor.launch.py
```

3. **保持第三步的 `panda_robot_model.launch.py` 运行**（若已 Ctrl+C，重新运行），不需要启动第二个模型发布器。
4. Windows PowerShell 启动原来的 `ros2_demo_runner.py`，参数 `--controller dls` 或 `--controller osc`，保持之前已验证的 `--wsl-host $WslIP`、`--port 15555` 等设置。可附加 `--viewer` 在 Windows 查看 MuJoCo 真正的物理模型。
5. Ubuntu 终端 C 调用重置服务：

```bash
ros2 service call /panda/reset_episode std_srvs/srv/Trigger '{}'
```

此时 RViz2 应出现彩色完整 Panda 机械臂、方块和末端轨迹，并随着动作更新，包括夹爪两指。

## 五、重新录制可回放完整 Panda 的 OSC bag

录制时新加 `/panda/gripper_opening`，保留原始 7D `/joint_states`。由于回放期间已经运行 `robot_state_publisher`，为了避免两个发布者重复向 `/tf` 写入同名可视化 TF，下面的**新录制默认不记录 `/tf`、`/tf_static`**；回放时由模型 Launch 重新生成。方块、末端和轨迹的 RViz Marker 仍然能从 `/panda/markers` 直接显示（其坐标系是 `world`）。

```bash
source /opt/ros/humble/setup.bash
mkdir -p ~/panda_ros_bags
ros2 bag record --use-sim-time \
  -o ~/panda_ros_bags/panda_osc_full_robot_01 \
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

保持 ROS2 Bridge、Panda Robot Model Launch、Windows Demo Runner 运行；录制开始后再调用 Service 开始抓放。结束后 recorder Ctrl+C。注意：`rosbag2` 不等同于 MP4 视频；需要展示视频时，另用 OBS 录制 RViz2 画面。

离线回放：停止 Windows Runner 与原 Bridge（避免 `/clock` 冲突），**保持 `panda_robot_model.launch.py` + RViz2 运行**，然后：

```bash
ros2 bag play ~/panda_ros_bags/panda_osc_full_robot_01
```

新袋内只有原始关节数据，回放时由 adapter + robot_state_publisher 重新生成完整 Panda 的视觉 TF。可重复回放，不需要 Windows MuJoCo。

## 六、坐标系对齐校验（重要）

完整机器人可视化 URDF 使用 MoveIt 的机器人描述；原仿真为 Menagerie MJCF。因此，即使机械臂能运动，也需要在**实时运行**时检查两份末端位姿是否一致：

```bash
# 终端 1
ros2 run tf2_ros tf2_echo world panda_hand
# 终端 2
ros2 run tf2_ros tf2_echo world viz_panda_hand
```

对比位置（单位米）和四元数。若位置存在固定偏差，可通过模型 Launch 的 base_x/base_y/base_z/base_yaw 参数调整静态基座的可视化位置，例如：

```bash
ros2 launch panda_ros2_bridge panda_robot_model.launch.py \
  base_x:=0.0 base_y:=0.0 base_z:=0.0 base_yaw:=0.0
```

只调整显示端的坐标变换，不会改变 MuJoCo 物理模型。若误差随姿态明显变化，不能简单靠调基座弥补，需排查 URDF/MJCF 的坐标约定及关节映射。

## 七、排查

- **RobotModel 报错 `No transform`**：确保 `/panda/urdf_joint_states` 有 9 关节、model launch 正在运行、RViz `Fixed Frame = world`、`use_sim_time` 为 true。
- **RobotModel 报错 `Could not load resource`**：核对安装 `moveit_resources_panda_description`，且 WSL 下 URDF 的 `package://...` 网格文件存在。
- **RobotModel 不显示**：检查 `Description Source = Topic`、`Description Topic = /robot_description`；用 `ros2 topic info /robot_description` 验证 publisher；必要时删除并重新添加 RobotModel。
- **关节不动但 Marker 会动**：`ros2 topic echo /panda/urdf_joint_states --once`，检查 adapter 是否仍在运行；确认 `ros2 bag play` 实际发布 `/joint_states`。
- **新 bag 没有夹爪动作**：检查实时 `ros2 topic echo /panda/gripper_opening --once`；如果旧 Bridge 未重建，它不会发布该 Topic。
- **正常录制但回放跳帧**：ROS2 bag 回放时尝试 `--rate 0.5`，并确认 `/clock` 没有多处发布。

## 验证边界与模型许可

本补丁完成了 Python 编译 / 纯 Python 关节转换测试，**没有在你的 Ubuntu/WSL2 + RViz2 环境进行端到端图形测试**。原 SAC checkpoint、DLS/OSC、成功判据、评估脚本均不变。

MoveIt 提供的 Panda URDF 和网格来自上游资源包，有其自己的许可条件。本补丁只引用 ROS2 系统安装目录中的官方资源，**不复制 mesh/URDF 文件**；在 GitHub README 中请说明视觉模型来源与与物理模型的区别。
