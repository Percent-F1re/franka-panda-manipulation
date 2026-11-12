# V4 ROS 2 Bridge：Windows robosuite + WSL2 Ubuntu ROS 2

## 设计与研究边界

这是在**原有 robosuite V3 上增加的可选演示层**，不替换原单 SAC、DLS、OSC、奖励、固定种子或统一成功判据。Windows 仍持有原始 Menagerie Panda MjModel/MjData 并执行原来的 `env.step()`；WSL2 仅订阅状态并提供重置演示的 Service。基线 84% / 81% 和 V3 鲁棒性结果不应重新标记为“经 ROS2 重新评估”。

```
Windows Python .venv                          WSL2 Ubuntu / ROS 2
┌──────────────────────────────────┐         ┌────────────────────────────────────┐
│ SAC 46D → 4D action             │         │ panda_ros2_bridge / rclpy         │
│ PandaDualControllerEnv           │ --UDP--> │ /joint_states, /panda/*, /tf       │
│ DLS | OSC → robosuite/MuJoCo     │         │ /clock, /panda/markers            │
│ original model & reward untouched│ <--UDP-- │ /panda/reset_episode (Trigger)     │
└──────────────────────────────────┘         │ monitor Subscriber, rosbag2, RViz2 │
                                           └────────────────────────────────────┘
```

`Service` 使用 UDP 发送**重置请求**；成功响应表示“请求已发送”，**不是**“物理 reset 已完成”。`/panda/task_status` 中的 `reset_applied` 才是已处理确认。服务只针对**演示运行器**，不影响其他基准评估进程。

**确认原始仓库最新 `main` 包含** `envs/panda_dual_controller.py` 和 `legacy_sac_checkpoint.py`。本补丁只新增 `ros2_demo_runner.py` 与 `ros2_ws/`，不覆盖任何旧文件。

## A. Windows：将补丁安装到本地项目（PowerShell）

```powershell
$Repo = 'D:\github\franka-panda-manipulation-v3-publish'
$Zip = Join-Path $env:USERPROFILE 'Downloads\ros2_wsl_panda_bridge_patch.zip'
Expand-Archive -LiteralPath $Zip -DestinationPath $Repo -Force
Test-Path "$Repo\ros2_demo_runner.py"
Test-Path "$Repo\ros2_ws\src\panda_ros2_bridge\package.xml"
```

假设已经有可以跑 DLS/OSC 的原 Windows `.venv`：

```powershell
$Py = 'D:\zhuomian\project\robot_projects\.venv\Scripts\python.exe'
$Checkpoint = 'D:\zhuomian\project\robot_simulation\sac_pickplace\results_pickplace\seed_42\best_model.zip'
Set-Location $Repo
& $Py .\ros2_demo_runner.py --help
```

## B. WSL2 Ubuntu：构建纯 ROS 2 功能包（Ubuntu bash，不是 PowerShell）

```bash
ls /opt/ros
# 按上面列出的实际版本选择：例如安装 Jazzy 则执行下一行
source /opt/ros/jazzy/setup.bash
echo "$ROS_DISTRO"
sudo apt update
sudo apt install -y python3-colcon-common-extensions ros-${ROS_DISTRO}-rosbag2 ros-${ROS_DISTRO}-rviz2
mkdir -p ~/panda_ros2_ws
cp -a /mnt/d/github/franka-panda-manipulation-v3-publish/ros2_ws/. ~/panda_ros2_ws/
cd ~/panda_ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 pkg executables panda_ros2_bridge
```

说明：若你安装的是 Lyrical/Humble，而不是 Jazzy，请将本文所有 `source /opt/ros/jazzy/setup.bash` 替换为自己的实际发行版。依赖包名中的 `${ROS_DISTRO}` 会在正确 source 后自动设置。

请在 WSL 用户的 Linux 家目录构建，避免直接在 `/mnt/d`（Windows NTFS）上 `colcon --symlink-install` 产生权限、链接和性能问题。若后续修改包源码，再重新复制到 WSL 并构建。

## C. 启动 ROS Bridge（WSL 窗口 A）

```bash
source /opt/ros/jazzy/setup.bash       # 将 jazzy 改成你实际安装的发行版
source ~/panda_ros2_ws/install/setup.bash
ros2 launch panda_ros2_bridge panda_monitor.launch.py
```

默认监听 UDP `15555`；使用 `token=local_demo_only`，仅用于本机开发，勿在不受信任网络中开放该 UDP 端口。服务对外开放前应增加身份认证或限制网络访问。

## D. 获取 WSL 的 IP，并启动 Windows robosuite（PowerShell 窗口 B）

```powershell
$Repo = 'D:\github\franka-panda-manipulation-v3-publish'
$Py = 'D:\zhuomian\project\robot_projects\.venv\Scripts\python.exe'
$Checkpoint = 'D:\zhuomian\project\robot_simulation\sac_pickplace\results_pickplace\seed_42\best_model.zip'
$WslIP = (((wsl.exe hostname -I) | Out-String).Trim() -split '\s+')[0]
Write-Host "WSL IP = $WslIP"
Set-Location $Repo
& $Py .\ros2_demo_runner.py --model $Checkpoint --controller dls --seed 42000 `
    --wsl-host $WslIP --port 15555 --viewer --realtime
```

默认处于**待命**状态；不会自动执行 SAC，便于先启动 rosbag2 后再通过 Service 开始。取消 `--viewer` 可以只使用 RViz2 观察。如果使用了 WSL2 `mirrored` 网络模式，`--wsl-host 127.0.0.1` 也可能可用；默认 NAT 优先使用 `wsl hostname -I` 中实际的 IPv4。WSL 地址可能在重启后改变。<https://learn.microsoft.com/windows/wsl/networking>

## E. Topic、Service、TF2（WSL 窗口 C）

```bash
source /opt/ros/jazzy/setup.bash
source ~/panda_ros2_ws/install/setup.bash
ros2 node list
ros2 topic list
ros2 topic echo /panda/task_status
ros2 topic echo /panda/ee_pose --once
ros2 topic echo /joint_states --once
ros2 service list
ros2 service call /panda/reset_episode std_srvs/srv/Trigger '{}'
```

`reset` 后，WSL 窗口 A/C 会显示 `reset_applied`、`approaching`、`lifting`、`transferred`、`success`/`timeout` 等信息；这段信息是**演示状态**，不是本项目额外 1 秒统一物理判据。

ROS 2 发布内容：

| Topic | 消息类型 | 含义 |
|---|---|---|
| `/clock` | `rosgraph_msgs/Clock` | 单调演示时间；每次重置不倒退 |
| `/joint_states` | `sensor_msgs/JointState` | 7 个关节角和速度（不虚构 URDF 名称） |
| `/panda/ee_pose`, `/panda/cube_pose`, `/panda/target_pose` | `geometry_msgs/PoseStamped` | 世界系中的末端、方块和目标位姿 |
| `/panda/observation`, `/panda/sac_action` | `std_msgs/Float32MultiArray` | 46D 观测及 4D SAC 动作，只读 |
| `/tf` | `tf2_msgs/TFMessage` | world→panda_hand / panda_cube / panda_target |
| `/panda/markers` | `visualization_msgs/MarkerArray` | Cube、target、末端轨迹线 |
| `/panda/task_status` | `std_msgs/String` | 任务阶段、episode、reset_applied |

## F. RViz2 可视化（WSL 窗口 D）

```bash
source /opt/ros/jazzy/setup.bash
rviz2
```

在 RViz2 中：
1. `Global Options → Fixed Frame = world`。
2. `Add → By topic → /panda/markers → MarkerArray`：橙色方块、绿色目标、蓝色末端路径。
3. `Add → By topic → /tf → TF`：显示 `panda_hand`、`panda_cube`、`panda_target` 坐标系。
4. 可选择增加 `/panda/ee_pose → Pose` 便于展示末端方向。
5. `ros2 param set /rviz2 use_sim_time true`；若节点名称不同，先用 `ros2 node list` 查看。
6. File → Save Config As，将 `.rviz` 配置保存在自己的 ROS2 目录中，并可提交 GitHub。

本项目**没有提供原始 Menagerie Panda 对应的 URDF**；这一版本只显示 Marker 和 TF2，不能宣称已在 RViz2 中完成完整机器人 URDF 可视化。若 WSLg/OpenGL 导致 RViz2 失败，先通过 `ros2 topic echo` 和 TF2 确认发布功能，再单独排查图形驱动。

## G. rosbag2 记录和回放

WSL 窗口 E，在**第一次 Service 重置前**启动：

```bash
source /opt/ros/jazzy/setup.bash
mkdir -p ~/panda_ros_bags
ros2 bag record --use-sim-time -o ~/panda_ros_bags/panda_dls_seed42000 \
  /clock /joint_states /tf /panda/ee_pose /panda/cube_pose /panda/target_pose \
  /panda/markers /panda/task_status /panda/sac_action
```

然后在 WSL 窗口 C 调用 `ros2 service call /panda/reset_episode std_srvs/srv/Trigger '{}'`。回合结束后在 recorder 终端 `Ctrl+C`。

```bash
ros2 bag info ~/panda_ros_bags/panda_dls_seed42000
```

**回放时先停止 Windows Runner 与 ROS2 Bridge**，避免实时 `/clock` 和回放 `/clock` 冲突：

```bash
# WSL 另开终端，启动 RViz2（Fixed Frame=world，use_sim_time=true）
rviz2
# 假设在另一 WSL 终端：
ros2 bag play ~/panda_ros_bags/panda_dls_seed42000 --clock
```

回放若需要观察 `task_status`，另开终端 `ros2 topic echo /panda/task_status`。为避免覆盖旧数据，第二次录制时换一个 bag 目录名。

## H. 切换 OSC、验证没有改变 V3 结果

停止 Windows demo runner (`Ctrl+C`) 后，将 `--controller dls` 改为 `--controller osc`。**不要在同一回合中途切换执行器。**

项目旧的正式评估脚本 `compare_dls_vs_osc.py` 和 `run_robustness.py` 均不调用 `ros2_demo_runner.py`，结果不依赖 ROS2 是否启动；**此设计保护已有实验，但仍应以 Windows Smoke Test 验证真实联动。**

## I. Github 与简历

在仓库根目录的 `README.md` 增加本文中的 ROS2 示意图和一段 RViz2 实际运行截图/GIF。提交新增文件即可，不要改写 84%/81% 的实验结果。

`package.xml` 与 `setup.py` 的许可现在暂写 `Proprietary`（表示用户尚未授权开放自己的 ROS2 扩展代码），**它不等同于上游 Menagerie 模型的 Apache-2.0 许可**。发布前请按你自己选择的开源许可证或使用条件修改，并填写真实的维护者信息；示例 `maintainer@example.invalid` 需要替换。

在你完成实测、保存 rosbag2 和 RViz2 实际截图之后，简历可以写：

> 基于 ROS2（WSL2/Ubuntu）构建 robosuite 仿真状态监控与任务管理模块，通过 rclpy 实现关节状态、末端/方块位姿及 TF2 发布，利用 Service 实现任务重置，使用 Launch、RViz2 与 rosbag2 完成运行状态可视化和数据录制回放；保持原 SAC–DLS/OSC 控制闭环不变。

## 故障排查

- `ros2 pkg executables` 没有找到节点：重新 `source /opt/ros/<distro>/setup.bash`、`source ~/panda_ros2_ws/install/setup.bash`，检查 `colcon build` 输出。
- Windows Demo 正常但 ROS Topic 没数据：确认 WSL Bridge 在先，Windows 传的是 `wsl hostname -I` 首个正确 IPv4；检查 WSL/Windows 防火墙及 NAT/mirrored 模式。
- Topic 正常但 Service 重置没有动作：核对 `reset_applied` 事件和 Windows 终端是否出现 `[RESET]`；Windows 防火墙可能阻挡 UDP 回包；也要确认 WSL 源端端口 15555 未被重启后改变。
- Service 无法触发，Windows 自己继续跑：先关闭 `--autostart`；`reset` 应由 ROS 服务发送。
- `ros2 bag record --use-sim-time` 没数据：确认 `/clock` 已在发布；本程序在待命时也以 2Hz 发布快照。
- 轮询只对窗口状态有效：ROS 2 不参与策略动作决策，服务只作用于 Demo Runner，非正式 Benchmark。

## 版本 / 验证状态

- 已根据仓库 `PandaDualControllerEnv` 46D/4D、`reset(seed=...)`、`_pose`、`data.xpos/xquat`、`legacy_sac_checkpoint.load_legacy_sac` 的接口编写。
- 已运行静态编译检查和独立 UDP 协议测试（见 `tests`）。
- **尚未在用户 Windows + WSL2 ROS2/MuJoCo 环境做端到端物理运行验证**；须完成 Smoke、Topic/Service、RViz 和 rosbag 测试后才可在简历中写为实际使用经验。
