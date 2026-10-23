
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

# ========== 1. 加载模型 ==========

model_path = (
    "mujoco_menagerie/"
    "franka_emika_panda/scene.xml"
)

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_KEY, "home"
)

mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

# ========== 2. 获取关节与末端索引 ==========

hand_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "hand"
)

joint_names = [f"joint{i}" for i in range(1, 8)]

joint_ids = []
dof_ids = []

for name in joint_names:
    jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name
    )
    joint_ids.append(jid)
    dof_ids.append(model.jnt_dofadr[jid])

# ========== 3. 设置三维目标 ==========

start_xyz = data.xpos[hand_id].copy()

# 从当前位置移动：X +4cm，Z +3cm
target_xyz = start_xyz + np.array([
    0.03, 0.03, -0.02
])

print("起点 XYZ (m):", start_xyz)
print("目标 XYZ (m):", target_xyz)

# ========== 4. 设置控制器参数 ==========

duration = 6.0
dt = model.opt.timestep

K = 2.0          # 末端误差反馈增益
damping = 0.05   # DLS 阻尼系数
max_qvel = 0.5   # 关节指令速度限制 rad/s

# 七个关节的期望位置
q_command = data.ctrl[:7].copy()

# 获取关节位置限制
joint_limits = model.jnt_range[joint_ids]
ctrl_limits = model.actuator_ctrlrange[:7]

lower = np.maximum(
    joint_limits[:, 0], ctrl_limits[:, 0]
)
upper = np.minimum(
    joint_limits[:, 1], ctrl_limits[:, 1]
)

# ========== 5. 创建记录列表 ==========

times = []
actual_xyz_list = []
error_list = []

# Jacobian: 3 行、所有自由度列
jacp = np.zeros((3, model.nv))

# ========== 6. 开始控制 ==========

with mujoco.viewer.launch_passive(
    model, data
) as viewer:

    while viewer.is_running() and data.time < duration:

        start_time = time.perf_counter()

        # 读取当前末端位置
        current_xyz = data.xpos[hand_id].copy()

        # 计算 XYZ 误差
        error = target_xyz - current_xyz

        # 计算 hand 的平移 Jacobian
        mujoco.mj_jacBody(
            model, data, jacp, None, hand_id
        )

        # 只使用七个机械臂关节
        J = jacp[:, dof_ids]

        # DLS 计算关节速度
        A = J @ J.T + damping**2 * np.eye(3)

        desired_xyz_velocity = K * error

        dq = J.T @ np.linalg.solve(
            A, desired_xyz_velocity
        )

        # 限制关节指令速度
        dq = np.clip(
            dq, -max_qvel, max_qvel
        )

        # 积分得到新的关节目标
        q_command += dq * dt

        # 防止超过关节限位
        q_command = np.clip(
            q_command, lower, upper
        )

        # 交给 Panda 内置位置控制器
        data.ctrl[:7] = q_command

        # 记录当前时刻
        times.append(data.time)
        actual_xyz_list.append(current_xyz)
        error_list.append(np.linalg.norm(error))

        # 物理仿真步进
        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.perf_counter() - start_time
        remaining = dt - elapsed

        if remaining > 0:
            time.sleep(remaining)

# ========== 7. 整理结果 ==========

times = np.array(times)
actual_xyz = np.array(actual_xyz_list)
errors = np.array(error_list)

if len(times) == 0:
    raise RuntimeError("未采集到仿真数据")

output_dir = Path("reach_3_3_-2_results")
output_dir.mkdir(exist_ok=True)

# ========== 8. 保存 CSV ==========

with open(
    output_dir / "reach_data.csv",
    "w", newline="", encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "time_s",
        "actual_x_m",
        "actual_y_m",
        "actual_z_m",
        "target_x_m",
        "target_y_m",
        "target_z_m",
        "error_m"
    ])

    for i, t in enumerate(times):
        writer.writerow([
            t,
            *actual_xyz[i],
            *target_xyz,
            errors[i]
        ])

# ========== 9. 绘制 XYZ 轨迹 ==========

plt.figure(figsize=(10, 5))

for j, axis in enumerate(["X", "Y", "Z"]):
    plt.plot(
        times, actual_xyz[:, j],
        label=f"Actual {axis}"
    )
    plt.axhline(
        target_xyz[j],
        linestyle="--",
        label=f"Target {axis}"
    )

plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Panda End-Effector XYZ Tracking")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    output_dir / "reach_xyz.png", dpi=200
)

# ========== 10. 绘制末端误差 ==========

plt.figure(figsize=(10, 5))

plt.plot(times, errors * 1000)

plt.xlabel("Time (s)")
plt.ylabel("Position Error (mm)")
plt.title("Panda End-Effector Position Error")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    output_dir / "reach_error.png", dpi=200
)

plt.show()

print("\n最终记录的末端位置:")
print(actual_xyz[-1])

print("目标位置:")
print(target_xyz)

print(
    f"最终记录误差: {errors[-1]*1000:.3f} mm"
)

print("结果已保存到:", output_dir)

# 最终误差（毫米）
final_error_mm = errors[-1] * 1000

# 成功阈值：5 mm
threshold_mm = 5.0

success = final_error_mm < threshold_mm

print(f"最终误差: {final_error_mm:.6f} mm")
print(f"是否成功: {success}")

# 计算全程 RMSE
rmse_mm = np.sqrt(np.mean(errors ** 2)) * 1000

print(f"全程位置误差 RMSE: {rmse_mm:.4f} mm")