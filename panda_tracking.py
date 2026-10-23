
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

# ---------- 1. 加载模型 ----------
# 使用相对路径加载机械臂模型
model_path = "mujoco_menagerie/franka_emika_panda/scene_payload.xml"

model = mujoco.MjModel.from_xml_path(model_path)

# 输出文件仍然保存到项目目录
base_dir = Path(__file__).resolve().parent
data = mujoco.MjData(model)

# 官方 home 初始姿态
key_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_KEY, "home"
)
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)


# ---------- 2. 找到七个关节 ----------

joint_names = [f"joint{i}" for i in range(1, 8)]

qpos_ids = []
dof_ids = []

for name in joint_names:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )

    qpos_ids.append(model.jnt_qposadr[joint_id])
    dof_ids.append(model.jnt_dofadr[joint_id])

print("关节位置索引数量:", len(qpos_ids))
print("关节自由度索引数量:", len(dof_ids))

# ---------- 3. 设置目标轨迹 ----------

home = data.ctrl[:7].copy()
target = home.copy()

# 只改变第 1 和第 3 个关节
target[0] += 0.3
target[2] += 0.2

duration = 6.0
move_duration = 2.0

# ---------- 4. 记录数据 ----------

times = []
actual_list = []
target_list = []
torque_list = []

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running() and data.time < duration:

        start = time.perf_counter()

        # 目标从 home 平滑变化到 target
        alpha = min(data.time / move_duration, 1.0)

        desired = home + alpha * (target - home)

        # 前七个执行器控制七个旋转关节
        data.ctrl[:7] = desired

        # 读取七个关节的实际角度
        actual = data.qpos[qpos_ids].copy()

        # 保存目标和实际角度（转换为度）
        times.append(data.time)
        actual_list.append(np.rad2deg(actual))
        target_list.append(np.rad2deg(desired.copy()))

        mujoco.mj_step(model, data)
        # 读取本次仿真步执行器产生的关节力矩
        joint_torques = data.qfrc_actuator[dof_ids].copy()
        torque_list.append(joint_torques)
        viewer.sync()

        # 尽量按照真实时间显示动画
        remaining = (
            model.opt.timestep
            - (time.perf_counter() - start)
        )

        if remaining > 0:
            time.sleep(remaining)

# ---------- 5. 整理数据 ----------

times = np.array(times)
actual = np.array(actual_list)
targets = np.array(target_list)

errors = targets - actual

output_dir = base_dir / "panda_results_payload_05_2s"
output_dir.mkdir(exist_ok=True)

# ---------- 6. 保存 CSV ----------

torques = np.array(torque_list)

torque_path = output_dir / "panda_torque.csv"

with open(
    torque_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        ["time_s"] +
        [f"joint{i}_torque_Nm" for i in range(1, 8)]
    )

    writer.writerows(
        zip(times, *torques.T)
    )

csv_path = output_dir / "panda_joint_data.csv"

with open(
    csv_path, "w", newline="", encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    headers = ["time_s"]

    for i in range(7):
        headers.extend([
            f"joint{i+1}_target_deg",
            f"joint{i+1}_actual_deg",
            f"joint{i+1}_error_deg",
        ])

    writer.writerow(headers)

    for k, t in enumerate(times):
        row = [t]

        for i in range(7):
            row.extend([
                targets[k, i],
                actual[k, i],
                errors[k, i],
            ])

        writer.writerow(row)

# ---------- 7. 画第 1、3 关节的跟踪曲线 ----------

plt.figure(figsize=(10, 5))

for i in [0, 2]:
    plt.plot(
        times, targets[:, i], "--",
        label=f"Joint {i+1} target"
    )

    plt.plot(
        times, actual[:, i],
        label=f"Joint {i+1} actual"
    )

plt.xlabel("Time (s)")
plt.ylabel("Joint angle (deg)")
plt.title("Panda Joint Tracking")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    output_dir / "panda_tracking.png", dpi=200
)

# ---------- 8. 画七个关节的误差曲线 ----------

plt.figure(figsize=(10, 5))

for i in range(7):
    plt.plot(
        times, errors[:, i],
        label=f"Joint {i+1}"
    )

plt.xlabel("Time (s)")
plt.ylabel("Tracking error (deg)")
plt.title("Panda Joint Tracking Errors")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    output_dir / "panda_error.png", dpi=200
)

# ---------- 9. 自动计算误差 ----------

# 运动结束后的稳态保持阶段
hold_mask = times >= move_duration

print("\n七个关节跟踪结果：")

for i in range(7):
    rmse = np.sqrt(np.mean(errors[:, i] ** 2))

    hold_rmse = np.sqrt(
        np.mean(errors[hold_mask, i] ** 2)
    )

    print(
        f"Joint {i+1}: "
        f"全程 RMSE={rmse:.3f} deg, "
        f"保持阶段 RMSE={hold_rmse:.3f} deg"
    )

print("\n实验完成！")
print("结果位置:", output_dir)

plt.show()