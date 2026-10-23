
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
import mujoco


# ========== 1. 实验参数 ==========

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    "mujoco_menagerie/"
    "franka_emika_panda/scene_payload.xml"
)

OUTPUT_DIR = BASE_DIR / "random_reach_results_payload_05_4cm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_EPISODES = 50
RANDOM_SEED = 42
TARGET_RANGE = 0.04       # ±4 cm

DURATION = 6.0
K = 2.0
DAMPING = 0.05
MAX_QVEL = 0.5           # rad/s

SUCCESS_THRESHOLD = 0.005  # 5 mm
HOLD_WINDOW = 0.2          # 最后0.2秒


# ========== 2. 加载机器人模型 ==========

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

dt = model.opt.timestep

home_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_KEY, "home"
)

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

assert home_id >= 0, "找不到 home 关键帧"
assert hand_id >= 0, "找不到 hand 刚体"
assert len(set(joint_ids)) == 7, "关节索引异常"

joint_ids = np.array(joint_ids)
dof_ids = np.array(dof_ids)


# ========== 3. 检查关节限制 ==========

# Panda 模型中前七个执行器为位置控制
assert model.nu >= 7, "执行器数量不足7个"

# 需要确保每个关节都有有效的位置限制
assert np.all(model.jnt_limited[joint_ids]), (
    "部分关节未定义位置限制，请检查模型"
)

assert np.all(model.actuator_ctrllimited[:7]), (
    "部分执行器未定义控制范围，请检查模型"
)

joint_limits = model.jnt_range[joint_ids]
ctrl_limits = model.actuator_ctrlrange[:7]

lower = np.maximum(
    joint_limits[:, 0],
    ctrl_limits[:, 0]
)

upper = np.minimum(
    joint_limits[:, 1],
    ctrl_limits[:, 1]
)

assert np.all(lower < upper), "关节上下限设置异常"


# ========== 4. 获得固定的 home 位置 ==========

def reset_robot():
    # 每次从相同的初始状态开始
    mujoco.mj_resetDataKeyframe(
        model, data, home_id
    )
    mujoco.mj_forward(model, data)


reset_robot()

home_xyz = data.xpos[hand_id].copy()

print("Home XYZ:", home_xyz)


# ========== 5. 固定随机种子生成目标 ==========

rng = np.random.default_rng(RANDOM_SEED)

offsets = rng.uniform(
    low=-TARGET_RANGE,
    high=TARGET_RANGE,
    size=(NUM_EPISODES, 3)
)

targets = home_xyz[None, :] + offsets


# ========== 6. 单次 Reach 实验 ==========

def run_reach(target_xyz):

    reset_robot()

    q_command = data.ctrl[:7].copy()

    times = []
    errors = []

    jacp = np.zeros((3, model.nv))

    limit_hit = False

    # 无可视化，快速进行物理仿真
    while data.time < DURATION:

        current_xyz = data.xpos[hand_id].copy()
        error = target_xyz - current_xyz

        # 计算平移 Jacobian
        mujoco.mj_jacBody(
            model, data,
            jacp, None, hand_id
        )

        J = jacp[:, dof_ids]

        # Damped Least Squares
        A = (
            J @ J.T
            + DAMPING**2 * np.eye(3)
        )

        desired_velocity = K * error

        dq = J.T @ np.linalg.solve(
            A, desired_velocity
        )

        # 限制单关节指令速度
        dq = np.clip(
            dq, -MAX_QVEL, MAX_QVEL
        )

        # 积分为新的位置目标
        proposed = q_command + dq * dt

        # 判断有没有发生指令限位裁剪
        if np.any(
            (proposed < lower) |
            (proposed > upper)
        ):
            limit_hit = True

        q_command = np.clip(
            proposed, lower, upper
        )

        data.ctrl[:7] = q_command

        # 推进仿真
        mujoco.mj_step(model, data)

        # 记录步进之后的实际状态
        current_xyz = data.xpos[hand_id].copy()
        distance = np.linalg.norm(
            target_xyz - current_xyz
        )

        times.append(data.time)
        errors.append(distance)

    times = np.array(times)
    errors = np.array(errors)

    # 判断每个采样时刻的误差是否不超过5mm
    within = errors <= SUCCESS_THRESHOLD

    # 从后往前判断：
    # 当前时刻以及之后所有时刻是否均满足要求
    stays_inside = np.logical_and.accumulate(
        within[::-1]
    )[::-1]

    if np.any(stays_inside):
        first_index = np.flatnonzero(stays_inside)[0]
        reach_time = float(times[first_index])
    else:
        reach_time = np.nan

    # 最终位置误差
    final_error = float(errors[-1])

    # 全程距离 RMSE
    rmse = float(
        np.sqrt(np.mean(errors**2))
    )

    # 最后0.2秒的误差
    hold_mask = (
        times >= times[-1] - HOLD_WINDOW
    )

    hold_max_error = float(
        np.max(errors[hold_mask])
    )

    # 不仅最终到达，还要短时间稳定
    success = (
        final_error <= SUCCESS_THRESHOLD
        and hold_max_error <= SUCCESS_THRESHOLD
    )

    return {
        "success": success,
        "final_error_mm": final_error * 1000,
        "rmse_mm": rmse * 1000,
        "hold_max_error_mm": hold_max_error * 1000,
        "limit_hit": limit_hit,
        "times": times,
        "errors_mm": errors * 1000,
        "time_to_5mm_s": reach_time,
    }


# ========== 7. 批量运行50次 ==========

results = []
all_curves = []

for episode in range(NUM_EPISODES):

    target = targets[episode]
    offset = offsets[episode]

    result = run_reach(target)

    row = {
        "episode": episode + 1,
        "offset_x_m": offset[0],
        "offset_y_m": offset[1],
        "offset_z_m": offset[2],
        "target_x_m": target[0],
        "target_y_m": target[1],
        "target_z_m": target[2],
        "success": result["success"],
        "final_error_mm": result["final_error_mm"],
        "rmse_mm": result["rmse_mm"],
        "hold_max_error_mm": result["hold_max_error_mm"],
        "limit_hit": result["limit_hit"],
        "time_to_5mm_s": result["time_to_5mm_s"],
    }

    results.append(row)

    all_curves.append((
        result["times"],
        result["errors_mm"],
        result["success"]
    ))

    print(
        f"Episode {episode+1:02d}/{NUM_EPISODES} | "
        f"Final={result['final_error_mm']:.3f} mm | "
        f"RMSE={result['rmse_mm']:.3f} mm | "
        f"Success={result['success']}"
    )


# ========== 8. 保存CSV ==========

csv_path = OUTPUT_DIR / "reach_summary.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# ========== 9. 计算总体指标 ==========

successes = np.array([
    r["success"] for r in results
])

final_errors = np.array([
    r["final_error_mm"] for r in results
])

rmses = np.array([
    r["rmse_mm"] for r in results
])

success_rate = np.mean(successes) * 100

print("\n========== 测试结果 ==========")
print(f"测试次数: {NUM_EPISODES}")
print(f"成功次数: {np.sum(successes)}")
print(f"成功率: {success_rate:.1f}%")
print(
    f"平均最终误差: "
    f"{np.mean(final_errors):.4f} mm"
)
print(
    f"最大最终误差: "
    f"{np.max(final_errors):.4f} mm"
)
print(
    f"平均全程RMSE: "
    f"{np.mean(rmses):.4f} mm"
)
print(
    f"触及指令限位的次数: "
    f"{sum(r['limit_hit'] for r in results)}"
)


# ========== 10. 绘制50条误差曲线 ==========

plt.figure(figsize=(11, 6))

for times, errors, success in all_curves:
    plt.plot(
        times,
        errors,
        alpha=0.30,
        color="tab:blue" if success else "tab:red"
    )

plt.axhline(
    SUCCESS_THRESHOLD * 1000,
    color="black",
    linestyle="--",
    label="5 mm threshold"
)

plt.xlabel("Time (s)")
plt.ylabel("Position Error (mm)")
plt.title("Random Target Reach - 50 Episodes")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_curves.png",
    dpi=200
)


# ========== 11. 最终误差分布 ==========

plt.figure(figsize=(9, 5))

plt.hist(
    final_errors,
    bins=15,
    edgecolor="black"
)

plt.axvline(
    SUCCESS_THRESHOLD * 1000,
    linestyle="--",
    color="red",
    label="5 mm threshold"
)

plt.xlabel("Final Position Error (mm)")
plt.ylabel("Number of Episodes")
plt.title("Final Error Distribution")
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "final_error_hist.png",
    dpi=200
)

plt.show()

print("\n结果保存位置:", OUTPUT_DIR)