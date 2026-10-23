
import time
import numpy as np
import mujoco
import mujoco.viewer

# 1. 加载机械臂模型
model_path = (
    "mujoco_menagerie/"
    "franka_emika_panda/scene.xml"
)

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 2. 使用官方 home 初始姿态
key_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home"
)

mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

# 3. 读取七个关节的初始目标
home = data.ctrl[:7].copy()

# 4. 指定新的关节目标
target = home.copy()
target[0] += 0.3
target[2] += 0.2

print("机械臂自由度数:", model.nv)
print("执行器数量:", model.nu)
print("初始目标:", home)
print("新目标:", target)

# 5. 打开 MuJoCo 仿真窗口
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running() and data.time < 10.0:

        start = time.time()

        # 前 3 秒平滑移动到新目标
        alpha = min(data.time / 3.0, 1.0)

        desired = home + alpha * (target - home)

        # 官方模型的前七个执行器：
        # 输入各个关节的目标位置
        data.ctrl[:7] = desired

        mujoco.mj_step(model, data)
        viewer.sync()

        # 尽量按照仿真时间步实时显示
        remaining = model.opt.timestep - (
            time.time() - start
        )

        if remaining > 0:
            time.sleep(remaining)

print("仿真结束")