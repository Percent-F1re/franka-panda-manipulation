# Windows 安装、运行和复现指南

## 一、准备本地代码

在 `franka-panda-manipulation/` 根目录保留原始四个 Pick-and-Place Stage 文件及之前的基础控制脚本；将 `sac_pickplace/`、`sac_reach/` 原目录保留为子目录。**不要先改变 Python 文件之间的导入关系。**

本项目的 `panda_pickplace_env.py` 会从父目录查找 `panda_pickplace_stage4_random.py`；Stage 4 会导入 Stage 1、2、3。

将训练好的两个策略权重分别手动复制到：

- `models/sac_grasp_best.zip`（对应原来的 `sac_pickplace/results_grasp/seed_42/best_model.zip`）
- `models/sac_pickplace_best.zip`（对应原来的 `sac_pickplace/results_pickplace/seed_42/best_model.zip`）

两个名字的区别很重要；不要混淆、覆盖或将一个模型当作另一个模型。

## 二、安装依赖与 Panda 模型

打开仓库根目录 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git mujoco_menagerie
git -C mujoco_menagerie sparse-checkout set franka_emika_panda

git -C mujoco_menagerie rev-parse HEAD
```

将最后打印的 Menagerie Git commit 记录到 `docs/environment_versions.md`，以免上游更新导致实验难复现。该目录由 `.gitignore` 排除，不应把整个外部模型库再次推送到自己的 GitHub 仓库。

如果已经在本地有完全匹配的 Menagerie Panda 模型，也可以直接拷贝整个 `franka_emika_panda` 子目录（含 assets、LICENSE、panda.xml、scene.xml），但应记录来源和上游版本。

## 三、运行传统专家

```powershell
python panda_pickplace_stage4_random.py
```

它将根据 Stage 1～3 自动构建新的场景 XML 并输出 50 组随机操作结果。注意：新的环境版本、随机种子不同，成功率不保证精确复现已发布数字。

## 四、评测 SAC：推荐从统一物理基准开始

```powershell
cd sac_pickplace
python -m unittest -v test_benchmark_common
python benchmark_pickplace.py --model ../models/sac_pickplace_best.zip --episodes 5 --seed 40000 --out results_benchmark/smoke_5
python benchmark_pickplace.py --model ../models/sac_pickplace_best.zip --episodes 100 --seed 45000 --out results_benchmark/reproduce_100
```

`--out` 必须是一个全新的空目录。测试使用相同初始位置配对运行专家/SAC，最终强制统一物理判定，并保留 1 秒停止决策后的观察。

## 五、训练新的完整 SAC 策略（可选，耗时）

```powershell
python train_sac_pickplace.py --pretrained ../models/sac_grasp_best.zip --steps 500000 --seed 42 --eval-every 25000 --eval-episodes 20 --out results_pickplace/reproduction_42
```

不要把 `results_pickplace/reproduction_42` 与旧结果目录重名。训练时使用新的 Replay Buffer；这不是完全续训之前的全部历史状态。

## 六、常见问题

- **找不到 Stage 4 / Stage 1～3**：确保四个 `panda_pickplace_stage*.py` 与 `sac_pickplace/` 在同一仓库根目录。
- **找不到 XML / Assets**：检查 `mujoco_menagerie/franka_emika_panda/scene.xml` 与 `panda.xml` 是否存在。
- **找不到权重**：确认 `models/sac_pickplace_best.zip` 是由 Stable-Baselines3 保存的训练模型，而不是项目源码压缩包。
- **结果不同**：核对 MuJoCo、SB3、模型文件版本、训练种子和测试种子；不要通过提高成功率阈值掩盖差异。
- **PowerShell 无法激活 venv**：可以直接运行 `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`，之后用相同 Python 路径运行脚本。
