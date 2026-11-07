# 原始 Franka Panda 单 SAC → robosuite 1.5.2 精确场景迁移版 v1

> **这是对旧项目的迁移，不是分阶段 SAC 新项目。** 保留原始 46D→4D 单 SAC、每个物理子步 6D DLS、原始 Panda Menagerie 执行器及夹爪、原始奖励/接触/成功逻辑。保留原有模型文件；不运行三阶段 Lift/Transfer/Place Actor。

## 与此前 robosuite Lift 迁移版的根本不同

此前 `envs/panda_legacy_pickplace.py` 创建的是 robosuite 默认 Panda + `JOINT_POSITION` 阻抗控制器，还把夹爪拆成两个位置执行器。因此旧策略在不同的动力学下失败，并不奇怪。

**本版本**采用自定义 `LegacyMenageriePandaRobosuite(MujocoEnv)`：

- 在 robosuite 1.5.2 的真正 `MujocoEnv` 环境生命周期中运行（初始化、reset、step、MjSim）；
- 使用 `MjSim(mujoco.MjModel.from_xml_path(...))` 加载**原始** `scene_pickplace_stage3.xml` / `panda_pickplace.xml` 和用户提供的 67 个 STL/OBJ；
- 保留原始 7 个 `general` 关节位置伺服执行器 + `actuator8`（tendon `split`, 控制范围 0～255）；
- 保留原始连杆质量/惯量、接触几何、关节阻尼、armature、夹爪 equality、XML 的 `implicitfast` 积分器；
- `MujocoEnv.step` 在每个 0.04 秒策略步内调用 20 次物理步。`_pre_action` 每步调用**原版 DLS**，然后由 `MjSim.step()` 执行原始 MuJoCo 控制信号；
- `envs/panda_legacy_exact.py` 来源于原始 `panda_pickplace_env.py`。`_get_obs`, `_physical_state`, `_info`, `render` 和其他物理判据函数与旧版保持相同的源代码结构。变动仅涉及引擎初始化、reset、step、close 等后端适配；
- **不修改用户原始 SAC checkpoint / Actor，也不加入额外夹爪速度限制。**

研究表述应准确写成“基于 robosuite `MujocoEnv` 的自定义 Franka Panda 场景，复用原始 MuJoCo Menagerie 模型”，**不是**声称“使用 robosuite 内置 Lift/默认 Panda 进行基准测试”。后者与本项目不是一回事。

## 1. 安装：不要重复嵌套项目目录

ZIP **内部已经包含** `panda_robosuite_project/` 一级目录，因此应解压到 `robot_projects`，而不是再次解压到 `robot_projects/panda_robosuite_project`。

在 PowerShell：

```powershell
cd D:\zhuomian\project\robot_projects

$Zip = Join-Path $env:USERPROFILE "Downloads\legacy_sac_robosuite_exact_v1.zip"
Test-Path $Zip
# 上面应为 True；如果不是，请将 $Zip 改为实际下载文件的完整路径
Expand-Archive -LiteralPath $Zip -DestinationPath "." -Force
```

主要新增：

```
panda_robosuite_project/
├── envs/
│   ├── robosuite_menagerie_core.py     # 真实 robosuite MujocoEnv + 原版原生执行器
│   ├── original_stage4_compat.py       # 原 Stage1-4 参数与 MJCF 索引
│   └── panda_legacy_exact.py          # 原始 46D/4D SAC Gym 任务，改为 robosuite 引擎
├── legacy_original_model/franka_emika_panda/
│   ├── panda_pickplace.xml
│   ├── scene_pickplace.xml
│   ├── scene_pickplace_stage3.xml
│   └── assets/                       # 67 个原始 OBJ/STL
├── legacy_reference/                 # 原始 Python 文件的参考副本；不在迁移环境执行
├── numpy_pickle_compat.py            # 用户上传的旧 checkpoint 兼容模块
├── legacy_sac_checkpoint.py          # 保留原权重的 46D/4D 加载器
├── legacy_exact_rules.py             # 独立物理 Benchmark 规则
├── legacy_exact_benchmark.py         # 1 秒物理稳定验证
├── smoke_sac_robosuite_exact.py
├── eval_sac_robosuite_exact.py
├── compare_sac_robosuite_exact.py
├── train_sac_robosuite_exact.py      # **可选：物理验证通过后才考虑微调**
└── tests/test_exact_migration_offline.py
```

本包不覆盖之前的 `envs/panda_legacy_pickplace.py`、已有模型和分阶段训练日志。`numpy_pickle_compat.py` 内容来自此次用户上传；原文件如已存在，名称相同。

## 2. 静态检查（不需要物理仿真）

```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s ".\panda_robosuite_project\tests" -p "test_exact_migration_offline.py" -v
```

## 3. 真实物理 Smoke（先运行这个）

```powershell
& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\smoke_sac_robosuite_exact.py" --seed 42000
```

期待：`SMOKE PASSED: original Menagerie Panda running under robosuite MujocoEnv`。

若报错，请先发送完整回溯：**当前交付未在用户 Windows 环境实测**，不能预先承诺上述 smoke 已通过。

## 4. 加载原来的 78% 单 SAC（不训练，不改权重）

```powershell
$LegacyModel = "D:\zhuomian\project\robot_simulation\sac_pickplace\results_pickplace\seed_42\best_model.zip"
Test-Path $LegacyModel

& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\eval_sac_robosuite_exact.py" --model $LegacyModel --check-only
```

必须传原始 **46D/4D 完整 PickPlace** SAC ZIP，不是后来 39D 的分阶段模型。NumPy 2.x pickle 兼容层不会重新训练或覆盖权重。

## 5. 先比较物理轨迹，而不是立即跑 100 次

从旧实验中选取 Seed42000 的 Case 1（原失败）和 Case 11（原成功）：

```powershell
$OldEnv = "D:\zhuomian\project\robot_simulation\sac_pickplace\panda_pickplace_env.py"

& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\compare_sac_robosuite_exact.py" --model $LegacyModel --old-env $OldEnv --seed 42000 --index 1 --steps 220 --out ".\panda_robosuite_project\evaluation\exact_menagerie\compare42000_case1_v1"

& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\compare_sac_robosuite_exact.py" --model $LegacyModel --old-env $OldEnv --seed 42000 --index 11 --steps 220 --out ".\panda_robosuite_project\evaluation\exact_menagerie\compare42000_case11_v1"
```

重点检查 `initial_comparison.json`、`summary.json` 和 `paired_step_trace.csv`：初始 46D 差异、25/32/45 步夹爪开度、关节跟踪误差、实际抬升。旧环境的 Stage4 动态生成了几乎相同的场景；理论上差异应远小于之前默认 robosuite Panda 的 ~35 mm 末端差异。但由于 robosuite 的 reset / forward 调用顺序和数值细节可能不同，**不保证逐比特一致，也不保证恢复 78%。**

## 6. robosuite 中零样本测试与原始独立物理 Benchmark

首先只测试 1 次，验证从初始状态到完整放置及额外 1 秒冻结控制检查：

```powershell
& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\eval_sac_robosuite_exact.py" --model $LegacyModel --case 1 --episodes 1 --seed 42000 --save-traces --out ".\panda_robosuite_project\evaluation\exact_menagerie\zero42000_case1_v1"
```

如果真实物理测试和对照合理，再测试 20 次：

```powershell
& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\eval_sac_robosuite_exact.py" --model $LegacyModel --episodes 20 --seed 42000 --out ".\panda_robosuite_project\evaluation\exact_menagerie\zero42000_n20_v1"
```

最终要与原研究的 78/100 对比，使用原始独立 Benchmark 的随机种子 **40000**、100 次：

```powershell
& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\eval_sac_robosuite_exact.py" --model $LegacyModel --episodes 100 --seed 40000 --out ".\panda_robosuite_project\evaluation\exact_menagerie\zero40000_n100_v1"
```

- 每次自动增量写入 `episodes.csv`，Ctrl+C 不会丢失已完成的行；
- `unified_success` 是物理事件 + 1 秒后续物理检查，和 Gymnasium `is_success` 分开记录；
- 输出目录不可含原有数据，避免覆盖实验。

## 7. 可选：微调（**目前不要运行**）

只有确认 robosuite 上已复现合理的旧行为后才考虑：

```powershell
& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\train_sac_robosuite_exact.py" --pretrained $LegacyModel --steps 20000 --out ".\panda_robosuite_project\runs\exact_legacy_finetune_001"
```

仍使用**原始一个 SAC Actor**；初始模型会保存为 `initial_model.zip` 和初始 `best_model.zip`，后续评估差于基线的 checkpoint 不会覆盖 Best。旧 checkpoint 始终只读。

## 检查边界

已在交付环境进行：完整 ZIP 网格资源一致性校验、XML 执行器及积分器结构、Python 语法检查、8 项离线契约测试、与原版任务函数的 AST 一致性对照（除后端生命周期相关函数）。

尚**未**在你的 Windows robosuite 1.5.2 + MuJoCo 上跑真实仿真，不能声称成功率已经恢复。**如果运行错误，优先修复本迁移包本身，而不是退回增加训练步骤。**
