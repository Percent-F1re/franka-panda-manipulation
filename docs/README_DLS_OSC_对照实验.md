# V2 · Menagerie Panda + robosuite 环境框架：单 SAC、可切换 DLS / 官方 OSC

> **本补丁是覆盖安装在你已经跑通 `legacy_sac_robosuite_exact_v1.zip` 的 `panda_robosuite_project/` 上的新增功能。**
> 不覆盖 `envs/panda_legacy_exact.py`、原始 SAC `best_model.zip`、原始 Panda XML、原来 84/100 的测试结果，也**不**把策略改为三个 Actor。
>
> **运行状态**：已完成代码审查、AST 任务逻辑一致性检查、派生 XML 生成单元测试、7 个执行器增益/控制范围切换及恢复的离线测试；**尚未在你的 Windows robosuite 1.5.2 / MuJoCo 3.9 运行真实物理 smoke 或 100 次评估**。真实数值和 OSC 控制稳定性需以你运行结果为准。

## 1. 核心研究设计

| 项目 | DLS 模式 | OSC 模式 |
|---|---|---|
| 机器人 | 原始 Menagerie Panda XML | 相同原始 Panda XML |
| 质量、惯量、关节、方块、桌面 | 原始 | 原始 |
| SAC Actor | **同一份原始 46D → 4D 权重** | **同一份原始 46D → 4D 权重** |
| 第 1～3 维 SAC 动作 | 每策略步 `hand_target += 0.01 * action[:3]` | **完全相同** |
| 第 4 维 SAC 动作 | 原始 `actuator8` 0～255 绝对夹爪目标 | **完全相同** |
| 工作区裁剪 | 原始 | **完全相同** |
| 末端姿态 | 固定为 reset 后姿态 | **相同固定姿态** |
| 末端执行 | 原始 6D DLS + 7 个位置执行器 | **robosuite 官方 `OperationalSpaceController` (`OSC_POSE`)** + 7 个力矩电机 |
| 物理步长 / 策略周期 | 0.002s / 0.04s | 0.002s / 0.04s |
| 成功判定 | `legacy_exact_rules.BenchmarkRules` | **相同** |
| 放置后评价 | 冻结末端目标，保留位置伺服闭环 1 秒 | 冻结末端目标，保留 OSC 力矩反馈闭环 1 秒 |

### 必须说明的控制差异

原始 Menagerie 七个 `general` 执行器是 **关节位置伺服**。官方 robosuite `OperationalSpaceController` 生成的却是 **关节力矩**，不能直接输入这些位置执行器。所以 `OSC` 模式下，**仅对编译完成的七个机械臂执行器**修改 MuJoCo `gaintype`、`biastype`、`gainprm`、`biasprm` 和 `ctrlrange`，使控制输入变为力矩；力矩上限取原始 `actuator_forcerange`。**第八个 tendon 夹爪执行器、所有质量和惯量、碰撞、关节、方块和任务逻辑完全不变**。每次 reset 前都会恢复位置伺服配置，完成原始 1.5s 沉降后再进入 OSC 力矩模式。退出也会恢复。

这是真实的 **DLS 位置控制链 vs OSC 力矩控制链**，不是“只改控制器名称却仍将 OSC 力矩当作位置命令”。两个组的机器人实体和环境相同，但执行器控制结构不可能完全相同——这本身就是控制器对照的自变量。

### 为什么 OSC 模式也有 46 维 `q_cmd`？

旧 SAC 的第 15～21 个观测量是原 DLS 算出的 `q_cmd`，已融入训练好的 Actor。为了不更改网络或 observation shape，OSC 模式每个物理步都会计算**一份影子 DLS 关节参考值**，仅放在 observation；**实际输出给七个关节的仍是官方 robosuite OSC 计算的力矩**。这一折中保持了 46D 接口，但 OSC 的 `q_cmd` 只是兼容特征，并非实际执行目标；后续论文必须如实说明。

### 为什么需两个额外的 site？

原始 Menagerie 模型没有 robosuite OSC 所要求的站点：在 `hand` body 原点添加质量为零、不参与碰撞的 `dual_osc_hand_ref`；在 worldbody 添加固定参考 `hand_center`。两组使用同一个派生场景，其余 XML 物理数据不变。脚本只生成 `panda_pickplace_dual_controllers.xml` 和 `scene_pickplace_dual_controllers.xml`，**不编辑原始 XML**。

## 2. 下载和安装

下载 `dls_osc_controller_comparison_patch.zip`，保存至 Windows Downloads。在 PowerShell 里：

```powershell
cd D:\zhuomian\project\robot_projects

$Patch = Join-Path $env:USERPROFILE "Downloads\dls_osc_controller_comparison_patch.zip"
Test-Path $Patch
# 应为 True，不然把 $Patch 改为实际下载位置

Expand-Archive -LiteralPath $Patch -DestinationPath ".\panda_robosuite_project" -Force
```

压缩包内部是**平铺到项目根目录的结构**：

```text
envs/
    dual_scene_core.py
    panda_dual_controller.py
    controllers/
        __init__.py
        legacy_dls.py
        osc_bridge.py
legacy_dual_benchmark.py
smoke_dls_osc.py
compare_dls_vs_osc.py
trace_dls_vs_osc.py
tests/
    test_dual_controller_static.py
    test_derived_scene_offline.py
    test_actuator_switch_offline.py
README_DLS_OSC_对照实验.md
```

这些全部是**新增文件**，不包含旧 `panda_legacy_exact.py`，也不包含原始 Actor 权重。

需要的旧文件必须仍在原项目位置：

```text
panda_robosuite_project/
    envs/robosuite_menagerie_core.py
    envs/original_stage4_compat.py
    envs/panda_legacy_exact.py
    legacy_sac_checkpoint.py
    legacy_exact_rules.py
    legacy_exact_benchmark.py
    numpy_pickle_compat.py
    legacy_original_model/franka_emika_panda/
        panda_pickplace.xml
        scene_pickplace_stage3.xml
        assets/   # 67 个 OBJ/STL
```

## 3. 必须按顺序做的 4 个检查

### A. 不需仿真的静态测试

```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s ".\panda_robosuite_project\tests" -p "test_*dual*.py" -v
& ".\.venv\Scripts\python.exe" -m unittest discover -s ".\panda_robosuite_project\tests" -p "test_derived_scene_offline.py" -v
& ".\.venv\Scripts\python.exe" -m unittest discover -s ".\panda_robosuite_project\tests" -p "test_actuator_switch_offline.py" -v
```

### B. 真正的 robosuite 双模式物理冒烟测试

```powershell
& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\smoke_dls_osc.py" --mode both --seed 42000 --steps 5
```

期望在通过时输出 `DUAL CONTROLLER SMOKE PASSED`，并确认原始 8 个执行器、OSC 七个力矩控制和两次 reset 都正常。**如果失败请提供完整报错；不要跳过 smoke 直接跑 100 次**。

### C. 原模型权重（不要选错 SAC）

```powershell
$LegacyModel = "D:\zhuomian\project\robot_simulation\sac_pickplace\results_pickplace\seed_42\best_model.zip"
Test-Path $LegacyModel
```

这是你先前 84% 零样本评估所用的**原始完整 PickPlace 46D/4D SAC**，不是 39D 或分阶段模型。

### D. 同动作逐步诊断（只测试一个案例）

先用原始 DLS 的观测产生一段 SAC 动作，再把相同动作输入 OSC。这个程序只用来判断控制动态差异，**不是闭环成功率评估**。

```powershell
& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\trace_dls_vs_osc.py" `
  --model $LegacyModel --seed 42000 --case 1 --steps 220 `
  --out ".\panda_robosuite_project\evaluation\dls_osc\replay_case1_v1"
```

输出：`initial.json`、`paired_step_trace.csv`、`summary.json`。重点检查初始 46D 差异、末端差异、关节差异、OSC 力矩饱和比例与方块抬升。

## 4. 正式同种子配对闭环实验

### 先只跑一个 Seed 42000 案例

```powershell
& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\compare_dls_vs_osc.py" `
  --model $LegacyModel --controller both --episodes 1 --seed 42000 `
  --post-hold target --save-traces `
  --out ".\panda_robosuite_project\evaluation\dls_osc\closedloop_n1_v1"
```

这一次 SAC 在两个环境中分别根据**自己的 46D 观测**独立生成 4D 动作；相同的只是 checkpoint、物体初始位置、seed、任务和判据。这才是主要的控制器比较结果。

### 验证正常后跑 20 次

```powershell
& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\compare_dls_vs_osc.py" `
  --model $LegacyModel --controller both --episodes 20 --seed 42000 `
  --post-hold target `
  --out ".\panda_robosuite_project\evaluation\dls_osc\closedloop_n20_v1"
```

### 最后跑同种子 100 次

```powershell
& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\compare_dls_vs_osc.py" `
  --model $LegacyModel --controller both --episodes 100 --seed 42000 `
  --post-hold target `
  --out ".\panda_robosuite_project\evaluation\dls_osc\closedloop_n100_v1"
```

CSV 将随着每个 episode **增量保存**：
- `episodes_dls.csv`：100 次 DLS 闭环与最终 1s 物理检查；
- `episodes_osc.csv`：100 次官方 OSC 闭环与同样规则；
- `paired.csv`：逐对 DLS/OSC 共同成功、单组成功和 XY 误差；
- `summary.json`：两组 Wilson 95% CI、Lift/Transfer/Place、失败条件计数、OSC 饱和统计、精确 McNemar 配对检验；
- `metadata.json`：随机种子、原始 checkpoint SHA256、XML SHA256、版本、控制器参数、post-hold 方案等。

`--case 13` 可以只运行指定第 13 个随机位置（而不是第一个），注意另设一个全新 `--out` 路径。

### 关于终止后额外 1 秒的公平性

旧版 `legacy_exact_benchmark.py` 是冻结原始 `data.ctrl` 一秒。对于 DLS 关节位置伺服，此法等价于保持目标不变；但对于 OSC，**固定一瞬间的关节力矩并不是保持末端目标**，会人为惩罚 OSC。因此默认 `--post-hold target` 是：

- DLS：保持最终 `q_cmd` 和原夹爪目标，由原物理位置伺服保持；
- OSC：保持最终 6D 末端目标和夹爪目标，继续每 0.002s 更新力矩；
- 两组：**不再调用 SAC、不发新动作、不救援方块；两组都继续 1.0s 真实物理仿真，并用一模一样的 `BenchmarkRules` 判定。**

如需研究这一评估选择对 OSC 的影响，另用 `--post-hold raw_actuator` 做**敏感性实验**，但不要与默认 `target` 结果混在一起。`raw_actuator` 的 DLS 结果应该可与原来 84/100 的评估口径直接比较；OSC 的该口径是“锁住最后一帧力矩”，不是等效的目标保持控制。

## 5. 合理的科研结论与边界

- 可回答：**将预训练原始 SAC 权重、同一四维动作和同一 Menagerie Panda 物理模型固定后，DLS 与 robosuite OSC_POSE 控制适配的闭环性能有何不同？**
- 不能回答：在任意机械臂或所有训练设置下，DLS/OSC 哪个算法普遍更优。
- OSC 组没有重新训练，对预训练 DLS 控制器分布存在**策略-控制器协同偏差**；这应明确称作 **zero-shot controller swap**。
- OSC 的默认 `kp=150`、`damping_ratio=1` 来源于 robosuite 默认参数，可通过 `--osc-kp`、`--osc-damping-ratio` 另做预先声明的灵敏度实验。**不要只根据最终成功率无限调参**。每个参数组合都用不同的输出目录和相同 seed。
- 这不是 robosuite 内置 Panda / Lift 任务；**是使用 robosuite 环境框架、原始 Menagerie Panda 物理模型进行控制器比较**。
- 保持原始 84%（Seed 42000，100 次）迁移结果不改动。如果新 DLS 组没有大致复现旧基线，应先查清由新增参考 site、初始化或控制切换引入的差异，再评价 OSC。

## 6. 排错优先级

1. `ImportError: envs.robosuite_menagerie_core`：未安装上一个 `legacy_sac_robosuite_exact_v1` 包或解压目录不对。
2. `Missing ...dual_osc_hand_ref`：派生 XML site 未创建成功；检查原始 `panda_pickplace.xml` 和 `scene_pickplace_stage3.xml` 是否在 `legacy_original_model/franka_emika_panda/` 下。
3. `Unexpected non-unit joint gear`：当前原始 Panda XML 与我们假定的 Menagerie 原始执行器不同，**不能跳过检查直接发送 OSC 力矩**。
4. 官方 OSC 报错：发送完整 Python traceback 和 `mujoco.__version__`、`robosuite.__version__`。
5. OSC 可运行但表现很差：先看 `torque_saturation_fraction`、末端轨迹、接触次数，**不要先改奖励函数或重新训练 SAC**。

---

**成功验收标准**：两个模式都跑通 smoke；DLS 模式基本复现已验证的同种子基线；OSC 使用官方控制器、相同原始 4D SAC 和物理任务；20/100 次数据具备相同随机位置、成功阈值及可复现的 controller 配置；OSC 成功率无论高低都如实报告。
