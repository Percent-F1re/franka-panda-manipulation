# V3 鲁棒性实验 + GitHub 完整发布准备

**当前状态：代码与采样设计已打包，10 项离线测试通过；真实 MuJoCo/robosuite 物理仿真仍需在你的 Windows 环境运行。不要把尚未运行的 V3 实验写成结果。**

本包接着已经验证的：

- V1 Menagerie Panda + robosuite `MujocoEnv` 原样迁移；
- V2 46D/4D 原始单 SAC，DLS/官方 OSC 控制器可切换；
- V2 Seed 42000、100 组配对测试：DLS 84/100，OSC 81/100，McNemar p≈0.690；
- 两者同一个原始 SAC 权重，未分别重训练。

**这是扩展补丁，不是独立完整环境包。** 必须先安装并通过 V1 和 V2。V3 完成后用 `prepare_github_release.py` 从你的完整本地项目制作独立可发布 GitHub 目录。

## A. 实验设计

### 固定变量

- 同一 46D/4D SAC checkpoint，deterministic=True，**不重新训练**；
- 同一原始 Menagerie Panda、7 机械臂控制执行链和原 `actuator8` tendon gripper；
- DLS / OSC `kp=150` / 阻尼比 `1.0`，不调参数；
- 0.002 s 物理步、0.04 s 策略周期、600 步上限；
- 同一 `BenchmarkRules`，每回合结束额外 1 s 目标保持与物理检验；
- 同 scenario / case index：DLS 与 OSC 使用同一质量、摩擦、起点、目标扰动，但各自**独立闭环**选择 SAC 动作。

### 随机化组别

| Scenario | 质量 / 惯量 | 接触摩擦 | 初始方块 XY | 目标 XY（相对原目标） |
|---|---|---|---|---|
| `baseline` | 1× | 1× | ±4 cm | 不变 |
| `mass_random` | 0.6～1.4× | 1× | ±4 cm | 不变 |
| `friction_random` | 1× | 0.5～1.5× | ±4 cm | 不变 |
| `object_position_wide` | 1× | 1× | ±6 cm | 不变 |
| `target_position_random` | 1× | 1× | ±4 cm | ±2 cm |
| `combined_mild` | 0.8～1.2× | 0.8～1.2× | ±5 cm | ±1.5 cm |
| `combined_hard` | 0.6～1.4× | 0.5～1.5× | ±6 cm | ±3 cm |

随机化范围**是此阶段预注册的实验设置，不是已知最优范围或已验证硬件范围**。原 cube 0.08 kg，质量测试相当于 0.048～0.112 kg。方块尺寸、几何及重力不改变。质量变化连同全部惯性主矩同比例缩放。为确保 MuJoCo 接触摩擦混合公式真正感受到扰动，摩擦因子作用在**方块、桌面和两侧手指碰撞几何**的滑动/扭转/滚动摩擦系数上，不是只改 cube 的一项摩擦。此设计代表整个接触系统摩擦变化，而非独立“只改变 cube 摩擦”的实验。目标位置从 `cube_start.xy + [0,+0.06] m` 加独立 XY 扰动，反映任务目标泛化。

初始目标、质量、摩擦和姿态均保持在原本状态后再运行相应 episode。除了目标显示 marker，不修改 XML 文件。**每次 reset 先恢复原始模型质量、惯量、摩擦，再应用当回合扰动，并在原 1.5 s 沉降之前完成。** 这样避免回合间污染。DLS/OSC 控制代码、SAC 权重、奖励函数和成功判据均不改。

### 对应研究问题

- Q1：质量变化对保持抓取和放置成功率有何影响？
- Q2：接触摩擦变化是否提高释放后的残留指尖接触失败？
- Q3：初始方块位置分布扩大时，固定策略的工作空间泛化如何？
- Q4：目标 XY 与起始方块位置独立变化时，放置精度如何变化？
- Q5：四种扰动组合下哪种控制器表现更稳定？

注意：**这一版是 frozen Actor 的零样本物理鲁棒性实验，不是 domain randomization 训练，也不宣称 sim-to-real。**

## B. 代码安装（PowerShell）

本 ZIP 直接展开到已有 `panda_robosuite_project` 项目根目录：

```powershell
cd D:\zhuomian\project\robot_projects

$Patch = Join-Path $env:USERPROFILE "Downloads\robosuite_robustness_release_patch.zip"
Test-Path $Patch
Expand-Archive -LiteralPath $Patch -DestinationPath ".\panda_robosuite_project" -Force
```

会新增：

```text
envs/panda_domain_randomization.py
robustness_design.py
robustness_scenarios.json
smoke_robustness.py
run_robustness.py
plot_robustness.py
prepare_github_release.py
github/README.md
github/.gitignore
github/THIRD_PARTY_NOTICES.md
github/RELEASE_CHECKLIST.md
tests/test_robustness_offline.py
```

另外仅**小幅扩展**现有 `envs/panda_dual_controller.py`（插入两处 reset hook），以及 `legacy_dual_benchmark.py`（添加可选 `episode_options` 参数）。正常 DLS/OSC 不传这些选项时行为不变。其他 V1/V2 核心文件、XML、SAC 权重不修改。

## C. 先做 3 项验证

### 1. 离线检查

```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s ".\panda_robosuite_project\tests" -p "test_robustness_offline.py" -v
```

### 2. 真实物理 Smoke（两个控制器）

```powershell
& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\smoke_robustness.py" --mode both --seed 42000
```

期望：`ROBUSTNESS SMOKE PASSED`。验证 mass、inertia、friction 生效、目标观测更新、下次 reset 能精确恢复原始物理参数，以及 DLS / OSC 两种真实控制器均可运行。

### 3. 先重跑 **baseline 100 组** 确认不回退

```powershell
$LegacyModel = "D:\zhuomian\project\robot_simulation\sac_pickplace\results_pickplace\seed_42\best_model.zip"

& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\run_robustness.py" `
  --model $LegacyModel --scenarios baseline --controller both `
  --episodes 100 --seed 42000 --post-hold target `
  --out ".\panda_robosuite_project\evaluation\robustness\baseline_100_v3"
```

**预期**与 V2 一致：DLS 84/100，OSC 81/100。若不一致，先定位代码回退，不能直接用新结果覆盖旧基线。

## D. 鲁棒性实验

先运行每 scenario 5 个 episode 的小规模测试，排除碰撞异常、初始不可达、物理爆炸：

```powershell
& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\run_robustness.py" `
  --model $LegacyModel --scenarios mass_random,friction_random,object_position_wide,target_position_random,combined_mild,combined_hard `
  --controller both --episodes 5 --seed 42000 --post-hold target `
  --out ".\panda_robosuite_project\evaluation\robustness\pilot_5_v3"
```

通过后才运行正式 100 × 7 × 2 控制器，最多 1400 个 episode：

```powershell
& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\run_robustness.py" `
  --model $LegacyModel --scenarios all --controller both `
  --episodes 100 --seed 42000 --post-hold target `
  --out ".\panda_robosuite_project\evaluation\robustness\final_100_v3"
```

可用 `--case 1 --episodes 1 --save-traces` 调试单个案例，使用**新的** `--out`。

输出：

- `design.json`：所有 scenario/case 的**预先采样参数**；
- `metadata.json`：模型 SHA-256、场景、控制器、BenchmarkRules、版本、采样方法；
- `episodes.csv`：每个 episode 的原始判定、统一物理成功、失败原因、质量/摩擦/目标/初始位置；
- `summary.json`：每 scenario × controller 成功率、Wilson CI、失败条件频次、DLS/OSC 配对 McNemar、各 stress 相比 baseline 的同种子配对统计；
- `traces/`：仅在 `--save-traces` 时生成逐步轨迹。

如中途 Ctrl+C，已经完成的 episode 行会保存在 CSV 中；**程序不会自动续跑**，重新运行时需选新输出目录，不得合并不完整结果伪装成 100 次评估。

## E. 可视化

```powershell
& ".\.venv\Scripts\python.exe" ".\panda_robosuite_project\plot_robustness.py" `
  --episodes ".\panda_robosuite_project\evaluation\robustness\final_100_v3\episodes.csv" `
  --out ".\panda_robosuite_project\evaluation\robustness\final_100_v3\plots"
```

得到 `robustness_success.png`、`robustness_final_xy.png`。只发布**实际运行结果**，不要将设计范围误作实测成功率。

## F. GitHub 完整源码封装（在 V3 测试完成后）

先固定本机依赖：

```powershell
& ".\.venv\Scripts\python.exe" -m pip freeze > ".\panda_robosuite_project\requirements-lock.txt"
```

**模型资源与许可证：** 官方 [MuJoCo Menagerie Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda) 的模型为 Apache-2.0。发布衍生 XML 与 67 个原模型网格时必须保留上游 model `LICENSE`，并对修改的 XML 在 README 中注明派生。你的原模型目录可能还没有 LICENSE：可以使用本机原始 Menagerie 的 `franka_emika_panda\LICENSE` 作为 `--model-license`。仓库顶层的自有代码 License 请自行选择与核对，不应自动归属上游版权。

```powershell
$Project = ".\panda_robosuite_project"
$OriginalLicense = "D:\zhuomian\project\robot_simulation\mujoco_menagerie\franka_emika_panda\LICENSE"
Test-Path $OriginalLicense

& ".\.venv\Scripts\python.exe" "$Project\prepare_github_release.py" `
  --project $Project `
  --model-license $OriginalLicense `
  --out ".\release_staging\franka-panda-robosuite-v3"
```

该程序从**你的真实本地完整工程**收集 V1/V2/V3 正式源码、XML、67 个网格、模型 LICENSE、原始 checkpoint 加载器和 GitHub README 模板，并生成文件 SHA-256 清单。避免将实验日志、私有路径、完整 venv、原始 SAC ZIP、过时实验补丁上传到仓库。

若只是源码发布而不想同步网格文件，可以用 `--without-assets`，此时必须在 README 中说明用户需自行下载模型资源；**不可宣传为完全开箱即用**。

获得发布目录后，你可以将内容复制进已有 GitHub 仓库的工作树，再运行：

```powershell
git status
git add .
git commit -m "Add reproducible DLS/OSC benchmark and domain randomization suite"
git push
```

如尚未 clone 仓库，先 clone，之后将 staging 目录内容复制进去。复制时不要覆盖已确认的历史实验数据；GitHub Release 再创建 `v3.0` tag。**本发布脚本不会触网或自动推送 GitHub。**

## G. 研究结论写法

现在可以报告的事实：

> 在 robosuite 自定义环境中复用原始 Menagerie Panda 和相同 46D/4D SAC Actor，DLS 和 robosuite 官方 OSC 在 100 组配对任务上分别实现 84% 和 81% 的统一物理成功率，精确 McNemar p≈0.690，未发现显著成功率差异。

鲁棒性测试只有在运行后才可以写：

> 进一步在质量、接触摩擦、初始位置、目标位置等单因素和组合扰动条件下，通过控制器配对评估量化固定 SAC 策略的零样本鲁棒性。

**不要写成“在 robosuite 内置标准 Panda 上验证”或“已经进行域随机化训练”。**
