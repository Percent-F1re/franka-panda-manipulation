# V3 GitHub README：两个 SAC Demo + 两张实验结果图

本补丁**不会修改** SAC checkpoint、DLS / OSC 控制、robosuite 场景、奖励或物理成功判据。仅新增视频录制和静态科研可视化工具。

## 已完成、无需本机重算的部分

- `assets/plots/robustness_success_rate.png`：七种环境条件下 DLS / OSC 统一物理成功率（Wilson 95% CI），输入是已发布的 14 行 `results/V3_ROBUSTNESS_SUMMARY.csv`。
- `assets/plots/robustness_xy_error.png`：各场景 100 回合、所有回合的最终方块 XY 平均误差。
- `generate_readme_figures.py`：从仓库已有 CSV 重新生成两张图，不用重新运行 MuJoCo。

**注意：** 这两张图是分类实验结果图（柱状图）；不把不连续的扰动条件连接为“连续学习曲线”，以免造成误解。

## 一、将补丁解压到已推送的本地 GitHub 仓库

在 **PowerShell** 中执行：

```powershell
$Repo = "D:\github\franka-panda-manipulation-v3-publish"
$Py = "D:\zhuomian\project\robot_projects\.venv\Scripts\python.exe"
$Checkpoint = "D:\zhuomian\project\robot_simulation\sac_pickplace\results_pickplace\seed_42\best_model.zip"
$Patch = Join-Path $env:USERPROFILE "Downloads\robosuite_readme_media_patch.zip"

Test-Path $Repo
Test-Path $Py
Test-Path $Checkpoint
Test-Path $Patch

Expand-Archive -LiteralPath $Patch -DestinationPath $Repo -Force
Set-Location $Repo
```

四项路径均应为 `True`。不要再次执行旧项目迁移的 `git rm` 或 `prepare_github_release.py`。

## 二、生成或检查两张图

补丁已包含 PNG，可以直接用于 README。如需自行从你的正式 CSV 重绘：

```powershell
& $Py -m pip install matplotlib numpy
& $Py .\generate_readme_figures.py --data .\results\V3_ROBUSTNESS_SUMMARY.csv --out .\assets\plots

Test-Path .\assets\plots\robustness_success_rate.png
Test-Path .\assets\plots\robustness_xy_error.png
```

## 三、录制同一物体初始位置的 DLS 和 OSC Demo

```powershell
& $Py -m pip install imageio imageio-ffmpeg pillow

& $Py .\record_sac_demo.py `
  --model $Checkpoint `
  --mode both `
  --seed 42000 `
  --max-cases 30 `
  --out .\assets\demos
```

录制过程会先使用原始冻结 SAC **扫描最多 30 个基线案例**，寻找 DLS 和 OSC **都通过原有统一物理判据**的同一案例；之后使用同一种子、同一方块初始位置，分别独立闭环重新运行并录制。不会挑选测试集以外的未知成功率，也不会使用专家轨迹动作代替 SAC。自动选择的**成功展示案例不能代替 100 次统计结果**。

保存文件：

```text
assets/demos/sac_dls.mp4
assets/demos/sac_dls.gif
assets/demos/sac_osc.mp4
assets/demos/sac_osc.gif
assets/demos/demo_metadata.json
assets/demos/sac_dls_case_XXX_trace.csv
assets/demos/sac_osc_case_XXX_trace.csv
```

录制 MP4 时以 MuJoCo 离屏渲染取帧，输出为 960×544、25fps 的 H.264 视频；GIF 用于在 GitHub 首页**直接播放**。单个 GIF 如果太大，可以重新运行并添加 `--overwrite --gif-width 360 --gif-stride 3` 以减小体积。

提示：离屏渲染需要正常的 OpenGL 驱动和 Windows 图形桌面。当前补丁的**MuJoCo 视频录制尚未在你的 Windows 上实际验证**，但已有的 SAC 环境、Benchmark 逻辑不被替换。若遇到渲染/OpenGL 错误，先反馈异常日志，不需要修改原始控制器。

如只想录制某个已知成功的基线案例（case 从 1 开始，seed=42000）：

```powershell
& $Py .\record_sac_demo.py --model $Checkpoint --mode both --seed 42000 --case 1 --out .\assets\demos --overwrite
```

若该案例没有同时通过两种控制器的最终物理判定，脚本会停止并提示；推荐默认自动扫描模式。

## 四、自动把图片和视频嵌入 README

前提是两个 GIF、两个 MP4、两个 PNG 已存在，`results/V3_ROBUSTNESS_RESULTS.md` 也存在：

```powershell
& $Py .\update_readme_media.py
& $Py .\update_readme_media.py --apply
```

脚本会将 `README_MEDIA_SNIPPET.md` 插入到 `README.md` 的 `## System` 部分之前。重复执行会更新同一块标记区域，不会无限重复插入，也不会重写其他 README 章节。

## 五、发布到你现有的 GitHub main 分支

```powershell
git status --short

git add README.md README_MEDIA_SNIPPET.md README_DEMOS_AND_FIGURES.md `
  record_sac_demo.py generate_readme_figures.py update_readme_media.py `
  assets/plots/robustness_success_rate.png assets/plots/robustness_xy_error.png `
  assets/demos/sac_dls.gif assets/demos/sac_osc.gif `
  assets/demos/sac_dls.mp4 assets/demos/sac_osc.mp4 assets/demos/demo_metadata.json

git diff --cached --check
git diff --cached --stat

git commit -m "Add SAC DLS OSC demos and V3 robustness figures to README"
git push origin main
```

不要运行 `git push --force`。先查看 `git status --short`，确保没有误提交虚拟环境、大型无关文件或私密数据。若 MP4 超过 GitHub 单文件大小限制，可以只提交 GIF，将完整 MP4 放在 Releases 后修改 README 的 MP4 链接。若要发布每步 trace CSV，可以另外 `git add -f assets/demos/*_trace.csv`，但非必须。

确认已推送后查看：

https://github.com/Percent-F1re/franka-panda-manipulation

## 使用说明的边界

- 这里录制的是一个**自选成功的代表性案例**，非真实成功率统计。README 的成功率仍来自 100 次成对评估。
- 原始 SAC 在 DLS 下训练；OSC 是固定策略的控制器切换实验。
- 机器人为原 Menagerie Panda，经 robosuite 自定义环境管理，不是 robosuite 标准 Panda 动力学模型。
- 当前环境中没有 Windows 仿真运行时和原始 SAC checkpoint，因此没有虚构 Demo 视频。两张 PNG 则已经从正式记录生成。
