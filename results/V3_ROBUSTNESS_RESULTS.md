# V3：零样本域随机化鲁棒性实测结果

> 来源：2026-11-07 上传的 Windows 实验终端 `FINAL SUMMARY` 文本。已报告 7 scenario × 2 controller × 100 episodes。原始 `design.json`、`episodes.csv`、`summary.json`、`metadata.json` 尚未随本次日志上传，完整数据复核及可复现实验封存仍需这些原始文件。

## 实验条件

- 原始 Menagerie Panda 场景运行在 robosuite 自定义 `MujocoEnv` 中，非 robosuite 内置默认 Panda。
- 同一个**在 DLS 下训练**的 46D→4D SAC Actor；DLS/OSC 两组分别独立闭环执行，不重新训练、不调参。
- 100 组同种子配对案例，`seed=42000`；每个场景/控制器使用相同的扰动设计索引。
- 基线物体 XY 初始偏移 ±4 cm；统一物理成功判据包括额外 1 秒物理稳定检查。
- 质量因子作用于方块**质量和转动惯量**；摩擦因子同时作用于**方块、桌面和指尖碰撞体**三类摩擦分量；目标扰动是独立的 XY 变化。

## 100 次成功率结果

| 场景 | DLS | OSC | DLS 相对基线 | OSC 相对基线 |
|---|---:|---:|---:|---:|
| `baseline` · 原始基线 | **84%** | **81%** | +0 pp | +0 pp |
| `mass_random` · 质量随机化 | **88%** | **86%** | +4 pp | +5 pp |
| `friction_random` · 摩擦随机化 | **81%** | **83%** | -3 pp | +2 pp |
| `object_position_wide` · 扩大初始位置 | **69%** | **71%** | -15 pp | -10 pp |
| `target_position_random` · 目标位置随机化 | **87%** | **78%** | +3 pp | -3 pp |
| `combined_mild` · 综合轻度扰动 | **76%** | **85%** | -8 pp | +4 pp |
| `combined_hard` · 综合较强扰动 | **58%** | **53%** | -26 pp | -28 pp |

## 与基线的配对检验

| 场景 | DLS McNemar p | OSC McNemar p | 12 项 Bonferroni 后显著？ |
|---|---:|---:|---|
| `mass_random` | 0.387695 | 0.226562 | 否 |
| `friction_random` | 0.607239 | 0.814529 | 否 |
| `object_position_wide` | 0.0106738 | 0.0871586 | 否 |
| `target_position_random` | 0.663624 | 0.677639 | 否 |
| `combined_mild` | 0.168638 | 0.557197 | 否 |
| `combined_hard` | 0.000106045 | 4.25596e-06 | DLS、OSC 均显著 |

## 同场景 DLS/OSC 配对比较

| 场景 | 仅 DLS 成功 | 仅 OSC 成功 | McNemar p |
|---|---:|---:|---:|
| `baseline` | 14 | 11 | 0.6900 |
| `mass_random` | 10 | 8 | 0.8145 |
| `friction_random` | 12 | 14 | 0.8450 |
| `object_position_wide` | 15 | 17 | 0.8601 |
| `target_position_random` | 18 | 9 | 0.1221 |
| `combined_mild` | 10 | 19 | 0.1360 |
| `combined_hard` | 20 | 15 | 0.4996 |

## 实验结论

1. 质量 0.6–1.4× 与整个接触系统摩擦 0.5–1.5× 条件下，成功率相对基线没有统计显著变化；质量随机化后表面上升不能解释为策略变强。
2. 把物体 XY 初始范围从 ±4 cm 扩展到 ±6 cm，DLS 由 84% 降至 69%，OSC 由 81% 降至 71%。DLS 未校正 p=0.0107，但 12 项多重检验校正后不显著。
3. 综合较强扰动（质量 0.6–1.4×、接触摩擦 0.5–1.5×、物体 XY ±6 cm、目标 XY ±3 cm）下降最显著：DLS 58%，OSC 53%；相对基线分别 −26 和 −28 个百分点，校正后仍显著。
4. 所有场景下 DLS 与 OSC 的逐组配对成功率差异均未达到 p<0.05，不能宣布某一种控制器整体鲁棒性更优。
5. 更强扰动时最终 XY 位置失配明显增加：综合较强扰动中，两控制器均有 39 个 episode 未通过最终 XY 判据（计数可能与其他失败条件重叠）。

## 原始数据注意事项

- 本次**粘贴文本**的 `combined_hard.osc` 段落缺少 `unified_lift` 字段，也缺少 `paired_vs_baseline` 键名；该处仅转录终端可见的成功次数与 p 值，不补造缺失字段。
- 请在最终 GitHub Release 中将原始 `summary.json`、`metadata.json`、`design.json` 和完整 `episodes.csv` 按版本归档，以核对所有实际采样值及文件哈希。
- 这是固定策略在仿真中的**测试时参数随机化**，不是利用域随机化进行训练，更不等同于 sim-to-real 验证。
- 因为 SAC 原来在 DLS 下训练，OSC 组只可解释为固定策略切换控制器的泛化性能，不是两控制器各自训练后的性能上限。

统计说明：`paired_vs_baseline_p` 和 `paired_controller_p` 为每组 100 个配对案例的精确 McNemar 检验（来自终端摘要）。单因素+综合的 6×2=12 项对基线比较采用 Bonferroni 阈值 0.05/12≈0.00417 时，仅综合较强扰动 DLS 与 OSC 的下降仍显著。每种失败条件的计数可能重叠，不应相加解释为失败 episode 总数。

