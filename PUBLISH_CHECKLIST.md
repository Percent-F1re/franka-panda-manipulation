# 发布前检查表

- [ ] **使用单独的发布文件夹**；原来的 `robot_simulation` 项目备份，不在原项目上直接试错。
- [ ] Stage 1～4 文件和 `sac_pickplace/`、`sac_reach/` 均按原有相对位置复制。
- [ ] `mujoco_menagerie` Panda 模型可由 README 命令重新获取；记录上游 commit。
- [ ] `models/sac_grasp_best.zip` 与 `models/sac_pickplace_best.zip` 放入正确位置；检查文件尺寸。
- [ ] 填写 `docs/environment_versions.md`，确保没有 TODO。
- [ ] `python -m unittest -v test_benchmark_common` 通过。
- [ ] 规则式专家 Stage 4 能运行；SAC benchmark 5 例 smoke 测试能跑通。
- [ ] 最终 README 中的命令已从新目录实际执行。
- [ ] 100 组报告真实标明 SAC 严格成功率为 **78%**；不要只展示原环境内的 100%。
- [ ] README 展示至少一段操作 GIF/视频、训练曲线以及配对误差图（素材需自己确认）。
- [ ] 将第三方模型来源及 Apache-2.0 声明写清；核对版权许可，再选自己的 LICENSE。
- [ ] 检查 `git status --short` / `git diff --cached --name-only`，确认没有敏感路径、私人文件、数据集冗余、缓存、超大权重。
- [ ] 公开仓库后，从一个新目录 `git clone`，按照 README 完成一次安装与评测。
- [ ] 用 GitHub Release 打 `v0.1.0` 标签，注明当前 limitations 和可复现 benchmark。
