# Reproduction environment (complete before public release)

- OS used for published experiment: Windows (enter exact version)
- Python version: **TODO** (`python --version`)
- MuJoCo package version: **TODO** (`python -m pip show mujoco`)
- Gymnasium version: **TODO**
- Stable-Baselines3 version: **TODO**
- NumPy / SciPy versions: **TODO**
- MuJoCo Menagerie commit: **TODO** (`git -C mujoco_menagerie rev-parse HEAD`)
- SAC Pick-and-Place policy source: `results_pickplace/seed_42/best_model.zip`
- SAC Grasp pretrained source: `results_grasp/seed_42/best_model.zip`
- Pick-and-Place training budget: 500,000 fine-tuning steps
- Benchmark: 100 paired initial targets, seed 45000, XY ±0.04 m
- Known limitations: one training seed, privileged object state, single cube, fixed target offset, no physical robot
