# Unified physical Pick-and-Place benchmark

## Protocol

- Simulator: MuJoCo Franka Panda with a free, frictional cube and a tabletop.
- Methods: ten-stage scripted 6D-DLS expert vs state-based SAC policy with 6D-DLS low-level control.
- Test cases: 100 **paired** random cube initial XY offsets sampled uniformly in [-0.04, +0.04] m, seed 45000.
- Policy checkpoint: Pick-and-Place SAC best checkpoint selected from prior validation (not trained or selected using these 100 trials).
- Each method receives the **same initial cube pose** within each paired case (the audit checks settled XY agreement).
- Following control termination, the final arm-joint and gripper commands are **frozen** and the MuJoCo simulation advances for **1 second**. No forced gripper opening or active corrective motion is allowed.

## Success criteria (fixed before evaluating)

| Item | Threshold/condition |
|---|---|
| Valid lift | cube rises at least 40 mm and remains off table with bilateral gripper contact |
| Valid transfer | cube horizontally moves at least 35 mm while elevated |
| Event dwell | valid lift and transfer must each persist for at least 0.12 s |
| Final position | center XY error ≤ 20 mm |
| Final height | cube bottom-to-table gap absolute error ≤ 10 mm |
| Final support | cube contacts tabletop |
| Gripper release | neither finger contacts cube |
| Final speed | cube linear speed ≤ 0.03 m/s |
| Final drift | cube drift ≤ 10 mm over last 0.6 s of post-observation |

The criteria are evaluated from MuJoCo physical state, not from the RL reward or the environment's terminal flag. The 0.12 s event dwell is an added **benchmark-only** constraint; it is not claimed to be the original Stage 4 condition.

## Outcomes: 100 held-out paired trials

| Metric | Expert | SAC |
|---|---:|---:|
| Valid lift | 100 / 100 | 100 / 100 |
| Valid transfer | 100 / 100 | 100 / 100 |
| Unified stable place | 100 / 100 | 78 / 100 |
| Mean XY error | 0.0977 mm | 5.4565 mm |
| Mean simulated task time | 22.00 s | 3.3916 s |
| Mean post-observation drift | 0 mm | 0.1404 mm |

Per-case data are preserved in [episodes.csv](../results/unified_benchmark/episodes.csv) and source summary in [summary.json](../results/unified_benchmark/summary.json). See [paired XY error plot](../results/unified_benchmark/paired_xy_error_100.png).

### Failure analysis

22 of 100 SAC trials failed the unified final success condition. Based on episode-level checks, 21 involved residual finger–cube contact after the 1 s post-action window; one involved XY position out of tolerance. These observations motivate more careful release control and termination-condition analysis. They **do not establish** whether contact reappears due to gripper command, arm position, or another dynamics effect; that mechanism requires logging.

### Interpretation limitations

- Expert and SAC operate under different action-generation approaches and terminate differently; the 22 s vs ~3.4 s numbers describe these particular implementations, not an isolated speed improvement.
- The expert sees simulator cube state and uses a staged task plan; SAC uses simulator observations. Neither is a visual, real-robot policy.
- SAC 78/100 is an empirical estimate on one fixed model and target distribution; it is not universal success probability.
- Do not substitute the earlier 50/50 original-environment success number for the stricter unified score.
