"""SAC-v2 Reach: continue the learned policy for 1 s after first 3-step success.

Put this script next to panda_reach_env_v2.py in sac_v2_ablation/, with the
trained weight at results_v2/sac_panda_reach_v2.zip.

Commands:
    python evaluate_sac_v2_hold.py --episodes 50 --seed 2027
    python evaluate_sac_v2_hold.py --episodes 1 --seed 2060

'policy' continues SAC predictions. 'zero' holds the accumulated joint-position
command constant (action = zeros); it is a diagnosis baseline, not a learned policy.

Evaluation-only code intentionally simulates past Gymnasium's terminal flag, by
advancing MuJoCo directly. Do NOT use this continuation in training code.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from stable_baselines3 import SAC

from panda_reach_env_v2 import PandaReachEnvV2


def post_success_step(env: PandaReachEnvV2, action: np.ndarray):
    """Physically reproduce env.step's actuation, without ending the episode."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise ValueError("Expected finite SAC action with shape (7,)")
    action = np.clip(action, -1.0, 1.0)

    # These are exactly the position-command and MuJoCo integration operations
    # in PandaReachEnvV2.step(). Rewards/terminal flags are deliberately omitted.
    env._q_command = np.clip(
        env._q_command + action * env.action_delta_rad,
        env.lower,
        env.upper,
    )
    env.data.ctrl[:7] = env._q_command
    env.data.ctrl[7] = 255.0
    for _ in range(env.frame_skip):
        mujoco.mj_step(env.model, env.data)

    env.step_count += 1  # for consistent internal bookkeeping
    distance_m = env._distance()
    inside = distance_m <= env.distance_threshold
    env.success_streak = env.success_streak + 1 if inside else 0
    return env._get_obs(), distance_m * 1000.0


def run_one(model, env, seed, mode, hold_steps, threshold_mm):
    obs, _ = env.reset(seed=seed)
    initial_distance_mm = env._distance() * 1000.0
    reached = False
    reach_step = None
    for _ in range(env.max_episode_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated:
            reached = True
            reach_step = env.step_count
            break
        if truncated:
            break

    control_dt = float(env.model.opt.timestep * env.frame_skip)
    result = {
        "seed": int(seed),
        "mode": mode,
        "initial_distance_mm": float(initial_distance_mm),
        "reached_before_timeout": bool(reached),
        "reach_step": reach_step if reached else "",
        "reach_time_s": (reach_step * control_dt) if reached else "",
        "distance_at_success_mm": (env._distance() * 1000.0) if reached else "",
        "hold_success": False,
        "hold_max_mm": "",
        "hold_mean_mm": "",
        "hold_final_mm": "",
        "out_of_5mm_steps": "",
        "hold_max_action_abs": "",
        "hold_mean_action_l2": "",
    }
    traces = []
    if not reached:
        return result, traces

    # At time=0, original Gymnasium task has just terminated successfully.
    # A hold test assesses only the following hold_steps control samples.
    hold_distances = []
    hold_action_norms = []
    hold_action_abs = []
    for k in range(1, hold_steps + 1):
        if mode == "policy":
            action, _ = model.predict(obs, deterministic=True)
        elif mode == "zero":
            action = np.zeros(7, dtype=np.float32)
        else:
            raise ValueError("Unknown mode: " + mode)

        obs, distance_mm = post_success_step(env, action)
        action = np.asarray(action, dtype=np.float64)
        hold_distances.append(distance_mm)
        hold_action_norms.append(float(np.linalg.norm(action)))
        hold_action_abs.append(float(np.max(np.abs(action))))
        q_actual = env.data.qpos[env.qpos_ids]
        traces.append({
            "seed": int(seed),
            "mode": mode,
            "elapsed_since_success_s": k * control_dt,
            "distance_mm": distance_mm,
            "within_5mm": bool(distance_mm <= threshold_mm),
            "action_l2": hold_action_norms[-1],
            "action_max_abs": hold_action_abs[-1],
            "max_joint_cmd_tracking_error_rad": float(
                np.max(np.abs(env._q_command - q_actual))
            ),
        })

    hold_distances = np.asarray(hold_distances, dtype=np.float64)
    result.update({
        "hold_success": bool(np.all(hold_distances <= threshold_mm)),
        "hold_max_mm": float(hold_distances.max()),
        "hold_mean_mm": float(hold_distances.mean()),
        "hold_final_mm": float(hold_distances[-1]),
        "out_of_5mm_steps": int(np.sum(hold_distances > threshold_mm)),
        "hold_max_action_abs": float(np.max(hold_action_abs)),
        "hold_mean_action_l2": float(np.mean(hold_action_norms)),
    })
    return result, traces


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--hold-sec", type=float, default=1.0)
    parser.add_argument("--mode", choices=("policy", "zero", "both"), default="both")
    args = parser.parse_args()
    if args.episodes < 1 or args.hold_sec <= 0:
        parser.error("--episodes must be >= 1 and --hold-sec > 0")

    here = Path(__file__).resolve().parent
    path = here / "results_v2" / "sac_panda_reach_v2.zip"
    if not path.is_file():
        raise FileNotFoundError(f"Trained SAC-v2 weights not found: {path}")
    policy = SAC.load(str(path), device="cpu")
    env = PandaReachEnvV2()
    out = here / "results_v2" / "hold_evaluation"
    out.mkdir(parents=True, exist_ok=True)

    control_dt = float(env.model.opt.timestep * env.frame_skip)
    hold_steps = int(round(args.hold_sec / control_dt))
    if hold_steps < 1:
        raise ValueError("hold-sec is shorter than one control period")
    true_duration = hold_steps * control_dt
    threshold_mm = float(env.distance_threshold * 1000.0)
    modes = ["policy", "zero"] if args.mode == "both" else [args.mode]
    print(f"Control period: {control_dt:.4f}s; hold steps: {hold_steps}; "
          f"actual duration: {true_duration:.4f}s; threshold: {threshold_mm:g}mm")

    results = []
    traces = []
    try:
        for mode in modes:
            for ep in range(args.episodes):
                seed = args.seed + ep
                result, trace = run_one(policy, env, seed, mode, hold_steps, threshold_mm)
                results.append(result)
                traces.extend(trace)
                text = (f"max={result['hold_max_mm']:.2f}mm, "
                        f"outside={result['out_of_5mm_steps']}/{hold_steps}, "
                        f"hold_pass={result['hold_success']}") if result["reached_before_timeout"] else "not reached"
                print(f"{mode:6s} seed={seed} {ep + 1:02d}/{args.episodes} {text}")
    finally:
        env.close()

    write_csv(out / "hold_summary.csv", results, list(results[0].keys()))
    write_csv(
        out / "hold_trace.csv", traces,
        ["seed", "mode", "elapsed_since_success_s", "distance_mm", "within_5mm",
         "action_l2", "action_max_abs", "max_joint_cmd_tracking_error_rad"],
    )

    for mode in modes:
        subset = [r for r in results if r["mode"] == mode]
        reached = [r for r in subset if r["reached_before_timeout"]]
        passed = sum(bool(r["hold_success"]) for r in subset)
        print(f"\n{mode}: Reach {len(reached)}/{len(subset)}; "
              f"Reach + stable 1 s {passed}/{len(subset)} "
              f"({passed / len(subset) * 100:.1f}%); "
              f"Conditional hold rate {passed / len(reached) * 100:.1f}%"
              if reached else f"\n{mode}: no Reach successes")
        if reached:
            worst = max(reached, key=lambda r: r["hold_max_mm"])
            print(f"  Worst holding: seed={worst['seed']}, "
                  f"max error={worst['hold_max_mm']:.3f} mm")

        # Plot one error curve per successfully reached goal.
        plt.figure(figsize=(9, 5))
        for seed in range(args.seed, args.seed + args.episodes):
            line = [r for r in traces if r["mode"] == mode and r["seed"] == seed]
            if line:
                plt.plot(
                    [r["elapsed_since_success_s"] for r in line],
                    [r["distance_mm"] for r in line], alpha=0.35,
                )
        plt.axhline(threshold_mm, linestyle="--", label=f"{threshold_mm:g} mm threshold")
        plt.xlabel("Time after first 3-step Reach success (s)")
        plt.ylabel("Distance to target (mm)")
        plt.title(f"SAC-v2 continued hold: {mode}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"hold_error_{mode}.png", dpi=180)
        plt.close()

    print(f"\nSaved CSV and plots to: {out}")


if __name__ == "__main__":
    main()
