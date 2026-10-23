"""Evaluate a trained SAC policy on 50 held-out goals; save CSV."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from panda_reach_env import PandaReachEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / "results"
    model = SAC.load(str(out / "sac_panda_reach"), device="cpu")
    env = PandaReachEnv()
    records = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        last = {}
        for t in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, last = env.step(action)
            if terminated or truncated:
                break
        records.append({
            "episode": ep + 1,
            "goal_x": float(env.goal[0]),
            "goal_y": float(env.goal[1]),
            "goal_z": float(env.goal[2]),
            "success": bool(last["is_success"]),
            "final_distance_mm": float(last["distance_mm"]),
            "steps": int(last["step_count"]),
        })
        print(f"{ep+1:02d}/{args.episodes}: success={last['is_success']}, distance={last['distance_mm']:.2f} mm")
    env.close()
    out.mkdir(exist_ok=True)
    with (out / "evaluation.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Success rate: {100 * np.mean([r['success'] for r in records]):.1f}%")
    print(f"Mean final distance: {np.mean([r['final_distance_mm'] for r in records]):.2f} mm")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    errors = np.asarray([r["final_distance_mm"] for r in records])
    ax.bar(np.arange(1, len(records) + 1), errors, width=0.8)
    ax.axhline(5.0, linestyle="--", color="black", label="15 mm threshold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Final distance to goal (mm)")
    ax.set_title("SAC Panda Reach: evaluation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "evaluation_errors.png", dpi=180)
    plt.close(fig)
    print("CSV:", out / "evaluation.csv")
    print("Plot:", out / "evaluation_errors.png")


if __name__ == "__main__":
    main()
