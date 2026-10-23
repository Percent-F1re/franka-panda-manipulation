"""Train SAC on PandaReachEnv. Run: python sac_reach/train_sac.py --steps 100000"""
import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from panda_reach_env import PandaReachEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)

    probe = PandaReachEnv()
    check_env(probe, warn=True)
    obs, info = probe.reset(seed=args.seed)
    print("Environment OK; observation:", obs.shape, "action:", probe.action_space.shape)
    probe.close()

    env = Monitor(PandaReachEnv(), filename=str(out / "train_monitor.csv"))
    env.reset(seed=args.seed)
    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=256,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        device="auto",  # change to 'auto' if GPU PyTorch is configured
        tensorboard_log=None,
    )
    checkpoint = CheckpointCallback(
        save_freq=25_000, save_path=str(out / "checkpoints"), name_prefix="sac_panda"
    )
    model.learn(total_timesteps=args.steps, callback=checkpoint, progress_bar=False)
    saved = out / "sac_panda_reach"
    model.save(str(saved))
    env.close()
    print("Saved model:", str(saved) + ".zip")


if __name__ == "__main__":
    main()
