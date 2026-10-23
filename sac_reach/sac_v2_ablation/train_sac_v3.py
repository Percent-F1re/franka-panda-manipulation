"""Train SAC-v3 from scratch or warm-start compatible V2 policy weights.

Warm start copies policy parameters only (30->32 obs dims are incompatible).
Therefore this script intentionally trains from scratch.
"""
import argparse
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from panda_reach_env_v3 import PandaReachEnvV3


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=300_000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--hold-sec', type=float, default=1.0)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    out = root / 'results_v3' / f'seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    probe = PandaReachEnvV3(hold_seconds=args.hold_sec)
    check_env(probe, warn=True)
    print('Observation:', probe.observation_space.shape,
          'hold steps:', probe.hold_steps, 'dt:', probe.control_dt)
    probe.close()
    env = Monitor(PandaReachEnvV3(hold_seconds=args.hold_sec),
                  filename=str(out / 'train_monitor.csv'),
                  info_keywords=('phase', 'hold_count'))
    env.reset(seed=args.seed)
    model = SAC('MlpPolicy', env, learning_rate=3e-4,
                buffer_size=100_000, learning_starts=2_000,
                batch_size=256, gamma=0.99, train_freq=1,
                gradient_steps=1, policy_kwargs=dict(net_arch=[256, 256]),
                verbose=1, seed=args.seed, device='auto')
    callback = CheckpointCallback(save_freq=50_000,
                                  save_path=str(out / 'checkpoints'),
                                  name_prefix='sac_v3')
    try:
        model.learn(total_timesteps=args.steps, callback=callback)
        model.save(str(out / 'sac_panda_reach_v3'))
        print('Saved:', out / 'sac_panda_reach_v3.zip')
    finally:
        env.close()

if __name__ == '__main__':
    main()
