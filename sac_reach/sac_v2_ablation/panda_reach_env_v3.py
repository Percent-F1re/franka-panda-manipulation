"""Hold-aware SAC environment. Uses identical 30-D observations and actions as V2.

A successful Reach (3 consecutive samples within tolerance) starts a 25-step
hold phase. Only uninterrupted hold yields episode success. A violation resets
hold progress, but the episode continues until success or time limit.
"""
import numpy as np
from panda_reach_env_v2 import PandaReachEnvV2


class PandaReachEnvV3(PandaReachEnvV2):
    def __init__(self, hold_seconds=1.0, max_episode_steps=175,
                 hold_bonus=15.0, hold_step_bonus=0.25,
                 **kwargs):
        super().__init__(max_episode_steps=max_episode_steps, **kwargs)
        self.control_dt = float(self.model.opt.timestep * self.frame_skip)
        self.hold_steps = max(1, int(np.ceil(hold_seconds / self.control_dt)))
        self.hold_bonus = float(hold_bonus)
        self.hold_step_bonus = float(hold_step_bonus)
        self.phase = 0  # 0 approach, 1 hold
        self.hold_count = 0
        # Extra phase flag and normalized hold progress: 30 -> 32.
        from gymnasium import spaces
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(32,), dtype=np.float32)

    def _get_obs(self):
        base = super()._get_obs()
        return np.concatenate((base, np.array([
            float(self.phase), self.hold_count / self.hold_steps
        ], dtype=np.float32)))

    def reset(self, *, seed=None, options=None):
        self.phase = 0
        self.hold_count = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        # V2 advances the simulator once. Its terminal flag is overridden here.
        old_phase = self.phase
        obs, reward, v2_done, _v2_truncated, info = super().step(action)
        inside = info['distance_mm'] <= self.distance_threshold * 1000.0
        if old_phase == 0 and v2_done:
            self.phase = 1
            self.hold_count = 0
            # Remove V2's +10 early termination bonus.
            reward -= 10.0
        elif old_phase == 1:
            if inside:
                self.hold_count += 1
                reward += self.hold_step_bonus
            else:
                # Failed hold attempt: start another Reach / Hold cycle.
                self.phase = 0
                self.hold_count = 0
                self.success_streak = 0
                reward -= 1.0
        done = self.phase == 1 and self.hold_count >= self.hold_steps
        if done:
            reward += self.hold_bonus
        truncated = self.step_count >= self.max_episode_steps and not done
        info.update({
            'is_success': bool(done), 'phase': 'hold' if self.phase else 'reach',
            'hold_count': int(self.hold_count), 'hold_steps': int(self.hold_steps),
            'reach_triggered': bool(old_phase == 0 and v2_done),
        })
        # super().step constructed observation before phase updates.
        return self._get_obs(), float(reward), bool(done), bool(truncated), info
