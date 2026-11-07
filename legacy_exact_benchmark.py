"""Original reward-independent 1-second physical benchmark, robosuite owned sim."""
from __future__ import annotations
import numpy as np
from legacy_exact_rules import BenchmarkRules, audit_events, audit_final


def sample_physics(env) -> dict:
    s = env._physical_state()
    cube_xyz = np.asarray(s['cube_xyz_m'], dtype=float).copy()
    return dict(
        time_s=float(env.data.time),
        cube_xyz=cube_xyz,
        rise_mm=float(s['rise_m'] * 1000),
        horizontal_moved_mm=1000.0 * float(np.linalg.norm(cube_xyz[:2] - env.cube_start[:2])),
        xy_error_mm=float(s['cube_xy_error_m'] * 1000),
        bottom_gap_mm=float(s['bottom_gap_m'] * 1000),
        table_contacts=int(s['table_contact']),
        left_contacts=int(s['left_contact']),
        right_contacts=int(s['right_contact']),
        speed_m_s=float(s['cube_speed_m_s']),
    )


def hold_last_controls_for_one_second(env, rules=None):
    """Like original benchmark: freeze current q/gripper ctrl and continue physics.

    Never recompute DLS, send actions, teleport, detach or open gripper.
    """
    rules = rules or BenchmarkRules()
    held = env.data.ctrl.copy()
    dt = float(env.model.opt.timestep)
    samples=[]
    for _ in range(max(1, round(rules.post_observation_s / dt))):
        env.data.ctrl[:] = held
        env.raw_env.sim.step()
        samples.append(sample_physics(env))
    return samples


def evaluate_episode(model, env, *, seed: int, xy_offset=None,
                     rules=None, save_trace=False):
    rules = rules or BenchmarkRules()
    options = {'xy_offset': np.asarray(xy_offset, dtype=float)} if xy_offset is not None else None
    obs, info = env.reset(seed=seed, options=options)
    samples=[]
    trace=[]
    for i in range(env.max_episode_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        snap = sample_physics(env)
        samples.append(snap)
        if save_trace:
            trace.append({
                'step':i+1, 'time_s':snap['time_s'],
                'a_x':float(action[0]), 'a_y':float(action[1]),
                'a_z':float(action[2]), 'a_grip':float(action[3]),
                'cube_rise_mm':snap['rise_mm'],
                'xy_error_mm':snap['xy_error_mm'],
                'table_contacts':snap['table_contacts'],
                'left_contacts':snap['left_contacts'],
                'right_contacts':snap['right_contacts'],
                'finger_opening_mm':1000.0 * float(sum(env.data.qpos[j] for j in env.finger_qpos_ids)),
                'grasp_distance_mm':1000.0 * env._grasp_distance(),
                'joint_tracking_max_rad':float(np.max(np.abs(env.q_cmd-env.data.qpos[env.qpos_ids]))),
            })
        if terminated or truncated:
            break
    else:
        raise RuntimeError('Episode failed to terminate')
    lift, transfer = audit_events(samples, rules)
    post = hold_last_controls_for_one_second(env, rules)
    physical = audit_final(post, lift, transfer, rules)
    record = {
        'seed':int(seed),
        'steps':i+1,
        'original_is_success':bool(info['is_success']),
        'original_lift':bool(info['has_lifted']),
        'original_transfer':bool(info['has_transferred']),
        **physical,
    }
    return record, trace
