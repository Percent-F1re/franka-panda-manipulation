"""Stage 4: randomized physical Pick-and-Place benchmark for Franka Panda.

Requires the user's working Stage 1, 2, 3 scripts alongside this script.
Run: python panda_pickplace_stage4_random.py

Retains the Stage 3 controller, time constants, contact checks and grasp offset.
50 fixed-seed initial cube offsets (XY +/-4cm). The place target is the
*settled* cube XY plus Stage 3's fixed transfer offset (+6cm world Y).
This tests adaptive grasping, not fixed-world-target placement.
"""
import csv
import json
import os
import traceback
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

import panda_pickplace_stage3 as s3
from panda_pickplace_stage1 import CUBE_POS, CUBE_HALF_WIDTH
from panda_pickplace_stage2_grasp import (
    HAND_TO_GRASP_LOCAL, GRIPPER_OPEN, GRIPPER_CLOSED,
    JOINT_LIMIT_MARGIN, K_POS, K_ROT, DAMPING, MAX_JOINT_CMD_SPEED,
    mj_id, orientation_error_world, quintic, count_cube_contacts,
)

NUM_EPISODES = 50
RANDOM_SEED = 42
RANDOM_XY_RANGE_M = 0.04
SHOW_VIEWER = False           # Batch experiments: set True only for debugging
SAVE_EXPERT_DATA = False       # Turn on after validating the expert
SAVE_DETAILED_LOGS = False     # True: one CSV per episode (can be large)
EXPERT_SAMPLE_INTERVAL = 5    # Subsample every 5 physics steps for saved demonstrations

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'pickplace_stage4_results'
OUT.mkdir(parents=True, exist_ok=True)

# As a safety/diagnostic detail, place exactly as Stage3: relative 6cm transfer.
TRANSFER_XY = np.asarray(s3.TRANSFER_OFFSET_XY, dtype=float)

# ========== Model loaded once, no 50 XML recompilations ==========
scene_path = s3.make_stage3_scene()
os.chdir(ROOT)  # relative ASCII path handles the user's Chinese Windows directory
scene_relative = 'mujoco_menagerie/franka_emika_panda/' + scene_path.name
model = mujoco.MjModel.from_xml_path(scene_relative)
data = mujoco.MjData(model)

home = mj_id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
hand = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
cube = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
cube_geom = mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'cube_geom')
table_geom = mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'table_top')
left = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_finger')
right = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_finger')
left_joint = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint1')
right_joint = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint2')
cube_free = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_free')
grip_act = mj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'actuator8')
arm_joints = np.array([mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i}') for i in range(1, 8)])
qpos_ids = model.jnt_qposadr[arm_joints]
dof_ids = model.jnt_dofadr[arm_joints]
cube_qpos_addr = int(model.jnt_qposadr[cube_free])
cube_dof_addr = int(model.jnt_dofadr[cube_free])
finger_qpos_ids = [int(model.jnt_qposadr[left_joint]), int(model.jnt_qposadr[right_joint])]
qlo = np.maximum(model.jnt_range[arm_joints, 0], model.actuator_ctrlrange[:7, 0]) + JOINT_LIMIT_MARGIN
qhi = np.minimum(model.jnt_range[arm_joints, 1], model.actuator_ctrlrange[:7, 1]) - JOINT_LIMIT_MARGIN
if model.nu < 8 or grip_act != 7 or np.any(qlo >= qhi):
    raise RuntimeError('Unexpected actuator or joint limits; check Panda model')
dt = float(model.opt.timestep)
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))


def pose():
    return data.xpos[hand].copy(), data.xmat[hand].reshape(3, 3).copy()


def contacts():
    return count_cube_contacts(model, data, cube_geom, table_geom, left, right)


def reset_episode(initial_xy):
    """Fully restore keyframe, then move ONLY the cube's free joint pose.

    This is initial-condition randomization, not teleporting the cube while
    executing the pick-place task. Quaternion is kept unit [w,x,y,z].
    """
    mujoco.mj_resetDataKeyframe(model, data, home)
    data.qpos[cube_qpos_addr:cube_qpos_addr+3] = [*initial_xy, CUBE_POS[2]]
    data.qpos[cube_qpos_addr+3:cube_qpos_addr+7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    data.qacc_warmstart[:] = 0.0
    data.ctrl[:] = 0.0
    q_cmd = np.clip(data.qpos[qpos_ids].copy(), qlo, qhi)
    data.ctrl[:7] = q_cmd
    data.ctrl[grip_act] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)
    # Re-settle cube from its new randomized XY on the table, as in Stage3.
    for _ in range(round(s3.SETTLE_BEFORE_GRASP / dt)):
        data.ctrl[:7] = q_cmd
        data.ctrl[grip_act] = GRIPPER_OPEN
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return q_cmd


def expert_observation_action(stage, target_cube_xy, desired_p, q_cmd, command):
    """Extension point for Behavior Cloning: (observation, action).

    Expert action is the absolute position-control target for 7 Panda joints,
    followed by gripper control (0 or 255). Observations use simulated state
    and privileged cube position; no cameras and no policy training yet.
    Only call AFTER a step and store the action applied during that step.
    """
    p, rot = pose()
    obs = np.concatenate([
        data.qpos[qpos_ids].copy(), data.qvel[dof_ids].copy(),
        p, rot.reshape(-1), data.xpos[cube].copy(),
        np.array([sum(data.qpos[i] for i in finger_qpos_ids)]),
        np.asarray(target_cube_xy), np.asarray(desired_p)
    ])
    act = np.concatenate([np.asarray(q_cmd).copy(), [float(command)]])
    return obs, act


def run_episode(ep_no, xy_offset, save_expert=False, save_log=False):
    initial_xy = np.asarray(CUBE_POS[:2], dtype=float) + xy_offset
    qcmd = reset_episode(initial_xy)
    cube_start = data.xpos[cube].copy()  # actual SETTLED position, not nominal
    table_z = float(data.geom_xpos[table_geom, 2] + model.geom_size[table_geom, 2])
    _, R_target = pose()
    alignment = float(R_target[:, 2] @ np.array([0, 0, -1.0]))
    if alignment < 0.90:
        raise RuntimeError(f'Gripper not facing down: alignment={alignment:.3f}')

    grasp = cube_start - R_target @ HAND_TO_GRASP_LOCAL
    pregrasp = grasp + [0, 0, s3.PREGRASP_HEIGHT]
    lift = grasp + [0, 0, s3.LIFT_HEIGHT]
    shift = np.array([TRANSFER_XY[0], TRANSFER_XY[1], 0.0])
    transfer = lift + shift
    place = grasp + shift
    retreat = place + [0, 0, s3.RETREAT_HEIGHT]
    target_xy = cube_start[:2] + TRANSFER_XY

    stages = [
        ('approach', pregrasp, s3.APPROACH_TIME, GRIPPER_OPEN),
        ('descend', grasp, s3.DESCEND_TIME, GRIPPER_OPEN),
        ('close', grasp, s3.CLOSE_TIME, GRIPPER_CLOSED),
        ('lift', lift, s3.LIFT_TIME, GRIPPER_CLOSED),
        ('hold', lift, s3.HOLD_TIME, GRIPPER_CLOSED),
        ('transfer', transfer, s3.TRANSFER_TIME, GRIPPER_CLOSED),
        ('place', place, s3.PLACE_TIME, GRIPPER_CLOSED),
        ('release', place, s3.RELEASE_TIME, GRIPPER_OPEN),
        ('retreat', retreat, s3.RETREAT_TIME, GRIPPER_OPEN),
        ('settle', retreat, s3.FINAL_SETTLE_TIME, GRIPPER_OPEN),
    ]
    checks = {'lift': False, 'transfer': False}
    rows = []
    examples_obs, examples_act, examples_stages = [], [], []
    clip_steps = 0
    reason = ''
    completed = True
    step_counter = 0
    sample_every = max(1, round(0.01 / dt))  # compact diagnostics at ~100 Hz
    viewer_cm = mujoco.viewer.launch_passive(model, data) if SHOW_VIEWER else nullcontext(None)

    def record(stage, desired_p, applied_gripper):
        xyz = data.xpos[cube].copy()
        tbl, lc, rc = contacts()
        rows.append({
            'episode': ep_no, 'time_s': float(data.time), 'stage': stage,
            'cube_x_m': float(xyz[0]), 'cube_y_m': float(xyz[1]),
            'cube_z_m': float(xyz[2]),
            'cube_speed_m_s': float(np.linalg.norm(data.qvel[cube_dof_addr:cube_dof_addr+3])),
            'target_x_m': float(target_xy[0]), 'target_y_m': float(target_xy[1]),
            'hand_pos_error_mm': float(1000 * np.linalg.norm(desired_p - data.xpos[hand])),
            'opening_mm': float(1000 * sum(data.qpos[i] for i in finger_qpos_ids)),
            'table_contacts': int(tbl), 'left_contacts': int(lc), 'right_contacts': int(rc),
            'gripper_command': float(applied_gripper),
        })

    with viewer_cm as viewer:
        for stage, target, duration, grip_cmd in stages:
            if stage == 'close':
                hand_err = 1000 * float(np.linalg.norm(grasp - data.xpos[hand]))
                if hand_err > s3.POSE_ABORT_MM:
                    reason = f'pregrasp_alignment_error_{hand_err:.1f}mm'
                    completed = False
                    break
            if stage == 'transfer' and not checks['lift']:
                reason = 'lift_verification_failed'
                completed = False
                break

            p0, _ = pose()
            nsteps = max(1, round(duration / dt))
            for i in range(nsteps):
                if viewer is not None and not viewer.is_running():
                    reason = 'viewer_closed'
                    completed = False
                    break
                t = i * dt
                s, sd = quintic(t, duration)
                pdes = p0 + s * (target - p0)
                vff = sd * (target - p0)
                p, R = pose()
                ep = pdes - p
                er = orientation_error_world(R_target, R)
                mujoco.mj_jacBody(model, data, jacp, jacr, hand)
                J = np.vstack((jacp[:, dof_ids], jacr[:, dof_ids]))
                twist = np.hstack((vff + K_POS * ep, K_ROT * er))
                dq = J.T @ np.linalg.solve(J @ J.T + DAMPING**2 * np.eye(6), twist)
                peak = float(np.max(np.abs(dq)))
                if peak > MAX_JOINT_CMD_SPEED:
                    dq *= MAX_JOINT_CMD_SPEED / peak
                raw_qcmd = qcmd + dq * dt
                if np.any((raw_qcmd < qlo) | (raw_qcmd > qhi)):
                    clip_steps += 1
                qcmd = np.clip(raw_qcmd, qlo, qhi)
                data.ctrl[:7] = qcmd
                data.ctrl[grip_act] = grip_cmd
                # Snapshot PRE-step state + action, appropriate for causal BC.
                if save_expert and step_counter % EXPERT_SAMPLE_INTERVAL == 0:
                    obs, act = expert_observation_action(stage, target_xy, pdes, qcmd, grip_cmd)
                    examples_obs.append(obs)
                    examples_act.append(act)
                    examples_stages.append(stage)
                mujoco.mj_step(model, data)
                if step_counter % sample_every == 0 or i == nsteps-1:
                    record(stage, pdes, grip_cmd)
                step_counter += 1
                if viewer is not None:
                    viewer.sync()
            if not completed:
                break
            if stage == 'hold':
                tbl, lc, rc = contacts()
                rise_mm = 1000 * float(data.xpos[cube, 2] - cube_start[2])
                checks['lift'] = bool(rise_mm >= s3.MIN_LIFT_MM and tbl == 0 and lc > 0 and rc > 0)
            if stage == 'transfer':
                sample = [r for r in rows if r['stage'] == 'transfer']
                if sample:
                    min_z = min(r['cube_z_m'] for r in sample)
                    max_tbl = max(r['table_contacts'] for r in sample)
                else:
                    min_z, max_tbl = -np.inf, 1000
                moved = float(np.linalg.norm(data.xpos[cube, :2] - cube_start[:2]))
                checks['transfer'] = bool(
                    min_z - cube_start[2] >= s3.MIN_LIFT_MM / 1000
                    and max_tbl == 0 and moved > 0.035
                )
                if not checks['transfer']:
                    reason = 'cube_slipped_or_fell_during_transfer'
                    completed = False
                    break
            if stage == 'place':
                bottom_gap = data.xpos[cube, 2] - CUBE_HALF_WIDTH - table_z
                if bottom_gap > 0.015:
                    reason = f'placement_above_table_{bottom_gap*1000:.1f}mm'
                    completed = False
                    break

    final_cube = data.xpos[cube].copy()
    final_tbl, final_l, final_r = contacts()
    xy_err = float(1000 * np.linalg.norm(final_cube[:2] - target_xy))
    bottom_gap_mm = float(1000 * (final_cube[2] - CUBE_HALF_WIDTH - table_z))
    speed = float(np.linalg.norm(data.qvel[cube_dof_addr:cube_dof_addr+3]))
    settle_rows = [r for r in rows if r['stage'] == 'settle']
    last_window = [r for r in settle_rows if rows and r['time_s'] >= rows[-1]['time_s'] - 0.6]
    drift = float('inf')
    if len(last_window) >= 2:
        points = np.array([[r['cube_x_m'], r['cube_y_m'], r['cube_z_m']] for r in last_window])
        drift = float(1000 * np.max(np.linalg.norm(points-points[0], axis=1)))
    placed = bool(completed and checks['lift'] and checks['transfer']
                  and xy_err <= s3.XY_TOLERANCE_MM
                  and abs(bottom_gap_mm) <= s3.Z_TOLERANCE_MM
                  and final_tbl > 0 and final_l == 0 and final_r == 0
                  and speed <= s3.STABLE_SPEED_M_S and drift <= s3.STABLE_DRIFT_MM)
    if not placed and not reason:
        if not completed: reason = 'sequence_not_completed'
        elif not checks['lift']: reason = 'lift_failed'
        elif not checks['transfer']: reason = 'transfer_failed'
        elif xy_err > s3.XY_TOLERANCE_MM: reason = 'final_xy_out_of_tolerance'
        elif abs(bottom_gap_mm) > s3.Z_TOLERANCE_MM: reason = 'final_cube_height_invalid'
        elif final_tbl == 0: reason = 'cube_not_on_table'
        elif final_l > 0 or final_r > 0: reason = 'cube_still_touching_fingers'
        elif speed > s3.STABLE_SPEED_M_S: reason = 'cube_not_still'
        elif drift > s3.STABLE_DRIFT_MM: reason = 'cube_drifting'
        else: reason = 'unknown_failure'

    out = dict(
        episode=int(ep_no), offset_x_m=float(xy_offset[0]), offset_y_m=float(xy_offset[1]),
        initial_cube_x_m=float(cube_start[0]), initial_cube_y_m=float(cube_start[1]),
        target_cube_x_m=float(target_xy[0]), target_cube_y_m=float(target_xy[1]),
        final_cube_x_m=float(final_cube[0]), final_cube_y_m=float(final_cube[1]),
        final_cube_z_m=float(final_cube[2]),
        grasp_success=bool(checks['lift']), transfer_success=bool(checks['transfer']),
        place_success=placed, sequence_completed=bool(completed),
        final_xy_error_mm=xy_err, final_bottom_gap_mm=bottom_gap_mm,
        final_cube_speed_m_s=speed, last_0p6s_drift_mm=drift if np.isfinite(drift) else None,
        final_table_contacts=int(final_tbl), final_left_contacts=int(final_l), final_right_contacts=int(final_r),
        joint_command_clipping_steps=int(clip_steps), failure_reason=reason,
    )
    if save_log and rows:
        p = OUT / 'episode_logs'
        p.mkdir(exist_ok=True)
        with (p / f'episode_{ep_no:03d}.csv').open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    if save_expert and placed and examples_obs:
        p = OUT / 'expert_demonstrations'
        p.mkdir(exist_ok=True)
        np.savez_compressed(p / f'episode_{ep_no:03d}.npz',
                            observations=np.asarray(examples_obs, dtype=np.float32),
                            actions=np.asarray(examples_act, dtype=np.float32),
                            stages=np.asarray(examples_stages))
    return out


def save_summary(results, planned_offsets):
    with (OUT / 'episode_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    all_n = len(results)
    grasp_n = sum(r['grasp_success'] for r in results)
    transfer_n = sum(r['transfer_success'] for r in results)
    place_n = sum(r['place_success'] for r in results)
    errs = np.asarray([r['final_xy_error_mm'] for r in results], dtype=float)
    valid = errs[np.isfinite(errs)]
    failure_counts = dict(Counter(r['failure_reason'] for r in results if not r['place_success']))
    agg = {
        'settings': dict(episodes_requested=NUM_EPISODES, episodes_completed=all_n,
                         seed=RANDOM_SEED, offset_range_m=RANDOM_XY_RANGE_M,
                         transfer_offset_xy_m=TRANSFER_XY.tolist(),
                         xy_success_tolerance_mm=s3.XY_TOLERANCE_MM,
                         note='fixed expert parameters copied from working Stage 3'),
        'grasp_success_count': grasp_n, 'grasp_success_rate': grasp_n / all_n,
        'transfer_success_count': transfer_n, 'transfer_success_rate': transfer_n / all_n,
        'place_success_count': place_n, 'place_success_rate': place_n / all_n,
        'mean_final_xy_error_mm_all': float(valid.mean()) if len(valid) else None,
        'median_final_xy_error_mm_all': float(np.median(valid)) if len(valid) else None,
        'max_final_xy_error_mm_all': float(valid.max()) if len(valid) else None,
        'mean_final_xy_error_mm_success_only': (float(np.mean([r['final_xy_error_mm'] for r in results if r['place_success']]))
                                                 if place_n else None),
        'failure_reason_counts': failure_counts,
        'expert_data_saved': bool(SAVE_EXPERT_DATA),
    }
    (OUT / 'batch_summary.json').write_text(json.dumps(agg, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (OUT / 'random_offsets.json').write_text(json.dumps(planned_offsets.tolist(), indent=2), encoding='utf-8')
    return agg


def make_plots(results):
    ids = np.arange(1, len(results)+1)
    err = np.array([r['final_xy_error_mm'] for r in results], dtype=float)
    success = np.array([r['place_success'] for r in results])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(ids[success], err[success], s=24, label='Success', color='tab:green')
    ax.scatter(ids[~success], err[~success], s=24, label='Failure', color='tab:red')
    ax.axhline(s3.XY_TOLERANCE_MM, ls='--', color='black', label='XY tolerance')
    ax.set(xlabel='Episode', ylabel='Final cube XY error (mm)', title='Random Pick-and-Place final XY errors')
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(OUT/'final_xy_errors.png', dpi=180); plt.close(fig)

    names = ['Grasp + lift', 'Transfer', 'Place + settle']
    pct = [100*np.mean([r[k] for r in results]) for k in ('grasp_success', 'transfer_success', 'place_success')]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, pct)
    ax.set_ylim(0, 105); ax.set_ylabel('Success rate (%)')
    ax.set_title('Stage 4 benchmark: 50 randomized cube positions')
    for bar, value in zip(bars, pct):
        ax.text(bar.get_x()+bar.get_width()/2, value+1, f'{value:.1f}%', ha='center')
    fig.tight_layout(); fig.savefig(OUT/'success_rates.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ox = np.array([r['offset_x_m'] for r in results])*100
    oy = np.array([r['offset_y_m'] for r in results])*100
    ax.scatter(ox[success], oy[success], label='Success', color='tab:green')
    ax.scatter(ox[~success], oy[~success], label='Failure', color='tab:red')
    ax.set(xlabel='Initial X offset (cm)', ylabel='Initial Y offset (cm)', title='Cube initial offsets')
    ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/'target_scatter.png', dpi=180); plt.close(fig)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    offsets = rng.uniform(-RANDOM_XY_RANGE_M, RANDOM_XY_RANGE_M, size=(NUM_EPISODES, 2))
    results = []
    print(f'Model nq={model.nq} nv={model.nv} nu={model.nu}; 50 episodes, seed={RANDOM_SEED}')
    print('No figure/viewer in batch mode; this can take a while')
    for index, offset in enumerate(offsets, 1):
        try:
            result = run_episode(index, offset, save_expert=SAVE_EXPERT_DATA, save_log=SAVE_DETAILED_LOGS)
        except Exception as exc:
            # Preserve a failed sample rather than silently skipping it.
            result = dict(
                episode=index, offset_x_m=float(offset[0]), offset_y_m=float(offset[1]),
                initial_cube_x_m=None, initial_cube_y_m=None, target_cube_x_m=None, target_cube_y_m=None,
                final_cube_x_m=None, final_cube_y_m=None, final_cube_z_m=None,
                grasp_success=False, transfer_success=False, place_success=False, sequence_completed=False,
                final_xy_error_mm=float('nan'), final_bottom_gap_mm=float('nan'),
                final_cube_speed_m_s=float('nan'), last_0p6s_drift_mm=None,
                final_table_contacts=0, final_left_contacts=0, final_right_contacts=0,
                joint_command_clipping_steps=0,
                failure_reason='exception_' + type(exc).__name__ + ': ' + str(exc)[:180],
            )
            print('  Error details:', traceback.format_exc(limit=3))
        results.append(result)
        # Save progress every episode so a crash doesn't destroy previous work.
        with (OUT/'episode_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
        print(f'[{index:02d}/{NUM_EPISODES}] offset=({offset[0]*100:+.1f},{offset[1]*100:+.1f})cm '
              f'grasp={result["grasp_success"]} transfer={result["transfer_success"]} '
              f'place={result["place_success"]} xy_err={result["final_xy_error_mm"]:.2f}mm '
              f'reason={result["failure_reason"] or "none"}', flush=True)
    agg = save_summary(results, offsets)
    make_plots(results)
    print('\n====== Stage 4 summary ======')
    print(f'Grasp:    {agg["grasp_success_count"]}/{len(results)} ({100*agg["grasp_success_rate"]:.1f}%)')
    print(f'Transfer: {agg["transfer_success_count"]}/{len(results)} ({100*agg["transfer_success_rate"]:.1f}%)')
    print(f'Place:    {agg["place_success_count"]}/{len(results)} ({100*agg["place_success_rate"]:.1f}%)')
    print('Failure reasons:', agg['failure_reason_counts'])
    print('Results:', OUT)


if __name__ == '__main__':
    main()
