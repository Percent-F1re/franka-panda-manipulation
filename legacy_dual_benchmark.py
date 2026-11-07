"""Identical reward-independent physical rules for DLS and robosuite OSC.

One subtlety: the legacy 1-second auditor kept raw position servo controls
fixed. Under OSC, a *frozen torque* is NOT a held pose. Default comparison
holds the final HIGH-LEVEL target, with each controller continuing its native
feedback. Both controllers have no new SAC actions or reference changes.

Option raw_actuator_hold reproduces the old benchmark exactly even for OSC;
this is a sensitivity test, NOT the primary fair controller comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from legacy_exact_rules import BenchmarkRules, audit_events, audit_final
from legacy_exact_benchmark import sample_physics


def hold_final_target_for_one_second(env, rules=None, *, hold='target'):
    rules=rules or BenchmarkRules()
    if hold not in ('target','raw_actuator'):
        raise ValueError("hold must be 'target' or 'raw_actuator'")
    held_ctrl=env.data.ctrl.copy()
    n=max(1,int(round(rules.post_observation_s / float(env.model.opt.timestep))))
    samples=[]
    for _ in range(n):
        if hold=='target' and env.controller=='osc':
            # Last 6D desired pose + last gripper target remain unchanged;
            # robosuite OSC recalculates appropriate torque each physics step.
            env._osc_bridge.hold_goal_substep()
        else:
            # DLS: original servo q_cmd still holds pose physically.  OSC with
            # raw_actuator holds last *torque*, reported separately, never mixed
            # with the primary comparison.
            env.data.ctrl[:] = held_ctrl
        env.raw_env.sim.step()
        samples.append(sample_physics(env))
    return samples


def eval_one(model,env,*,seed:int,offset,post_hold:str='target',
             rules=None,save_trace=False,episode_options=None):
    rules=rules or BenchmarkRules()
    reset_options=dict(episode_options or {})
    reset_options['xy_offset']=np.asarray(offset,dtype=float)
    obs,info=env.reset(seed=seed,options=reset_options)
    samples=[]
    trace=[]
    max_hand_error_mm=0.
    max_joint_error_rad=0.
    for k in range(env.max_episode_steps):
        action,_=model.predict(obs,deterministic=True)
        obs,reward,term,trunc,info=env.step(action)
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError(f'Invalid {env.controller} 46D observation at step {k+1}')
        snap=sample_physics(env)
        samples.append(snap)
        err_mm=1000*float(np.linalg.norm(env.hand_target-env.data.xpos[env.hand]))
        joint_err=float(np.max(np.abs(env.q_cmd-env.data.qpos[env.qpos_ids])))
        max_hand_error_mm=max(max_hand_error_mm,err_mm)
        max_joint_error_rad=max(max_joint_error_rad,joint_err)
        if save_trace:
            trace.append({
                'step':k+1,
                'time_s':snap['time_s'],
                'action_x':float(action[0]),'action_y':float(action[1]),
                'action_z':float(action[2]),'action_gripper':float(action[3]),
                'hand_target_error_mm':err_mm,
                'q_cmd_minus_q_max_rad':joint_err,
                'cube_rise_mm':snap['rise_mm'],
                'cube_xy_error_mm':snap['xy_error_mm'],
                'table_contacts':snap['table_contacts'],
                'left_contacts':snap['left_contacts'],
                'right_contacts':snap['right_contacts'],
                'finger_aperture_mm':1000*sum(float(env.data.qpos[j]) for j in env.finger_qpos_ids),
                'cube_speed_m_s':snap['speed_m_s'],
            })
        if term or trunc:
            break
    else:
        raise RuntimeError(f'{env.controller} failed to terminate at max_episode_steps')
    lift,transfer=audit_events(samples,rules)
    post=hold_final_target_for_one_second(env,rules,hold=post_hold)
    phys=audit_final(post,lift,transfer,rules)
    result={
        'controller':env.controller,
        'seed':int(seed),
        'steps':k+1,
        'original_is_success':bool(info['is_success']),
        'original_lift':bool(info['has_lifted']),
        'original_transfer':bool(info['has_transferred']),
        'max_hand_target_error_mm':round(max_hand_error_mm,6),
        'max_q_cmd_minus_q_rad':round(max_joint_error_rad,8),
        'post_hold':post_hold,
        **phys,
    }
    cd=env.controller_diagnostics()
    result['torque_saturated_substeps']=cd.get('torque_saturated_substeps')
    result['torque_substeps']=cd.get('torque_substeps')
    result['torque_saturation_fraction']=cd.get('torque_saturation_fraction')
    result['peak_unclipped_torque_nm']=cd.get('peak_unclipped_torque_nm')
    return result, trace
