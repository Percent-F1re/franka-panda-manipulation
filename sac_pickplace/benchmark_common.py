"""Identical, reward-independent physics auditor for Stage4 and SAC rollouts."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BenchmarkRules:
    min_lift_mm: float = 40.0
    min_transfer_mm: float = 35.0
    xy_tol_mm: float = 20.0
    z_tol_mm: float = 10.0
    speed_tol_m_s: float = 0.03
    drift_tol_mm: float = 10.0
    event_dwell_s: float = 0.12
    post_observation_s: float = 1.0
    drift_window_s: float = 0.6


def sample_state(model, data, *, ids, cube_initial_xyz, target_xy, time_s):
    """Record a physical snapshot using the same MuJoCo indices for both policies."""
    cube = np.asarray(data.xpos[ids['cube']], dtype=np.float64).copy()
    qvel = np.asarray(data.qvel[ids['cube_dof_addr']:ids['cube_dof_addr']+3], dtype=np.float64)
    table, left, right = ids['contact_fn'](
        model, data, ids['cube_geom'], ids['table_geom'], ids['left'], ids['right']
    )
    table_top_z = float(data.geom_xpos[ids['table_geom'], 2] + model.geom_size[ids['table_geom'], 2])
    return {
        'time_s': float(time_s),
        'cube_xyz': cube,
        'rise_mm': float((cube[2] - cube_initial_xyz[2]) * 1000),
        'horizontal_moved_mm': float(np.linalg.norm(cube[:2] - cube_initial_xyz[:2]) * 1000),
        'xy_error_mm': float(np.linalg.norm(cube[:2] - target_xy) * 1000),
        'bottom_gap_mm': float((cube[2] - ids['cube_half_width'] - table_top_z) * 1000),
        'speed_m_s': float(np.linalg.norm(qvel)),
        'table_contacts': int(table),
        'left_contacts': int(left),
        'right_contacts': int(right),
    }


def audit_events(samples: list[dict], rules: BenchmarkRules):
    """Count sustained, chronological lift then transfer events.

    An eligible sample contributes the elapsed time since the previous sample;
    a failing sample resets the consecutive duration. Both methods use the
    same event thresholds in seconds, not number of logged rows.
    """
    lift, transfer = False, False
    lift_dwell, transfer_dwell = 0.0, 0.0
    prev_t = None
    for s in samples:
        t = float(s['time_s'])
        dt = 0.0 if prev_t is None else max(0.0, t - prev_t)
        prev_t = t
        # We capture the expert every ~0.01s and SAC every ~0.04s.
        # If logs ever contain a large gap, don't incorrectly count it as contact duration.
        dt = min(dt, 0.041)
        airborne = s['table_contacts'] == 0 and s['rise_mm'] >= rules.min_lift_mm
        bilat = s['left_contacts'] > 0 and s['right_contacts'] > 0
        lift_condition = airborne and bilat
        if not lift:
            lift_dwell = lift_dwell + dt if lift_condition else 0.0
            if lift_dwell + 1e-9 >= rules.event_dwell_s:
                lift = True
        if lift and not transfer:
            transfer_condition = airborne and bilat and s['horizontal_moved_mm'] >= rules.min_transfer_mm
            transfer_dwell = transfer_dwell + dt if transfer_condition else 0.0
            if transfer_dwell + 1e-9 >= rules.event_dwell_s:
                transfer = True
    return lift, transfer


def audit_final(post_samples: list[dict], lift: bool, transfer: bool, rules: BenchmarkRules):
    if not post_samples:
        raise ValueError('post_samples is empty')
    end = post_samples[-1]
    cutoff = end['time_s'] - rules.drift_window_s
    window = [s['cube_xyz'] for s in post_samples if s['time_s'] >= cutoff - 1e-9]
    # Use a time-based window; point samples are equally frequent in both post tests.
    drift_mm = float(1000 * max(np.linalg.norm(p - window[0]) for p in window)) if len(window) >= 2 else float('inf')
    physical_ok = {
        'xy_ok': end['xy_error_mm'] <= rules.xy_tol_mm,
        'z_ok': abs(end['bottom_gap_mm']) <= rules.z_tol_mm,
        'table_contact_ok': end['table_contacts'] > 0,
        'fingers_released_ok': end['left_contacts'] == 0 and end['right_contacts'] == 0,
        'speed_ok': end['speed_m_s'] <= rules.speed_tol_m_s,
        'drift_ok': drift_mm <= rules.drift_tol_mm,
    }
    stable = all(physical_ok.values())
    return {
        'unified_lift': bool(lift),
        'unified_transfer': bool(transfer),
        'unified_stable_place': bool(stable),
        'unified_success': bool(lift and transfer and stable),
        'final_xy_error_mm': round(end['xy_error_mm'], 6),
        'final_bottom_gap_mm': round(end['bottom_gap_mm'], 6),
        'final_cube_speed_m_s': round(end['speed_m_s'], 8),
        'post_drift_mm': round(drift_mm, 6),
        'final_table_contacts': end['table_contacts'],
        'final_left_contacts': end['left_contacts'],
        'final_right_contacts': end['right_contacts'],
        **physical_ok,
    }


def wilson_interval(success: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return [None, None]
    p = success / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total)) / denom
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denom
    return [max(0.0, center-half), min(1.0, center+half)]


def summarize(rows: list[dict], rules: BenchmarkRules):
    out = {'rules': asdict(rules), 'episodes_per_method': len(rows)//2, 'methods': {}}
    for method in ('expert', 'sac'):
        rs = [r for r in rows if r['method'] == method]
        n = len(rs)
        count = sum(bool(r['unified_success']) for r in rs)
        out['methods'][method] = {
            'n': n, 'unified_success_count': count,
            'unified_success_rate': count/n if n else None,
            'wilson_95_success_rate': wilson_interval(count,n),
            'unified_lift_rate': sum(bool(r['unified_lift']) for r in rs)/n if n else None,
            'unified_transfer_rate': sum(bool(r['unified_transfer']) for r in rs)/n if n else None,
            'unified_stable_place_rate': sum(bool(r['unified_stable_place']) for r in rs)/n if n else None,
            'mean_final_xy_error_mm_all': float(np.mean([r['final_xy_error_mm'] for r in rs])) if n else None,
            'mean_task_sim_time_s': float(np.mean([r['task_time_s'] for r in rs])) if n else None,
            'mean_post_drift_mm': float(np.mean([r['post_drift_mm'] for r in rs])) if n else None,
        }
    return out
