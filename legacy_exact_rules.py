"""Original user's reward-independent physical benchmark rules.

Algorithm and thresholds faithfully ported from
https://github.com/Percent-F1re/franka-panda-manipulation/blob/main/sac_pickplace/benchmark_common.py
This file needs only NumPy and does not construct a simulator.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class BenchmarkRules:
    min_lift_mm: float = 40.0
    min_transfer_mm: float = 35.0
    xy_tol_mm: float = 20.0
    z_tol_mm: float = 10.0
    speed_tol_m_s: float = .03
    drift_tol_mm: float = 10.0
    event_dwell_s: float = .12
    post_observation_s: float = 1.0
    drift_window_s: float = .6


def audit_events(samples: list[dict], rules: BenchmarkRules):
    lift, transfer = False, False
    lift_dwell, transfer_dwell = 0., 0.
    prev_t = None
    for s in samples:
        t = float(s['time_s'])
        dt = 0. if prev_t is None else max(0., t-prev_t)
        prev_t = t
        dt = min(dt, .041)
        airborne = s['table_contacts'] == 0 and s['rise_mm'] >= rules.min_lift_mm
        bilateral = s['left_contacts'] > 0 and s['right_contacts'] > 0
        if not lift:
            lift_dwell = lift_dwell+dt if airborne and bilateral else 0.
            if lift_dwell+1e-9 >= rules.event_dwell_s:
                lift=True
        if lift and not transfer:
            transfer_dwell = transfer_dwell+dt if (airborne and bilateral and
                s['horizontal_moved_mm'] >= rules.min_transfer_mm) else 0.
            if transfer_dwell+1e-9 >= rules.event_dwell_s:
                transfer=True
    return lift, transfer


def audit_final(post_samples: list[dict], lift: bool, transfer: bool, rules: BenchmarkRules):
    if not post_samples:
        raise ValueError('post_samples is empty')
    end=post_samples[-1]
    cutoff = end['time_s'] - rules.drift_window_s
    window=[s['cube_xyz'] for s in post_samples if s['time_s'] >= cutoff-1e-9]
    drift_mm = (float(1000*max(np.linalg.norm(p-window[0]) for p in window))
                if len(window)>=2 else float('inf'))
    physical_ok = {
        'xy_ok': end['xy_error_mm'] <= rules.xy_tol_mm,
        'z_ok': abs(end['bottom_gap_mm']) <= rules.z_tol_mm,
        'table_contact_ok': end['table_contacts'] > 0,
        'fingers_released_ok': end['left_contacts']==0 and end['right_contacts']==0,
        'speed_ok': end['speed_m_s'] <= rules.speed_tol_m_s,
        'drift_ok': drift_mm <= rules.drift_tol_mm,
    }
    stable=all(physical_ok.values())
    return {
        'unified_lift':bool(lift),
        'unified_transfer':bool(transfer),
        'unified_stable_place':bool(stable),
        'unified_success':bool(lift and transfer and stable),
        'final_xy_error_mm':round(end['xy_error_mm'],6),
        'final_bottom_gap_mm':round(end['bottom_gap_mm'],6),
        'final_cube_speed_m_s':round(end['speed_m_s'],8),
        'post_drift_mm':round(drift_mm,6),
        'final_table_contacts':end['table_contacts'],
        'final_left_contacts':end['left_contacts'],
        'final_right_contacts':end['right_contacts'],
        **physical_ok,
    }


def wilson_interval(success:int,total:int,z:float=1.959963984540054):
    if total==0:return [None,None]
    p=success/total
    denom=1+z*z/total
    center=(p+z*z/(2*total))/denom
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return [max(0.,center-half),min(1.,center+half)]
