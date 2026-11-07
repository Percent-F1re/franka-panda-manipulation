"""Selectable controller 1: EXACT original 6D DLS + Menagerie position servos."""
from __future__ import annotations


class LegacyDLSPositionController:
    name = 'dls'
    observation_q_cmd_semantics = 'real position-servo target'

    def __init__(self, task_env, original_substep_fn):
        self.env = task_env
        self.original_substep_fn = original_substep_fn

    def reset(self):
        return None

    def run_substep(self):
        # Use the original, unmodified environment DLS implementation, not an
        # approximation or a new kinematic algorithm.
        self.original_substep_fn()

    def diagnostics(self):
        return {
            'controller': self.name,
            'q_cmd_semantics': self.observation_q_cmd_semantics,
            'torque_saturated_substeps': None,
            'torque_substeps': 0,
        }
