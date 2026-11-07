"""Real-physics smoke: verify reversible mass/inertia/friction and target changes.
Run after V3 patch install, before running large robust benchmarks.
"""
from __future__ import annotations
import argparse
import numpy as np
from envs.panda_domain_randomization import RobustPandaDualControllerEnv
from robustness_design import scenario_catalog,generate_cases


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['dls','osc','both'],default='both')
    ap.add_argument('--seed',type=int,default=42000)
    args=ap.parse_args()
    modes=['dls','osc'] if args.mode=='both' else [args.mode]
    cs=generate_cases(args.seed,1,'combined_mild',scenario_catalog()['combined_mild'])[0]
    for mode in modes:
        env=RobustPandaDualControllerEnv(controller=mode,task='pickplace',
                                        max_episode_steps=600,random_xy_range_m=.06)
        try:
            mass0=float(env._nominal_physics['cube_mass'])
            inertia0=env._nominal_physics['cube_inertia'].copy()
            fric0=env._nominal_physics['friction'].copy()
            options={'xy_offset':np.asarray(cs['xy_offset']),
                     'mass_scale':cs['mass_scale'],
                     'friction_scale':cs['friction_scale'],
                     'target_delta_xy':np.asarray(cs['target_delta_xy'])}
            ob,info=env.reset(seed=cs['seed'],options=options)
            assert ob.shape==(46,) and np.all(np.isfinite(ob))
            assert np.isclose(env.model.body_mass[env.cube],mass0*cs['mass_scale'])
            assert np.allclose(env.model.body_inertia[env.cube],inertia0*cs['mass_scale'])
            assert np.allclose(env.model.geom_friction[env._affected_friction_geom_ids],
                               fric0*cs['friction_scale'])
            expected=env.cube_start[:2]+env.transfer_xy+np.asarray(cs['target_delta_xy'])
            assert np.allclose(env.target_xy,expected)
            assert np.allclose(ob[36:38],env.target_xy-env.data.xpos[env.cube,:2],atol=2e-7),(
                'target feature index mismatch',ob[36:38])
            before=env.data.qpos[env.qpos_ids].copy()
            ob,rew,term,trunc,info=env.step(np.array([0.,0.,0.,1.],dtype=np.float32))
            assert ob.shape==(46,) and np.all(np.isfinite(ob))
            print(f'[SMOKE] {mode.upper()} perturbed reset mass={mass0*cs["mass_scale"]:.4f}kg '
                  f'friction_x={cs["friction_scale"]:.3f} target_delta={cs["target_delta_xy"]}',flush=True)
            # Reset to true nominal, verify there was no accumulation/leakage.
            ob,info=env.reset(seed=cs['seed'],options={'xy_offset':np.asarray(cs['xy_offset'])})
            assert np.isclose(env.model.body_mass[env.cube],mass0)
            assert np.array_equal(env.model.body_inertia[env.cube],inertia0)
            assert np.array_equal(env.model.geom_friction[env._affected_friction_geom_ids],fric0)
            assert np.allclose(env.target_xy,env.cube_start[:2]+env.transfer_xy)
            print(f'[SMOKE] {mode.upper()} original physics restored at next reset',flush=True)
        finally:env.close()
    print('ROBUSTNESS SMOKE PASSED',flush=True)

if __name__=='__main__':main()
