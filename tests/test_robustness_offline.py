"""Offline schema, scenario sampling and no-regression tests (NumPy only)."""
from __future__ import annotations
import ast
import json
import pathlib
import sys
import unittest
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from robustness_design import scenario_catalog,generate_cases,summary,exact_mcnemar_pvalue,design_hash


class RobustnessOfflineTests(unittest.TestCase):
    def setUp(self):self.scenarios=scenario_catalog()

    def test_scenario_catalog_coverage(self):
        self.assertEqual(set(self.scenarios),{
            'baseline','mass_random','friction_random','object_position_wide',
            'target_position_random','combined_mild','combined_hard'})

    def test_exact_original_baseline_xy_rng(self):
        cases=generate_cases(42000,100,'baseline',self.scenarios['baseline'])
        expected=np.random.default_rng(42000).uniform(-.04,.04,size=(100,2))
        np.testing.assert_array_equal([c['xy_offset'] for c in cases],expected)
        self.assertAlmostEqual(cases[0]['xy_offset'][0],-.014797319770491351)
        self.assertAlmostEqual(cases[0]['xy_offset'][1],-.0049091368173379835)
        self.assertTrue(all(c['mass_scale']==c['friction_scale']==1. for c in cases))

    def test_paired_same_design_and_independent_streams(self):
        c=generate_cases(42000,100,'combined_hard',self.scenarios['combined_hard'])
        self.assertEqual(c,generate_cases(42000,100,'combined_hard',self.scenarios['combined_hard']))
        self.assertEqual(design_hash(c),design_hash(c))
        self.assertEqual(c[0]['seed'],42000)
        self.assertTrue(all(.6<=r['mass_scale']<=1.4 for r in c))
        self.assertTrue(all(.5<=r['friction_scale']<=1.5 for r in c))
        self.assertTrue(all(max(abs(x) for x in r['xy_offset'])<=.06 for r in c))
        self.assertTrue(all(max(abs(x) for x in r['target_delta_xy'])<=.03 for r in c))
        self.assertGreater(np.std([r['mass_scale'] for r in c]),.01)
        self.assertGreater(np.std([r['friction_scale'] for r in c]),.01)

    def test_only_one_axis_changes_in_single_factor_scenarios(self):
        for name,cfg in self.scenarios.items():
            cases=generate_cases(42000,20,name,cfg)
            for r in cases:
                if name not in ('mass_random','combined_mild','combined_hard'):self.assertEqual(r['mass_scale'],1.)
                if name not in ('friction_random','combined_mild','combined_hard'):self.assertEqual(r['friction_scale'],1.)
                if name not in ('target_position_random','combined_mild','combined_hard'):
                    self.assertEqual(r['target_delta_xy'],[0.,0.])
                if name not in ('object_position_wide','combined_mild','combined_hard'):
                    self.assertLessEqual(max(map(abs,r['xy_offset'])),.04)

    def test_exact_mcnemar_matches_existing_v2(self):
        self.assertAlmostEqual(exact_mcnemar_pvalue(14,11),.6900379657745361,places=12)

    def test_source_hooks_are_in_reset_before_settle_and_observation(self):
        src=(ROOT/'envs/panda_dual_controller.py').read_text()
        self.assertLess(src.index('self.raw_env.reset()'),src.index("model_hook = getattr(self, '_before_original_settle', None)"))
        self.assertLess(src.index("model_hook = getattr(self, '_before_original_settle', None)"),src.index('mujoco.mj_resetDataKeyframe'))
        self.assertLess(src.index('self.target_xy = self.cube_start[:2] + self.transfer_xy'),src.index("target_hook = getattr(self, '_after_original_target_set', None)"))
        self.assertLess(src.index("target_hook = getattr(self, '_after_original_target_set', None)"),src.rindex('self.prev_cube_target_distance ='))

    def test_source_restores_model_before_reapplying(self):
        src=(ROOT/'envs/panda_domain_randomization.py').read_text()
        self.assertLess(src.index("m.body_mass[self.cube] = baseline['cube_mass']"),src.index("m.body_mass[self.cube] *= p['mass_scale']"))
        self.assertLess(src.index("m.geom_friction[self._affected_friction_geom_ids,:] = baseline['friction']"),src.index("m.geom_friction[self._affected_friction_geom_ids,:] *= p['friction_scale']"))
        self.assertIn('mujoco.mj_setConst(m,self.data)',src)

    def test_all_patch_python_compiles(self):
        for p in ROOT.rglob('*.py'):
            with self.subTest(p=p.name):ast.parse(p.read_text(encoding='utf-8'))

if __name__=='__main__':unittest.main()
