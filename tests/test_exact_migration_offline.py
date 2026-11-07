"""No robosuite/MuJoCo installation required. Source / asset preservation tests."""
from __future__ import annotations
import ast
import pathlib
import unittest
import xml.etree.ElementTree as ET

ROOT=pathlib.Path(__file__).resolve().parents[1]
MOD=ROOT/'legacy_original_model'/'franka_emika_panda'

class ExactMigrationOfflineTests(unittest.TestCase):
    def test_all_python_files_compile(self):
        for p in ROOT.rglob('*.py'):
            ast.parse(p.read_text(encoding='utf-8'),filename=str(p))

    def test_compiled_scene_references_original_included_robot(self):
        for name in ('scene_pickplace.xml','scene_pickplace_stage3.xml'):
            tree=ET.parse(MOD/name)
            includes=[n.get('file') for n in tree.getroot().findall('include')]
            self.assertIn('panda_pickplace.xml',includes)

    def test_mesh_references_all_exist(self):
        root=ET.parse(MOD/'panda_pickplace.xml').getroot()
        meshes=root.findall('.//asset/mesh')
        self.assertGreaterEqual(len(meshes),65)
        for m in meshes:
            self.assertTrue((MOD/'assets'/m.get('file')).is_file(),m.get('file'))

    def test_original_general_actuators_all_8(self):
        root=ET.parse(MOD/'panda_pickplace.xml').getroot()
        motors=root.find('actuator')
        self.assertEqual(len(list(motors)),8)
        self.assertTrue(all(a.tag=='general' for a in motors))
        self.assertEqual(motors[-1].get('name'),'actuator8')
        self.assertEqual(motors[-1].get('ctrlrange'),'0 255')
        self.assertEqual(motors[-1].get('tendon'),'split')

    def test_integrator_tendon_equality(self):
        root=ET.parse(MOD/'panda_pickplace.xml').getroot()
        self.assertEqual(root.find('option').get('integrator'),'implicitfast')
        self.assertIsNotNone(root.find("./tendon/fixed[@name='split']"))
        self.assertIsNotNone(root.find('equality/joint'))

    def test_stage3_has_visual_target_only(self):
        root=ET.parse(MOD/'scene_pickplace_stage3.xml').getroot()
        marker=root.find("./worldbody/site[@name='place_target_marker']")
        self.assertIsNotNone(marker)
        self.assertEqual(marker.get('pos'),'0.58 0.06 0.383')
        self.assertEqual(len(root.findall('./worldbody/geom')),2)

    def test_still_uses_real_robosuite_base_and_mjsim(self):
        src=(ROOT/'envs/robosuite_menagerie_core.py').read_text()
        self.assertIn('class LegacyMenageriePandaRobosuite(MujocoEnv)',src)
        self.assertIn('mujoco.MjModel.from_xml_path',src)
        self.assertIn('self.sim = MjSim(compiled)',src)
        self.assertIn('self.step(np.zeros(',src)
        self.assertNotIn('suite.make(',src)
        self.assertNotIn('JOINT_POSITION',src.replace('JOINT_POSITION impedance controller',''))

    def test_original_actor_reward_predicates_preserved(self):
        src=(ROOT/'envs/panda_legacy_exact.py').read_text()
        # 46-dim observation, 4-dim action, unchanged lift/place reward thresholds.
        self.assertIn('return obs  # 46 dimensions',src)
        self.assertIn('shape=(4,)',src)
        self.assertIn('self.s3.MIN_LIFT_MM',src)
        self.assertIn('self.s3.XY_TOLERANCE_MM',src)
        self.assertIn('self.s3.STABLE_DRIFT_MM',src)
        self.assertIn('self.raw_env.advance_one_policy_step()',src)


    def test_original_task_functions_ast_identical(self):
        def methods(filename):
            tree=ast.parse(filename.read_text(encoding='utf-8'))
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PandaPickPlaceEnv')
            return {n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
        old=methods(ROOT/'legacy_reference/panda_pickplace_env.py')
        new=methods(ROOT/'envs/panda_legacy_exact.py')
        for name in ('_get_obs','_physical_state','_info','_pose','_contacts',
                     '_cube_velocity','_grasp_distance','render'):
            with self.subTest(function=name):
                self.assertEqual(ast.dump(old[name],include_attributes=False),
                                 ast.dump(new[name],include_attributes=False))

if __name__=='__main__': unittest.main()
