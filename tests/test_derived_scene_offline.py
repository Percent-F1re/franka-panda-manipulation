"""Exercise actual generated dual-site MJCF logic using a minimal fake model dir."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


class DerivedSceneOffline(unittest.TestCase):
    def test_sites_only_and_unchanged_orig(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)
            panda='<mujoco><worldbody><body name="hand" pos="0 0 0"><geom type="sphere" size="0.01"/></body></worldbody></mujoco>'
            scene='<mujoco><include file="panda_pickplace.xml"/><worldbody><site name="place_target_marker" pos="0.58 0.06 0.383" size="0.008"/></worldbody></mujoco>'
            (d/'panda_pickplace.xml').write_text(panda)
            (d/'scene_pickplace_stage3.xml').write_text(scene)
            fake=types.ModuleType('envs.robosuite_menagerie_core')
            fake.MODEL_DIR=d
            fake.LegacyMenageriePandaRobosuite=type('LegacyMenageriePandaRobosuite',(object,),{})
            key='envs.robosuite_menagerie_core'
            old=sys.modules.get(key)
            sys.modules[key]=fake
            try:
                spec=importlib.util.spec_from_file_location('offline_dual_scene',ROOT/'envs/dual_scene_core.py')
                mod=importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                out=mod.ensure_dual_scene()
                self.assertEqual(out,d/'scene_pickplace_dual_controllers.xml')
                dr=ET.parse(d/'panda_pickplace_dual_controllers.xml').getroot()
                ds=ET.parse(out).getroot()
                self.assertIsNotNone(dr.find(".//body[@name='hand']/site[@name='dual_osc_hand_ref']"))
                self.assertIsNotNone(ds.find(".//worldbody/site[@name='hand_center']"))
                self.assertEqual(ds.find('include').get('file'),'panda_pickplace_dual_controllers.xml')
                self.assertEqual((d/'panda_pickplace.xml').read_text(),panda)
                self.assertEqual((d/'scene_pickplace_stage3.xml').read_text(),scene)
                first=out.read_bytes()
                self.assertEqual(mod.ensure_dual_scene(),out)
                self.assertEqual(out.read_bytes(),first)
            finally:
                if old is None:sys.modules.pop(key,None)
                else:sys.modules[key]=old

if __name__=='__main__':unittest.main()
