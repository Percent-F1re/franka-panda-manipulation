"""Offline validation for clean GitHub staging, no real model or Git required."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prepare_github_release import FILES,MODEL_FILES,MODEL

class PackagingOfflineTests(unittest.TestCase):
    def test_no_private_outputs_staged(self):
        for f in FILES:
            self.assertNotIn('/evaluation/',f)
            self.assertNotIn('best_model',f)
            self.assertFalse(f.endswith('.zip'))
        self.assertIn('smoke_robustness.py',FILES)

    def test_stage_requires_license_or_no_assets(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t)
            project=t/'source';project.mkdir()
            out=t/'release'
            for filename in FILES:
                p=project/filename;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text('# fixture\n',encoding='utf-8')
            model=project/MODEL;model.mkdir(parents=True)
            for f in MODEL_FILES:(model/f).write_text('<mujoco/>')
            assets=model/'assets';assets.mkdir();(assets/'mesh.obj').write_text('# original fixture')
            import shutil
            shutil.copytree(ROOT/'github',project/'github')
            cli=[sys.executable,str(ROOT/'prepare_github_release.py'),'--project',str(project),'--out',str(out)]
            missing_license=subprocess.run(cli,capture_output=True,text=True)
            self.assertNotEqual(missing_license.returncode,0)
            self.assertIn('LICENSE required',missing_license.stderr)
            (model/'LICENSE').write_text('fixture license for unit test')
            complete=subprocess.run(cli,capture_output=True,text=True)
            self.assertEqual(complete.returncode,0,complete.stderr)
            self.assertTrue((out/MODEL/'assets'/'mesh.obj').is_file())
            self.assertTrue((out/MODEL/'LICENSE').is_file())
            self.assertTrue((out/'README.md').is_file())
            self.assertTrue((out/'.gitignore').is_file())
            manifest=json.loads((out/'release_manifest.json').read_text())
            self.assertNotIn(str(project),json.dumps(manifest))
            self.assertNotIn(str(out),json.dumps(manifest))

if __name__=='__main__':unittest.main()
