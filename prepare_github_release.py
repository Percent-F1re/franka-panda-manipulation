"""Stage a clean *complete* GitHub tree from a tested local Windows project.

This is a local staging tool, NOT a GitHub publisher. Nothing is uploaded.
Must run AFTER installing V1 + V2 + this V3 patch in panda_robosuite_project.
User's training weights, evaluation traces and local paths are deliberately
NOT included. Original Menagerie meshes may be published only with license.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

FILES=[
    'envs/robosuite_menagerie_core.py','envs/original_stage4_compat.py',
    'envs/panda_legacy_exact.py', 'envs/dual_scene_core.py',
    'envs/panda_dual_controller.py', 'envs/panda_domain_randomization.py',
    'envs/controllers/__init__.py','envs/controllers/legacy_dls.py',
    'envs/controllers/osc_bridge.py',
    'numpy_pickle_compat.py','legacy_sac_checkpoint.py',
    'legacy_exact_rules.py','legacy_exact_benchmark.py',
    'legacy_dual_benchmark.py','robustness_design.py',
    'robustness_scenarios.json',
    'smoke_sac_robosuite_exact.py','smoke_dls_osc.py',
    'compare_sac_robosuite_exact.py','eval_sac_robosuite_exact.py',
    'compare_dls_vs_osc.py','trace_dls_vs_osc.py',
    'train_sac_robosuite_exact.py',
    'run_robustness.py','plot_robustness.py','smoke_robustness.py','prepare_github_release.py',
]
MODEL='legacy_original_model/franka_emika_panda'
MODEL_FILES=['panda_pickplace.xml','scene_pickplace.xml','scene_pickplace_stage3.xml']
REFERENCE_FILES={
    'panda_pickplace_stage1.py','panda_pickplace_stage2_grasp.py',
    'panda_pickplace_stage3.py','panda_pickplace_stage4_random.py',
    'panda_pickplace_env.py','train_sac_pickplace.py',
    'evaluate_sac_pickplace.py','benchmark_pickplace.py',
}


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',required=True,help='Existing tested panda_robosuite_project directory')
    p.add_argument('--out',required=True,help='New empty staging directory, NOT existing Git repo')
    p.add_argument('--model-license',default=None,help='Original Menagerie franka_emika_panda/LICENSE file')
    p.add_argument('--without-assets',action='store_true',help='Do not vendor original meshes; users must download assets')
    a=p.parse_args()
    source=Path(a.project).expanduser().resolve();out=Path(a.out).expanduser().resolve()
    if not source.is_dir():p.error(f'Source project missing: {source}')
    if source==out or source in out.parents or out in source.parents:
        p.error('Stage outside the source project to avoid recursion and accidental deletion')
    if out.exists() and any(out.iterdir()):p.error('Stage output must be a NEW empty directory')
    missing=[f for f in FILES if not (source/f).is_file()]
    missing +=[str(Path(MODEL)/f) for f in MODEL_FILES if not (source/MODEL/f).is_file()]
    if missing:p.error('Missing release components: '+', '.join(missing))
    assetdir=source/MODEL/'assets'
    licensepath=Path(a.model_license).expanduser().resolve() if a.model_license else source/MODEL/'LICENSE'
    if not a.without_assets:
        if not assetdir.is_dir() or not any(assetdir.iterdir()):
            p.error(f'Original model assets missing: {assetdir}')
        if not licensepath.is_file():
            p.error('Original model LICENSE required before redistribution; use --model-license PATH (e.g. from original Menagerie folder), or --without-assets')
    template=source/'github'
    if not template.is_dir():p.error('Missing github/ templates; install V3 patch first')
    out.mkdir(parents=True,exist_ok=True)
    for f in FILES:
        dest=out/f;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/f,dest)
    for f in MODEL_FILES:
        dest=out/MODEL/f;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/MODEL/f,dest)
    # Derived, generated dual-controller XML need not be committed; generated
    # automatically by dual_scene_core on first run.
    if not a.without_assets:
        shutil.copytree(assetdir,out/MODEL/'assets')
        shutil.copy2(licensepath,out/MODEL/'LICENSE')
    for f in template.rglob('*'):
        if f.is_file():
            dest=out/f.relative_to(template)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(f,dest)
    if (source/'requirements-lock.txt').exists():
        shutil.copy2(source/'requirements-lock.txt',out/'requirements-lock.txt')
    else:
        print('WARNING: Save your Windows environment with pip freeze > requirements-lock.txt first')
    # Copy only small, non-sensitive descriptive README / tests; never random
    # huge evaluation logs, model weights, local config or hidden files.
    for test in (source/'tests').glob('test_*offline*.py'):
        dest=out/'tests'/test.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(test,dest)
    # Retain original MuJoCo benchmark as a reference, clearly separated
    # from the active robosuite v3 runtime package.
    reference=source/'legacy_reference'
    if reference.is_dir():
        for f in sorted(reference.iterdir()):
            if f.is_file() and f.name in REFERENCE_FILES:
                dest=out/'legacy_reference'/f.name
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(f,dest)
    for name in ('README_legacy_exact.md','README_DLS_OSC_对照实验.md','README_鲁棒性与GitHub发布.md'):
        if (source/name).is_file():
            dest=out/'docs'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dest)
    man={'source':'local tested project (path deliberately omitted)',
         'assets_vendored':not a.without_assets,
         'files':{str(x.relative_to(out)).replace('\\','/'):digest(x)
                  for x in sorted(out.rglob('*')) if x.is_file()}}
    (out/'release_manifest.json').write_text(json.dumps(man,indent=2,ensure_ascii=False),encoding='utf-8')
    print('STAGED',out,'files',len(man['files']))
    print('No GitHub repository was edited. Review the staged contents then commit manually or via a connected GitHub app.')

if __name__=='__main__':main()
