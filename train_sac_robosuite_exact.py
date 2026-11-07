"""OPTIONAL continued training of ORIGINAL single SAC Actor in robosuite.

Do NOT run until zero-shot fidelity is verified. Never changes old checkpoint.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from envs.panda_legacy_exact import PandaPickPlaceEnv
from legacy_sac_checkpoint import load_legacy_sac
from legacy_exact_benchmark import evaluate_episode
import numpy as np


class ValidationCallback(BaseCallback):
    def __init__(self, eval_env, output, freq, n, seed, incumbent):
        super().__init__(verbose=0)
        self.eval_env=eval_env;self.output=output;self.freq=freq;self.n=n;self.seed=seed
        self.best_score=incumbent
    def _on_step(self):
        if self.num_timesteps%self.freq != 0:return True
        scores=[]
        for k in range(self.n):
            r,_=evaluate_episode(self.model,self.eval_env,seed=self.seed+k)
            scores.append(r)
        score=(sum(x['unified_success'] for x in scores),
               sum(x['original_is_success'] for x in scores),
               sum(x['unified_transfer'] for x in scores),
               sum(x['unified_lift'] for x in scores))
        best=score>self.best_score
        if best:
            self.best_score=score
            self.model.save(str(self.output/'best_model'))
        with (self.output/'validation.csv').open('a',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f)
            if f.tell()==0:w.writerow(['timesteps','unified_success','original_success','transfer','lift','n','best'])
            w.writerow([self.num_timesteps,*score,self.n,best])
        print(f'[EVAL] step={self.num_timesteps} unified={score[0]}/{self.n} initial-preserved best={best}',flush=True)
        return True


def main():
    p=argparse.ArgumentParser(description='After fidelity confirmed: fine-tune same original SAC Actor')
    p.add_argument('--pretrained',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-every',type=int,default=5000)
    p.add_argument('--eval-episodes',type=int,default=10)
    p.add_argument('--eval-seed',type=int,default=23000)
    p.add_argument('--learning-starts',type=int,default=1000)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.steps<1 or args.eval_every<1 or args.eval_episodes<1:
        p.error('--steps, --eval-every and --eval-episodes must be positive')
    out=Path(args.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):p.error('Output directory not empty; use new --out')
    out.mkdir(parents=True,exist_ok=True)
    eval_env=PandaPickPlaceEnv(task='pickplace',max_episode_steps=600)
    train_env=None
    try:
        model=load_legacy_sac(args.pretrained,eval_env,args.device)
        (out/'config.json').write_text(json.dumps({**vars(args),'origin':'original 46D / 4D SAC, not staged HRL',
            'engine':'robosuite MujocoEnv + ORIGINAL Menagerie Panda MJCF'},ensure_ascii=False,indent=2),encoding='utf-8')
        baseline=[]
        for k in range(args.eval_episodes):
            r,_=evaluate_episode(model,eval_env,seed=args.eval_seed+k)
            baseline.append(r)
        incumbent=(sum(x['unified_success'] for x in baseline),
                   sum(x['original_is_success'] for x in baseline),
                   sum(x['unified_transfer'] for x in baseline),
                   sum(x['unified_lift'] for x in baseline))
        model.save(str(out/'initial_model'))
        model.save(str(out/'best_model'))
        print('[BASELINE] unified',incumbent[0],'/',len(baseline),'(initial saved as best)',flush=True)
        train_env=Monitor(PandaPickPlaceEnv(task='pickplace',max_episode_steps=600),
                          filename=str(out/'monitor.csv'),
                          info_keywords=('is_success','has_lifted','has_transferred'))
        train_env.reset(seed=args.seed)
        model.set_env(train_env)
        model.set_random_seed(args.seed)
        model.learning_starts=args.learning_starts
        cb=ValidationCallback(eval_env,out,args.eval_every,args.eval_episodes,args.eval_seed,incumbent)
        try:
            model.learn(total_timesteps=args.steps,callback=cb,reset_num_timesteps=True)
        finally:
            model.save(str(out/'final_model'))
        print('DONE',out)
    finally:
        if train_env is not None:train_env.close()
        eval_env.close()


if __name__=='__main__':main()
