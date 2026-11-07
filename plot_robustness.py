"""Plot grouped zero-shot success and final XY error from run_robustness.py CSV.
No policy rollouts; reads only recorded results. Requires matplotlib.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',required=True,help='Path to episodes.csv')
    p.add_argument('--out',required=True,help='New plots directory')
    args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=Path(args.episodes).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):p.error('Plot output dir must be new/empty')
    with path.open('r',encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if not rows:p.error('Episodes CSV is empty')
    labels=list(dict.fromkeys(r['scenario'] for r in rows))
    modes=[m for m in ('dls','osc') if any(r['controller']==m for r in rows)]
    grouped=defaultdict(list)
    for r in rows: grouped[(r['scenario'],r['controller'])].append(r)
    out.mkdir(parents=True,exist_ok=True)
    for field,ylabel,filename in (
        ('unified_success','Unified physical success rate (%)','robustness_success.png'),
        ('final_xy_error_mm','Mean final XY error (mm)','robustness_final_xy.png'),
    ):
        fig,ax=plt.subplots(figsize=(max(10,len(labels)*1.5),5.5))
        width=.37 if len(modes)==2 else .55
        x=np.arange(len(labels))
        for j,mode in enumerate(modes):
            vals=[];errors=[]
            for label in labels:
                group=grouped[(label,mode)]
                if not group:
                    vals.append(float('nan'));errors.append(0.);continue
                data=np.array([float(r[field]) for r in group],dtype=float)
                if field=='unified_success':data*=100
                vals.append(float(np.mean(data)))
                errors.append(float(np.std(data,ddof=1)/np.sqrt(len(data))) if len(data)>1 else 0.)
            pos=x+(j-(len(modes)-1)/2)*width
            ax.bar(pos,vals,width,label=mode.upper(),alpha=.8)
            if field=='final_xy_error_mm':ax.errorbar(pos,vals,yerr=errors,fmt='none',capsize=3,color='black',lw=1)
        ax.set_xticks(x,labels,rotation=30,ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title('Frozen original SAC; paired DLS vs OSC; same physics auditor')
        if field=='unified_success':ax.set_ylim(0,105)
        ax.legend();ax.grid(axis='y',alpha=.25);fig.tight_layout()
        fig.savefig(out/filename,dpi=180);plt.close(fig)
    print('Saved charts to',out)


if __name__=='__main__':main()
