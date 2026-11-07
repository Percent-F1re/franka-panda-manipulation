"""Pure NumPy deterministic paired robustness experiment design.

Do NOT draw perturbations from each environment's RNG. Draw ONCE, then reuse
identical case parameters for both DLS and OSC and across scenarios.

Legacy baseline preservation: np.random.default_rng(seed).uniform(-.04,.04,
size=(n,2)) matches the original V2 84/81 paired comparison exactly.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

SCENARIOS_PATH = Path(__file__).with_name('robustness_scenarios.json')


def scenario_catalog(path=SCENARIOS_PATH):
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(d,dict) or 'baseline' not in d:
        raise ValueError('Scenario catalog must be object with baseline')
    for name,c in d.items():
        if not isinstance(name,str) or not name or '/' in name or '\\' in name or name.startswith('.'):
            raise ValueError(f'Invalid scenario name: {name!r}')
        for field in ('mass_scale','friction_scale'):
            v=c.get(field)
            if not isinstance(v,list) or len(v)!=2 or not all(np.isfinite(float(x)) for x in v):
                raise ValueError(f'{name}.{field} must be two finite values')
            if not 0.25 <= float(v[0]) <= float(v[1]) <= 2.0:
                raise ValueError(f'{name}.{field} must stay within [0.25,2.0]')
        for field,limit in (('object_xy_half_range_m',.08),('target_xy_half_range_m',.05)):
            v=float(c[field])
            if not np.isfinite(v) or v<0 or v>limit:
                raise ValueError(f'{name}.{field} outside [0,{limit}]m')
    baseline=d['baseline']
    assert baseline['mass_scale']==[1.0,1.0] and baseline['friction_scale']==[1.0,1.0]
    assert baseline['object_xy_half_range_m']==.04 and baseline['target_xy_half_range_m']==0.0
    return d


def _uniform_from_seed(seed,n,stream,shape):
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(stream)]))
    return rng.uniform(-1.0,1.0,size=(n,)+shape)


def generate_cases(seed:int,n:int,scenario_name:str,scenario:dict):
    """A list of dicts with all random choices explicit and logged."""
    if n<1 or not 0<=seed<=2**32-1:
        raise ValueError('n must be >0 and seed must be a 32-bit nonnegative integer')
    # V2 baseline used this exact generator and order.
    original_xy=np.random.default_rng(seed).uniform(-.04,.04,size=(n,2))
    xy=original_xy * (float(scenario['object_xy_half_range_m'])/.04)
    massu=_uniform_from_seed(seed,n,1051,())
    fricu=_uniform_from_seed(seed,n,1052,())
    targetu=_uniform_from_seed(seed,n,1053,(2,))
    def scale_from_uniform(u,bounds):
        lo,hi=map(float,bounds)
        return lo+.5*(u+1.0)*(hi-lo)
    masses=scale_from_uniform(massu,scenario['mass_scale'])
    frictions=scale_from_uniform(fricu,scenario['friction_scale'])
    delta=targetu*float(scenario['target_xy_half_range_m'])
    return [dict(index=i+1,seed=seed+i,scenario=scenario_name,
                 xy_offset=[float(z) for z in xy[i]],
                 mass_scale=float(masses[i]),
                 friction_scale=float(frictions[i]),
                 target_delta_xy=[float(z) for z in delta[i]])
            for i in range(n)]


def design_hash(cases):
    b=json.dumps(cases,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    return hashlib.sha256(b).hexdigest()


def exact_mcnemar_pvalue(b,c):
    from math import comb
    n=int(b)+int(c)
    if n==0:return 1.0
    k=min(int(b),int(c))
    return min(1.0,2*sum(comb(n,i) for i in range(k+1))/2**n)


def wilson(k,n,z=1.959963984540054):
    import math
    if not n:return [None,None]
    p=k/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-half,center+half]


def summary(rows,scenarios,modes):
    groups={}
    def success(r):return bool(r['unified_success'])
    for sc in scenarios:
        groups[sc]={}
        for mode in modes:
            sub=[r for r in rows if r['scenario']==sc and r['controller']==mode]
            n=len(sub)
            hits=sum(success(r) for r in sub)
            groups[sc][mode]={
                'n':n,'successes':hits,'success_rate':hits/n if n else None,
                'wilson_95ci':wilson(hits,n),
                'unified_lift':sum(bool(r['unified_lift']) for r in sub),
                'unified_transfer':sum(bool(r['unified_transfer']) for r in sub),
                'mean_final_xy_error_mm':float(np.mean([float(r['final_xy_error_mm']) for r in sub])) if n else None,
                'fail_reasons':{f:sum(not bool(r[f]) for r in sub) for f in (
                    'xy_ok','z_ok','table_contact_ok','fingers_released_ok','speed_ok','drift_ok')},
            }
        if all(mode in modes for mode in ('dls','osc')):
            a={int(r['index']):r for r in rows if r['scenario']==sc and r['controller']=='dls'}
            b={int(r['index']):r for r in rows if r['scenario']==sc and r['controller']=='osc'}
            ids=set(a)&set(b)
            dlsonly=sum(success(a[i]) and not success(b[i]) for i in ids)
            osconly=sum(success(b[i]) and not success(a[i]) for i in ids)
            groups[sc]['paired_controllers']={
                'n':len(ids),
                'both_success':sum(success(a[i]) and success(b[i]) for i in ids),
                'dls_only_success':dlsonly,'osc_only_success':osconly,
                'neither_success':sum(not success(a[i]) and not success(b[i]) for i in ids),
                'mcnemar_exact_p':exact_mcnemar_pvalue(dlsonly,osconly),
            }
    # Compare each stress condition with baseline for the SAME controller and
    # same underlying case index (paired cross-scenario comparison).
    if 'baseline' in scenarios:
        for sc in scenarios:
            if sc=='baseline':continue
            for mode in modes:
                base={int(r['index']):r for r in rows if r['scenario']=='baseline' and r['controller']==mode}
                test={int(r['index']):r for r in rows if r['scenario']==sc and r['controller']==mode}
                ids=set(base)&set(test)
                if not ids:continue
                base_only=sum(success(base[i]) and not success(test[i]) for i in ids)
                test_only=sum(not success(base[i]) and success(test[i]) for i in ids)
                groups[sc][mode]['paired_vs_baseline']={
                    'n':len(ids),'baseline_only_success':base_only,
                    'stress_only_success':test_only,
                    'success_rate_difference_stress_minus_baseline':(test_only-base_only)/len(ids),
                    'mcnemar_exact_p':exact_mcnemar_pvalue(base_only,test_only),
                }
    return groups
