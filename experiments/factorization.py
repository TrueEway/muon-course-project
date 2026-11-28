"""Nonconvex full-rank matrix factorization, target condition number 1e4."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import json
import time
import numpy as np
from .common import ROOT,completed,fingerprint,write_csv,write_json
from .optim import ns_numpy

OUT = ROOT/'output'/'revision'/'factorization'
GRIDS = {'sgd':[0.03,0.1,0.3], 'adamw':[0.001,0.01,0.05], 'muon':[0.005,0.02,0.05]}


def objective(u,v,target):
    residual = u@v.T-target
    return float(.5*np.sum(residual**2)),residual@v,residual.T@u


def run(method,seed,lr,budget,pilot=False):
    config = dict(method=method,seed=seed,lr=lr,budget=budget,pilot=pilot,
                  dimension=32,kappa=10000.0,weight_decay=0,ns_steps=5,initial_scale=0.03)
    folder = OUT/('pilot' if pilot else 'formal')
    stem = f'{method}_s{seed}_lr{lr:g}_T{budget}'
    path = folder/(stem+'.json')
    previous = completed(path,config)
    if previous:
        return previous
    rng = np.random.default_rng(seed)
    left,_ = np.linalg.qr(rng.normal(size=(32,32)))
    right,_ = np.linalg.qr(rng.normal(size=(32,32)))
    target = (left*np.geomspace(1,1e-4,32))@right.T
    u,v = rng.normal(size=(32,32))*.03,rng.normal(size=(32,32))*.03
    m = [np.zeros_like(u),np.zeros_like(v)]
    second = [np.zeros_like(u),np.zeros_like(v)]
    records,spectra,spectrum_steps = [],[],[]
    start = time.perf_counter()
    for step in range(1,budget+1):
        loss,gu,gv = objective(u,v,target)
        if not np.isfinite(loss) or loss > 1e12:
            result = dict(config=config,config_hash=fingerprint(config),valid=False,final_loss=None)
            write_json(path,result)
            return result
        records.append(dict(step=step,loss=loss,relative_reconstruction=np.sqrt(2*loss)/np.linalg.norm(target),
            gradient_nuclear=float(np.linalg.svd(gu,compute_uv=False).sum()+np.linalg.svd(gv,compute_uv=False).sum()),
            gradient_fro=float(np.sqrt(np.sum(gu**2)+np.sum(gv**2))),elapsed_seconds=time.perf_counter()-start))
        for i,(p,g) in enumerate([(u,gu),(v,gv)]):
            if method == 'sgd':
                m[i] = .9*m[i]+g
                update = lr*m[i]
            elif method == 'adamw':
                m[i] = .9*m[i]+.1*g
                second[i] = .95*second[i]+.05*g*g
                update = lr*(m[i]/(1-.9**step))/(np.sqrt(second[i]/(1-.95**step))+1e-8)
            else:
                m[i] = .95*m[i]+.05*g
                update = lr*ns_numpy(m[i],5)
            if i == 0 and (step == 1 or step % 100 == 0):
                spectra.append(np.linalg.svd(update,compute_uv=False))
                spectrum_steps.append(step)
            p -= update
    final_loss,_,_ = objective(u,v,target)
    write_csv(folder/(stem+'.csv'),records)
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/(stem+'.npz'),update_singular_values=np.stack(spectra),steps=spectrum_steps,target_singular_values=np.geomspace(1,1e-4,32))
    result = dict(config=config,config_hash=fingerprint(config),valid=True,initial_loss=records[0]['loss'],
        final_loss=final_loss,relative_reconstruction=float(np.sqrt(2*final_loss)/np.linalg.norm(target)),
        elapsed_seconds=time.perf_counter()-start,steps=budget)
    write_json(path,result)
    print(f'factorization {stem}: loss={final_loss:.6g}',flush=True)
    return result


def main():
    chosen,pilots = {},[]
    for method,grid in GRIDS.items():
        records = [run(method,42,lr,300,True) for lr in grid]
        valid = [r for r in records if r['valid']]
        if not valid:
            raise RuntimeError('All learning-rate candidates failed for '+method)
        chosen[method] = min(valid,key=lambda r:r['final_loss'])['config']['lr']
        pilots.extend(records)
    write_json(OUT/'selection.json',dict(chosen=chosen,criterion='final pilot objective after 300 steps',pilot_seed=42,candidates=pilots))
    for seed in [0,1,2]:
        for method in GRIDS:
            run(method,seed,chosen[method],3000)


if __name__ == '__main__':
    main()
