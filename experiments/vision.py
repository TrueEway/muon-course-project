"""Fixed-budget, full ResNet-18 CIFAR-10 comparison and K ablation.

Each training run has a fresh process so Windows peak working-set memory is
per-run. Test-set curves are evaluated after training and never select LRs.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import ctypes
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from .common import ROOT, completed, environment, fingerprint, write_csv, write_json
from .optim import ns_torch, polar_torch
from .vision_data import prepare_cifar
from .vision_model import ResNet18, vision_optimizers
from .resources import configure_cuda,synchronize,cooperate,check_headroom,hardware_metadata

OUT = ROOT / 'output' / 'revision'
PROTOCOL = dict(edition='v1.1-h20', model='canonical ResNet-18, original 7x7 stem, native 32x32 inputs',
    epochs=3, batch=128, threads=4, seeds=[0,1,2], pilot_seed=42, pilot_steps=96,
    threshold_val_accuracy_percent=40.0, k_values=[0,1,3,5,7],
    learning_rate_grids={'sgd':[0.03,0.1,0.3], 'adamw':[0.0003,0.001,0.003], 'muon':[0.005,0.02,0.05]},
    ablation_policy='K=5 pilot selects Muon LR; K=0,1,3,7 share the same selected LR and auxiliary optimizer',
    test_policy='Test curves evaluated post-training; LR selection and target hitting use validation only',
    k_zero='Frobenius-normalized momentum, zero polynomial applications',
    augmentation='zero-pad 4, random 32x32 crop, random horizontal flip',
    schedule='linear 50-step warmup; cosine decay to 0.1 of base LR',
    weight_decay=0.0005, reference_parameter='layer1.0.conv2.weight', device='cuda:0', dtype='float32',
    resource_policy=dict(single_visible_gpu=True,gpu_allocator_fraction=0.04,target_duty_cycle=0.35,
                         cpu_affinity_count=4,cpu_threads=4,interop_threads=1,data_loader_workers=0,
                         minimum_free_gpu_gib=12,tf32=False))


def preprocess(images, generator=None):
    x = images.float().div_(255.0)
    if generator is not None:
        b = len(x)
        padded = F.pad(x, (4,4,4,4))
        offsets = torch.randint(0,9,(2,b),generator=generator)
        rows = offsets[0,:,None,None,None] + torch.arange(32)[None,None,:,None]
        x = padded.gather(2, rows.expand(b,3,32,40))
        cols = offsets[1,:,None,None,None] + torch.arange(32)[None,None,None,:]
        x = x.gather(3, cols.expand(b,3,32,32))
        flip = torch.rand(b, generator=generator) < 0.5
        x[flip] = x[flip].flip(-1)
    mean = x.new_tensor([0.4914,0.4822,0.4465])[None,:,None,None]
    std = x.new_tensor([0.2470,0.2435,0.2616])[None,:,None,None]
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    images, labels = data
    device = next(model.parameters()).device
    total, correct = 0.0, 0
    for start in range(0,len(labels),256):
        tic = time.perf_counter()
        y = labels[start:start+256].to(device)
        logits = model(preprocess(images[start:start+256]).to(device))
        total += float(F.cross_entropy(logits,y,reduction='sum'))
        correct += int((logits.argmax(-1) == y).sum())
        synchronize()
        cooperate(time.perf_counter()-tic)
    model.train()
    return total / len(labels), 100.0 * correct / len(labels)


def peak_rss_mib():
    if os.name != 'nt':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    class Counters(ctypes.Structure):
        _fields_ = [('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)] + [(name,ctypes.c_size_t) for name in
            ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    counter = Counters()
    counter.cb = ctypes.sizeof(counter)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    get_memory.restype = ctypes.c_int
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if not get_memory(handle, ctypes.byref(counter), counter.cb):
        raise RuntimeError('Cannot measure Windows peak working set')
    return counter.PeakWorkingSetSize / (1024**2)


def condition_and_rank(s):
    positive = s[s > max(float(s[0])*1e-6, 1e-12)] if len(s) else s
    return (float(positive[0] / positive[-1]),len(positive)) if len(positive) else (None,0)


def worker(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    device = configure_cuda()
    data, metadata = prepare_cifar()
    config = dict(protocol=PROTOCOL, method=args.method, seed=args.seed, lr=args.lr, auxiliary_lr=args.aux_lr,
                  k=args.k, pilot=args.pilot, dataset_sha256=metadata['sha256'])
    category = 'pilot' if args.pilot else 'formal'
    name = f'{args.method}_K{args.k}_s{args.seed}_lr{args.lr:g}'
    path = OUT / 'vision' / category / (name + '.json')
    previous = completed(path,config)
    if previous:
        print('cached ' + name,flush=True)
        return
    model = ResNet18().to(device=device,memory_format=torch.channels_last)
    optimizers, names = vision_optimizers(model,args.method,args.lr,args.aux_lr,args.k)
    reference = dict(model.named_parameters())[PROTOCOL['reference_parameter']]
    data_rng = torch.Generator().manual_seed(args.seed+700)
    augmentation_rng = torch.Generator().manual_seed(args.seed+900)
    images, labels = data['train']
    batches = math.ceil(len(labels)/128)
    budget = PROTOCOL['pilot_steps'] if args.pilot else batches*PROTOCOL['epochs']
    batch_logs, epoch_logs, saved, spectra = [],[],[],[]
    compute_seconds, steps = 0.0,0
    best_val, best_epoch = math.inf,0
    checkpoint_folder = OUT / 'checkpoints' / name
    checkpoint_folder.mkdir(parents=True,exist_ok=True)
    wall_start = time.perf_counter()
    for epoch in range(1,PROTOCOL['epochs']+1):
        permutation = torch.randperm(len(labels),generator=data_rng)
        loss_sum, count = 0.0,0
        for start in range(0,len(labels),128):
            steps += 1
            diagnostic = (steps == 1 or steps % batches == 0)
            before = reference.detach().clone() if diagnostic and not args.pilot else None
            synchronize()
            tic = time.perf_counter()
            index = permutation[start:start+128]
            x, y = preprocess(images[index],augmentation_rng).to(device), labels[index].to(device)
            factor = (steps/50 if steps <= 50 else 0.1+0.9*0.5*(1+math.cos(math.pi*(steps-50)/max(1,budget-50))))
            for optimizer, base_lr in optimizers:
                for group in optimizer.param_groups:
                    group['lr'] = base_lr*factor
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits,y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss in {name} step {steps}')
            loss.backward()
            for optimizer,_ in optimizers:
                optimizer.step()
            synchronize()
            duration = time.perf_counter()-tic
            compute_seconds += duration
            loss_value = float(loss.detach())
            loss_sum += loss_value*len(y)
            count += len(y)
            batch_row = dict(step=steps,epoch=epoch,loss=loss_value,lr_factor=factor,step_seconds=duration,
                             train_compute_seconds=compute_seconds)
            if before is not None:
                delta = (before-reference.detach()).reshape(64,-1).cpu().double()
                singular = torch.linalg.svdvals(delta).cpu().numpy()
                gradient_s = torch.linalg.svdvals(reference.grad.reshape(64,-1).detach().cpu().double()).numpy()
                condition, rank = condition_and_rank(singular)
                diagnostic_row = dict(step=steps,epoch=(steps/batches),update_condition=condition,rank=rank,
                                      gradient_nuclear=float(gradient_s.sum()),delta_operator_error=None)
                if args.method == 'muon':
                    opt = optimizers[-1][0]
                    matrix = reference.grad.lerp(opt.state[reference]['momentum'],0.95).reshape(64,-1)
                    q = matrix/(matrix.norm()+1e-7) if args.k == 0 else ns_torch(matrix,args.k)
                    error = float(torch.linalg.matrix_norm(q.detach().cpu().double()-polar_torch(matrix.detach().cpu()).double(),ord=2))
                    diagnostic_row['delta_operator_error'] = error
                spectra.append((diagnostic_row,singular,gradient_s))
            synchronize()
            batch_row['throttle_sleep_seconds'] = cooperate(time.perf_counter()-tic)
            batch_row['wall_seconds'] = time.perf_counter()-wall_start
            batch_logs.append(batch_row)
            if steps % 100 == 0 or steps == budget:
                check_headroom()
                print(f'{category} {name} step={steps}/{budget} loss={loss_value:.4f} compute={compute_seconds:.1f}s',flush=True)
            if steps >= budget:
                break
        val_loss,val_acc = evaluate(model,data['val'])
        epoch_logs.append(dict(epoch=epoch,step=steps,train_loss=loss_sum/count,val_loss=val_loss,val_accuracy=val_acc,
                               train_compute_seconds=compute_seconds,wall_seconds=time.perf_counter()-wall_start,
                               test_loss=None,test_accuracy=None))
        if val_loss < best_val:
            best_val,best_epoch = val_loss,epoch
        if not args.pilot:
            checkpoint = checkpoint_folder / f'epoch{epoch}.pt'
            torch.save(model.state_dict(),checkpoint)
            saved.append(checkpoint)
        print(f'{name} epoch={epoch} val_acc={val_acc:.2f}% val_loss={val_loss:.4f}',flush=True)
        if steps >= budget:
            break
    wall_training = time.perf_counter()-wall_start
    # Fixed protocol and training have finished before the held-out test is read.
    if not args.pilot:
        for row,checkpoint in zip(epoch_logs,saved):
            model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
            row['test_loss'],row['test_accuracy'] = evaluate(model,data['test'])
        for index,checkpoint in enumerate(saved,1):
            if index != best_epoch:
                checkpoint.unlink()
    hits = [row for row in epoch_logs if row['val_accuracy'] >= PROTOCOL['threshold_val_accuracy_percent']]
    path.parent.mkdir(parents=True,exist_ok=True)
    if spectra:
        write_csv(path.with_name(path.stem+'_spectrum.csv'),[item[0] for item in spectra])
        np.savez_compressed(path.with_suffix('.npz'), update_singular_values=np.stack([x[1] for x in spectra]),
                            gradient_singular_values=np.stack([x[2] for x in spectra]),steps=np.array([x[0]['step'] for x in spectra]))
    write_csv(path.with_suffix('.csv'),batch_logs)
    write_csv(path.with_name(path.stem+'_epochs.csv'),epoch_logs)
    selected = epoch_logs[best_epoch-1]
    summary = dict(config=config,config_hash=fingerprint(config),parameter_count=sum(p.numel() for p in model.parameters()),
        parameter_groups=names,steps=steps,epochs=len(epoch_logs),best_val_epoch=best_epoch,
        final_val_loss=epoch_logs[-1]['val_loss'],final_val_accuracy=epoch_logs[-1]['val_accuracy'],
        final_train_loss=epoch_logs[-1]['train_loss'],test_loss=selected['test_loss'],test_accuracy=selected['test_accuracy'],
        final_epoch_test_accuracy=epoch_logs[-1]['test_accuracy'],train_compute_seconds=compute_seconds,
        wall_training_seconds=wall_training,step_median_ms=1000*float(np.median([r['step_seconds'] for r in batch_logs])),
        step_mean_ms=1000*compute_seconds/steps,peak_process_rss_mib=peak_rss_mib(),
        peak_cuda_memory_mib=torch.cuda.max_memory_allocated(0)/(1024**2),
        peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(0)/(1024**2),
        throttle_sleep_seconds=sum(x['throttle_sleep_seconds'] for x in batch_logs),
        threshold_steps=hits[0]['step'] if hits else None,threshold_wall_seconds=hits[0]['wall_seconds'] if hits else None)
    write_json(path,summary)
    print(f'completed {category} {name}',flush=True)


def dispatch(method,seed,lr,aux_lr,k,pilot):
    command = [sys.executable,'-m','experiments.vision','--worker','--method',method,'--seed',str(seed),
               '--lr',str(lr),'--aux-lr',str(aux_lr),'--k',str(k),'--threads',str(PROTOCOL['threads'])]
    if pilot:
        command.append('--pilot')
    subprocess.run(command,cwd=ROOT,check=True)
    category = 'pilot' if pilot else 'formal'
    return json.loads((OUT/'vision'/category/f'{method}_K{k}_s{seed}_lr{lr:g}.json').read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--method',choices=['sgd','adamw','muon'])
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--lr',type=float,default=0.02)
    parser.add_argument('--aux-lr',type=float,default=0.001)
    parser.add_argument('--k',type=int,default=5)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--pilot',action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    torch.set_num_threads(PROTOCOL['threads'])
    old_protocol = OUT/'protocol.json'
    if old_protocol.exists() and json.loads(old_protocol.read_text(encoding='utf-8')) != PROTOCOL:
        raise RuntimeError('The frozen protocol changed; use a new edition rather than mix results')
    write_json(old_protocol,PROTOCOL)
    metadata = environment()
    metadata.update(hardware_metadata())
    write_json(OUT/'environment.json',metadata)
    prepare_cifar()
    chosen,pilots = {},[]
    for method in ['sgd','adamw','muon']:
        records = [dispatch(method,42,lr,chosen.get('adamw',0.001),5,True) for lr in PROTOCOL['learning_rate_grids'][method]]
        pilots.extend(records)
        chosen[method] = min(records,key=lambda r:r['final_val_loss'])['config']['lr']
    write_json(OUT/'vision'/'selection.json',dict(chosen=chosen,pilot_seed=42,metric='final validation CE after 96 steps',
        candidates=[dict(method=r['config']['method'],lr=r['config']['lr'],val_loss=r['final_val_loss'],val_accuracy=r['final_val_accuracy']) for r in pilots]))
    configurations = [('sgd',5),('adamw',5),('muon',5),('muon',0),('muon',1),('muon',3),('muon',7)]
    for index,seed in enumerate(PROTOCOL['seeds']):
        order = configurations[index:]+configurations[:index]
        for method,k in order:
            dispatch(method,seed,chosen[method],chosen['adamw'],k,False)
    print('CIFAR-10 comparison and K ablation complete.',flush=True)


if __name__ == '__main__':
    main()
