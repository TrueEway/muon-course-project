from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .common import completed, fingerprint, write_csv, write_json
from .optim import ns_numpy, polar_numpy


def problem(seed: int, kappa: int):
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    v, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    a = (u * np.geomspace(1.0, kappa ** -0.5, 32)) @ v.T
    hessian = a.T @ a
    target = rng.normal(size=(32, 16)) / np.sqrt(32 * 16)
    displacement = rng.normal(size=target.shape)
    displacement /= np.linalg.norm(displacement)
    return hessian, target, target + displacement


def value_grad(w, target, hessian):
    error = w - target
    gradient = hessian @ error
    return float(0.5 * np.sum(error * gradient)), gradient


def run_deterministic(out: Path, seeds: list[int], overwrite=False):
    base = out / 'deterministic'
    for seed in seeds:
        for kappa in (1, 10, 100):
            hessian, target, initial = problem(seed, kappa)
            for horizon in (100, 300, 1000, 3000):
                for method, beta in [('gd', 0.0), ('momentum', 0.9), ('muon0', 0.0), ('muon9', 0.9)]:
                    config = dict(suite='deterministic', seed=seed, kappa=kappa, horizon=horizon,
                                  method=method, beta=beta, eta=0.1 / np.sqrt(horizon) if method.startswith('muon') else (0.9 if method == 'gd' else 0.1),
                                  shape=[32, 16], initialization='M1=G1', weight_decay=0)
                    stem = f's{seed}_k{kappa}_T{horizon}_{method}'
                    summary_path = base / (stem + '.json')
                    if completed(summary_path, config, overwrite):
                        continue
                    w, momentum = initial.copy(), None
                    eta = config['eta']
                    records = []
                    max_violation = 0.0
                    start = time.perf_counter()
                    for t in range(1, horizon + 1):
                        loss, gradient = value_grad(w, target, hessian)
                        momentum = gradient.copy() if momentum is None else beta * momentum + (1 - beta) * gradient
                        direction = polar_numpy(momentum) if method.startswith('muon') else momentum
                        nuclear = float(np.linalg.svd(gradient, compute_uv=False).sum())
                        error = float(np.linalg.norm(momentum - gradient))
                        records.append(dict(step=t, loss=loss, grad_fro=float(np.linalg.norm(gradient)),
                                            grad_nuclear=nuclear, tracking_error=error))
                        next_w = w - eta * direction
                        if method.startswith('muon'):
                            next_loss, _ = value_grad(next_w, target, hessian)
                            rhs = loss - eta * nuclear + 2 * eta * np.sqrt(16) * error + 0.5 * 16 * eta ** 2
                            max_violation = max(max_violation, next_loss - rhs)
                        w = next_w
                    elapsed = time.perf_counter() - start
                    final_loss, final_gradient = value_grad(w, target, hessian)
                    delta = records[0]['loss']
                    alpha = 1 - beta
                    theorem_bound = delta / (eta * horizon) + 16 * eta * (0.5 + 2 * beta / alpha)
                    write_csv(base / (stem + '.csv'), records)
                    write_json(summary_path, dict(config=config, config_hash=fingerprint(config),
                        initial_loss=delta, final_loss=final_loss, final_grad_fro=float(np.linalg.norm(final_gradient)),
                        avg_grad_fro=float(np.mean([x['grad_fro'] for x in records])),
                        avg_grad_nuclear=float(np.mean([x['grad_nuclear'] for x in records])),
                        theorem_bound=theorem_bound if method.startswith('muon') else None,
                        max_descent_violation=max_violation, elapsed_seconds=elapsed))
            print(f'deterministic seed={seed} kappa={kappa} complete', flush=True)


def run_stochastic(out: Path, seeds: list[int], overwrite=False):
    base = out / 'stochastic'
    for seed in seeds:
        hessian, target, initial = problem(seed, 10)
        for sigma in (0.01, 0.1):
            for batch in (1, 16):
                for horizon in (256, 1024, 4096, 16384):
                    alpha = horizon ** -0.5
                    beta = 1 - alpha
                    eta = 0.1 * horizon ** -0.75
                    config = dict(suite='stochastic', seed=seed, kappa=10, sigma=sigma, batch=batch,
                                  horizon=horizon, alpha=alpha, beta=beta, eta=eta, shape=[32, 16],
                                  initialization='M1=G1', weight_decay=0, noise='iid Gaussian, E||noise||F^2=sigma^2/B')
                    stem = f's{seed}_sigma{sigma}_B{batch}_T{horizon}'
                    summary_path = base / (stem + '.json')
                    if completed(summary_path, config, overwrite):
                        continue
                    rng = np.random.default_rng(seed + 10000)
                    w, momentum = initial.copy(), None
                    records = []
                    start = time.perf_counter()
                    for t in range(1, horizon + 1):
                        loss, gradient = value_grad(w, target, hessian)
                        sample = gradient + rng.normal(size=w.shape) * sigma / np.sqrt(batch * w.size)
                        momentum = sample.copy() if momentum is None else beta * momentum + alpha * sample
                        records.append(dict(step=t, loss=loss, grad_fro=float(np.linalg.norm(gradient)),
                                            grad_nuclear=float(np.linalg.svd(gradient, compute_uv=False).sum()),
                                            tracking_error=float(np.linalg.norm(momentum - gradient))))
                        w = w - eta * polar_numpy(momentum)
                    elapsed = time.perf_counter() - start
                    delta = records[0]['loss']
                    tracking_bound = (sigma / (horizon * alpha * np.sqrt(batch))
                                      + sigma * np.sqrt(alpha / batch) + beta * eta * np.sqrt(16) / alpha)
                    bound = delta / (eta * horizon) + 16 * eta / 2 + 2 * np.sqrt(16) * tracking_bound
                    final_loss, _ = value_grad(w, target, hessian)
                    write_csv(base / (stem + '.csv'), records)
                    write_json(summary_path, dict(config=config, config_hash=fingerprint(config),
                        final_loss=final_loss, avg_grad_fro=float(np.mean([x['grad_fro'] for x in records])),
                        avg_grad_nuclear=float(np.mean([x['grad_nuclear'] for x in records])),
                        avg_tracking_error=float(np.mean([x['tracking_error'] for x in records])),
                        tracking_bound=tracking_bound, theorem_bound=bound, elapsed_seconds=elapsed))
                print(f'stochastic seed={seed} sigma={sigma} B={batch} complete', flush=True)


def run_orthogonalization(out: Path, seeds: list[int], overwrite=False):
    base = out / 'orthogonalization'
    config = dict(suite='orthogonalization', seeds=seeds, steps=[1, 3, 5, 10],
                  shapes=[[32, 32], [64, 16], [16, 64]], spectra=['flat', 'ill_conditioned', 'rank_deficient'],
                  dtype='float64', eps=1e-7, repeats=30)
    summary_path = base / 'summary.json'
    if completed(summary_path, config, overwrite):
        return
    records = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for m, n in config['shapes']:
            r = min(m, n)
            u, _ = np.linalg.qr(rng.normal(size=(m, r)))
            v, _ = np.linalg.qr(rng.normal(size=(n, r)))
            for spectrum in config['spectra']:
                s = np.ones(r) if spectrum == 'flat' else np.geomspace(1, 1e-6, r)
                if spectrum == 'rank_deficient':
                    s[r // 2:] = 0
                matrix = (u * s) @ v.T
                exact = polar_numpy(matrix)
                start = time.perf_counter()
                for _ in range(config['repeats']):
                    polar_numpy(matrix)
                svd_ms = 1000 * (time.perf_counter() - start) / config['repeats']
                for steps in config['steps']:
                    approximate = ns_numpy(matrix, steps)
                    start = time.perf_counter()
                    for _ in range(config['repeats']):
                        ns_numpy(matrix, steps)
                    ns_ms = 1000 * (time.perf_counter() - start) / config['repeats']
                    records.append(dict(seed=seed, rows=m, cols=n, spectrum=spectrum, steps=steps,
                        relative_error=float(np.linalg.norm(approximate - exact) / np.linalg.norm(exact)),
                        cosine=float(np.sum(approximate * exact) / (np.linalg.norm(approximate) * np.linalg.norm(exact))),
                        output_spectral=float(np.linalg.norm(approximate, 2)), svd_ms=svd_ms, ns_ms=ns_ms))
    write_csv(base / 'measurements.csv', records)
    write_json(summary_path, dict(config=config, config_hash=fingerprint(config), records=len(records)))
    print('orthogonalization complete', flush=True)
