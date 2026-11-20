from __future__ import annotations

import argparse
import os
from pathlib import Path

# Set before importing NumPy/PyTorch. Small matrix experiments need one BLAS
# thread; PyTorch intra-op threads are controlled separately below.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import torch

from .common import ROOT, environment, write_json
from .language import profile_language, run_language
from .synthetic import run_deterministic, run_orthogonalization, run_stochastic


def main():
    parser = argparse.ArgumentParser(description='Reproduce the Muon course report experiments')
    parser.add_argument('--suite', choices=['all', 'synthetic', 'deterministic', 'stochastic', 'orthogonalization', 'language', 'profile'], default='all')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'results')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.output = args.output.resolve()
    write_json(args.output / f'environment_{args.suite}.json', environment())
    print(f'suite={args.suite} threads={args.threads} seeds={args.seeds}', flush=True)
    if args.suite in {'all', 'synthetic', 'deterministic'}:
        run_deterministic(args.output, args.seeds, args.overwrite)
    if args.suite in {'all', 'synthetic', 'stochastic'}:
        run_stochastic(args.output, args.seeds, args.overwrite)
    if args.suite in {'all', 'synthetic', 'orthogonalization'}:
        run_orthogonalization(args.output, args.seeds, args.overwrite)
    if args.suite in {'all', 'language'}:
        run_language(args.output, args.seeds, args.overwrite)
    if args.suite == 'profile':
        profile_language(args.output)


if __name__ == '__main__':
    main()
