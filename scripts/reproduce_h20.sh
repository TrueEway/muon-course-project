#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MUON_GPU:?Select one GPU after checking current load}"
: "${MUON_CPUSET:?Select four logical CPUs after checking current load}"
bash scripts/run_h20.sh -m unittest discover -s tests -v
bash scripts/run_h20.sh -m experiments.factorization
bash scripts/run_h20.sh train.py
bash scripts/finalize_h20.sh
