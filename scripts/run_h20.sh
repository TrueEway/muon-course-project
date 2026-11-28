#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${MUON_GPU:-2}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="$PWD/tmp" XDG_CACHE_HOME="$PWD/.cache" MPLCONFIGDIR="$PWD/.cache/matplotlib"
export CUDA_CACHE_PATH="$PWD/.cache/cuda" PIP_CACHE_DIR="$PWD/.cache/pip"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$PIP_CACHE_DIR"
python_bin="${MUON_PYTHON:-python}"
cpu_set="${MUON_CPUSET:-0,1,2,3}"
if [ "$#" -eq 0 ]; then
  set -- train.py
fi
exec nice -n 10 ionice -c 3 taskset -c "$cpu_set" "$python_bin" -u "$@"
