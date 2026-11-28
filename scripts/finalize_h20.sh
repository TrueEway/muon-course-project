#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Optional project-local PDF audit dependency; never replaces server libraries.
if [ -d "$PWD/.cache/audit/pypdf" ]; then
  export PYTHONPATH="$PWD/.cache/audit${PYTHONPATH:+:$PYTHONPATH}"
fi
bash scripts/run_h20.sh scripts/capture_h20_environment.py
bash scripts/run_h20.sh plot.py
nice -n 10 ionice -c 3 taskset -c "${MUON_CPUSET:?Select four logical CPUs}" bash scripts/build_report.sh
bash scripts/run_h20.sh scripts/verify_artifacts.py
bash scripts/run_h20.sh scripts/render_review.py
echo 'POSTPROCESS_COMPLETE'
