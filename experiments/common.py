from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('Cannot write an empty experiment log')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment() -> dict:
    import torch
    return {
        'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
        'platform': platform.platform(), 'processor': platform.processor(),
        'cpu_count': os.cpu_count(), 'torch_threads': torch.get_num_threads(),
        'device': 'cpu', 'source_sha256': {
            str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'experiments').glob('*.py'))
        },
    }


def completed(path: Path, config: dict, overwrite=False) -> dict | None:
    if not path.exists() or overwrite:
        return None
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('config_hash') != fingerprint(config):
        raise RuntimeError(f'Configuration changed for {path}; choose another output directory or --overwrite')
    return value
