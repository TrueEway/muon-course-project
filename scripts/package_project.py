"""Package tracked deliverables, preserving relative paths and raw file bytes."""
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag',default='v1.1')
    args = parser.parse_args()
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    output = ROOT / 'dist' / ('muon-course-project-'+args.tag+'.zip')
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=8) as archive:
        for relative in sorted(filter(None, tracked)):
            archive.write(ROOT / relative, arcname='muon-course-project/' + relative)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(checksum + '  ' + output.name + '\n', encoding='ascii')
    print(json.dumps({'archive': str(output), 'bytes': output.stat().st_size, 'sha256': checksum}))


if __name__ == '__main__':
    main()
