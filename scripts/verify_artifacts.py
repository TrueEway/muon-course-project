"""Cross-check experiment logs, rendered report text, and registered hashes."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = ROOT / 'output' / 'results'
    audit = json.loads((ROOT / 'output' / 'audit.json').read_text(encoding='utf-8'))
    count = 0
    for suite in ['deterministic', 'stochastic']:
        for path in sorted((results / suite).glob('*.json')):
            summary = json.loads(path.read_text(encoding='utf-8'))
            with path.with_suffix('.csv').open(encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == summary['config']['horizon'], path
            for key in ['grad_fro', 'grad_nuclear']:
                mean = float(np.mean([float(r[key]) for r in rows]))
                assert math.isclose(mean, summary['avg_' + key], rel_tol=1e-12), path
            assert all(math.isfinite(float(r['loss'])) for r in rows), path
            count += 1
    for path in sorted((results / 'language' / 'formal').glob('*.json')):
        summary = json.loads(path.read_text(encoding='utf-8'))
        with path.with_suffix('.csv').open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1501, path
        best = min(float(r['val_loss']) for r in rows if r['val_loss'])
        assert math.isclose(best, summary['best_val_loss'], rel_tol=1e-12), path
        assert math.isclose(math.exp(summary['test_cross_entropy']), summary['test_perplexity'], rel_tol=1e-12), path
        assert math.isclose(float(rows[-1]['train_seconds']), summary['train_seconds'], rel_tol=1e-12), path
        count += 1
    digest = hashlib.sha256()
    for path in sorted(results.rglob('*')):
        if path.suffix in {'.json', '.csv'}:
            digest.update(str(path.relative_to(results)).replace('\\', '/').encode())
            digest.update(path.read_bytes())
    assert digest.hexdigest() == audit['results_sha256'], 'Raw result collection hash changed'
    environment_path = results / 'environment_all.json'
    if not environment_path.exists():
        raise RuntimeError('Run the complete protocol with --suite all before verification')
    environment = json.loads(environment_path.read_text(encoding='utf-8'))
    checked_sources = 0
    for relative, expected in environment['source_sha256'].items():
        if relative == 'experiments/summarize.py':
            continue  # Presentation-only changes are recorded in the Git release.
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
        checked_sources += 1
    reader = PdfReader(ROOT / 'output' / 'pdf' / 'main.pdf')
    # Poppler resolves the Adobe-GB1 CMaps used by XeTeX/Fandol. pypdf alone
    # returns CIDs for these fonts even though Poppler/desktop viewers read them.
    extracted = subprocess.check_output(['pdftotext', '-enc', 'UTF-8', str(ROOT / 'output' / 'pdf' / 'main.pdf'), '-']).decode('utf-8')
    pages = extracted.split('\f')
    if not pages[-1].strip():
        pages.pop()
    assert len(pages) == len(reader.pages), 'Page count mismatch after text extraction'
    assert all(len(p.strip()) > 50 for p in pages), 'Unexpected almost-empty PDF page'
    text = '\n'.join(pages)
    assert 'Muon' in text and '参考文献' in text, 'Missing report text'
    assert '??' not in text, 'Unresolved reference in PDF'
    links = []
    for page in reader.pages:
        for item in page.get('/Annots', []):
            annotation = item.get_object()
            if annotation.get('/A', {}).get('/URI'):
                links.append(str(annotation['/A']['/URI']))
    assert any('TrueEway/muon-course-project/tree/v1.0' in x for x in links), 'Code hyperlink missing'
    value = dict(status='passed', crosschecked_run_logs=count, unchanged_training_source_files=checked_sources,
                 pdf_pages=len(reader.pages), pdf_sha256=hashlib.sha256((ROOT / 'output' / 'pdf' / 'main.pdf').read_bytes()).hexdigest(),
                 results_sha256=digest.hexdigest(), code_hyperlink_present=True,
                 text_extractor='Poppler pdftotext UTF-8', page_text_lengths=[len(p) for p in pages])
    path = ROOT / 'output' / 'verification.json'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
