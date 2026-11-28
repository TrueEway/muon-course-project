"""Numerical and provenance audit for the frozen v1.1 additions."""
from pathlib import Path
import csv
import hashlib
import json
import math
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def csv_rows(path):
    with path.open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-11)


def verify_revision():
    rev = ROOT/'output'/'revision'
    audit = json.loads((ROOT/'output'/'revision_audit.json').read_text(encoding='utf-8'))
    protocol = json.loads((rev/'protocol.json').read_text(encoding='utf-8'))
    environment = json.loads((rev/'environment.json').read_text(encoding='utf-8'))
    memory_cap_mib = environment['gpu_total_memory_mib']*protocol['resource_policy']['gpu_allocator_fraction']
    files = sorted((rev/'vision'/'formal').glob('*.json'))
    assert len(files) == 21, 'Incomplete ResNet protocol'
    actual = set()
    for path in files:
        record = json.loads(path.read_text(encoding='utf-8'))
        config = record['config']
        actual.add((config['seed'],config['method'],config['k']))
        assert config['protocol'] == protocol, path
        assert record['parameter_count'] == 11181642, path
        assert record['steps'] == 1056 and record['epochs'] == 3, path
        batches = csv_rows(path.with_suffix('.csv'))
        epochs = csv_rows(path.with_name(path.stem+'_epochs.csv'))
        assert len(batches) == 1056 and len(epochs) == 3, path
        assert [int(x['step']) for x in batches] == list(range(1,1057)), path
        assert all(math.isfinite(float(x['loss'])) for x in batches), path
        for index,row in enumerate(epochs):
            current = batches[index*352:(index+1)*352]
            mean = sum(float(x['loss'])*(72 if int(x['step'])%352==0 else 128) for x in current)/45000
            assert close(mean,row['train_loss']), path
            assert int(row['step']) == (index+1)*352, path
        best = min(epochs,key=lambda x:float(x['val_loss']))
        assert int(best['epoch']) == record['best_val_epoch'], path
        assert close(best['test_accuracy'],record['test_accuracy']), path
        assert close(best['test_loss'],record['test_loss']), path
        hits = [x for x in epochs if float(x['val_accuracy']) >= 40.0]
        assert record['threshold_steps'] == (int(hits[0]['step']) if hits else None), path
        assert close(sum(float(x['step_seconds']) for x in batches),record['train_compute_seconds']), path
        assert record['peak_process_rss_mib'] > 0 and record['peak_cuda_memory_mib'] > 0, path
        assert record['peak_cuda_reserved_mib'] >= record['peak_cuda_memory_mib'], path
        assert record['peak_cuda_reserved_mib'] < memory_cap_mib, path
        assert close(sum(float(x['throttle_sleep_seconds']) for x in batches),record['throttle_sleep_seconds']), path
        assert record['wall_training_seconds'] >= record['train_compute_seconds'], path
        arrays = np.load(path.with_suffix('.npz'),allow_pickle=False)
        assert arrays['update_singular_values'].shape == (4,64), path
        assert arrays['steps'].tolist() == [1,352,704,1056], path
        for spectrum,row in zip(arrays['update_singular_values'],csv_rows(path.with_name(path.stem+'_spectrum.csv'))):
            nonzero = spectrum[spectrum>max(float(spectrum[0])*1e-6,1e-12)]
            assert len(nonzero) == int(row['rank']), path
            if len(nonzero):
                assert close(nonzero[0]/nonzero[-1],row['update_condition']), path
    expected = {(seed,method,k) for seed in [0,1,2] for method,k in [('sgd',5),('adamw',5)]+[('muon',k) for k in [0,1,3,5,7]]}
    assert actual == expected, 'Run/seed/K mismatch'
    factors = sorted((rev/'factorization'/'formal').glob('*.json'))
    assert len(factors) == 9
    for path in factors:
        record = json.loads(path.read_text(encoding='utf-8'))
        rows = csv_rows(path.with_suffix('.csv'))
        assert len(rows) == 3000 and record['valid'], path
        assert record['config']['kappa'] == 10000, path
        assert close(record['initial_loss'],rows[0]['loss']), path
        assert all(math.isfinite(float(x['gradient_nuclear'])) for x in rows), path
        singular = np.load(path.with_suffix('.npz'))['target_singular_values']
        assert close(singular[0]/singular[-1],10000), path
    digest = hashlib.sha256()
    for path in sorted(rev.rglob('*')):
        if path.suffix in {'.csv','.json','.npz'}:
            digest.update(str(path.relative_to(rev)).replace('\\','/').encode())
            digest.update(path.read_bytes())
    assert digest.hexdigest() == audit['revision_results_sha256'], 'New raw records changed'
    assert protocol['device'] == 'cuda:0' and protocol['threads'] == 4
    assert environment['cpu_affinity_count'] == 4
    critical = ['vision.py','vision_model.py','vision_data.py','factorization.py','optim.py','common.py','resources.py']
    for name in critical:
        relative = 'experiments/'+name
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == environment['source_sha256'][relative], relative
    figure_fonts = {}
    for index in range(1,5):
        path = ROOT/'output'/'figures'/f'fig{index}.pdf'
        fonts = subprocess.check_output(['pdffonts',str(path)],text=True)
        assert 'Type 3' not in fonts and 'TrueType' in fonts, path
        assert all(row.split()[-5] == 'yes' for row in fonts.splitlines()[2:] if row.strip()), path
        figure_fonts[path.name] = 'embedded TrueType; no Type 3'
    return dict(revision_formal_runs=30,vision_formal_runs=21,factorization_formal_runs=9,
                revision_results_sha256=digest.hexdigest(),revision_training_source_files_checked=len(critical),
                per_run_parameters=11181642,cifar_split=[45000,5000,10000],figure_font_checks=figure_fonts)


if __name__ == '__main__':
    print(json.dumps(verify_revision(),indent=2))
