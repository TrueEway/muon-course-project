from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import ROOT, write_csv, write_json

COLORS = ['#2563a6', '#b45c21', '#2c8065', '#8356a6']
LABELS = {'gd': 'Gradient descent', 'momentum': 'Momentum GD', 'muon0': r'Muon ($\beta=0$)',
          'muon9': r'Muon ($\beta=0.9$)', 'adamw': 'AdamW', 'muon_svd': 'Muon-SVD + AdamW', 'muon_ns': 'Muon-NS + AdamW'}


def read_jsons(path):
    return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(path.glob('*.json'))]


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def stats(values):
    x = np.asarray(values, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def pm(values, scientific=False):
    mean, std = stats(values)
    if scientific:
        def tex_scientific(value, digits):
            mantissa, exponent = f'{value:.{digits}e}'.split('e')
            return mantissa + r'\times10^{' + str(int(exponent)) + '}'
        return '$' + tex_scientific(mean, 2) + r'\,\pm\,' + tex_scientific(std, 1) + '$'
    return rf'${mean:.3f}\,\pm\,{std:.3f}$'


def finish(fig, path):
    fig.tight_layout(pad=0.7)
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)


def write_tex_table(path, columns, header, rows):
    # Keep the alignment environment and its final rule in the same input file;
    # LaTeX file hooks between a final row break and \bottomrule break noalign.
    text = ('\\begin{tabular}{' + columns + '}\\toprule\n'
            + header + r' \\ \midrule' + '\n'
            + '\n'.join(rows) + '\n' + r'\bottomrule\end{tabular}' + '\n')
    path.write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'output' / 'results')
    parser.add_argument('--output', type=Path, default=ROOT / 'output')
    args = parser.parse_args()
    out, result = args.output, args.input
    figures, tables = out / 'figures', out / 'tables'
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.18,
                         'pdf.fonttype': 42, 'mathtext.fontset': 'stix', 'lines.linewidth': 1.6})
    det = read_jsons(result / 'deterministic')
    stochastic = read_jsons(result / 'stochastic')
    formal = read_jsons(result / 'language' / 'formal')
    if len(det) != 144 or len(stochastic) != 48 or len(formal) != 9:
        raise RuntimeError(f'Expected 144 deterministic, 48 stochastic and 9 formal LM runs; got {len(det)}, {len(stochastic)}, {len(formal)}')
    groups = defaultdict(list)
    for record in det:
        c = record['config']
        groups[(c['kappa'], c['horizon'], c['method'])].append(record)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45))
    for ax, kappa in zip(axes, [1, 10, 100]):
        for index, method in enumerate(['gd', 'momentum', 'muon0', 'muon9']):
            curves = []
            for s in range(3):
                rows = read_csv(result / 'deterministic' / f's{s}_k{kappa}_T3000_{method}.csv')
                curves.append([float(x['loss']) for x in rows])
            mean = np.maximum(np.mean(curves, axis=0), 1e-18)
            ax.semilogy(np.arange(1, 3001), mean, color=COLORS[index], label=LABELS[method])
        ax.set(title=rf'$\kappa={kappa}$', xlabel='Iteration', ylim=(1e-18, 1))
    axes[0].set_ylabel('Objective gap (mean)')
    axes[1].legend(fontsize=7, loc='lower left')
    finish(fig, figures / 'deterministic_loss')

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    horizons = [100, 300, 1000, 3000]
    for ax, method in zip(axes, ['muon0', 'muon9']):
        observed, bound = [], []
        for t in horizons:
            records = groups[(10, t, method)]
            observed.append(stats([r['avg_grad_nuclear'] for r in records]))
            bound.append(np.mean([r['theorem_bound'] for r in records]))
        means, stds = np.array(observed).T
        ax.errorbar(horizons, means, yerr=stds, marker='o', color=COLORS[0], label='Observed nuclear gradient')
        ax.plot(horizons, bound, '--s', color=COLORS[1], label='Finite-horizon upper bound')
        ax.plot(horizons, means[0] * (np.array(horizons) / horizons[0]) ** -0.5, ':', color='gray', label=r'$T^{-1/2}$ reference')
        ax.set(xscale='log', yscale='log', xlabel='Horizon T', title=LABELS[method])
    axes[0].set_ylabel('Average gradient nuclear norm')
    axes[0].legend(fontsize=6.8)
    finish(fig, figures / 'deterministic_rate')

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.1), sharex=True)
    stochastic_groups = defaultdict(list)
    for record in stochastic:
        c = record['config']
        stochastic_groups[(c['sigma'], c['batch'], c['horizon'])].append(record)
    for row, sigma in enumerate([0.01, 0.1]):
        for col, batch in enumerate([1, 16]):
            ax = axes[row, col]
            ts = [256, 1024, 4096, 16384]
            means, stds, bounds = [], [], []
            for t in ts:
                records = stochastic_groups[(sigma, batch, t)]
                mean, std = stats([r['avg_grad_nuclear'] for r in records])
                means.append(mean)
                stds.append(std)
                bounds.append(np.mean([r['theorem_bound'] for r in records]))
            ax.errorbar(ts, means, yerr=stds, marker='o', color=COLORS[0], label='Observed')
            ax.plot(ts, bounds, '--s', color=COLORS[1], label='Upper bound')
            ax.plot(ts, means[0] * (np.array(ts) / ts[0]) ** -0.25, ':', color='gray', label=r'$T^{-1/4}$ reference')
            ax.set(xscale='log', yscale='log', title=rf'$\sigma={sigma},\ B={batch}$')
            if row == 1:
                ax.set_xlabel('Horizon T')
            if col == 0:
                ax.set_ylabel('Avg. nuclear gradient')
    axes[0, 0].legend(fontsize=7)
    finish(fig, figures / 'stochastic_rate')

    orth = read_csv(result / 'orthogonalization' / 'measurements.csv')
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55))
    for index, spectrum in enumerate(['flat', 'ill_conditioned', 'rank_deficient']):
        points = []
        for steps in [1, 3, 5, 10]:
            values = [float(r['relative_error']) for r in orth if r['spectrum'] == spectrum and int(r['steps']) == steps]
            points.append(stats(values))
        points = np.asarray(points)
        axes[0].plot([1, 3, 5, 10], points[:, 0], '-o', color=COLORS[index], label=spectrum.replace('_', ' '))
        axes[1].plot([1, 3, 5, 10], [np.mean([float(r['cosine']) for r in orth if r['spectrum'] == spectrum and int(r['steps']) == s]) for s in [1, 3, 5, 10]], '-o', color=COLORS[index])
    for index, shape in enumerate([(32, 32), (64, 16), (16, 64)]):
        axes[2].plot([1, 3, 5, 10], [np.mean([float(r['ns_ms']) / float(r['svd_ms']) for r in orth if (int(r['rows']), int(r['cols'])) == shape and int(r['steps']) == s]) for s in [1, 3, 5, 10]], '-o', color=COLORS[index], label=f'{shape[0]} x {shape[1]}')
    axes[0].set(ylabel='Relative polar error', xlabel='NS steps')
    axes[1].set(ylabel='Cosine with exact polar', xlabel='NS steps')
    axes[2].set(ylabel='NS / SVD runtime', xlabel='NS steps')
    axes[2].axhline(1, color='gray', linestyle=':')
    axes[0].legend(fontsize=6)
    axes[2].legend(fontsize=6)
    finish(fig, figures / 'orthogonalization')

    lm_groups = defaultdict(list)
    for record in formal:
        lm_groups[record['config']['method']].append(record)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for index, method in enumerate(['adamw', 'muon_svd', 'muon_ns']):
        val_curves, time_curves = [], []
        for record in lm_groups[method]:
            c = record['config']
            path = result / 'language' / 'formal' / f"{method}_s{c['seed']}_lr{c['lr']:g}_T{c['horizon']}.csv"
            rows = [r for r in read_csv(path) if r['val_loss']]
            steps = np.array([int(r['step']) for r in rows])
            val_curves.append([float(r['val_loss']) for r in rows])
            time_curves.append([float(r['train_seconds']) for r in rows])
        means = np.mean(val_curves, axis=0)
        stds = np.std(val_curves, axis=0, ddof=1)
        axes[0].plot(steps, means, color=COLORS[index], label=LABELS[method])
        axes[0].fill_between(steps, means - stds, means + stds, color=COLORS[index], alpha=0.12)
        # Points are averaged at the same iteration, then placed at mean compute
        # time. This is descriptive, not interpolation at matched wall time.
        axes[1].plot(np.mean(time_curves, axis=0), means, color=COLORS[index], label=LABELS[method])
    axes[0].set(xlabel='Training iteration', ylabel='Validation cross entropy')
    axes[1].set(xlabel='Training compute seconds (mean)', ylabel='Validation cross entropy')
    axes[0].legend(fontsize=7)
    finish(fig, figures / 'language_learning')

    det_rows = []
    for method in ['gd', 'momentum', 'muon0', 'muon9']:
        records = groups[(10, 3000, method)]
        det_rows.append(LABELS[method] + ' & ' + pm([r['final_loss'] for r in records], True) + ' & ' + pm([r['avg_grad_nuclear'] for r in records]) + r' \\')
    write_tex_table(tables / 'deterministic.tex', 'lcc', '方法 & 末点目标差 & 平均梯度核范数', det_rows)
    lm_rows, aggregate_rows = [], []
    for method in ['adamw', 'muon_svd', 'muon_ns']:
        records = lm_groups[method]
        lr = records[0]['config']['lr']
        lm_rows.append(LABELS[method].replace(' + ', '+') + f' & {lr:g} & ' + pm([r['test_cross_entropy'] for r in records])
                       + ' & ' + pm([r['test_perplexity'] for r in records]) + ' & ' + pm([r['train_seconds'] for r in records]) + r' \\')
        row = dict(method=method, lr=lr, seeds=len(records))
        for key in ['best_val_loss', 'test_cross_entropy', 'test_perplexity', 'train_seconds', 'tokens_per_second']:
            row[key + '_mean'], row[key + '_std'] = stats([r[key] for r in records])
        aggregate_rows.append(row)
    write_tex_table(tables / 'language.tex', 'lcccc', '方法 & 学习率 & 测试交叉熵 & 测试困惑度 & 训练计算耗时 / 秒', lm_rows)
    write_csv(tables / 'language_summary.csv', aggregate_rows)
    pilot = json.loads((result / 'language' / 'selection.json').read_text(encoding='utf-8'))
    pilot_rows = [LABELS[r['method']].replace(' + ', '+') + f" & {r['lr']:g} & {r['val_loss']:.4f}" + r' \\' for r in pilot['candidates']]
    write_tex_table(tables / 'pilot.tex', 'lcc', '方法 & 学习率 & 300 步验证交叉熵', pilot_rows)
    stochastic_rows = []
    for sigma in [0.01, 0.1]:
        for batch in [1, 16]:
            records = stochastic_groups[(sigma, batch, 16384)]
            stochastic_rows.append(f'{sigma:g} & {batch} & ' + pm([r['avg_grad_nuclear'] for r in records])
                + ' & ' + pm([r['theorem_bound'] for r in records]) + ' & ' + pm([r['avg_tracking_error'] for r in records]) + r' \\')
    write_tex_table(tables / 'stochastic.tex', 'ccccc', r'$\sigma$ & $B$ & 平均梯度核范数 & 有限步界 & 平均跟踪误差', stochastic_rows)
    orth_rows = []
    for spectrum, name in [('flat', '平坦谱'), ('ill_conditioned', '病态谱'), ('rank_deficient', '秩亏谱')]:
        records = [r for r in orth if r['spectrum'] == spectrum and int(r['steps']) == 5]
        orth_rows.append(name + ' & ' + pm([float(r['relative_error']) for r in records])
                        + ' & ' + pm([float(r['cosine']) for r in records]) + r' \\')
    write_tex_table(tables / 'orthogonalization.tex', 'lcc', '输入谱 & 相对 Frobenius 误差 & 方向余弦', orth_rows)

    exact_runs = [r for r in det if r['config']['method'].startswith('muon')]
    ratios = [r['avg_grad_nuclear'] / r['theorem_bound'] for r in exact_runs + stochastic]
    empirical_over_bound = max(ratios)
    if empirical_over_bound > 1 + 1e-10:
        raise RuntimeError('An empirical average exceeded the bound; investigate before reporting')
    digest = hashlib.sha256()
    for path in sorted(result.rglob('*')):
        if path.suffix in {'.json', '.csv'}:
            digest.update(str(path.relative_to(result)).replace('\\', '/').encode())
            digest.update(path.read_bytes())
    metadata = json.loads((ROOT / 'data' / 'manifest.json').read_text(encoding='utf-8'))
    audit = dict(deterministic_runs=len(det), stochastic_runs=len(stochastic), formal_lm_runs=len(formal),
                 pilot_lm_runs=len(pilot['candidates']), max_empirical_over_theorem_bound=empirical_over_bound,
                 max_one_step_descent_violation=max(r['max_descent_violation'] for r in exact_runs),
                 results_sha256=digest.hexdigest(), language_summary=aggregate_rows,
                 dataset=metadata)
    write_json(out / 'audit.json', audit)
    macros = {
        'ParameterCount': str(formal[0]['parameter_count']), 'VocabSize': str(metadata['vocabulary_size']),
        'TrainChars': str(metadata['split_sizes']['train']), 'ValChars': str(metadata['split_sizes']['val']),
        'TestChars': str(metadata['split_sizes']['test']), 'DataSHA': metadata['sha256'],
        'ResultDigest': digest.hexdigest()[:16], 'MaxBoundRatio': f'{empirical_over_bound:.6f}',
        'MaxDescentViolation': f"{audit['max_one_step_descent_violation']:.2e}",
        'AdamLR': f"{pilot['selected_lrs']['adamw']:g}", 'MuonSVDLR': f"{pilot['selected_lrs']['muon_svd']:g}",
        'MuonNSLR': f"{pilot['selected_lrs']['muon_ns']:g}",
    }
    for row in aggregate_rows:
        prefix = {'adamw': 'Adam', 'muon_svd': 'SVD', 'muon_ns': 'NS'}[row['method']]
        for key, suffix in [('test_cross_entropy_mean', 'CE'), ('test_perplexity_mean', 'PPL'), ('train_seconds_mean', 'Time')]:
            macros[prefix + suffix] = f'{row[key]:.3f}'
    macro_lines = ['\\newcommand{\\' + k + '}{' + v + '}' for k, v in macros.items()]
    (tables / 'macros.tex').write_text('\n'.join(macro_lines) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k not in ['dataset', 'language_summary']}, indent=2))
    print(json.dumps(aggregate_rows, indent=2))


if __name__ == '__main__':
    main()
