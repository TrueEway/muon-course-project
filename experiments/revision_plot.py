"""Independent four-figure renderer for the frozen v1.1 experiment protocol."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .common import ROOT,write_csv,write_json
from .summarize import read_csv,stats,pm,write_tex_table

REV = ROOT/'output'/'revision'
FIG = ROOT/'output'/'figures'
TABLE = ROOT/'output'/'tables'
COLORS = {'muon':'#b34e36','adamw':'#245fa3','sgd':'#777777'}
LABELS = {'muon':'Muon hybrid (K=5)','adamw':'AdamW','sgd':'SGD + momentum'}


def jsons(folder):
    return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('*.json'))]


def save(fig,name):
    fig.tight_layout(pad=.8)
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)


def vision_stem(r):
    c = r['config']
    return f"{c['method']}_K{c['k']}_s{c['seed']}_lr{c['lr']:g}"


def paired_speedups(records,baseline):
    ratios = []
    for r in records:
        b = next(x for x in baseline if x['config']['seed'] == r['config']['seed'])
        if b['threshold_steps'] is not None and r['threshold_steps'] is not None:
            ratios.append(b['threshold_steps']/r['threshold_steps'])
    return ratios


def compact_pm(values,digits):
    mean,std = stats(values)
    return '$'+f'{mean:.{digits}f}'+r'\pm'+f'{std:.{digits}f}'+'$'


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    formal = jsons(REV/'vision'/'formal')
    factor = jsons(REV/'factorization'/'formal')
    if len(formal) != 21 or len(factor) != 9:
        raise RuntimeError(f'Full protocol required: 21 ResNet and 9 factorization runs; found {len(formal)} and {len(factor)}')
    expected = {(s,m,k) for s in [0,1,2] for m,k in [('sgd',5),('adamw',5)]+[('muon',k) for k in [0,1,3,5,7]]}
    assert {(r['config']['seed'],r['config']['method'],r['config']['k']) for r in formal} == expected
    FIG.mkdir(parents=True,exist_ok=True)
    TABLE.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,
                         'lines.linewidth':1.7,'mathtext.fontset':'stix'})
    primary = {method:[r for r in formal if r['config']['method']==method and r['config']['k']==5] for method in LABELS}
    # Figure 1: actual nonconvex factorization, not a quadratic renamed as such.
    fig,axes = plt.subplots(1,2,figsize=(7.2,2.65))
    for method in ['sgd','adamw','muon']:
        records = [r for r in factor if r['config']['method']==method]
        curves = []
        for r in records:
            c = r['config']
            path = REV/'factorization'/'formal'/f"{method}_s{c['seed']}_lr{c['lr']:g}_T3000.csv"
            curves.append(read_csv(path))
        for ax,key in zip(axes,['relative_reconstruction','gradient_nuclear']):
            values = np.array([[float(row[key]) for row in curve] for curve in curves])
            mean,std = values.mean(0),values.std(0,ddof=1)
            steps = np.arange(1,len(mean)+1)
            ax.semilogy(steps,np.maximum(mean,1e-15),color=COLORS[method],label=LABELS[method].replace('hybrid (K=5)','(NS5)'))
            ax.fill_between(steps,np.maximum(mean-std,1e-15),mean+std,color=COLORS[method],alpha=.12)
            ax.set_xlabel('Iteration')
    axes[0].set_ylabel('Relative reconstruction error')
    axes[1].set_ylabel('Gradient nuclear norm')
    axes[0].legend(fontsize=7)
    save(fig,'fig1')

    # Figure 2: test curves are evaluated after frozen training is complete.
    fig,axes = plt.subplots(1,3,figsize=(7.4,2.65))
    for method,records in primary.items():
        batch_curves,epoch_curves = [],[]
        for r in records:
            stem = vision_stem(r)
            batch_curves.append(read_csv(REV/'vision'/'formal'/(stem+'.csv')))
            epoch_curves.append(read_csv(REV/'vision'/'formal'/(stem+'_epochs.csv')))
        losses = np.array([[float(x['loss']) for x in c] for c in batch_curves])
        smoothed = np.stack([np.convolve(x,np.ones(32)/32,mode='valid') for x in losses])
        mean,std = smoothed.mean(0),smoothed.std(0,ddof=1)
        x = np.arange(32,32+len(mean))
        axes[0].plot(x,mean,color=COLORS[method],label=LABELS[method])
        axes[0].fill_between(x,mean-std,mean+std,color=COLORS[method],alpha=.1)
        acc = np.array([[float(x['test_accuracy']) for x in c] for c in epoch_curves])
        epochs = np.arange(1,4)
        axes[1].errorbar(epochs,acc.mean(0),yerr=acc.std(0,ddof=1),marker='o',color=COLORS[method])
        clock = np.array([[float(x['wall_seconds']) for x in c] for c in epoch_curves])
        axes[2].plot(clock.mean(0),acc.mean(0),marker='o',color=COLORS[method])
    axes[0].set(xlabel='Training step',ylabel='Train loss (32-step mean)')
    axes[1].set(xlabel='Epoch',ylabel='Post-training test accuracy (%)',xticks=[1,2,3])
    axes[2].set(xlabel='Training wall time (s)',ylabel='Test accuracy (%)')
    axes[0].legend(fontsize=6.4)
    speed_text = []
    for method in ['sgd','adamw','muon']:
        pairs = paired_speedups(primary[method],primary['sgd'])
        text = f'{np.mean(pairs):.2f}x' if len(pairs)==3 else 'NR'
        speed_text.append(text)
    axes[1].text(.02,.98,'40% validation target\nSGD / AdamW / Muon step ratios\n'+' / '.join(speed_text),
                 transform=axes[1].transAxes,fontsize=7,va='top',
                 bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    save(fig,'fig2')

    # Figure 3: spectra of the actual representative weight delta, including decay.
    fig,axes = plt.subplots(1,2,figsize=(7.2,2.65))
    for method,records in primary.items():
        spectra,conditions = [],[]
        for r in records:
            stem = vision_stem(r)
            arrays = np.load(REV/'vision'/'formal'/(stem+'.npz'))
            s = arrays['update_singular_values'][-1]
            spectra.append(s/max(s[0],1e-15))
            rows = read_csv(REV/'vision'/'formal'/(stem+'_spectrum.csv'))
            conditions.append([float(x['update_condition']) if x['update_condition'] else np.nan for x in rows])
            epochs = [float(x['epoch']) for x in rows]
        axes[0].semilogy(np.arange(1,65),np.maximum(np.mean(spectra,0),1e-12),color=COLORS[method],label=LABELS[method])
        axes[1].semilogy(epochs,np.nanmean(conditions,0),'-o',color=COLORS[method])
    axes[0].set(xlabel='Singular-value index',ylabel=r'Update spectrum $s_i/s_1$')
    axes[1].set(xlabel='Epoch',ylabel='Effective update condition number')
    axes[0].legend(fontsize=7)
    save(fig,'fig3')

    # Figure 4: K does not get separately tuned; all non-K variables are held fixed.
    fig,ax = plt.subplots(figsize=(6.8,2.65))
    other = ax.twinx()
    ks = [0,1,3,5,7]
    time_stats,accuracy_stats,val_accuracy = [],[],[]
    for k in ks:
        records = [r for r in formal if r['config']['method']=='muon' and r['config']['k']==k]
        time_stats.append(stats([r['step_median_ms'] for r in records]))
        accuracy_stats.append(stats([r['test_accuracy'] for r in records]))
        val_accuracy.append(np.mean([r['final_val_accuracy'] for r in records]))
    times,tstd = np.array(time_stats).T
    accuracies,astd = np.array(accuracy_stats).T
    bars = ax.bar(np.arange(5),times,yerr=tstd,color='#999999',alpha=.6,width=.55,label='Median step time')
    accuracy_line = other.errorbar(np.arange(5),accuracies,yerr=astd,color=COLORS['muon'],marker='D',linewidth=1.8,label='Selected test accuracy')
    ax.set(xticks=np.arange(5),xticklabels=ks,xlabel='NS iteration count K',ylabel='Median step time (ms)')
    other.set_ylabel('Test accuracy (%)',color=COLORS['muon'])
    other.grid(False)
    other.spines['right'].set_visible(True)
    fig.legend([bars,accuracy_line],['Median step time','Selected test accuracy'],loc='lower center',
               bbox_to_anchor=(.5,1.0),ncol=2,frameon=False,fontsize=8)
    save(fig,'fig4')

    selection = json.loads((REV/'vision'/'selection.json').read_text(encoding='utf-8'))
    config_rows = [
        r'模型 / 参数量 & \multicolumn{3}{c}{完整 ResNet-18 / 11,181,642} \\',
        r'输入 / stem & \multicolumn{3}{c}{$32\times32$ / 原始 $7\times7$、步长 2，保留 max-pool} \\',
        r'训练 / 验证 / 测试 & \multicolumn{3}{c}{45,000 / 5,000 / 10,000} \\',
        r'批量 / epoch / 种子 & \multicolumn{3}{c}{128 / 3 / 0、1、2} \\',
        r'硬件 / 精度 & \multicolumn{3}{c}{单张 H20-3e，Xeon 6759P-C；float32、4 个 CPU 线程} \\',
        r'共享资源限制 & \multicolumn{3}{c}{GPU 分配器上限 4\%，协作占空目标 35\%，TF32 关闭} \\',
        '选定学习率 & '+ ' & '.join(f"{selection['chosen'][m]:g}" for m in ['sgd','adamw','muon'])+r' \\',
        r'动量 / 矩参数 & $0.9$ & $(0.9,0.95)$ & $0.95$；辅助 AdamW $(0.9,0.95)$ \\',
        r'矩阵权重衰减 & $5\times10^{-4}$ & $5\times10^{-4}$ & $5\times10^{-4}$ \\',
        r'调度 / 调参预算 & \multicolumn{3}{c}{50 步预热、余弦衰减；各 3 个 LR、种子 42、96 步} \\',
    ]
    write_tex_table(TABLE/'cifar_config.tex','lccc','配置项 & SGD+M & AdamW & Muon 混合',config_rows)
    performance_rows,aggregates = [],[]
    for method in ['sgd','adamw','muon']:
        records = primary[method]
        hits = [r['threshold_steps'] for r in records]
        hit_text = compact_pm(hits,0) if all(x is not None for x in hits) else f"未全达标（{sum(x is not None for x in hits)}/3）"
        pairs = paired_speedups(records,primary['sgd'])
        speed = compact_pm(pairs,1) if len(pairs)==3 else '---'
        performance_rows.append({'sgd':'SGD+M','adamw':'AdamW','muon':'Muon 混合'}[method]+' & '
            +compact_pm([r['final_train_loss'] for r in records],3)+' & '+compact_pm([r['test_accuracy'] for r in records],2)+' & '
            +hit_text+' & '+speed+' & '+compact_pm([r['step_median_ms'] for r in records],1)+' & '
            +compact_pm([r['peak_process_rss_mib'] for r in records],0)+' & '
            +compact_pm([r['peak_cuda_memory_mib'] for r in records],0)+r' \\')
        row = dict(method=method)
        for key in ['final_train_loss','test_accuracy','step_median_ms','peak_process_rss_mib','peak_cuda_memory_mib','peak_cuda_reserved_mib','throttle_sleep_seconds','wall_training_seconds']:
            row[key+'_mean'],row[key+'_std'] = stats([r[key] for r in records])
        row['hits'] = sum(x is not None for x in hits)
        aggregates.append(row)
    write_tex_table(TABLE/'cifar_performance.tex','lccccccc',
        r'方法 & 末轮 Loss & Acc / \% & 达标步数 & 步数比 & ms/step & RSS / MiB & 显存 / MiB',performance_rows)
    write_csv(TABLE/'cifar_summary.csv',aggregates)
    k_rows = []
    for k,(tm,ts),(am,ass) in zip(ks,time_stats,accuracy_stats):
        records = [r for r in formal if r['config']['method']=='muon' and r['config']['k']==k]
        k_rows.append(f'{k} & '+pm([r['final_val_accuracy'] for r in records])+' & '
            +pm([r['test_accuracy'] for r in records])+' & '+pm([r['step_median_ms'] for r in records])+r' \\')
    write_tex_table(TABLE/'k_ablation.tex','cccc',r'$K$ & 末轮验证 Acc / \% & 选定测试 Acc / \% & ms/step',k_rows)
    factor_rows = []
    for method in ['sgd','adamw','muon']:
        records = [r for r in factor if r['config']['method']==method]
        factor_rows.append(LABELS[method].replace('hybrid (K=5)','(NS5)')+' & '+pm([r['final_loss'] for r in records],True)
                           +' & '+pm([r['relative_reconstruction'] for r in records],True)+r' \\')
    write_tex_table(TABLE/'factorization_summary.tex','lcc','方法 & 末点目标值 & 相对重构误差',factor_rows)
    digest = hashlib.sha256()
    for path in sorted(REV.rglob('*')):
        if path.suffix in {'.json','.csv','.npz'}:
            digest.update(str(path.relative_to(REV)).replace('\\','/').encode())
            digest.update(path.read_bytes())
    audit = dict(edition='v1.1',vision_formal_runs=len(formal),vision_pilot_runs=len(jsons(REV/'vision'/'pilot')),
                 factorization_formal_runs=len(factor),factorization_pilot_runs=len(jsons(REV/'factorization'/'pilot')),
                 revision_results_sha256=digest.hexdigest(),primary_summary=aggregates,
                 best_final_validation_k=ks[int(np.argmax(val_accuracy))])
    write_json(ROOT/'output'/'revision_audit.json',audit)
    macros = {'RevisionDigest':digest.hexdigest()[:16],'CifarBestValK':str(audit['best_final_validation_k'])}
    for row in aggregates:
        prefix = {'sgd':'CifarSGD','adamw':'CifarAdam','muon':'CifarMuon'}[row['method']]
        macros[prefix+'Acc'] = f"{row['test_accuracy_mean']:.2f}"
        macros[prefix+'Step'] = f"{row['step_median_ms_mean']:.1f}"
        macros[prefix+'Wall'] = f"{row['wall_training_seconds_mean']:.1f}"
    pairs = paired_speedups(primary['muon'],primary['sgd'])
    macros['CifarMuonSpeed'] = f'{np.mean(pairs):.2f}' if len(pairs)==3 else '未全达标'
    (TABLE/'revision_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
