"""Standalone research figures from actual pooled evaluation CSVs."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'research/round4070_learning_traversability'
with (R/'MODEL_MEAN_STD.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
methods=['StatsMLP','HistoryMLP','GRU','GRUShuffled','TCN','ResidualGRU','M2']
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for ax,mode,title in zip(axes,['forced','selective'],['Forced decision: coverage = 1','Selective: thresholds 0.2 / 0.8']):
    rr=[next(r for r in rows if r['method']==m and r['mode']==mode) for m in methods]
    x=np.arange(len(rr)); means=[float(r['false_pass_n_mean']) for r in rr];err=[float(r['false_pass_n_std']) for r in rr]
    ax.bar(x,means,yerr=err,color=['#879bb3']*5+['#416a94','#bd7b36'],capsize=3)
    ax.set_xticks(x,methods,rotation=35,ha='right');ax.set_ylabel('False-pass count / 26 BLOCKED');ax.set_title(title)
    ax.set_ylim(0,10);ax.axhline(1,color='#a95555',linestyle='--',linewidth=1,label='E2 corrected: 1 FP, 0.25 coverage')
    if mode=='selective':
        for i,r in enumerate(rr): ax.text(i,9.0,f"{float(r['coverage_mean']):.2f}",ha='center',fontsize=8)
        ax.text(0,9.5,'Coverage:',fontsize=8)
    ax.legend(fontsize=7,loc='upper right' if mode=='forced' else 'center right')
fig.suptitle('Round 4070-F exploratory holdouts | 2 PASSABLE + 26 BLOCKED | 3 initializations',fontsize=11)
fig.tight_layout();fig.savefig(R/'MODEL_COMPARISON.png',dpi=170);plt.close(fig)

curves=ROOT/'outputs/round4070_learning_traversability/phase_a_valid/training_curves.csv'
with curves.open(newline='',encoding='utf-8') as f: cs=list(csv.DictReader(f))
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
for ax,fold in zip(axs,['family_fold_0','family_fold_1']):
    for method in ['GRU','ResidualGRU','M2']:
        rr=[r for r in cs if r['fold']==fold and r['method']==method and r['seed']=='4070']
        ax.plot([int(r['epoch']) for r in rr],[float(r['train_loss']) for r in rr],label=method+' train')
        ax.plot([int(r['epoch']) for r in rr],[float(r['validation_loss']) for r in rr],ls='--',label=method+' val')
    ax.set_title(fold+' | validation: BLOCKED only');ax.set_xlabel('Epoch');ax.set_ylabel('BCE');ax.set_yscale('log');ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(R/'TRAINING_CURVES.png',dpi=150)
plt.close(fig)
if (R/'PHASE_C_MODEL_MEAN_STD.csv').exists():
    with (R/'PHASE_C_MODEL_MEAN_STD.csv').open(newline='',encoding='utf-8') as f: phase_c=list(csv.DictReader(f))
    with (R/'STRICT_INSTANT_MODEL_MEAN_STD.csv').open(newline='',encoding='utf-8') as f: instant=list(csv.DictReader(f))
    compare=['GRU','TCN','ResidualGRU','M2','GRUShuffled','Instantaneous']
    fig,ax=plt.subplots(figsize=(10,4.8));x=np.arange(len(compare))
    for offset,data,n,color,label in [(-.18,rows+instant,26,'#527da8','Phase A: 2 PASSABLE / 26 BLOCKED'),(.18,phase_c,29,'#bf8237','Phase C: 7 PASSABLE / 29 BLOCKED')]:
        rr=[next(r for r in data if r['method']==m and r['mode']=='forced') for m in compare]
        means=[float(r['false_pass_n_mean'])/n for r in rr];errs=[float(r['false_pass_n_std'])/n for r in rr]
        ax.bar(x+offset,means,width=.34,yerr=errs,color=color,label=label,capsize=3)
    ax.set_xticks(x,compare,rotation=20,ha='right');ax.set_ylabel('False-pass / BLOCKED');ax.set_ylim(0,.25)
    ax.set_title('M2 advantage did not persist after adding physical data\nForced decisions, coverage = 1; error bars = initialization SD')
    ax.legend(fontsize=9)
    fig.text(.02,.02,'Exploratory, overlapping samples; 4 positive physical configurations, all positive episodes use the right hand.',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(R/'PHASE_C_COMPARISON.png',dpi=170);plt.close(fig)
print('Saved actual-data figures')
