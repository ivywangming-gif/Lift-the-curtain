"""Pool disjoint configuration holdouts by seed, not unweighted fold accuracies."""
import argparse, csv, json, statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'research/round4070_learning_traversability'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
def main():
    a=argparse.ArgumentParser();a.add_argument('--run',default='phase_a_valid');a.add_argument('--prefix',default='');args=a.parse_args()
    run=ROOT/'outputs/round4070_learning_traversability'/args.run
    raw=read(run/'metrics.csv'); groups=defaultdict(list)
    for r in raw:
        if r['fold'] in ['family_fold_0','family_fold_1']:groups[r['method'],r['mode'],r['seed']].append(r)
    pooled=[]
    for (method,mode,seed),rr in groups.items():
        def total(k):return sum(int(r[k]) for r in rr)
        n,P,N=total('n'),total('passable_n'),total('blocked_n')
        tp,tn,fp,fn,u=[total(k) for k in ['true_pass_n','true_block_n','false_pass_n','false_reject_n','uncertain_n']]
        pooled.append(dict(method=method,mode=mode,seed=seed,n=n,passable_n=P,blocked_n=N,
            true_pass_n=tp,true_block_n=tn,false_pass_n=fp,false_reject_n=fn,uncertain_n=u,
            false_pass=fp/N if N else None,false_reject=fn/P if P else None,
            passable_recall=tp/P if P else None,blocked_recall=tn/N if N else None,
            balanced_accuracy=(tp/P+tn/N)/2 if P and N else None,coverage=1-u/n,uncertain_rate=u/n,
            brier=sum(float(r['brier'])*int(r['n']) for r in rr)/n,
            parameters_min=min(int(r['params']) for r in rr),parameters_max=max(int(r['params']) for r in rr),
            train_n_min=min(int(r['train_n']) for r in rr),train_n_max=max(int(r['train_n']) for r in rr),
            model_forward_ms_mean=statistics.mean(float(r['inference_ms']) for r in rr),
            train_seconds=sum(float(r['train_seconds']) for r in rr)))
    base={'Logistic','GradientBoosting','StatsMLP','HistoryMLP','GRU','TCN','ResidualGRU','M2'}
    write(REPORT/(args.prefix+'LEARNING_MODEL_COMPARISON.csv'),[r for r in pooled if r['method'] in base])
    write(REPORT/(args.prefix+'ABLATION_RESULTS.csv'),[r for r in pooled if r['method'] not in base])
    write(REPORT/(args.prefix+'SIDE_EVALUATION.csv'),[r for r in raw if r['fold'] not in ['family_fold_0','family_fold_1']])
    stats=[]; gg=defaultdict(list)
    for r in pooled:gg[r['method'],r['mode']].append(r)
    for (m,mode),rr in gg.items():
        row=dict(method=m,mode=mode,seeds=len(rr))
        for k in ['false_pass','false_pass_n','false_reject','passable_recall','blocked_recall','coverage','balanced_accuracy','brier','model_forward_ms_mean']:
            values=[r[k] for r in rr];row[k+'_mean']=statistics.mean(values);row[k+'_std']=statistics.stdev(values) if len(values)>1 else 0
        stats.append(row)
    write(REPORT/(args.prefix+'MODEL_MEAN_STD.csv'),stats)
    print(json.dumps([r for r in stats if r['mode']=='forced'],indent=2))
if __name__=='__main__':main()
