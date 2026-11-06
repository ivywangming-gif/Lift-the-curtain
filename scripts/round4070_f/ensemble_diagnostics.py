"""Three-initialization disagreement; a diagnostic, not calibrated uncertainty."""
import csv, statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/round4070_learning_traversability'
REPORT=ROOT/'research/round4070_learning_traversability'
rows=[]
for stage in ['phase_a_valid','phase_c','phase_c_final']:
    path=OUT/stage/'predictions.csv'
    if not path.exists():continue
    groups=defaultdict(list)
    with path.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['fold'] in ['family_fold_0','family_fold_1']:groups[r['method'],r['trial_id']].append(r)
    for (method,trial),rr in groups.items():
        assert len(rr)==len({r['seed'] for r in rr})==3
        assert len({r['y'] for r in rr})==1
        pp=[float(r['p']) for r in rr];mean=statistics.mean(pp)
        decisions=['PASSABLE' if p>=.8 else 'BLOCKED' if p<.2 else 'UNCERTAIN' for p in pp]
        rows.append(dict(stage=stage,method=method,trial_id=trial,y_evaluation_only=rr[0]['y'],
            p_mean=mean,p_std=statistics.stdev(pp),p_min=min(pp),p_max=max(pp),
            distinct_seed_decisions=len(set(decisions)),
            fixed_threshold_ensemble_decision='PASSABLE' if mean>=.8 else 'BLOCKED' if mean<.2 else 'UNCERTAIN',
            interpretation='initialization disagreement only; no risk guarantee'))
with (REPORT/'ENSEMBLE_UNCERTAINTY.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
print(f'{len(rows)} per-episode diagnostic rows; no fitting or threshold selection')
