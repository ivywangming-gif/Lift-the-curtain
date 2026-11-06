"""Frozen E0/E1/E2 adaptations to G's legal forward-body channels.

Only training empty controls supply reference responses. True timestamps supply
stall duration. Thresholds were frozen historically; no G test labels are read.
The original online 200 Hz rule is a separate audit, never a CSV replacement.
"""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/round4070_fixed_probe_benchmark'))
from evaluate_benchmark import e0_decision, e2_decision
from lift_curtain.control.deployment_observation import ProprioceptiveThresholds
import json


def evaluate_rules(episodes,raw,times,names,train_indices):
    """Return (list[{'E0','E1','E2','features'}], auditable train-only references)."""
    index={n:i for i,n in enumerate(names)}
    q=[i for i,n in enumerate(names) if n.startswith('q/')]
    cmd=[index['q_cmd/'+names[i][2:]] for i in q]
    effort=[i for i,n in enumerate(names) if n.startswith('effort/')]
    if len(q)!=29 or len(cmd)!=29 or len(effort)!=29:raise ValueError('Official29 input required')
    cfg=ProprioceptiveThresholds(**json.loads((ROOT/'configs/proprioceptive_thresholds.json').read_text()))
    references={};audit={}
    for side in ('left','right'):
        ids=[i for i in train_indices if episodes[i].get('is_control') and episodes[i]['y']==1 and episodes[i]['side']==side]
        if not ids:
            references[side]=None;audit[side]={'status':'NO_TRAIN_SAME_HAND_EMPTY_REFERENCE','train_ids':[]};continue
        progress=index[side+'_hand_delta_initial_b/x']
        references[side]=dict(effort=np.mean(np.concatenate([raw[i][:,effort] for i in ids]),axis=0),
            progress=float(np.mean([raw[i][-1,progress]-raw[i][0,progress] for i in ids])),
            tracking=float(np.mean([np.sqrt(np.mean((raw[i][:,cmd]-raw[i][:,q])**2)) for i in ids])))
        audit[side]=dict(status='TRAIN_ONLY_REFERENCE',train_ids=[episodes[i]['trial_id'] for i in ids],
                        progress_m=references[side]['progress'],tracking_rms_rad=references[side]['tracking'])
    results=[]
    for e,x,t in zip(episodes,raw,times):
        side=e['side'];ref=references[side];j=index[side+'_hand_delta_initial_b/x'];v=index[side+'_hand_velocity_initial_b/x']
        p=float(x[-1,j]-x[0,j]);r=None if ref is None or abs(ref['progress'])<=1e-6 else p/ref['progress']
        baseline=x[0,effort] if ref is None else ref['effort']
        features=dict(sample_count=len(t),hand_forward_progress_m=p,hand_progress_ratio=r,
            effort_deviation=float(np.mean(np.linalg.norm(x[:,effort]-baseline,axis=1))),
            upper_tracking_rms=float(np.sqrt(np.mean((x[:,cmd]-x[:,q])**2))),
            hand_low_speed_duration_s=float(np.sum(np.diff(t)*(np.abs(x[:-1,v])<=.004))),
            engineering_coordinate='initial robot body forward x; measured FK velocity at real control log timestamps',
            baseline_source='train same-hand empty only' if ref else 'no reference; E1 uses first effort sample and no progress ratio')
        d='UNCERTAIN'
        if len(t)>=cfg.minimum_samples:
            if p>=cfg.responsive_progress_m and (r is None or r>=cfg.responsive_progress_ratio_min):d='PASSABLE'
            elif (p<=cfg.blocked_progress_max_m or (r is not None and r<=cfg.blocked_progress_ratio_max)) and features['effort_deviation']>=cfg.blocked_effort_deviation_min and features['hand_low_speed_duration_s']>=cfg.blocked_stall_duration_s:d='BLOCKED'
        results.append(dict(E0=e0_decision(features)[0],E1=d,
            E2=e2_decision(features,{'upper_tracking_rms':None if ref is None else ref['tracking']})[0],features=features))
    return results,dict(references=audit,threshold_source='configs/proprioceptive_thresholds.json and frozen E RULES',
                         stall_integration='sum(diff(actual_time)*low_speed_at_left_endpoint); no terminal extrapolation',
                         learned_thresholds=False,source_online_rule_distinct=True)
