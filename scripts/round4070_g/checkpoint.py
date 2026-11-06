"""Materialize an honest, read-only campaign checkpoint from actual artifacts."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import re
import psutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/round4070_comprehensive_traversability'
DOC=ROOT/'research/round4070_comprehensive_traversability'


def read(path, default=None):
    try: return json.loads(path.read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError): return default


def main():
    candidates=list(OUT.glob('campaign_manifest_v[0-9]*.json'))
    candidates=[p for p in candidates if re.fullmatch(r'campaign_manifest_v\d+\.json',p.name)]
    source=max(candidates,key=lambda p:int(re.search(r'_v(\d+)',p.name)[1]))
    manifest=read(source)
    actual_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    assert actual_hash==read(source.with_suffix('.sha256.json'))['sha256']
    gates=[]
    for path in (OUT/'physics').glob('*/*/summary.json'):
        d=read(path,{})
        diagnosis=read(path.parent/'apparatus_diagnosis.json',{})
        gates.append(dict(path=str(path.relative_to(ROOT)),physical_config_id=d.get('physical_config_id'),
            scene_id=d.get('scene_id'),seed=d.get('seed'),physics_valid=d.get('physics_valid',False),
            failed_checks=[k for k,v in d.get('checks',{}).items() if not v],
            failure_reason=d.get('failure_reason'),apparatus_diagnosis=diagnosis))
    valid_by_config={}
    for gate in gates:
        if gate['physics_valid']:
            valid_by_config.setdefault(gate['physical_config_id'],set()).add(gate['seed'])
    ledger=[];outcomes=[]
    for plan in manifest['trial_plans']:
        folder=OUT/'trials'/plan['trial_id']
        attempts=list(folder.glob('*/campaign_launch.json'))
        results=[read(p,{}) for p in folder.glob('*/g_outcome.json')]
        best=next((d for d in results if d.get('runtime_valid') and d.get('probe_valid')),
                  results[-1] if results else {})
        status='PLANNED'
        if len(valid_by_config.get(plan['physical_config_id'],set()))>=3: status='PHYSICS_VALID'
        if attempts: status='G1_LAUNCHED'
        if best.get('runtime_valid'): status='RUNTIME_VALID'
        if best.get('probe_valid'): status='PROBE_VALID'
        if best.get('label_valid'): status='LABEL_VALID'
        if best: outcomes.append(best)
        ledger.append(dict(trial_id=plan['trial_id'],scene_id=plan['scene_id'],
            physical_config_id=plan['physical_config_id'],seed=plan['seed'],probe_side=plan['probe_side'],
            experiment_role=plan['experiment_role'],attempts_launched=len(attempts),stage=status,
            runtime_valid=bool(best.get('runtime_valid')),probe_valid=bool(best.get('probe_valid')),
            label_valid=bool(best.get('label_valid')),label=best.get('label',''),
            reason=best.get('label_reason',best.get('status',''))))
    learning=[]
    for p in OUT.glob('learning*/task_queue.json'):
        queue=read(p,{})
        protocol=read(p.parent/'frozen_protocol.json',{})
        learning.append(dict(run=p.parent.name,scope=protocol.get('scope'),
            status=queue.get('status'),task_counts=dict(Counter(t.get('status') for t in queue.get('tasks',[])))))
    dataset=read(OUT/'dataset/manifest.json',{})
    episodes=dataset.get('episodes',[])
    active={}
    for name in ['workflow_lock.json','campaign_lock.json','resource_lock.json']:
        lock=read(OUT/name,{})
        if lock: active[name]={**lock,'pid_currently_exists':psutil.pid_exists(int(lock['pid']))}
    state=dict(status='IN_PROGRESS_NOT_COMPLETE',checked_utc=datetime.now(timezone.utc).isoformat(),
        frozen_manifest=str(source.relative_to(ROOT)),manifest_sha256=actual_hash,
        planned_g1_trials=len(ledger),max_g1_launches=manifest['max_g1_launches'],
        physics_attempts_launched=sum(1 for _ in (OUT/'physics').glob('*/*/campaign_launch.json')),
        physics_summaries=len(gates),physics_gates_passed=sum(g['physics_valid'] for g in gates),
        physical_configs_with_three_valid_seeds=sum(len(s)>=3 for s in valid_by_config.values()),
        g1_attempts_launched=sum(r['attempts_launched'] for r in ledger),
        g1_finalized_trial_ids=len(outcomes),g1_runtime_valid=sum(r['runtime_valid'] for r in ledger),
        g1_probe_valid=sum(r['probe_valid'] for r in ledger),g1_label_valid=sum(r['label_valid'] for r in ledger),
        g1_labels=dict(Counter(r['label'] for r in ledger if r['label'])),
        imported_history_count=len(episodes),
        imported_core_labels=dict(Counter(e['label'] for e in episodes if not e.get('is_control'))),
        learning_runs=learning,active_locks=active,
        completion_warning='No full benchmark or method advantage is claimed. Planned work is not executed evidence.',
        resume_command="powershell -NoProfile -ExecutionPolicy Bypass -File .\\run_round4070_g.ps1 -Mode all -RetryRuntimeFailures")
    DOC.mkdir(parents=True,exist_ok=True)
    (DOC/'PROGRESS_CHECKPOINT.json').write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (DOC/'TRIAL_PROGRESS_LEDGER.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
    (OUT/'physics_checkpoint_evidence.json').write_text(json.dumps(gates,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in state.items() if k not in ['active_locks','learning_runs']},indent=2,ensure_ascii=False))


if __name__=='__main__': main()
