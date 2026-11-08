"""Materialize an honest, read-only campaign checkpoint from actual artifacts."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import psutil
from budget_ledger import launch_ledger_summary, trial_budget_state

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/round4070_comprehensive_traversability'
DOC=ROOT/'research/round4070_comprehensive_traversability'


def read(path, default=None):
    try: return json.loads(path.read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError): return default


def current_campaign():
    pointer=read(DOC/'CURRENT_CAMPAIGN.json',{})
    name=pointer.get('manifest','')
    if not name or Path(name).name!=name:
        raise RuntimeError('Missing or invalid current frozen campaign pointer')
    source=OUT/name
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=pointer.get('sha256') or digest!=read(source.with_suffix('.sha256.json'),{}).get('sha256'):
        raise RuntimeError('Current frozen campaign SHA256 mismatch')
    return source,digest


def main():
    source,actual_hash=current_campaign()
    manifest=read(source)
    e1_baseline=read(DOC/'IF_ELSE_BASELINE_STATUS.json',{})
    budget=launch_ledger_summary(manifest,repo_root=ROOT,ledger_path=DOC/'G1_LAUNCH_LEDGER.json')
    primary_ids={p['trial_id'] for p in manifest['trial_plans']}
    dataset=read(OUT/'dataset/manifest.json',{})
    episodes=[e for e in dataset.get('episodes',[]) if e.get('planned_trial_id',e.get('trial_id','')) in primary_ids]
    admitted={e.get('planned_trial_id',e['trial_id']):e for e in episodes}
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
        budget_state=trial_budget_state(plan['trial_id'],budget)
        results=[read(p,{}) for p in folder.glob('*/g_outcome.json')]
        best=next((d for d in results if d.get('runtime_valid') and d.get('probe_valid')),
                  results[-1] if results else {})
        formal=admitted.get(plan['trial_id'],{})
        status='PLANNED'
        if len(valid_by_config.get(plan['physical_config_id'],set()))>=3: status='PHYSICS_VALID'
        if budget_state['known_attempts']: status='G1_LAUNCHED'
        if best.get('runtime_valid'): status='RUNTIME_VALID'
        if best.get('probe_valid'): status='PROBE_VALID'
        if formal.get('label_valid'): status='LABEL_VALID'
        if budget_state['withhold']: status='BLOCKED_WITH_REASON'
        if best: outcomes.append(best)
        failed_gates=[g for g in gates if g['physical_config_id']==plan['physical_config_id'] and not g['physics_valid']]
        if len(valid_by_config.get(plan['physical_config_id'],set()))>=3:
            admission_reason='G1_ADMISSION_PAUSED' if (OUT/'PAUSE_G1.flag').exists() else 'AWAITING_G1_OR_INDEPENDENT_ADMISSION'
        elif failed_gates:
            admission_reason='PHYSICS_GATE_FAILED_OR_NOT_VERIFIED:'+','.join(sorted({c for g in failed_gates for c in g['failed_checks']}))
        else:
            admission_reason='THREE_DISTINCT_PHYSICS_SEEDS_PENDING'
        ledger.append(dict(trial_id=plan['trial_id'],scene_id=plan['scene_id'],
            physical_config_id=plan['physical_config_id'],seed=plan['seed'],probe_side=plan['probe_side'],
            experiment_role=plan['experiment_role'],attempts_launched=budget_state['known_attempts'],
            local_launch_files=len(attempts),missing_raw_budget_withhold=budget_state['withhold'],stage=status,
            runtime_valid=bool(best.get('runtime_valid')),probe_valid=bool(best.get('probe_valid')),
            label_valid=bool(formal.get('label_valid')),label=formal.get('label',''),
            online_outcome_label_diagnostic_only=best.get('label',''),
            reason=budget_state['reason'] if budget_state['withhold'] else formal.get('label_reason','PRIMARY_IMPORT_AND_BODY_AUDIT_PENDING' if best else admission_reason)))
    learning=[]
    for p in OUT.glob('learning*/task_queue.json'):
        queue=read(p,{})
        protocol=read(p.parent/'frozen_protocol.json',{})
        tasks=queue.get('tasks',[])
        learning.append(dict(run=p.parent.name,scope=protocol.get('scope'),
            status=queue.get('status'),
            task_counts=dict(Counter(t.get('status') for t in tasks if t.get('evaluation_scope')!='PARTIAL_HELDOUT_SINGLE_CLASS')),
            partial_single_class_task_counts=dict(Counter(t.get('status') for t in tasks if t.get('evaluation_scope')=='PARTIAL_HELDOUT_SINGLE_CLASS')),
            all_scopes_task_counts=dict(Counter(t.get('status') for t in tasks))))
    actual_launches=list((OUT/'trials').glob('*/*/campaign_launch.json'))
    local_launch_keys={p.relative_to(ROOT).as_posix() for p in actual_launches}
    known_launch_keys={r['launch_file'] for r in budget['entries']}
    pilot_launches=[p for p in actual_launches if p.parent.parent.name not in primary_ids]
    pilot_rows=[]
    for path in pilot_launches:
        outcome=read(path.parent/'g_outcome.json',{})
        pilot_rows.append(dict(trial_id=path.parent.parent.name,attempt=path.parent.name,
            protocol_status='SUPERSEDED_DIAGNOSTIC_ONLY_NOT_PRIMARY',
            original_outcome_label=outcome.get('label',''),runtime_valid=outcome.get('runtime_valid',False),
            probe_valid=outcome.get('probe_valid',False),raw_path=str(path.parent.relative_to(ROOT))))
    for key in sorted(known_launch_keys-local_launch_keys):
        path=Path(key)
        if path.parent.parent.name in primary_ids:continue
        pilot_rows.append(dict(trial_id=path.parent.parent.name,attempt=path.parent.name,
            protocol_status='HISTORICAL_LAUNCH_BUDGET_PRESERVED_RAW_NOT_PRESENT_LOCALLY',
            original_outcome_label='',runtime_valid='',probe_valid='',raw_path=str(path.parent)))
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
        g1_attempts_launched=len(known_launch_keys),g1_launch_files_locally_available=len(actual_launches),
        historical_launches_not_present_locally=len(budget['missing_raw_launches']),
        durable_launch_ledger_entries=budget['durable_entry_count'],
        budget_withheld_trial_ids=budget['withheld_trial_ids'],
        remaining_actual_g1_launch_budget=budget['remaining'],
        primary_g1_attempts_launched=sum(r['attempts_launched'] for r in ledger),
        superseded_pilot_attempts_launched=len(pilot_rows),
        primary_g1_finalized_trial_ids=len(outcomes),g1_runtime_valid=sum(r['runtime_valid'] for r in ledger),
        g1_probe_valid=sum(r['probe_valid'] for r in ledger),g1_label_valid=sum(r['label_valid'] for r in ledger),
        g1_labels=dict(Counter(r['label'] for r in ledger if r['label'])),
        imported_history_count=len(episodes),
        imported_core_labels=dict(Counter(e['label'] for e in episodes if not e.get('is_control'))),
        learning_runs=learning,active_locks=active,
        e1_fixed_baseline=e1_baseline if e1_baseline.get('schema') == 'round4070-g.if_else_baseline_status.v1' else None,
        g1_admission_paused=(OUT/'PAUSE_G1.flag').exists(),
        completion_warning='No full benchmark or method advantage is claimed. Planned work is not executed evidence.',
        resume_command="powershell -NoProfile -ExecutionPolicy Bypass -File .\\run_round4070_g.ps1 -Mode all -RetryRuntimeFailures")
    DOC.mkdir(parents=True,exist_ok=True)
    (DOC/'PROGRESS_CHECKPOINT.json').write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (DOC/'TRIAL_PROGRESS_LEDGER.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
    if pilot_rows:
        with (DOC/'PILOT_PROGRESS_LEDGER.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(pilot_rows[0]));writer.writeheader();writer.writerows(pilot_rows)
    (OUT/'physics_checkpoint_evidence.json').write_text(json.dumps(gates,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in state.items() if k not in ['active_locks','learning_runs']},indent=2,ensure_ascii=False))


if __name__=='__main__': main()
