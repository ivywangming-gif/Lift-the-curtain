"""Serial, durable, resumable G physics/G1 campaign with bounded retries."""
from pathlib import Path
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import shutil
import time
import psutil
from resource_lock import resource_lock
from physics_admission import trial_state, retry_authorized, empty_dependency, auditor_source_hashes
from budget_ledger import launch_ledger_summary,trial_budget_state,append_actual_launch
from g1_quarantine import quarantine_state

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/round4070_comprehensive_traversability'
GUARD=ROOT/'scripts/round4070_fixed_probe_benchmark/run_guarded_with_resources.ps1'
FROZEN_FALCON=Path('C:/Robotics/FALCON-upstream/sim2real/models/falcon/g1_29dof.onnx')
FROZEN_FALCON_SHA256='8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0'


def read(path,default=None):
    try:return json.loads(path.read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return default


def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8');os.replace(temp,path)


def event(kind,**fields):
    OUT.mkdir(parents=True,exist_ok=True)
    row=dict(time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),event=kind,pid=os.getpid(),**fields)
    with (OUT/'campaign_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(row),flush=True)


def training_active():
    w=read(OUT/'training_window.json',{})
    try:return w.get('allow_training') is True and dt.datetime.fromisoformat(w['expires_utc'].replace('Z','+00:00'))>dt.datetime.now(dt.timezone.utc)
    except (KeyError,ValueError,TypeError):return False


def resource_ready():
    available=psutil.virtual_memory().available/1024**2
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    free=float(raw.strip().splitlines()[0]);disk_free=shutil.disk_usage(OUT).free
    return available>=2048 and free>=4096 and disk_free>=20*1024**3,dict(available_ram_mb=available,free_vram_mb=free,
        output_disk_free_bytes=disk_free,output_disk_free_gib=disk_free/1024**3,minimum_output_disk_free_gib=20)


def g1_launch_count(manifest):
    # This tracked ledger survives a fresh checkout even when newer raw does
    # not. Missing known raw never refunds budget or permits the same replay.
    return launch_ledger_summary(manifest)['count']


def g1_budget_admission(manifest,trial_id,run):
    summary=launch_ledger_summary(manifest)
    state=trial_budget_state(trial_id,summary)
    if summary['budget_exhausted']:
        return dict(ready=False,reason='G1_BUDGET_EXHAUSTED',launches=summary['count'])
    if state['withhold']:
        return dict(ready=False,reason=state['reason'],budget_state=state)
    requested=int(Path(run).name.removeprefix('attempt'))
    if requested<state['minimum_next_attempt_number']:
        return dict(ready=False,reason='KNOWN_ATTEMPT_PATH_REUSE_FORBIDDEN',budget_state=state)
    return dict(ready=True,budget_state=state,launches=summary['count'])


def archive_source_files(paths,run):
    """Preserve original source bytes before guard/App startup can fail."""
    run=Path(run).resolve();archive={};digests={}
    for source in dict.fromkeys(Path(p).resolve() for p in paths):
        if not source.is_relative_to(ROOT):raise RuntimeError('Launch source outside repository: '+str(source))
        relative=source.relative_to(ROOT);content=source.read_bytes();digest=hashlib.sha256(content).hexdigest()
        archive_relative=Path('generated')/source.relative_to(run) if source.is_relative_to(run) else relative
        target=run/'launch_sources'/archive_relative
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as output:output.write(content)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise RuntimeError('Launch source archive byte mismatch')
        digests[str(relative)]=digest
        archive[relative.as_posix()]=dict(path=target.relative_to(run).as_posix(),sha256=digest,size_bytes=len(content))
    return digests,archive


def launch_source_paths(script,invocation):
    folder=ROOT/'scripts/round4070_g'
    required=['physics_campaign.py','physics_scene_builder.py','physics_scenes.py','physics_admission.py','resource_lock.py',
        'physics_native_bind.py','physics_history.py','physics_g1_collision_audit.py','collision_geometry.py','collision_clearance.py','physics_wall_evidence.py','budget_ledger.py']
    # The plan contract helper lives in g1_trial.py. Preserve every G1 module
    # so later pre-App failures cannot omit a newly factored G1 helper.
    return [script,GUARD,ROOT/'scripts/round4070_fixed_probe_benchmark/run_episode.py',invocation,
            *[folder/name for name in required],*sorted(folder.glob('g1_*.py'))]


def launch(script,run,args,kind,trial_id,*,manifest=None):
    while True:
        if (OUT/'PAUSE.flag').exists() or training_active():event('WAIT_TRAINING_OR_PAUSE',trial_id=trial_id);time.sleep(15);continue
        try:
            with resource_lock('physics',trial_id):
                if training_active() or (OUT/'PAUSE.flag').exists():continue
                if kind=='G1' and (OUT/'PAUSE_G1.flag').exists():
                    event('G1_ADMISSION_WITHHELD_BEFORE_LAUNCH',trial_id=trial_id,reason='PAUSE_G1.flag',g1_budget_consumed=0)
                    return None
                if kind=='G1':
                    if manifest is None:raise RuntimeError('FROZEN_MANIFEST_REQUIRED_FOR_G1_BUDGET')
                    budget=g1_budget_admission(manifest,trial_id,run)
                    if not budget['ready']:
                        event('G1_DURABLE_BUDGET_WITHHELD_BEFORE_LAUNCH',trial_id=trial_id,budget=budget,g1_budget_consumed=0)
                        return None
                ready,resources=resource_ready()
                if not ready:event('WAIT_RESOURCE',trial_id=trial_id,**resources)
                else:
                    if kind=='G1':
                        policy_sha=hashlib.sha256(FROZEN_FALCON.read_bytes()).hexdigest()
                        if policy_sha!=FROZEN_FALCON_SHA256:raise RuntimeError('FROZEN_FALCON_SHA256_MISMATCH; no G1 launch permitted')
                        resources['frozen_falcon_sha256']=policy_sha
                    run.mkdir(parents=True,exist_ok=False)
                    invocation=run/'launch_guard.ps1'
                    def quote(value):return "'"+str(value).replace("'","''")+"'"
                    cmd='& '+quote(GUARD)+' -ScriptPath '+quote(script)+' -RunRoot '+quote(run)+' -OverallTimeoutSeconds 1200 -ScriptArgs @('+','.join(quote(v) for v in args)+')\n'
                    invocation.write_text(cmd,encoding='utf-8')
                    source_hashes,source_archive=archive_source_files(launch_source_paths(script,invocation),run)
                    atomic(run/'campaign_launch.json',dict(kind=kind,trial_id=trial_id,queue_pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),guard=str(GUARD),script=str(script),args=args,resources=resources,
                        source_sha256=source_hashes,launch_source_archive=source_archive,source_archive_complete_before_guard=True))
                    if kind=='G1':
                        # Failure here aborts before guard/App startup. The
                        # immutable local launch intent still costs budget.
                        ledger=append_actual_launch(run/'campaign_launch.json',manifest=manifest)
                        event('G1_LAUNCH_DURABLY_RECORDED',trial_id=trial_id,g1_launches=ledger['count'])
                    event(kind+'_LAUNCHED',trial_id=trial_id,run_root=str(run),**resources)
                    with (run/'guard_console.log').open('w',encoding='utf-8') as console:
                        result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(invocation)],cwd=ROOT,stdout=console,stderr=subprocess.STDOUT)
                    watchdog=read(run/'watchdog_summary.json',{})
                    event(kind+'_GUARD_EXIT',trial_id=trial_id,returncode=result.returncode,watchdog_summary=str(run/'watchdog_summary.json'),wall_time_s=watchdog.get('wall_time_s'),timeout_reason=watchdog.get('timeout_reason'),owned_pid=watchdog.get('pid_started_by_round4070_problem_discovery'))
                    return result.returncode
        except InterruptedError:event('WAIT_RESOURCE_LOCK',trial_id=trial_id)
        time.sleep(15)


def summarize(manifest):
    formal=read(OUT/'dataset/manifest.json',{}).get('episodes',[])
    formal_by_plan={e.get('planned_trial_id'):e for e in formal if e.get('primary_admission')}
    states=[]
    for sid,spec in manifest['scene_specs'].items():
        rep=next((r for r in manifest['representative_scene_ids'] if manifest['scene_specs'][r]['physical_config_id']==spec['physical_config_id']),sid)
        summary_paths=sorted((OUT/'physics'/rep).glob('*/summary.json'))
        summaries=[read(p,{}) for p in summary_paths]
        valid_seeds={s.get('seed') for s in summaries if s.get('physics_valid')}
        attempts=sorted((OUT/'physics'/rep).glob('seed*_attempt*'))
        evidence=[]
        for path in attempts:
            value=read(path/'summary.json',{});diagnosis=read(path/'apparatus_diagnosis.json',{});watchdog=read(path/'watchdog_summary.json',{})
            failed_checks=[k for k,v in value.get('checks',{}).items() if v is False]
            reason=value.get('failure_reason') or ('INTERACTION_EVIDENCE_NOT_DEMONSTRATED_NOT_MATERIAL_INSTABILITY' if failed_checks and set(failed_checks)<={'contact_demonstrated','release_clear'} else 'FAILED_NUMERICAL_OR_GEOMETRIC_GATE' if failed_checks else watchdog.get('timeout_reason'))
            if value.get('physics_valid') is True:reason=None
            evidence.append(dict(seed=value.get('seed'),physics_valid=value.get('physics_valid'),failed_checks=failed_checks,
                reason=reason,cleanup_or_watchdog_reason=watchdog.get('timeout_reason'),execution_failure=value.get('failure_type'),apparatus_invalid=diagnosis.get('apparatus_invalid',False),
                resolved_by_valid_same_seed=value.get('seed') in valid_seeds,
                summary_path=(path/'summary.json').relative_to(ROOT).as_posix() if value else None,
                attempt_path=path.relative_to(ROOT).as_posix()))
        unresolved=[e for e in evidence if not e['resolved_by_valid_same_seed']]
        if len(valid_seeds)>=3:physics_status='PHYSICS_VALID'
        elif not attempts:physics_status='NOT_EXECUTED'
        elif any(e['failed_checks'] and not e['apparatus_invalid'] for e in unresolved):physics_status='PHYSICS_GATE_FAILED'
        elif any(e['apparatus_invalid'] for e in unresolved):physics_status='APPARATUS_REPAIR_REQUIRED'
        elif any(e['execution_failure'] for e in unresolved):physics_status='PHYSICS_EXECUTION_FAILED'
        elif not summaries:physics_status='EXECUTION_INCOMPLETE'
        else:physics_status='PHYSICS_SEEDS_INCOMPLETE'
        trial_rows=[]
        for plan in manifest['trial_plans']:
            if plan['physical_config_id']!=spec['physical_config_id']:continue
            for path in (OUT/'trials'/plan['trial_id']).glob('*/g_outcome.json'):
                trial_rows.append(read(path,{}))
        runtime=sum(bool(x.get('runtime_valid')) for x in trial_rows);probe=sum(bool(x.get('probe_valid')) for x in trial_rows)
        admitted=[formal_by_plan[p['trial_id']] for p in manifest['trial_plans'] if p['physical_config_id']==spec['physical_config_id'] and p['trial_id'] in formal_by_plan]
        labels=sum(bool(x.get('label_valid')) for x in admitted)
        states.append(dict(scene_id=sid,family='CONTROL_EMPTY' if sid=='EMPTY' else spec['family'],name=spec['name'],physical_config_id=spec['physical_config_id'],representative=rep,material_fidelity=spec['material_fidelity'],is_control=sid=='EMPTY',physics_attempts=len(attempts),physics_valid_seeds=len(valid_seeds),physics_valid=len(valid_seeds)>=3,
            physics_failed_seeds=json.dumps(sorted({e['seed'] for e in unresolved if e['seed'] is not None and (e['failed_checks'] or e['execution_failure'])})),physics_attempt_evidence=json.dumps(evidence,separators=(',',':')),
            g1_completed=len(trial_rows),runtime_valid=runtime,probe_valid=probe,label_valid=labels,passable=sum(x.get('label') in ['PASSABLE','PASSABLE_FOR_ACTION'] for x in admitted),blocked=sum(x.get('label')=='BLOCKED' for x in admitted),label_unknown=sum(x.get('label')=='LABEL_UNKNOWN' for x in admitted),primary_audited=len(admitted),label_pending=max(0,len(trial_rows)-len(admitted)),raw_online_passable=sum(x.get('label') in ['PASSABLE','PASSABLE_FOR_ACTION'] for x in trial_rows),raw_online_blocked=sum(x.get('label')=='BLOCKED' for x in trial_rows),physics_status=physics_status,g1_status='LABEL_VALID' if labels else 'AUDITED_UNKNOWN' if admitted else 'LABEL_PENDING' if trial_rows else 'PENDING',learning_status='PENDING_EXTERNAL_LEARNING_LEDGER'))
    snapshot=dict(status='IN_PROGRESS_NOT_COMPLETE',time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),g1_launches=g1_launch_count(manifest),scene_states=states)
    atomic(OUT/'campaign_progress.json',snapshot)
    path=ROOT/'research/round4070_comprehensive_traversability/SCENE_COVERAGE_MATRIX.csv'
    with path.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(states[0]));w.writeheader();w.writerows(states)
    return snapshot


def main():
    default_manifest=next((OUT/f'campaign_manifest_v{v}.json' for v in [5,4,3] if (OUT/f'campaign_manifest_v{v}.json').exists() and (OUT/f'campaign_manifest_v{v}.sha256.json').exists()),OUT/'campaign_manifest_v3.json')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=default_manifest);p.add_argument('--mode',choices=['all','physics','g1','status'],default='all');p.add_argument('--scene');p.add_argument('--max-jobs',type=int,default=0);p.add_argument('--retry-runtime-failures',action='store_true');a=p.parse_args()
    manifest=read(a.manifest);expected=read(a.manifest.with_suffix('.sha256.json'))
    assert hashlib.sha256(a.manifest.read_bytes()).hexdigest()==expected['sha256'],'Frozen manifest changed'
    jobs=0;blocked=[]
    def budget_stop():return a.max_jobs>0 and jobs>=a.max_jobs
    if a.mode=='status':print(json.dumps(summarize(manifest),indent=2));return
    # The queue itself owns a separate exclusive lock to prevent duplicate
    # launch selection while releasing the physics/training lock at boundaries.
    with resource_lock('campaign',f'queue-{os.getpid()}',OUT/'campaign_lock.json'):
        event('CAMPAIGN_STARTED',manifest=str(a.manifest),manifest_sha256=expected['sha256'],mode=a.mode,scene=a.scene)
        specs=manifest['scene_specs'];config_dir=OUT/manifest.get('config_directory','configs');ordered=['EMPTY']+manifest['representative_scene_ids']+[s for s in specs if s.startswith('DR')]
        if a.scene:ordered=[a.scene]
        def gate(sid):
            nonlocal jobs
            spec=specs[sid];root=OUT/'physics'/sid
            current_sources={name:hashlib.sha256((ROOT/'scripts/round4070_g'/name).read_bytes()).hexdigest() for name in ['physics_scene_builder.py','physics_gate.py']}
            execution_block=read(root/'execution_block.json',{})
            if execution_block.get('source_sha256')==current_sources:
                event('SCENE_EXECUTION_BLOCK_RETAINED',scene_id=sid,failure=execution_block.get('failure_reason'));return False
            for seed in manifest['physics_seeds']:
                previous=sorted(root.glob(f'seed{seed}_attempt*'));summaries=[read(x/'summary.json',{}) for x in previous]
                if any(x.get('physics_valid') for x in summaries):continue
                diagnosed=bool(previous and all(read(x/'apparatus_diagnosis.json',{}).get('apparatus_invalid') for x in previous))
                if previous and (not a.retry_runtime_failures or len(previous)>=2 or (any(x.get('checks') for x in summaries) and not diagnosed)):continue
                if budget_stop():return False
                run=root/f'seed{seed}_attempt{len(previous)+1:02d}'
                launch(ROOT/'scripts/round4070_g/physics_gate.py',run,['--scene-config-json',str(config_dir/f'{sid}.json'),'--run-root',str(run),'--seed',str(seed),'--seconds',str(manifest['physics_seconds']),'--headless'],'PHYSICS',f'{sid}-{seed}')
                jobs+=1;summarize(manifest)
                result=read(run/'summary.json',{})
                if result.get('failure_type')=='IMPLEMENTATION_OR_RUNTIME_EXCEPTION':
                    atomic(root/'execution_block.json',dict(source_sha256=current_sources,failure_reason=result.get('failure_reason'),attempt=str(run),status='BLOCKED_WITH_REASON_PENDING_IMPLEMENTATION_REPAIR',not_material_invalid=True))
                    event('SCENE_IMPLEMENTATION_REPAIR_REQUIRED',scene_id=sid,failure=result.get('failure_reason'));break
            valid={s.get('seed') for s in [read(x,{}) for x in root.glob('*/summary.json')] if s.get('physics_valid')}
            return len(valid)>=3
        g1_admission_paused=(OUT/'PAUSE_G1.flag').exists() and a.mode=='all'
        if a.mode=='physics' or g1_admission_paused:
            for sid in ordered:
                if budget_stop():break
                if not gate(sid):blocked.append(sid)
        else:
            gate_cache={}
            def priority(plan):
                if plan['scene_id']=='EMPTY':return (0 if plan['repetition_index']==0 else 2,plan['repetition_index'])
                if plan['phase']=='coverage':return (1 if plan['repetition_index']==0 else 3,plan['repetition_index'])
                return (4 if plan['phase']=='robot_variation' else 5 if plan['phase']=='combination' else 6,plan['repetition_index'])
            global_errors=[]
            for plan in sorted(manifest['trial_plans'],key=priority):
                if (OUT/'PAUSE_G1.flag').exists():
                    g1_admission_paused=True;break
                sid=plan['scene_id']
                if a.scene and sid!=a.scene:continue
                if budget_stop():break
                known_budget=trial_budget_state(plan['trial_id'],launch_ledger_summary(manifest))
                if known_budget['withhold']:
                    event('G1_KNOWN_RAW_MISSING_WITHHELD',trial_id=plan['trial_id'],budget_state=known_budget,g1_budget_consumed=0)
                    continue
                root=OUT/'trials'/plan['trial_id'];previous=sorted(root.glob('attempt*'))
                # An explicitly reviewed local uncertainty can withhold this
                # exact plan only. Validate before automatic derived re-audit
                # so no source change silently replaces its bound evidence.
                quarantine=quarantine_state(previous[-1] if previous else root/'attempt01',plan)
                if quarantine['status']=='GLOBAL_PAUSE_REQUIRED':
                    (OUT/'PAUSE_G1.flag').write_text('Quarantine evidence requires explicit review: '+quarantine['reason'],encoding='utf-8')
                    event('G1_QUARANTINE_EVIDENCE_GLOBAL_PAUSE',trial_id=plan['trial_id'],quarantine=quarantine,g1_budget_consumed=0)
                    g1_admission_paused=True;break
                if quarantine['status']=='VALIDATED_WITHHOLD':
                    event('G1_EXACT_PLAN_QUARANTINE_WITHHELD',trial_id=plan['trial_id'],quarantine=quarantine,g1_budget_consumed=0)
                    continue
                auditor=ROOT/'scripts/round4070_g/physics_g1_collision_audit.py'
                if auditor.exists():
                    for old in previous:
                        existing=read(old/'g_collision_audit.json',{})
                        audit_sources=auditor_source_hashes()
                        native=old/'g_native_geometry_verified.json'
                        native_changed=native.exists() and existing.get('source_files',{}).get(native.name)!=hashlib.sha256(native.read_bytes()).hexdigest()
                        if ((old/'g_body_collision_history_audit_only.npz').exists() or (old/'g_body_collision_history_index.json').exists()) and (not existing or existing.get('auditor_source_sha256')!=audit_sources or native_changed):
                            event('G1_EXISTING_RAW_REAUDIT',trial_id=plan['trial_id'],attempt=str(old),new_g1_budget_consumed=0)
                            subprocess.run([os.sys.executable,str(auditor),'--run-root',str(old)],cwd=ROOT,check=False)
                states=[trial_state(x,plan) for x in previous]
                if any(s['status']=='PHYSICS_INVALID' for s in states):
                    atomic(OUT/'physics'/sid/'g1_physics_block.json',dict(reason='ACTUAL_ROBOT_PENETRATION',attempts=[str(x) for x in previous],physical_config_id=plan['physical_config_id']))
                if (OUT/'physics'/sid/'g1_physics_block.json').exists():
                    event('G1_SCENE_WITHHELD_PHYSICS_INVALID',scene_id=sid,trial_id=plan['trial_id']);blocked.append(sid);continue
                if any(s['status']=='COMPLETE' for s in states):continue
                if previous and (not a.retry_runtime_failures or len(previous)>=2 or not retry_authorized(previous[-1],plan)):
                    event('G1_RAW_REAUDIT_OR_REPAIR_REQUIRED',trial_id=plan['trial_id'],states=[{k:s[k] for k in ['status','reason']} for s in states],recapture_permitted=False)
                    if any(s['status']=='NOT_VERIFIED' for s in states):
                        (OUT/'PAUSE_G1.flag').write_text('Independent collision/RNG evidence requires repair and raw re-audit before recapture',encoding='utf-8');g1_admission_paused=True;break
                    continue
                dependency=empty_dependency(plan,manifest,OUT)
                if not dependency['ready']:
                    event('G1_EMPTY_DEPENDENCY_WITHHELD',trial_id=plan['trial_id'],scene_id=sid,dependency=dependency,g1_budget_consumed=0);continue
                if sid not in gate_cache:
                    gate_cache[sid]=gate(sid) if a.mode=='all' else len({read(x,{}).get('seed') for x in (OUT/'physics'/sid).glob('*/summary.json') if read(x,{}).get('physics_valid')})>=3
                if not gate_cache[sid]:
                    blocked.append(sid);continue
                if budget_stop():break
                launched=g1_launch_count(manifest)
                if launched>=manifest['max_g1_launches']:event('G1_BUDGET_EXHAUSTED',launches=launched);break
                evidence=[dict(path=str(x.relative_to(ROOT)),sha256=hashlib.sha256(x.read_bytes()).hexdigest(),physical_config_id=plan['physical_config_id']) for x in (OUT/'physics'/sid).glob('*/summary.json') if read(x,{}).get('physics_valid')]
                assert len({read(ROOT/e['path'],{}).get('seed') for e in evidence})>=3
                assert all(read(ROOT/e['path'],{}).get('physical_config_id')==plan['physical_config_id'] for e in evidence)
                next_attempt=max(known_budget['minimum_next_attempt_number'],1+max((int(p.name.removeprefix('attempt')) for p in previous),default=0))
                run=root/f'attempt{next_attempt:02d}';runtime_plan=dict(plan,physics_valid=True,physics_gate_status='THREE_DISTINCT_SEEDS_VALID',physics_gate_evidence=evidence,empty_control_admission=dependency)
                plan_path=OUT/'runtime_plans'/f'{plan["trial_id"]}_attempt{next_attempt:02d}.json';atomic(plan_path,runtime_plan)
                result=launch(ROOT/'scripts/round4070_g/g1_trial.py',run,['--scene-config-json',str(config_dir/f'{sid}.json'),'--trial-config-json',str(plan_path),'--run-root',str(run),'--headless'],'G1',plan['trial_id'],manifest=manifest)
                if result is None:g1_admission_paused=True;break
                jobs+=1;summarize(manifest)
                if plan.get('body_collision_audit_required'):
                    auditor=ROOT/'scripts/round4070_g/physics_g1_collision_audit.py'
                    if auditor.exists():subprocess.run([os.sys.executable,str(auditor),'--run-root',str(run)],cwd=ROOT,check=False)
                    state=trial_state(run,plan)
                    if state['status']=='PHYSICS_INVALID':
                        atomic(OUT/'physics'/sid/'g1_physics_block.json',dict(reason='ACTUAL_ROBOT_PENETRATION',attempt=str(run),physical_config_id=plan['physical_config_id']))
                    elif state['status']=='NOT_VERIFIED':
                        (OUT/'PAUSE_G1.flag').write_text('Independent collision/RNG audit not verified: '+state['reason'],encoding='utf-8')
                        event('G1_GLOBAL_AUDIT_ADMISSION_PAUSE',trial_id=plan['trial_id'],state=state,g1_budget_consumed=1)
                        g1_admission_paused=True;break
                stderr=(run/'process_stderr.log').read_text(encoding='utf-8',errors='replace') if (run/'process_stderr.log').exists() else ''
                errors=re.findall(r'(?m)^(NameError|AttributeError|ImportError|SyntaxError|TypeError|UnboundLocalError|RuntimeError|ValueError|AssertionError): (.+)$',stderr)
                if errors:global_errors.append(errors[-1])
                else:global_errors=[]
                if len(global_errors)>=2 and global_errors[-1]==global_errors[-2]:
                    event('GLOBAL_IMPLEMENTATION_ERROR_PAUSE',failure=global_errors[-1],status='INCOMPLETE_REPAIR_REQUIRED');break
        snapshot=summarize(manifest)
        if g1_admission_paused:
            event('G1_ADMISSION_PAUSED',reason='PAUSE_G1.flag',jobs_launched_this_invocation=jobs,g1_launches=snapshot['g1_launches'],next_action='CONTINUE_PHYSICS_ONLY' if jobs else 'WAIT_FOR_G1_ADMISSION',training_permitted=False,status='INCOMPLETE_CHECKPOINTED')
        event('CAMPAIGN_BOUNDARY',jobs_launched_this_invocation=jobs,blocked_scenes=sorted(set(blocked)),g1_launches=snapshot['g1_launches'],status='INCOMPLETE_CHECKPOINTED',g1_admission_paused=g1_admission_paused)


if __name__=='__main__':main()
