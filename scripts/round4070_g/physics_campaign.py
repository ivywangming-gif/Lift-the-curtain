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
import time
import psutil
from resource_lock import resource_lock

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/round4070_comprehensive_traversability'
GUARD=ROOT/'scripts/round4070_fixed_probe_benchmark/run_guarded_with_resources.ps1'


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
    free=float(raw.strip().splitlines()[0]);return available>=2048 and free>=4096,dict(available_ram_mb=available,free_vram_mb=free)


def launch(script,run,args,kind,trial_id):
    while True:
        if (OUT/'PAUSE.flag').exists() or training_active():event('WAIT_TRAINING_OR_PAUSE',trial_id=trial_id);time.sleep(15);continue
        try:
            with resource_lock('physics',trial_id):
                if training_active():continue
                ready,resources=resource_ready()
                if not ready:event('WAIT_RESOURCE',trial_id=trial_id,**resources)
                else:
                    run.mkdir(parents=True,exist_ok=False)
                    invocation=run/'launch_guard.ps1'
                    def quote(value):return "'"+str(value).replace("'","''")+"'"
                    cmd='& '+quote(GUARD)+' -ScriptPath '+quote(script)+' -RunRoot '+quote(run)+' -OverallTimeoutSeconds 1200 -ScriptArgs @('+','.join(quote(v) for v in args)+')\n'
                    invocation.write_text(cmd,encoding='utf-8')
                    atomic(run/'campaign_launch.json',dict(kind=kind,trial_id=trial_id,queue_pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),guard=str(GUARD),script=str(script),args=args,resources=resources,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [script,ROOT/'scripts/round4070_g/physics_scene_builder.py',ROOT/'scripts/round4070_g/physics_scenes.py',GUARD]}))
                    event(kind+'_LAUNCHED',trial_id=trial_id,run_root=str(run),**resources)
                    with (run/'guard_console.log').open('w',encoding='utf-8') as console:
                        result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(invocation)],cwd=ROOT,stdout=console,stderr=subprocess.STDOUT)
                    watchdog=read(run/'watchdog_summary.json',{})
                    event(kind+'_GUARD_EXIT',trial_id=trial_id,returncode=result.returncode,watchdog_summary=str(run/'watchdog_summary.json'),wall_time_s=watchdog.get('wall_time_s'),timeout_reason=watchdog.get('timeout_reason'),owned_pid=watchdog.get('pid_started_by_round4070_problem_discovery'))
                    return result.returncode
        except InterruptedError:event('WAIT_RESOURCE_LOCK',trial_id=trial_id)
        time.sleep(15)


def summarize(manifest):
    states=[]
    for sid,spec in manifest['scene_specs'].items():
        rep=next((r for r in manifest['representative_scene_ids'] if manifest['scene_specs'][r]['physical_config_id']==spec['physical_config_id']),sid)
        summaries=[read(p,{}) for p in (OUT/'physics'/rep).glob('*/summary.json')]
        valid_seeds={s.get('seed') for s in summaries if s.get('physics_valid')}
        trial_rows=[]
        for plan in manifest['trial_plans']:
            if plan['physical_config_id']!=spec['physical_config_id']:continue
            for path in (OUT/'trials'/plan['trial_id']).glob('*/g_outcome.json'):
                trial_rows.append(read(path,{}))
        runtime=sum(bool(x.get('runtime_valid')) for x in trial_rows);probe=sum(bool(x.get('probe_valid')) for x in trial_rows);labels=sum(bool(x.get('label_valid')) for x in trial_rows)
        states.append(dict(scene_id=sid,family='CONTROL_EMPTY' if sid=='EMPTY' else spec['family'],name=spec['name'],physical_config_id=spec['physical_config_id'],representative=rep,material_fidelity=spec['material_fidelity'],is_control=sid=='EMPTY',physics_attempts=len(summaries),physics_valid_seeds=len(valid_seeds),physics_valid=len(valid_seeds)>=3,g1_completed=len(trial_rows),runtime_valid=runtime,probe_valid=probe,label_valid=labels,passable=sum(x.get('label') in ['PASSABLE','PASSABLE_FOR_ACTION'] for x in trial_rows),blocked=sum(x.get('label')=='BLOCKED' for x in trial_rows),label_unknown=sum(x.get('label')=='LABEL_UNKNOWN' for x in trial_rows),physics_status='PHYSICS_VALID' if len(valid_seeds)>=3 else 'PHYSICS_PENDING_OR_FAILED',g1_status='LABEL_VALID' if labels else 'PROBE_VALID' if probe else 'RUNTIME_VALID' if runtime else 'G1_RESULT_AVAILABLE' if trial_rows else 'PENDING',learning_status='PENDING_EXTERNAL_LEARNING_LEDGER'))
    snapshot=dict(status='IN_PROGRESS_NOT_COMPLETE',time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),g1_launches=sum(1 for _ in (OUT/'trials').glob('*/*/campaign_launch.json')),scene_states=states)
    atomic(OUT/'campaign_progress.json',snapshot)
    path=ROOT/'research/round4070_comprehensive_traversability/SCENE_COVERAGE_MATRIX.csv'
    with path.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(states[0]));w.writeheader();w.writerows(states)
    return snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=OUT/'campaign_manifest_v3.json');p.add_argument('--mode',choices=['all','physics','g1','status'],default='all');p.add_argument('--scene');p.add_argument('--max-jobs',type=int,default=0);p.add_argument('--retry-runtime-failures',action='store_true');a=p.parse_args()
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
        if a.mode=='physics':
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
                sid=plan['scene_id']
                if a.scene and sid!=a.scene:continue
                if budget_stop():break
                root=OUT/'trials'/plan['trial_id'];previous=sorted(root.glob('attempt*'))
                summaries=[read(x/'g_outcome.json',{}) for x in previous]
                if any(s.get('runtime_valid') for s in summaries):continue
                if previous and (not a.retry_runtime_failures or len(previous)>=2):continue
                if sid not in gate_cache:
                    gate_cache[sid]=gate(sid) if a.mode=='all' else len({read(x,{}).get('seed') for x in (OUT/'physics'/sid).glob('*/summary.json') if read(x,{}).get('physics_valid')})>=3
                if not gate_cache[sid]:
                    blocked.append(sid);continue
                if budget_stop():break
                launched=sum(1 for _ in (OUT/'trials').glob('*/*/campaign_launch.json'))
                if launched>=manifest['max_g1_launches']:event('G1_BUDGET_EXHAUSTED',launches=launched);break
                evidence=[dict(path=str(x.relative_to(ROOT)),sha256=hashlib.sha256(x.read_bytes()).hexdigest(),physical_config_id=plan['physical_config_id']) for x in (OUT/'physics'/sid).glob('*/summary.json') if read(x,{}).get('physics_valid')]
                assert len({read(ROOT/e['path'],{}).get('seed') for e in evidence})>=3
                assert all(read(ROOT/e['path'],{}).get('physical_config_id')==plan['physical_config_id'] for e in evidence)
                run=root/f'attempt{len(previous)+1:02d}';runtime_plan=dict(plan,physics_valid=True,physics_gate_status='THREE_DISTINCT_SEEDS_VALID',physics_gate_evidence=evidence)
                plan_path=OUT/'runtime_plans'/f'{plan["trial_id"]}_attempt{len(previous)+1:02d}.json';atomic(plan_path,runtime_plan)
                launch(ROOT/'scripts/round4070_g/g1_trial.py',run,['--scene-config-json',str(config_dir/f'{sid}.json'),'--trial-config-json',str(plan_path),'--run-root',str(run),'--headless'],'G1',plan['trial_id'])
                jobs+=1;summarize(manifest)
                stderr=(run/'process_stderr.log').read_text(encoding='utf-8',errors='replace') if (run/'process_stderr.log').exists() else ''
                errors=re.findall(r'(?m)^(NameError|AttributeError|ImportError|SyntaxError|TypeError|UnboundLocalError): (.+)$',stderr)
                if errors:global_errors.append(errors[-1])
                else:global_errors=[]
                if len(global_errors)>=2 and global_errors[-1]==global_errors[-2]:
                    event('GLOBAL_IMPLEMENTATION_ERROR_PAUSE',failure=global_errors[-1],status='INCOMPLETE_REPAIR_REQUIRED');break
        snapshot=summarize(manifest);event('CAMPAIGN_BOUNDARY',jobs_launched_this_invocation=jobs,blocked_scenes=sorted(set(blocked)),g1_launches=snapshot['g1_launches'],status='INCOMPLETE_CHECKPOINTED')


if __name__=='__main__':main()
