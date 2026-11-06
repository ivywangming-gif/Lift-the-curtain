"""Run the approved isolated collider diagnostic under the normal owned guard."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
from resource_lock import resource_lock
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/round4070_comprehensive_traversability'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--kind',choices=['collider','wall'],default='collider');a=p.parse_args()
    assert Path(a.run_id).name==a.run_id and a.run_id not in ['.','..']
    assert (OUT/'PAUSE.flag').exists() and (OUT/'PAUSE_G1.flag').exists(),'Diagnostic requires an explicit safe campaign handoff'
    run=OUT/'physics_diagnostics'/a.run_id;run.mkdir(parents=True,exist_ok=False)
    script=ROOT/'scripts/round4070_g'/('physics_wall_smoke.py' if a.kind=='wall' else 'physics_collision_smoke.py');guard=ROOT/'scripts/round4070_fixed_probe_benchmark/run_guarded_with_resources.ps1'
    q=lambda value:"'"+str(value).replace("'","''")+"'"
    inv=run/'launch_guard.ps1';inv.write_text('& '+q(guard)+' -ScriptPath '+q(script)+' -RunRoot '+q(run)+' -OverallTimeoutSeconds 1200 -ScriptArgs @(\'--run-root\','+q(run)+',\'--headless\')',encoding='utf-8')
    with resource_lock('physics',a.run_id):
        from physics_campaign import archive_source_files,launch_source_paths,resource_ready
        ready,resources=resource_ready()
        if not ready:raise RuntimeError('Diagnostic resource threshold not satisfied: '+str(resources))
        runtime=run/'runtime_sources';runtime.mkdir()
        (runtime/'physics_pvd_extract.py').write_bytes((ROOT/'scripts/round4070_g/physics_pvd_extract.py').read_bytes())
        hashes,archives=archive_source_files([*launch_source_paths(script,inv),Path(__file__),ROOT/'scripts/round4070_g/physics_pvd_extract.py'],run)
        (run/'campaign_launch.json').write_text(json.dumps(dict(kind='PHYSICS_DIAGNOSTIC',diagnostic_kind=a.kind,g1_budget_consumed=0,resources=resources,source_sha256=hashes,launch_source_archive=archives,source_archive_complete_before_guard=True),indent=2),encoding='utf-8')
        print('DIAGNOSTIC_GUARD_START',os.getpid(),a.kind,str(run),flush=True)
        with (run/'guard_console.log').open('w',encoding='utf-8') as f:
            result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(inv)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        print('DIAGNOSTIC_RETURN',result.returncode,str(run),flush=True)
        return result.returncode


if __name__=='__main__':raise SystemExit(main())
