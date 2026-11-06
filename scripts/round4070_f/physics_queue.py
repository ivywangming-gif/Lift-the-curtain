"""Serial F collection. Run only after offline Phase A; never mutates E outputs.

Usage: python scripts/round4070_f/physics_queue.py --stage screen|confirm|posture|p1|rail_mobility|left_completion
The legacy CLI name ``confirm`` denotes new-size exploration, not confirmation.
All simulations use the frozen E watchdog and headless renderer-free loop.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'outputs/round4070_learning_traversability/physics'
E = REPO / 'scripts/round4070_fixed_probe_benchmark'
RETRY_TAG = ''
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    temp.replace(path)


def call_guard(script, root, args, timeout=600):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'summary.json').exists():
        print('PRESERVE_EXISTING', root, flush=True)
        return read(root/'summary.json')
    if (root/'command.json').exists():
        raise RuntimeError(f'Incomplete previous output preserved: {root}')
    args = ['--run-root', str(root), '--headless'] + args
    write(root/'command.json', dict(script=str(script), args=args, guard=str(E/'run_guarded_with_resources.ps1')))
    def psq(value):
        return "'"+str(value).replace("'", "''")+"'"
    expression = '& '+psq(E/'run_guarded_with_resources.ps1')+' -ScriptPath '+psq(script)+' -RunRoot '+psq(root)+' -OverallTimeoutSeconds '+str(timeout)+' -ScriptArgs @('+','.join(map(psq,args))+')'
    print('START', root.name, datetime.now(timezone.utc).isoformat(), flush=True)
    proc = subprocess.run(['powershell','-NoProfile','-Command',expression], cwd=REPO, capture_output=True, text=True,
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    (root/'guard_stdout.log').write_text(proc.stdout, encoding='utf-8')
    (root/'guard_stderr.log').write_text(proc.stderr, encoding='utf-8')
    print('END', root.name, 'exit=',proc.returncode, 'status=',read(root/'summary.json').get('status'), flush=True)
    return read(root/'summary.json')


def g1_plan(stage):
    plans=[]
    if stage=='screen':
        entries=[('nominal','right'),('bottom_right_anchor','right'),('nominal','left'),('bottom_right_anchor','left')]
        for n,(scenario,side) in enumerate(entries):
            plans.append(dict(trial_id=f'F_SCREEN_{scenario}_{side}_S{420101+n}',scenario=scenario,probe_side=side,seed=420101+n,curtain_length=2.0,reset_root_y=-.004,reset_root_yaw=-.004))
    elif stage=='confirm':
        entries=[('nominal','right',1.95),('bottom_right_anchor','right',1.95),('curtain_hidden_rigid','right',2.0),('curtain_hidden_rigid','left',2.0)]
        for n,(scenario,side,length) in enumerate(entries):
            plans.append(dict(trial_id=f'F_CONFIRM_{scenario}_{side}_S{420201+n}',scenario=scenario,probe_side=side,seed=420201+n,curtain_length=length,reset_root_y=-.004,reset_root_yaw=-.004))
    elif stage=='left_completion':
        for n,scenario in enumerate(['nominal','bottom_right_anchor']):
            plans.append(dict(trial_id=f'F_LEFT_COMPLETION_{scenario}_left_S{420401+n}',scenario=scenario,probe_side='left',seed=420401+n,curtain_length=1.95,reset_root_y=-.004,reset_root_yaw=-.004))
    else:
        for n,scenario in enumerate(['nominal','bottom_right_anchor','curtain_hidden_rigid']):
            plans.append(dict(trial_id=f'F_POSTURE_{scenario}_right_S{420301+n}',scenario=scenario,probe_side='right',seed=420301+n,curtain_length=2.0,reset_root_y=.006,reset_root_yaw=.008))
    for p in plans:
        p.update(phase='F_'+stage.upper(),condition_id='F_'+p['scenario'].upper()+'_L'+str(p['curtain_length']),condition_family='hidden_wall' if p['scenario']=='curtain_hidden_rigid' else 'fabric',formal_core=True,target_depth_m=.1,probe_10cm_joint_step_rad=.85,structural_topology='A0_TOP_EDGE_ONLY',physics_gate_id='C_GAP05' if p['scenario']=='curtain_hidden_rigid' else 'A1_BOTTOM_RIGHT_ANCHOR' if p['scenario']=='bottom_right_anchor' else 'A0_TOP_EDGE_ONLY')
        p['equivalent_e_condition_id']={'nominal':'FABRIC_NOMINAL','bottom_right_anchor':'FABRIC_BOTTOM_RIGHT_ANCHOR','curtain_hidden_rigid':'FABRIC_WALL_GAP_05'}[p['scenario']]
        if RETRY_TAG:
            p['trial_id'] += '_' + RETRY_TAG
            p['execution_context_retry'] = RETRY_TAG
    return plans


def run_g1(stage):
    if stage=='posture' and not any(r.get('label')=='PASSABLE' for r in read(OUT/'supplementary_labels.json')):
        raise RuntimeError('Posture expansion withheld: screening has no validated positive')
    plans=g1_plan(stage)
    write(OUT/f'{stage}_manifest.json',dict(frozen_before_collection=True,split_role='supplementary_exploratory_no_model_selection',plans=plans))
    for p in plans:
        scene=p['scenario']; length=p['curtain_length']
        if length==2.0:
            gates=read(REPO/'configs/round4070_fixed_probe_benchmark/PHYSICS_GATE_STATUS.json')['gates']
            inherited=gates[p['physics_gate_id']]
            gate_root=Path(inherited['evidence_root'])
            evidence_paths=[gate_root/m/'summary.json' for m in ['no_probe','probe']]
            if inherited['status']!='PASS_E_PHYSICS' or any(read(x).get('status')!='PASS_PHYSICS_SANITY' for x in evidence_paths):
                raise RuntimeError('Inherited gate lacks verified pass evidence: '+str(gate_root))
            p['inherited_gate_sha256']={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in evidence_paths}
        if length!=2.0:
            gate=OUT/'p0_gate'/f'{scene}_L{length}'
            for mode in ['no_probe','probe']:
                extra=['--scenario',scene,'--mode',mode,'--seed',str(p['seed']),'--curtain-length',str(length),'--steps','720']
                if mode=='probe': extra+=['--matched-no-probe-summary',str(gate/'no_probe/summary.json')]
                gate_result=call_guard(E/'validate_single_panel.py',gate/mode,extra)
                if mode=='no_probe' and gate_result.get('status')!='PASS_PHYSICS_SANITY':
                    write(gate/'probe/withheld.json',dict(status='WITHHELD_PHYSICS_INVALID',reason='Initialization gate did not pass; probe not run'))
                    break
            if any(read(gate/m/'summary.json').get('status')!='PASS_PHYSICS_SANITY' for m in ['no_probe','probe']):
                write(OUT/'trials'/p['trial_id']/'withheld.json',dict(reason='PHYSICS_INVALID',plan=p));continue
        root=OUT/'trials'/p['trial_id']
        if not (root/'command.json').exists() and len(list((OUT/'trials').glob('*/command.json'))) >= 64:
            raise RuntimeError('User-authorized first-batch hard cap of 64 G1 launches reached')
        # Match E hidden-wall controls: stop after the same probe and safe
        # retract. Their blocked truth comes from independent rigid geometry,
        # not an unnecessary attempted whole-body collision with the wall.
        meta=dict(p,physics_gate_status='PASS_E_PHYSICS' if length==2.0 else 'PASS_F_PHYSICS',probe_validation_only=scene=='curtain_hidden_rigid',probe_contract='FIXED_ONE_SHOT_NOMINAL_0.10M_ACTUAL_FEEDBACK_RECORDED')
        write(root/'trial_metadata.json',meta)
        args=['--seed',str(p['seed']),'--probe-side',p['probe_side'],'--high-level-policy','r1','--reset-root-y',str(p['reset_root_y']),'--reset-root-yaw',str(p['reset_root_yaw']),'--precontact-offset','0.40','--precontact-region-x-min','0.36','--precontact-region-x-max','0.48','--precontact-region-y-half-width','0.10','--precontact-region-speed-max','0.06','--round15-progress-monitor','--research-feasibility-post-clearance-replan','--condition-id',p['condition_id'],'--curtain-length',str(length),'--density-scale-multiplier','1','--probe-depth-level','PROBE_10CM','--probe-10cm-joint-step-rad','0.85','--structural-topology','A0_TOP_EDGE_ONLY','--surface-thickness','0.002','--scenario',scene]
        if scene=='curtain_hidden_rigid': args+=['--hidden-rigid-x','0.92','--probe-validation-only']
        call_guard(E/'run_episode.py',root,args,900)
        summarize()


def run_p1(only_scene=None, first_seed=420071):
    from physics_p1_specs import scene_specs
    if only_scene and only_scene not in scene_specs():raise ValueError('Unknown P1 scene: '+only_scene)
    for scene in scene_specs():
        if only_scene and only_scene!=scene:continue
        for seed in [first_seed,first_seed+1]:
            root=OUT/'p1'/scene/f'S{seed}'
            for mode in ['no_probe','probe']:
                args=['--scene',scene,'--mode',mode,'--seed',str(seed),'--steps','720','--sample-every','1']
                if mode=='probe': args+=['--matched-no-probe-summary',str(root/'no_probe/summary.json')]
                gate_result=call_guard(REPO/'scripts/round4070_f/physics_p1.py',root/mode,args)
                if mode=='no_probe' and gate_result.get('status')!='PASS_PHYSICS_SANITY':
                    write(root/'probe/withheld.json',dict(status='WITHHELD_PHYSICS_INVALID' if gate_result else 'WITHHELD_EXECUTION_INVALID',reason='Initialization gate did not pass; probe not run'))
                    break
            # An invalid first repetition is sufficient to withhold G1; repeated
            # stability is measured only after both initial gates actually pass.
            if any(read(root/m/'summary.json').get('status')!='PASS_PHYSICS_SANITY' for m in ['no_probe','probe']):break
    summarize()


def summarize():
    spec=importlib.util.spec_from_file_location('f_frozen_eval',E/'evaluate_benchmark.py')
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    rows=[]
    for path in sorted((OUT/'trials').glob('*/summary.json')):
        s=read(path);p=read(path.parent/'trial_metadata.json')
        ok,detail=ev.protocol_audit(p,s,p)
        # The E label function accepts PASS; new F gates were checked before launch.
        gate_ok=p.get('physics_gate_status') in {'PASS_E_PHYSICS','PASS_F_PHYSICS'}
        label,valid,reason,explanation,evidence=ev.independent_label(p,s,'PASS' if gate_ok else 'MISSING',ok,detail)
        rows.append(dict(trial_id=p['trial_id'],path=str(path.parent),condition_id=p['condition_id'],condition_family=p['condition_family'],probe_side=p['probe_side'],seed=p['seed'],runtime_valid=s.get('runtime_valid'),probe_valid=s.get('probe_valid'),label=label,label_valid=valid,label_reason=reason,protocol_match=ok,protocol_detail=detail,actual_displacement_m=s.get('actual_hand_displacement_m'),whole_body_clearance=s.get('whole_body_candidate_clearance'),evidence=evidence))
    previous=read(OUT/'supplementary_labels.json')
    order={r['trial_id']:i for i,r in enumerate(previous)} if isinstance(previous,list) else {}
    rows.sort(key=lambda r:(order.get(r['trial_id'],len(order)),r['trial_id']))
    write(OUT/'supplementary_labels.json',rows)
    if rows:
        with (OUT/'supplementary_labels.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('LABELS',[(r['trial_id'],r['label']) for r in rows],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['screen','confirm','posture','left_completion','p1','rail_mobility','summarize'],required=True);parser.add_argument('--retry-tag',default='');parser.add_argument('--p1-scene',default=None);parser.add_argument('--p1-first-seed',type=int,default=420071);args=parser.parse_args()
    if args.retry_tag and not args.retry_tag.replace('_','').isalnum():raise ValueError('retry tag must be alphanumeric')
    RETRY_TAG=args.retry_tag
    OUT.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(r'C:\Robotics\FALCON-upstream\sim2real\models\falcon\g1_29dof.onnx')
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0', 'Frozen FALCON changed'
    sources=[E/'run_episode.py',E/'run_guarded_with_resources.ps1',E/'structural_curtain_asset.py',E/'validate_p1_flexible_scene.py',E/'validate_single_panel.py']
    sources += sorted((REPO/'scripts/round4070_f').glob('physics*.py'))
    write(OUT/'frozen_environment.json',dict(python=r'C:\conda_envs\isaaclab\python.exe',falcon_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),dt=.005,headless=True,render=False,max_new_g1=64,source_hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    if args.stage=='p1':run_p1(args.p1_scene,args.p1_first_seed)
    elif args.stage=='rail_mobility':
        assert all(read(OUT/'p1/SLIDING_TOP_TRACK'/f'S{seed}'/mode/'summary.json').get('status')=='PASS_PHYSICS_SANITY' for seed in [420071,420072] for mode in ['no_probe','probe']), 'Base rail gates must pass before mobility stimulus'
        call_guard(REPO/'scripts/round4070_f/physics_p1.py',OUT/'p1/SLIDING_TOP_TRACK_MOBILITY/S420073/no_probe',['--scene','SLIDING_TOP_TRACK','--mode','no_probe','--seed','420073','--steps','720','--sample-every','1','--rail-mobility'])
    elif args.stage=='summarize':summarize()
    else:run_g1(args.stage)
