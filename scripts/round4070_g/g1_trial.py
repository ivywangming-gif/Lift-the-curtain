"""G-only audited adapter around the frozen E G1 control loop.

Historical source is read, exact hooks are checked, and a patched copy is
compiled in memory. No E/F source, checkpoint, timing or safety limit is edited.
Only the physics queue may launch this executable (one Isaac process at a time).
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from g1_recording import Recorder, action_spec, write_json, ACTION_ID, CONTRACT

SOURCE=ROOT/'scripts/round4070_fixed_probe_benchmark/run_episode.py'


def replace_once(source,old,new):
    if source.count(old)!=1: raise RuntimeError(f'Frozen runner hook changed ({source.count(old)} matches): {old[:100]}')
    return source.replace(old,new,1)


def patch_source(source):
    source=replace_once(source,'                    retain_accelerations=False,',
        '                    retain_accelerations=True,  # G recorder: retain measured state only; policy unchanged')
    source=replace_once(source,'APP = AppLauncher(ARGS).app',
        'ARGS.scene_kind = {"rigid_board":"rigid", "empty":"none"}.get(_G_SPEC["geometry"]["kind"], "curtain")\n'
        'ARGS.physical_rigid = ARGS.scene_kind == "rigid"\n'
        'APP = AppLauncher(ARGS).app')
    source=replace_once(source,'PREPROBE_UPPER = make_side_upper(ARGS.probe_side, -0.45, 0.45)',
        'PREPROBE_UPPER = make_side_upper(ARGS.probe_side, -0.45, 0.45)\nPREPROBE_UPPER += np.asarray(_G_UPPER_OFFSET, dtype=np.float32)')
    source=replace_once(source,'PROBE_TARGET_UPPER = make_side_upper(ARGS.probe_side, -0.55 - _depth_step, 0.55 + _depth_step)',
        'PROBE_TARGET_UPPER = make_side_upper(ARGS.probe_side, -0.55 - _depth_step, 0.55 + _depth_step)\nPROBE_TARGET_UPPER += np.asarray(_G_UPPER_OFFSET, dtype=np.float32)')
    source=replace_once(source,'if not math.isfinite(CURTAIN_TOP_Z_M) or abs(CURTAIN_TOP_Z_M - float(ARGS.curtain_length) - 0.02) > 1.0e-6:',
        'if not math.isfinite(CURTAIN_TOP_Z_M) or CURTAIN_TOP_Z_M <= 0:')
    source=replace_once(source,'        (0.79, 0.91), (-1.00, 1.00), clear_margin_m=float(ARGS.candidate_clear_margin)',
        '        (float(_G_SPEC["geometry"]["center_x_m"])-.06, float(_G_SPEC["geometry"]["center_x_m"])+.06), (-1.00, 1.00), clear_margin_m=float(ARGS.candidate_clear_margin)')
    source=replace_once(source,'    candidate_xy = (0.85, 0.0)',
        '    candidate_xy = (float(_G_SPEC["geometry"]["center_x_m"]), float(_G_SPEC["geometry"]["center_y_m"]))')
    start=source.index('        if ARGS.scene_kind == "curtain":\n            from lift_curtain.envs.task_scale_scene')
    end=source.index('        actuators = {',start)
    source=source[:start]+'''        _G_VERIFY_SCENE_SOURCE()
        from physics_scene_builder import create_scene
        _G_SCENE = create_scene(stage, _G_SPEC, prim_root="/World/GScene")
        curtain_info = _G_SCENE.aggregate_info
        anchor_indices = [] if curtain_info is None else list(curtain_info.requested_bottom_anchor_indices)
        report["g_scene_evaluation_only"] = _G_SCENE.evaluation_metadata
'''+source[end:]
    source=replace_once(source,'            curtain_deformable = DeformablePrim(curtain_info.prim_path, deformable_type="surface")',
        '            curtain_deformable = _G_SCENE.make_deformable()')
    source=replace_once(source,'        curtain_snapshot_times: list[float] = []',
        '        curtain_snapshot_times: list[float] = []\n        _g_node_velocity_snapshots = []')
    source=replace_once(source,'                curtain_snapshot_times.append(sim_step * DT)',
        '                curtain_snapshot_times.append((sim_step + 1) * DT)\n                _g_node_velocity_snapshots.append(node_velocities.copy())')
    source=replace_once(source,'                last_curtain_stretch = max_neighbor_stretch(current_nodes, node_rest)',
        '                last_curtain_stretch = _G_SCENE.max_neighbor_stretch(current_nodes, node_rest)')
    start=source.index('                if top_indices:\n                    last_curtain_top_drift =')
    end=source.index('                if pre_probe_curtain is not None:',start)
    source=source[:start]+'''                last_curtain_top_drift, last_curtain_anchor_drift = _G_SCENE.constraint_drifts(current_nodes, curtain_initial)
'''+source[end:]
    source=replace_once(source,'            if local_step == 0 or sim_step % DECIMATION == 0:\n                started = time.perf_counter()',
        '            if local_step == 0 or sim_step % DECIMATION == 0:\n                _G_RECORDER.before_control(adapter, sim_step)\n                started = time.perf_counter()')
    source=replace_once(source,'            hand_velocity = hand_velocities[ARGS.probe_side]\n            if telemetry_sample:',
        '''            hand_velocity = hand_velocities[ARGS.probe_side]
            _G_RECORDER.observe(phase=phase,local_step=local_step,sim_step=sim_step,q=q_official,qdot=qdot_official,
                q_cmd=target_official,effort=effort_official,after=after,hands=hand_positions,velocities=hand_velocities,rpy=(roll,pitch,yaw),
                adapter=adapter,robot=robot,upper=requested_upper,preprobe_upper=PREPROBE_UPPER,
                target_upper=PROBE_TARGET_UPPER,command=command,walk_mode=walk_mode)
            if telemetry_sample:''')
    # The prescribed followup is independent of the candidate classifier. All
    # existing probe safety stops and R1 balance/progress filters remain active.
    source=replace_once(source,'        workflow.record_probe(ProbeDecision(decision.value))',
        '        _G_RULE_DECISION = decision.value\n'
        '        decision = ContactDecision.RESPONSIVE\n'
        '        workflow.record_probe(ProbeDecision(decision.value))')
    source=replace_once(source,'        if decision == ContactDecision.RESPONSIVE and not ARGS.probe_validation_only:',
        '        if (decision == ContactDecision.RESPONSIVE and not ARGS.probe_validation_only\n'
        '                and not conservative_stop and fall_reason is None and external_safety_reason is None):')
    source=replace_once(source,'            or probe_safety_stop_reason is not None\n        )',
        '            or probe_safety_stop_reason is not None\n            or external_safety_reason is not None\n            or fall_reason is not None\n        )')
    # Existing physics safety conditions used legacy phase names and omitted
    # the actual bounded advance name. Apply the same limits to that phase too.
    source=replace_once(source,'            if phase.startswith(("curtain_traverse", "responsive_traverse")) and curtain_deformable is not None:',
        '            if phase.startswith(("physical_probe", "curtain_traverse", "responsive_traverse", "round26_cautious_advance")) and curtain_deformable is not None:')
    source=replace_once(source,'            if phase.startswith("responsive_traverse") and ARGS.scene_kind == "rigid":',
        '            if phase.startswith(("responsive_traverse", "round26_cautious_advance")) and ARGS.scene_kind == "rigid":')
    source=replace_once(source,'        report["physical_probe_result"] = clean(physical_probe_result)\n        events.append({',
        '''        report["physical_probe_result"] = clean(physical_probe_result)
        return _G_RECORDER.finish(probe_result=clean(physical_probe_result),scene_metadata=_G_SCENE.evaluation_metadata,
            physics_valid=bool(_G_PLAN.get("physics_valid",False)),source_probe_valid=bool(probe_valid),fall_reason=fall_reason,
            external_safety_reason=external_safety_reason,events=clean(events),telemetry=clean(rows),
            evaluation_arrays={"nodes_world":curtain_snapshots,"node_velocities_world":_g_node_velocity_snapshots,
                "time_s":curtain_snapshot_times,"runtime_rest_positions":runtime_rest,
                "requested_fixed_indices":[] if curtain_info is None else curtain_info.requested_fixed_vertex_indices},
            rule_decision=_G_RULE_DECISION,logger=LOGGER)
        events.append({''')
    ast.parse(source)
    return source


def frozen_args(plan,spec,run_root,extra):
    allowed={'--headless','--device','--enable_cameras','--livestream','--experience'}
    if any(s.startswith('--') and s.split('=',1)[0] not in allowed for s in extra):
        raise ValueError('Only AppLauncher options are accepted outside the frozen trial contract')
    robot=plan.get('robot',plan.get('robot_conditions',{}))
    side=plan.get('probe_side',robot.get('probe_side','right'))
    duration=float(robot.get('probe_duration_s',1.5))
    if not 1.2 <= duration <= 1.8: raise ValueError('Unvalidated probe speed range')
    offset=robot.get('upper_offset_rad',[0.]*14)
    if len(offset)!=14 or max(abs(float(x)) for x in offset)>.05: raise ValueError('Unvalidated initial arm offset')
    inactive=offset[7:] if side=='left' else offset[:7]
    if any(float(x)!=0 for x in inactive): raise ValueError('Inactive arm offset needs a separate validated protocol')
    g=spec['geometry'];geom_top=float(g['height_m'])+float(g['bottom_z_m'])
    # AppLauncher flags may be supplied by queue; all action/safety arguments
    # below are last and authoritative. No selected network controls collection.
    args=list(extra)+['--run-root',str(run_root),'--scenario','nominal','--seed',str(plan['seed']),
        '--probe-side',side,'--high-level-policy','r1','--condition-id',str(spec['scene_id']),
        '--curtain-length',str(g['height_m']),'--curtain-top-z',str(geom_top),
        '--reset-root-y',str(robot.get('reset_root_y',plan.get('reset_root_y',-.004))),
        '--reset-root-yaw',str(robot.get('reset_root_yaw',plan.get('reset_root_yaw',-.004))),
        '--precontact-offset','0.40','--precontact-region-x-min','0.36','--precontact-region-x-max','0.48',
        '--precontact-region-y-half-width','0.10','--precontact-region-speed-max','0.06',
        '--round15-progress-monitor','--probe-depth-level','PROBE_10CM','--probe-10cm-joint-step-rad','0.85',
        '--probe-duration',str(duration),'--probe-hold-duration',str(2.0-duration),
        '--preprobe-duration',str(robot.get('preprobe_duration_s',1.5)),
        '--preprobe-hold-duration','0.4','--max-bounded-advance-segments','24',
        '--segment-timeout','7.0','--cruise-speed','0.25','--vx-max','0.30',
        '--traverse-exit-x','1.30','--traverse-exit-y','0.0','--candidate-clear-margin','0.04',
        '--retract-duration','1.5','--neutralize-duration','1.5','--stand-duration','1.5','--opening-mode','none']
    if plan.get('withhold_followup_for_geometric_safety',False): args+=['--probe-validation-only']
    return args,offset


def verify_gate_evidence(plan,spec):
    from physics_scenes import physical_hash
    if plan.get('physical_config_id')!=physical_hash(spec):raise RuntimeError('Trial physical config does not match frozen scene file')
    if plan.get('followup_action_id')!=ACTION_ID or plan.get('label_contract')!=CONTRACT:
        raise RuntimeError('Trial metadata must identify the exact frozen G action/label contract')
    seeds=set();verified=[]
    if not plan.get('physics_valid',False):raise RuntimeError('G1 launch requires independently completed physics gate')
    for item in plan.get('physics_gate_evidence',[]):
        path=Path(item['path']);path=path if path.is_absolute() else ROOT/path
        if not path.resolve().is_relative_to((ROOT/'outputs/round4070_comprehensive_traversability').resolve()):
            raise RuntimeError('Physics gate evidence must come from new G outputs')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('Physics gate summary hash changed')
        gate=json.loads(path.read_text(encoding='utf-8-sig'))
        if not gate.get('physics_valid') or gate.get('physical_config_id')!=plan['physical_config_id']:
            raise RuntimeError('Invalid/mismatched actual physics gate')
        if gate.get('schema')!='round4070-g.physics-gate.v1' or not gate.get('checks') or not all(gate['checks'].values()):
            raise RuntimeError('Physics gate checks missing or failed')
        source_hashes=gate.get('source_sha256',{})
        source_record=path
        if not source_hashes:
            source_record=path.parent/'campaign_launch.json'
            source_hashes=json.loads(source_record.read_text(encoding='utf-8-sig')).get('source_sha256',{})
        source_hashes={Path(name).name:value for name,value in source_hashes.items()}
        for name in ('physics_scene_builder.py','physics_scenes.py'):
            if name not in source_hashes:raise RuntimeError('Physics launch source hash missing: '+name)
            if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=source_hashes[name]:
                raise RuntimeError('Physics source changed after gate: '+name)
        seeds.add(gate['seed']);verified.append(dict(path=str(path),sha256=item['sha256'],seed=gate['seed'],source_sha256=source_hashes,
            source_evidence_path=str(source_record),source_evidence_sha256=hashlib.sha256(source_record.read_bytes()).hexdigest()))
    if len(seeds)<3:raise RuntimeError('At least three actual distinct-seed physics gates are required')
    return verified


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-config-json',type=Path,required=True)
    parser.add_argument('--trial-config-json',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args,extra=parser.parse_known_args()
    spec=json.loads(args.scene_config_json.read_text(encoding='utf-8-sig'))
    plan=json.loads(args.trial_config_json.read_text(encoding='utf-8-sig'))
    action_spec(plan)
    source=SOURCE.read_text(encoding='utf-8-sig');patched=patch_source(source)
    argv,offset=frozen_args(plan,spec,args.run_root,extra)
    sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
    args.run_root.mkdir(parents=True,exist_ok=True)
    proof=dict(original_source=str(SOURCE),original_source_sha256=sha(source),patched_source_sha256=sha(patched),
               original_source_file_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               action=action_spec(plan),source_edit_mode='exact checked in-memory hooks; original file unchanged',argv=argv,
               physics_dt_s=.005,control_decimation=4,geometry_decimation=20,record_decimation=4,
               dry_run=args.dry_run,followup_policy_uses_classifier=False)
    write_json(args.run_root/'g_runner_contract.json',proof)
    if args.dry_run:
        print(json.dumps(proof,indent=2));return 0
    proof['verified_physics_gate_evidence']=verify_gate_evidence(plan,spec)
    proof['recording_source_sha256']=hashlib.sha256((Path(__file__).parent/'g1_recording.py').read_bytes()).hexdigest()
    write_json(args.run_root/'g_runner_contract.json',proof)
    # Save the exact executable source once per version, in new G outputs only.
    source_path=ROOT/'outputs/round4070_comprehensive_traversability/runtime_sources'/f'{sha(patched)}.py'
    source_path.parent.mkdir(parents=True,exist_ok=True)
    if not source_path.exists():source_path.write_text(patched,encoding='utf-8')
    proof['runtime_source_archive']={}
    for name in ('g1_trial.py','g1_recording.py','physics_scene_builder.py','physics_scenes.py'):
        content=(Path(__file__).parent/name).read_bytes();digest=hashlib.sha256(content).hexdigest()
        archived=source_path.parent/f'{Path(name).stem}_{digest}.py'
        if not archived.exists():archived.write_bytes(content)
        proof['runtime_source_archive'][name]=dict(path=str(archived),sha256=digest)
    write_json(args.run_root/'g_runner_contract.json',proof)
    recorder=Recorder(args.run_root,plan,spec,sha(patched))
    def verify_live_scene_source():
        for name in ('physics_scene_builder.py','physics_scenes.py'):
            if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=proof['runtime_source_archive'][name]['sha256']:
                raise RuntimeError('Scene construction source changed during simulator startup: '+name)
    namespace={'__name__':'_g_frozen_runner','__file__':str(SOURCE),'_G_RECORDER':recorder,
               '_G_SPEC':spec,'_G_PLAN':plan,'_G_UPPER_OFFSET':offset,'_G_VERIFY_SCENE_SOURCE':verify_live_scene_source}
    sys.argv=[str(SOURCE),*argv];sys.path.insert(0,str(SOURCE.parent))
    try:
        exec(compile(patched,str(source_path),'exec'),namespace)
        return namespace['main']()
    finally:
        recorder.close_incomplete()


if __name__=='__main__':raise SystemExit(main())
