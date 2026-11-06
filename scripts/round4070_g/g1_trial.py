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
from g1_recording import Recorder, action_spec, write_json, clean, ACTION_ID, CONTRACT
from g1_randomization import ResetRandomization
from g1_audit import BodyCollisionRecorder
from g1_action_contract_v2 import PRIMARY_PROTOCOL as V5_PROTOCOL,load_plan_contract,ActionTubeGuard
from lift_curtain.control.falcon_contract import EXPECTED_ONNX_SHA256

SOURCE=ROOT/'scripts/round4070_fixed_probe_benchmark/run_episode.py'
FALCON_ONNX=Path(r'C:\Robotics\FALCON-upstream\sim2real\models\falcon\g1_29dof.onnx')
CAMPAIGN_V4=ROOT/'outputs/round4070_comprehensive_traversability/campaign_manifest_v4.json'
CAMPAIGN_V4_SHA256='6cbe2d9858a114d53d312912076317ae68dd809f18403db9d34a6e68e8bd7271'


def replace_once(source,old,new):
    if source.count(old)!=1: raise RuntimeError(f'Frozen runner hook changed ({source.count(old)} matches): {old[:100]}')
    return source.replace(old,new,1)


def patch_source(source):
    source=replace_once(source,'        sim = SimulationContext(SimulationCfg(dt=DT, render_interval=DECIMATION, device="cuda:0"))',
        '        _g_sim_cfg = SimulationCfg(dt=DT, render_interval=DECIMATION, device="cuda:0")\n'
        '        _G_CONFIGURE_SCENE_QUERY(_g_sim_cfg)\n'
        '        sim = SimulationContext(_g_sim_cfg)\n'
        '        _G_RECORD_SCENE_QUERY(sim, _g_sim_cfg)')
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
        '            # G deformable was initialized and randomized before FIRST_PHYSICS_STEP')
    source=replace_once(source,'            zero_runtime_velocity(curtain_deformable, curtain_positions.shape[0])',
        '            # Preserve the evidenced one-time G1_RNG_V1 reset velocity injection')
    source=replace_once(source,'        robot.write_data_to_sim()\n        sim.step(render=False)\n        robot.update(DT)\n        LOGGER.mark("FIRST_PHYSICS_STEP")',
        '''        robot.write_data_to_sim()
        import omni.kit.app
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.core.experimental.prims", True)
        if curtain_info is not None:
            curtain_deformable = _G_SCENE.make_deformable()
            _g_reset_nodes, _, _ = read_runtime(curtain_deformable)
            _g_fixed = sorted({int(i) for c in _G_SCENE.constraints for i in c["global_vertex_indices"]}
                | set(curtain_info.requested_fixed_vertex_indices))
            _G_RANDOMIZATION.apply_nodes(curtain_deformable,len(_g_reset_nodes),_g_fixed,simulation_time_s=sim.current_time,
                actual_root_pose_wxyz=_G_RANDOMIZATION.actual_root_pose(robot))
        else:
            _G_RANDOMIZATION.apply_nodes(None,0,[],simulation_time_s=sim.current_time,
                actual_root_pose_wxyz=_G_RANDOMIZATION.actual_root_pose(robot))
        _G_AUDIT.initialize(stage,robot,_G_SCENE)
        if curtain_deformable is None:
            _g_initial_nodes = _g_initial_vel = np.zeros((0,3),dtype=np.float32)
        else:
            _g_initial_nodes, _g_initial_vel, _ = read_runtime(curtain_deformable)
        _G_AUDIT.capture_initial_state(robot=robot,nodes=_g_initial_nodes,node_velocities=_g_initial_vel,actual_simulation_time_s=sim.current_time)
        @_G_AUDIT.track_phase
        def _g_first_physics_step(phase):
            sim.step(render=False)
            robot.update(DT)
            _G_AUDIT.note_executed_step(phase,1,DT)
            if curtain_deformable is None:
                nodes = velocities = np.zeros((0,3),dtype=np.float32)
            else:
                nodes, velocities, _ = read_runtime(curtain_deformable)
            _G_AUDIT.observe(robot=robot,poses=body_poses(robot),nodes=nodes,node_velocities=velocities,
                phase=phase,physics_step=1,time_s=DT,actual_simulation_time_s=sim.current_time)
        _g_first_physics_step("initialization_physics_step")
        LOGGER.mark("FIRST_PHYSICS_STEP")''')
    source=replace_once(source,'        curtain_snapshot_times: list[float] = []',
        '        curtain_snapshot_times: list[float] = []\n        _g_node_velocity_snapshots = []')
    source=replace_once(source,'                curtain_snapshot_times.append(sim_step * DT)',
        '                curtain_snapshot_times.append((sim_step + 1) * DT)\n                _g_node_velocity_snapshots.append(node_velocities.copy())')
    source=replace_once(source,'                last_curtain_stretch = max_neighbor_stretch(current_nodes, node_rest)',
        '                last_curtain_stretch = _G_SCENE.max_neighbor_stretch(current_nodes, node_rest)')
    source=replace_once(source,'                last_geometry = geometry_snapshot(body_poses(robot), bbox_template, float(candidate_xy[0]), hand_name=ACTIVE_HAND_NAME)',
        '                _g_body_poses = body_poses(robot)\n                last_geometry = geometry_snapshot(_g_body_poses, bbox_template, float(candidate_xy[0]), hand_name=ACTIVE_HAND_NAME)')
    source=replace_once(source,'            telemetry_sample = bool(local_step == 0 or sim_step % EVALUATION_DECIMATION == 0)',
        '            _G_AUDIT.note_executed_step(phase,sim_step+2,(sim_step+2)*DT)\n            telemetry_sample = bool(local_step == 0 or sim_step % EVALUATION_DECIMATION == 0)')
    start=source.index('                if top_indices:\n                    last_curtain_top_drift =')
    end=source.index('                if pre_probe_curtain is not None:',start)
    source=source[:start]+'''                last_curtain_top_drift, last_curtain_anchor_drift = _G_SCENE.constraint_drifts(current_nodes, curtain_initial)
'''+source[end:]
    source=replace_once(source,'            curtain_top_drift = last_curtain_top_drift',
        '''            if _G_AUDIT.should_sample(sim_step,local_step):
                if not telemetry_sample:
                    _g_body_poses = body_poses(robot)
                if curtain_deformable is None:
                    _g_audit_nodes = _g_audit_velocities = np.zeros((0,3),dtype=np.float32)
                elif telemetry_sample:
                    _g_audit_nodes, _g_audit_velocities = current_nodes, node_velocities
                else:
                    _g_audit_nodes, _g_audit_velocities, _ = read_runtime(curtain_deformable)
                _G_AUDIT.observe(robot=robot,poses=_g_body_poses,nodes=_g_audit_nodes,node_velocities=_g_audit_velocities,
                    phase=phase,physics_step=sim_step+2,time_s=(sim_step+2)*DT,actual_simulation_time_s=sim.current_time)
                _g_tube_reason = _G_ACTION_GUARD.observe(_G_AUDIT.last_sample)
                if _g_tube_reason is not None and external_safety_reason is None:
                    external_safety_reason = _g_tube_reason
            curtain_top_drift = last_curtain_top_drift''')
    source=replace_once(source,'        def execute_step(',
        '''        def _g_capture_boundary(phase):
            if curtain_deformable is None:
                nodes = velocities = np.zeros((0,3),dtype=np.float32)
            else:
                nodes, velocities, _ = read_runtime(curtain_deformable)
            _G_AUDIT.observe(robot=robot,poses=body_poses(robot),nodes=nodes,node_velocities=velocities,
                phase=phase,physics_step=sim_step+1,time_s=(sim_step+1)*DT,actual_simulation_time_s=sim.current_time,boundary=True)

        def execute_step(''')
    for phase_function in ('run_walk_to_waypoint','run_stand','run_upper_stand'):
        source=replace_once(source,f'        def {phase_function}(',f'        @_G_AUDIT.track_phase\n        def {phase_function}(')
    source=replace_once(source,'                    final_geometry = geometry_snapshot(',
        '                    _g_capture_boundary("responsive_traverse_stand")\n                    final_geometry = geometry_snapshot(')
    for recovery_phase in ('physical_post_retract_stand','responsive_release_stand'):
        original=f'            post_retract_stand = run_stand("{recovery_phase}", ARGS.stand_duration)'
        source=replace_once(source,original,original+f'\n            _g_capture_boundary("{recovery_phase}")')
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
        '        _g_capture_boundary("physical_probe_hold")\n'
        '        _g_tube_entry_reason = _G_ACTION_GUARD.begin(_G_AUDIT.last_sample)\n'
        '        if _g_tube_entry_reason is not None and external_safety_reason is None:\n'
        '            external_safety_reason = _g_tube_entry_reason\n'
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
        _G_AUDIT.finish(complete=True,expected_total_steps=sim_step+1,final_actual_simulation_time_s=sim.current_time)
        _G_WRITE_JSON(_G_RUN_ROOT/'g_action_tube_runtime.json',_G_ACTION_GUARD.evidence())
        _g_finished_result = _G_RECORDER.finish(probe_result=clean(physical_probe_result),scene_metadata=_G_SCENE.evaluation_metadata,
            physics_valid=bool(_G_PLAN.get("physics_valid",False)),source_probe_valid=bool(probe_valid),fall_reason=fall_reason,
            external_safety_reason=external_safety_reason,events=clean(events),telemetry=clean(rows),
            evaluation_arrays={"nodes_world":curtain_snapshots,"node_velocities_world":_g_node_velocity_snapshots,
                "time_s":curtain_snapshot_times,"runtime_rest_positions":runtime_rest,
                "requested_fixed_indices":[] if curtain_info is None else curtain_info.requested_fixed_vertex_indices},
            rule_decision=_G_RULE_DECISION,logger=LOGGER)
        _G_POST_FINALIZE_CLEANUP(sim)
        return _g_finished_result
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
    randomization=plan.get('g1_reset_randomization',{})
    actual_y=float(robot.get('reset_root_y',plan.get('reset_root_y',-.004)))+float(randomization.get('frozen_robot_delta_y_m',0.))
    actual_yaw=float(robot.get('reset_root_yaw',plan.get('reset_root_yaw',-.004)))+float(randomization.get('frozen_robot_delta_yaw_rad',0.))
    # AppLauncher flags may be supplied by queue; all action/safety arguments
    # below are last and authoritative. No selected network controls collection.
    args=list(extra)+['--onnx',str(FALCON_ONNX),'--run-root',str(run_root),'--scenario','nominal','--seed',str(plan['seed']),
        '--probe-side',side,'--high-level-policy','r1','--condition-id',str(spec['scene_id']),
        '--curtain-length',str(g['height_m']),'--curtain-top-z',str(geom_top),
        '--reset-root-y',str(actual_y),'--reset-root-yaw',str(actual_yaw),
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
    action=action_spec(plan)
    if plan.get('followup_action_id')!=action['action_id'] or plan.get('label_contract')!=action['label_contract']:
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


def assert_runtime_plan_matches(frozen,plan,*,verified_gate_seeds):
    """Only an independently established admission flag may be promoted.

    Callers supply seeds only after validating the actual gate files, hashes,
    physical configuration and source evidence. No experiment field is mutable.
    """
    if frozen is None:raise RuntimeError('Runtime trial is absent from frozen campaign')
    if plan.get('physics_valid') is not True or len(set(verified_gate_seeds))<3:
        raise RuntimeError('Runtime physics_valid requires three independently verified gate seeds')
    for key,value in frozen.items():
        if key=='physics_valid' and value is False and plan.get(key) is True:continue
        if plan.get(key)!=value:raise RuntimeError('Runtime plan differs from frozen campaign: '+key)


def post_finalize_cleanup(sim,recorder,body_audit,run_root):
    """Suppress IsaacLab's interactive stop loop only after complete recording."""
    run_root=Path(run_root)
    if not recorder.completed or not recorder.stream.closed or not body_audit.saved:
        raise RuntimeError('Post-finalize cleanup cannot precede recorder completion/close')
    for name in ('summary.json','g_outcome.json','g_body_collision_capture.json','g_action_tube_runtime.json'):
        if not (run_root/name).is_file():raise RuntimeError('Post-finalize cleanup missing completed artifact: '+name)
    outcome=json.loads((run_root/'g_outcome.json').read_text(encoding='utf-8-sig'))
    capture=json.loads((run_root/'g_body_collision_capture.json').read_text(encoding='utf-8-sig'))
    if outcome.get('status')!='G_FIXED_ACTION_FINISHED' or capture.get('episode_finalized') is not True:
        raise RuntimeError('Post-finalize cleanup requires finalized outcome and actual history')
    sim._disable_app_control_on_stop_handle=True
    write_json(run_root/'post_finalize_cleanup.json',dict(protocol='G_POST_FINALIZE_DISABLE_INTERACTIVE_STOP_RENDER_V1',
        applied_after_summary_and_finalize=True,recorders_closed=True,body_history_finalized=True,
        final_physics_step=capture.get('execution_anchors',{}).get('last_physics_step'),
        rollout_or_thresholds_changed=False,scope='Only prevents interactive rendering during final sim.stop/APP.close'))


def verify_empty_evidence(plan):
    from physics_admission import empty_dependency
    campaign=CAMPAIGN_V4;campaign_sha=CAMPAIGN_V4_SHA256
    if plan.get('primary_protocol')==V5_PROTOCOL:
        pointer=json.loads((ROOT/'research/round4070_comprehensive_traversability/CURRENT_CAMPAIGN.json').read_text(encoding='utf-8-sig'))
        if pointer.get('primary_protocol')!=V5_PROTOCOL or pointer.get('manifest')!='campaign_manifest_v5.json':
            raise RuntimeError('V5 must be explicitly activated before launch')
        campaign=CAMPAIGN_V4.parent/pointer['manifest'];campaign_sha=pointer['sha256']
    if hashlib.sha256(campaign.read_bytes()).hexdigest()!=campaign_sha:
        raise RuntimeError('Frozen campaign changed; cannot establish EMPTY dependency')
    manifest=json.loads(campaign.read_text(encoding='utf-8-sig'))
    frozen=next((p for p in manifest['trial_plans'] if p['trial_id']==plan['trial_id']),None)
    gates=verify_gate_evidence(plan,manifest['scene_specs'][plan['scene_id']])
    assert_runtime_plan_matches(frozen,plan,verified_gate_seeds=[g['seed'] for g in gates])
    expected=empty_dependency(plan,manifest,campaign.parent)
    supplied=plan.get('empty_control_admission',{})
    if not expected.get('ready') or not supplied.get('ready'):
        raise RuntimeError('Matching hand/robot-variant independently validated EMPTY is required before G1 candidate launch')
    for key in ('reason','trial_id','path','audit_sha256'):
        if supplied.get(key)!=expected.get(key):raise RuntimeError('EMPTY admission evidence changed or does not match actual completed trial: '+key)
    if expected.get('path'):
        content=(Path(expected['path'])/'g_collision_audit.json').read_bytes()
        digest=hashlib.sha256(content).hexdigest()
        if digest!=expected['audit_sha256']:raise RuntimeError('EMPTY audit changed during launch admission')
        archive=CAMPAIGN_V4.parent/'runtime_sources'/f'empty_collision_audit_{digest}.json'
        archive.parent.mkdir(parents=True,exist_ok=True)
        if archive.exists():
            if archive.read_bytes()!=content:raise RuntimeError('Immutable EMPTY audit archive conflict')
        else:archive.write_bytes(content)
        expected['immutable_audit_path']=str(archive)
    return {**expected,'frozen_campaign_path':str(campaign),'frozen_campaign_sha256':campaign_sha}


def scene_query_required(plan,spec):
    return bool(plan.get('primary_protocol')==V5_PROTOCOL and
                (spec.get('wall') is not None or spec.get('geometry',{}).get('kind')=='rigid_board'))


def configure_scene_query(cfg,required):
    """Enable only the native query manager, before SimulationContext exists."""
    before=clean(cfg.to_dict())
    if required:cfg.enable_scene_query_support=True
    after=clean(cfg.to_dict())
    key='enable_scene_query_support'
    if {k:v for k,v in before.items() if k!=key}!={k:v for k,v in after.items() if k!=key}:
        raise RuntimeError('Scene-query configuration unexpectedly changed another simulation setting')
    if after.get(key) is not bool(required):
        raise RuntimeError('Unexpected default scene-query configuration for this frozen scene')
    return dict(schema='round4070-g.scene_query_configuration.v1',required_for_actual_wall_evidence=bool(required),
        configuration_before=before,configuration_after=after,
        changed_configuration_fields=[k for k in sorted(set(before)|set(after)) if before.get(k)!=after.get(k)],
        scope='Native scene-query facilities only; no solver, contact, dt, policy, action or scene changes',
        model_input=False,configured_before_simulation_context=True)


def record_scene_query_configuration(sim,cfg,record,proof,run_root):
    actual=sim.get_physics_context().get_enable_scene_query_support()
    if actual not in (True,False):raise RuntimeError('Native scene-query getter did not return a boolean')
    actual=bool(actual)
    if actual!=record['required_for_actual_wall_evidence'] or actual!=bool(cfg.enable_scene_query_support):
        raise RuntimeError('Native scene-query flag differs from the requested actual configuration')
    record={**record,'actual_native_readback_enabled':actual,
        'readback_api':'SimulationContext.get_physics_context().get_enable_scene_query_support',
        'readback_stage':'AFTER_SIMULATION_CONTEXT_BEFORE_SCENE_CONSTRUCTION',
        'source_sha256':proof['runtime_source_archive']['g1_trial.py']['sha256']}
    path=Path(run_root)/'g_scene_query_configuration.json'
    if path.exists():raise FileExistsError('Never replace actual scene-query configuration evidence')
    write_json(path,record)
    proof['scene_query_configuration_evidence']=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        requested_enabled=record['required_for_actual_wall_evidence'],actual_enabled=actual)
    write_json(Path(run_root)/'g_runner_contract.json',proof)
    return record


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
    action_v2=load_plan_contract(plan) if plan.get('primary_protocol')==V5_PROTOCOL else None
    randomization=ResetRandomization(plan,args.run_root)
    falcon_sha256=hashlib.sha256(FALCON_ONNX.read_bytes()).hexdigest()
    if falcon_sha256!=EXPECTED_ONNX_SHA256:raise RuntimeError('Frozen FALCON ONNX checksum mismatch; no simulator launch allowed')
    source=SOURCE.read_text(encoding='utf-8-sig');patched=patch_source(source)
    argv,offset=frozen_args(plan,spec,args.run_root,extra)
    sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
    args.run_root.mkdir(parents=True,exist_ok=True)
    proof=dict(original_source=str(SOURCE),original_source_sha256=sha(source),patched_source_sha256=sha(patched),
               original_source_file_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               falcon_onnx_path=str(FALCON_ONNX),falcon_onnx_sha256=falcon_sha256,falcon_hash_check='LAUNCH_TIME_ASSERTED',
               action=action_spec(plan),source_edit_mode='exact checked in-memory hooks; original file unchanged',argv=argv,
               physics_dt_s=.005,control_decimation=4,geometry_decimation=20,record_decimation=4,
               collision_audit_record_decimation=1 if action_v2 else 4,collision_audit_model_input=False,
               primary_protocol=plan['primary_protocol'],g1_reset_randomization=plan['g1_reset_randomization'],
               dry_run=args.dry_run,followup_policy_uses_classifier=False,
               scene_query_support_requested=scene_query_required(plan,spec),
               scene_query_configuration_scope='Native query manager enabled only for V5 actual-wall evidence; no physics/controller parameter changes')
    write_json(args.run_root/'g_runner_contract.json',proof)
    if args.dry_run:
        print(json.dumps(proof,indent=2));return 0
    proof['verified_physics_gate_evidence']=verify_gate_evidence(plan,spec)
    proof['verified_empty_control_admission']=verify_empty_evidence(plan)
    proof['recording_source_sha256']=hashlib.sha256((Path(__file__).parent/'g1_recording.py').read_bytes()).hexdigest()
    write_json(args.run_root/'g_runner_contract.json',proof)
    # Save the exact executable source once per version, in new G outputs only.
    source_path=ROOT/'outputs/round4070_comprehensive_traversability/runtime_sources'/f'{sha(patched)}.py'
    source_path.parent.mkdir(parents=True,exist_ok=True)
    if not source_path.exists():source_path.write_text(patched,encoding='utf-8')
    proof['runtime_source_archive']={}
    helpers=['g1_trial.py','g1_recording.py','g1_audit.py','g1_randomization.py','g1_action_contract_v2.py','physics_native_bind.py',
             'physics_history.py','collision_geometry.py','collision_clearance.py','physics_scene_builder.py','physics_scenes.py','physics_admission.py']
    if (Path(__file__).parent/'physics_wall_evidence.py').is_file():helpers.append('physics_wall_evidence.py')
    for name in helpers:
        content=(Path(__file__).parent/name).read_bytes();digest=hashlib.sha256(content).hexdigest()
        archived=source_path.parent/f'{Path(name).stem}_{digest}.py'
        if not archived.exists():archived.write_bytes(content)
        proof['runtime_source_archive'][name]=dict(path=str(archived),sha256=digest)
    write_json(args.run_root/'g_runner_contract.json',proof)
    recorder=Recorder(args.run_root,plan,spec,sha(patched))
    body_audit=BodyCollisionRecorder(args.run_root,nominal_hz=200 if action_v2 else 50,action_contract=action_v2)
    action_guard=ActionTubeGuard(action_v2)
    query_record={}
    def configure_query(cfg):
        query_record.update(configure_scene_query(cfg,proof['scene_query_support_requested']))
    def verify_live_scene_source():
        for name in ('physics_scene_builder.py','physics_scenes.py'):
            if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=proof['runtime_source_archive'][name]['sha256']:
                raise RuntimeError('Scene construction source changed during simulator startup: '+name)
    namespace={'__name__':'_g_frozen_runner','__file__':str(SOURCE),'_G_RECORDER':recorder,
               '_G_SPEC':spec,'_G_PLAN':plan,'_G_UPPER_OFFSET':offset,'_G_VERIFY_SCENE_SOURCE':verify_live_scene_source,
               '_G_RANDOMIZATION':randomization,'_G_AUDIT':body_audit,'_G_ACTION_GUARD':action_guard,
               '_G_CONFIGURE_SCENE_QUERY':configure_query,
               '_G_RECORD_SCENE_QUERY':lambda sim,cfg:record_scene_query_configuration(sim,cfg,query_record,proof,args.run_root),
               '_G_WRITE_JSON':write_json,'_G_RUN_ROOT':args.run_root,
               '_G_POST_FINALIZE_CLEANUP':lambda sim:post_finalize_cleanup(sim,recorder,body_audit,args.run_root)}
    sys.argv=[str(SOURCE),*argv];sys.path.insert(0,str(SOURCE.parent))
    try:
        exec(compile(patched,str(source_path),'exec'),namespace)
        return namespace['main']()
    finally:
        body_audit.finish(complete=False)
        recorder.close_incomplete()


if __name__=='__main__':raise SystemExit(main())
