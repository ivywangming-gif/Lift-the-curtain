"""Incrementally import actual Round G histories, with frozen causal splits.

This never launches simulation, trains, modifies a source trial, or invents a
label for an unfinished trial. An output snapshot used by training must be kept
under its own --output path. Historical F stays a separate response-only source.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from lift_curtain.control.falcon_contract import OFFICIAL_POLICY_JOINT_ORDER as JOINTS, UPPER_JOINTS
from g1_recording import ACTION_ID, CONTRACT, PROBE_PHASES, ONLINE_LABEL_REVISION, action_spec, assign_label, clean, write_json, wall_crossing_audited
from physics_scenes import physical_hash
from g1_randomization import PRIMARY_PROTOCOL
from g1_action_contract_v2 import PRIMARY_PROTOCOL as V5_PROTOCOL,ACTION_ID as V2_ACTION,LABEL_CONTRACT as V2_CONTRACT,validate_contract
from g1_trial import assert_runtime_plan_matches

LABEL_VALIDITY_VERSION='G_BODY_COLLISION_VALIDITY_V1'
ACTION_EVIDENCE_REVISION=ONLINE_LABEL_REVISION
V5_ACTION_EVIDENCE_REVISION='G_DISCRETE_200HZ_ACTION_TUBE_V2'

OUT=ROOT/'outputs/round4070_comprehensive_traversability'
REPORT=ROOT/'research/round4070_comprehensive_traversability'
VECTORS=[('q',JOINTS,'rad'),('qdot',JOINTS,'rad/s'),('q_cmd',JOINTS,'rad'),('effort',JOINTS,'N.m applied_torque proxy'),
         ('previous_control_action',JOINTS,'normalized policy action'),('current_policy_action',JOINTS,'normalized policy action'),
         ('left_hand_fk_b','xyz','m'),('right_hand_fk_b','xyz','m'),
         ('left_hand_delta_initial_b','xyz','m'),('right_hand_delta_initial_b','xyz','m'),
         ('left_hand_velocity_initial_b','xyz','m/s'),('right_hand_velocity_initial_b','xyz','m/s'),
         ('imu_rpy',('roll','pitch','yaw'),'rad'),('imu_angular_velocity_b','xyz','rad/s'),
         ('imu_projected_gravity_b','xyz','unit vector'),('imu_specific_force_b','xyz','m/s2 simulator specific-force proxy'),
         ('base_velocity_b','xyz','m/s'),('commanded_upper_reference',UPPER_JOINTS,'rad'),
         ('planned_followup_upper_reference',UPPER_JOINTS,'rad'),('commanded_body_velocity',('vx','vy','wz'),'m/s,m/s,rad/s')]
SCALARS=[('imu_specific_force_available','bool'),('command_progress/fraction','unitless'),('command_walk_mode','bool'),
         ('probe_side/right','bool'),('followup_hand/right','bool'),('elapsed_s','s'),('dt_s','s'),('control_age_s','s'),
         ('followup_command/cruise_speed_mps','m/s'),('followup_command/vx_max_mps','m/s'),
         ('followup_command/max_segments','count'),('followup_command/segment_step_m','m'),
         ('followup_command/segment_timeout_s','s'),('followup_command/release_s','s')]


def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def stable_int(value):return int(hashlib.sha256(str(value).encode()).hexdigest()[:12],16)


def actual_launch_inventory(trials_root):
    """Count actual launch artifacts independently of usable episode metadata."""
    rows=[]
    for launch_path in sorted(Path(trials_root).rglob('campaign_launch.json')):
        directory=launch_path.parent;errors=[]
        def optional(name):
            path=directory/name
            if not path.exists():return {}
            try:
                value=load(path)
                if not isinstance(value,dict):raise ValueError('Expected JSON object')
                return value
            except (ValueError,OSError) as exc:
                errors.append(name+': '+type(exc).__name__+': '+str(exc));return {}
        launch=optional('campaign_launch.json');outcome=optional('g_outcome.json');watchdog=optional('watchdog_summary.json')
        metadata_present=(directory/'g_trial_metadata.json').is_file()
        final_present=outcome.get('status')=='G_FIXED_ACTION_FINISHED'
        outcome_present=(directory/'g_outcome.json').is_file()
        state='FINALIZED_OUTCOME_PRESENT' if final_present else 'STOPPED_WITHOUT_FINAL_OUTCOME' if watchdog else 'OPEN_OR_UNFINALIZED'
        missing=[]
        if not metadata_present:missing.append('g_trial_metadata.json')
        if not final_present:missing.append('FINALIZED_g_outcome.json')
        stderr=directory/'process_stderr.log';tail=[]
        if stderr.is_file():tail=[line for line in stderr.read_text(encoding='utf-8-sig',errors='replace').splitlines() if line.strip()][-12:]
        error=outcome.get('error') or watchdog.get('timeout_reason') or (tail[-1] if tail and not final_present else None)
        source_files={p.name:sha(p) for p in (launch_path,directory/'watchdog_summary.json',directory/'g_outcome.json',stderr) if p.is_file()}
        rows.append(dict(attempt_path=rel(directory),trial_id=launch.get('trial_id',directory.parent.name),attempt_id=directory.name,
            kind=launch.get('kind'),launch_utc=launch.get('utc'),status=state,metadata_present=metadata_present,
            outcome_present=outcome_present,final_outcome_present=final_present,outcome_status=outcome.get('status'),
            missing_episode_artifacts=missing,error=error,stderr_tail='\n'.join(tail) if tail else None,read_errors=errors,
            watchdog_exit_code=watchdog.get('exit_code'),watchdog_timeout_reason=watchdog.get('timeout_reason'),
            last_valid_stage=watchdog.get('last_valid_stage'),last_stage_status=watchdog.get('last_stage_status'),source_sha256=source_files))
    return dict(schema='round4070-g.actual_launch_inventory.v1',scope='All actual campaign_launch directories, including pre-App failures; not a model episode index',
        actual_launch_count=len(rows),finalized_outcome_count=sum(r['final_outcome_present'] for r in rows),
        no_metadata_count=sum(not r['metadata_present'] for r in rows),no_final_outcome_count=sum(not r['final_outcome_present'] for r in rows),
        incomplete_launch_count=sum(bool(r['missing_episode_artifacts']) for r in rows),
        stopped_without_final_outcome_count=sum(r['status']=='STOPPED_WITHOUT_FINAL_OUTCOME' for r in rows),
        actual_launches=rows,incomplete_launches=[r for r in rows if r['missing_episode_artifacts']])


def archive_derived_evidence(record,output):
    """Keep a byte-exact content-addressed copy for frozen training snapshots."""
    if not record or not record.get('source_evidence_path'):return record
    original=ROOT/record['source_evidence_path'];digest=record['source_evidence_sha256']
    content=original.read_bytes()
    if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('DERIVED_EVIDENCE_CHANGED_DURING_IMPORT')
    target=output/'evidence'/f'{original.stem}_{digest}.json';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.read_bytes()!=content:raise ValueError('CONTENT_ADDRESSED_EVIDENCE_CONFLICT')
    else:target.write_bytes(content)
    return {**record,'original_derived_evidence_path':rel(original),'source_files_base':rel(original.parent),
            'source_evidence_path':rel(target)}


def save_csv(path,rows,fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    temp.replace(path)


def schema():
    names=[];blocks={};units={}
    for key,suffixes,unit in VECTORS:
        start=len(names);names.extend(key+'/'+s for s in suffixes);blocks[key]=list(range(start,len(names)));units[key]=unit
    for name,unit in SCALARS:
        blocks[name]=[len(names)];names.append(name);units[name]=unit
    return names,blocks,units


def legal_sequence(rows,action,cutoff):
    selected=[r for r in rows if r.get('phase') in PROBE_PHASES and float(r['time_s'])<=cutoff+1e-9]
    if len(selected)<2:raise ValueError('INSUFFICIENT_REAL_PROBE_OBSERVATIONS')
    t=np.array([r['time_s'] for r in selected],dtype=float)
    if not np.isfinite(t).all() or not np.all(np.diff(t)>0):raise ValueError('NONMONOTONE_REAL_TIMESTAMPS')
    selected=[r for r in selected if float(r['time_s'])-t[0]<=2.0+1e-9]
    t=np.array([r['time_s'] for r in selected],dtype=float)
    effort_sources=sorted({r.get('effort_source','UNAVAILABLE') for r in selected})
    if effort_sources!=['applied_torque']:raise ValueError('EFFORT_SOURCE_NOT_VALIDATED:'+str(effort_sources))
    arrays=[];raw={}
    for key,suffixes,_ in VECTORS:
        if key=='imu_specific_force_b':
            # Explicit missing sensor mask: zero is never represented as a
            # measured value. Original unavailable rows remain in raw JSONL.
            a=np.array([r[key] if r.get(key) is not None else [0.]*3 for r in selected],float)
        else:a=np.array([r[key] for r in selected],float)
        if a.shape!=(len(t),len(suffixes)) or not np.isfinite(a).all():raise ValueError('INVALID_VECTOR:'+key)
        arrays.append(a);raw[key]=a.astype(np.float32)
    scalars=np.array([[float(r.get('imu_specific_force_b') is not None),float(r['command_progress']),float(r['command_walk_mode']),
                      float(r['probe_side']=='right'),float(r['followup_hand']=='right'),float(r['time_s'])-t[0],
                      0. if i==0 else t[i]-t[i-1],float(r['time_s'])-float(r['control_command_time_s']),
                      action['cruise_speed_mps'],action['vx_max_mps'],action['maximum_forward_segments'],
                      action['maximum_segment_step_m'],action['segment_timeout_s'],action['release_s']] for i,r in enumerate(selected)],float)
    arrays.append(scalars);seq=np.concatenate(arrays,axis=1).astype(np.float32)
    if not np.isfinite(seq).all():raise ValueError('NONFINITE_FEATURES')
    previous_verified=0
    for a,b in zip(selected,selected[1:]):
        delta=int(b['control_update_index'])-int(a['control_update_index'])
        if delta<0:raise ValueError('CONTROL_INDEX_BACKWARDS')
        if delta==1:
            if not np.allclose(b['previous_control_action'],a['current_policy_action'],atol=1e-7,rtol=0):
                raise ValueError('PREVIOUS_ACTION_TEMPORAL_MISMATCH')
            previous_verified+=1
    if np.any(scalars[:,7]<0) or np.any(scalars[:,7]>.0250001):raise ValueError('CONTROL_TIMESTAMP_ALIGNMENT')
    expected=raw['q_cmd'][:,15:]
    if not np.allclose(raw['commanded_upper_reference'],expected,atol=1e-7):raise ValueError('UPPER_COMMAND_NOT_ACTUALLY_EXECUTED')
    return seq,t,np.array([r['phase'] for r in selected]),raw,dict(effort_sources=effort_sources,
        true_previous_action_adjacent_updates_verified=previous_verified,imu_acceleration_missing_rows=int(np.sum(scalars[:,0]==0)),
        dt_median_s=float(np.median(np.diff(t))),dt_min_s=float(np.min(np.diff(t))),dt_max_s=float(np.max(np.diff(t))),
        sample_count=len(t),duration_s=float(t[-1]-t[0]),actual_hz=1/float(np.median(np.diff(t))))


def independently_verify_wall_evidence(directory,geometry,audit=None):
    """Replay actual wall entity/source evidence; old no-wall raw is compatible."""
    if geometry.get('wall') is None and geometry.get('wall_evidence') is None:return
    from physics_wall_evidence import verified_wall_for_audit
    sources={};actual=verified_wall_for_audit(directory,geometry,sources)
    if audit is not None:
        if any(audit.get('source_files',{}).get(name)!=digest for name,digest in sources.items()):
            raise ValueError('WALL_ENTITY_AUDIT_SOURCE_HASHES_MISSING')
        expected=dict(status='VERIFIED' if actual is not None else 'NOT_APPLICABLE',actual_geometry=actual)
        if audit.get('actual_wall_entity_audit')!=expected:
            raise ValueError('ACTUAL_WALL_ENTITY_AUDIT_REPLAY_MISMATCH')


def frozen_evidence(directory,meta,plan):
    proof_path=directory/'g_runner_contract.json';proof=load(proof_path)
    frozen_plan=None
    if proof.get('action')!=action_spec(plan) or meta.get('action_contract')!=action_spec(plan):
        raise ValueError('ACTION_SPEC_RUNTIME_METADATA_MISMATCH')
    if proof.get('dry_run') or proof.get('followup_policy_uses_classifier') is not False:
        raise ValueError('ACTION_COLLECTION_CONTRACT_NOT_VERIFIED')
    if proof.get('patched_source_sha256')!=meta.get('runtime_source_sha256'):
        raise ValueError('RUNTIME_SOURCE_METADATA_MISMATCH')
    source=OUT/'runtime_sources'/f"{proof['patched_source_sha256']}.py"
    normalized=hashlib.sha256(source.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
    if normalized!=proof['patched_source_sha256']:raise ValueError('EXECUTED_SOURCE_ARCHIVE_HASH_MISMATCH')
    recorder=OUT/'runtime_sources'/f"g1_recording_{proof['recording_source_sha256']}.py"
    if sha(recorder)!=proof['recording_source_sha256']:raise ValueError('RECORDER_SOURCE_ARCHIVE_HASH_MISMATCH')
    if plan.get('primary_protocol') in (PRIMARY_PROTOCOL,V5_PROTOCOL):
        v5=plan['primary_protocol']==V5_PROTOCOL
        if proof.get('primary_protocol')!=plan['primary_protocol'] or proof.get('g1_reset_randomization')!=plan.get('g1_reset_randomization'):
            raise ValueError('PRIMARY_RUNTIME_RANDOMIZATION_CONTRACT_MISMATCH')
        if proof.get('collision_audit_record_decimation')!=(1 if v5 else 4) or proof.get('geometry_decimation')!=20 or proof.get('collision_audit_model_input') is not False:
            raise ValueError('PRIMARY_AUDIT_AND_SAFETY_CLOCK_CONTRACT_MISMATCH')
        archives=proof.get('runtime_source_archive',{})
        required_helpers=['g1_trial.py','g1_recording.py','g1_audit.py','g1_randomization.py','physics_scene_builder.py','physics_scenes.py','physics_admission.py']
        if v5:required_helpers+=['g1_action_contract_v2.py','physics_native_bind.py','physics_history.py','collision_geometry.py','collision_clearance.py']
        for name in required_helpers:
            archived=archives.get(name,{})
            if not archived.get('path') or sha(archived['path'])!=archived.get('sha256'):
                raise ValueError('PRIMARY_RUNTIME_HELPER_ARCHIVE_MISMATCH:'+name)
        geometry_path=directory/'g_robot_collision_geometry.json'
        if geometry_path.exists() and load(geometry_path).get('exporter_source_sha256')!=archives['g1_audit.py']['sha256']:
            raise ValueError('COLLIDER_EXPORTER_SOURCE_DIFFERED_FROM_LAUNCH_ARCHIVE')
        if v5 and geometry_path.exists():
            independently_verify_wall_evidence(directory,load(geometry_path))
            if load(geometry_path).get('native_property_query_inventory',{}).get('source_sha256')!=archives['physics_native_bind.py']['sha256']:
                raise ValueError('NATIVE_QUERY_SOURCE_DIFFERED_FROM_LAUNCH_ARCHIVE')
            native_path=directory/'g_native_geometry_verified.json'
            if native_path.exists() and load(native_path).get('binder_source_sha256')!=archives['physics_native_bind.py']['sha256']:
                raise ValueError('NATIVE_BINDER_SOURCE_DIFFERED_FROM_LAUNCH_ARCHIVE')
        empty=proof.get('verified_empty_control_admission',{})
        if v5:
            validate_contract(plan['action_contract_v2'])
            campaign=Path(empty['frozen_campaign_path'])
            if sha(campaign)!=empty['frozen_campaign_sha256']:raise ValueError('FROZEN_V5_CAMPAIGN_HASH_MISMATCH')
            manifest=load(campaign)
            if manifest.get('g1_primary_protocol')!=V5_PROTOCOL or manifest.get('action_contract_v2_sha256')!=plan.get('action_contract_v2_sha256') or manifest.get('action_contract_v2')!=plan.get('action_contract_v2'):
                raise ValueError('FROZEN_V5_ACTION_CONTRACT_MISMATCH')
            frozen_plan=next((p for p in manifest['trial_plans'] if p['trial_id']==plan['trial_id']),None)
            if frozen_plan is None:raise ValueError('FROZEN_V5_TRIAL_PLAN_MISSING')
        if not empty.get('ready'):raise ValueError('PRIMARY_MATCHING_EMPTY_ADMISSION_MISSING')
        if plan.get('scene_id')=='EMPTY':
            if empty.get('reason')!='EMPTY_SELF_VALIDATION':raise ValueError('EMPTY_SELF_VALIDATION_PROVENANCE_MISMATCH')
        else:
            if empty.get('reason')!='MATCHING_EMPTY_VALIDATED' or not empty.get('immutable_audit_path'):
                raise ValueError('PRIMARY_MATCHING_EMPTY_PROVENANCE_MISSING')
            if sha(empty['immutable_audit_path'])!=empty.get('audit_sha256'):
                raise ValueError('PRIMARY_MATCHING_EMPTY_AUDIT_ARCHIVE_MISMATCH')
    seeds=set();hashes={}
    for item in proof.get('verified_physics_gate_evidence',[]):
        path=Path(item['path']);path=path if path.is_absolute() else ROOT/path
        if sha(path)!=item['sha256']:raise ValueError('PHYSICS_EVIDENCE_HASH_CHANGED')
        gate=load(path)
        if not gate.get('physics_valid') or gate.get('physical_config_id')!=plan['physical_config_id'] or not all(gate.get('checks',{}).values()):
            raise ValueError('PHYSICS_EVIDENCE_INVALID')
        seeds.add(gate['seed']);hashes[rel(path)]=item['sha256']
        source_record=Path(item['source_evidence_path'])
        if sha(source_record)!=item['source_evidence_sha256']:raise ValueError('PHYSICS_SOURCE_EVIDENCE_HASH_CHANGED')
    if len(seeds)<3:raise ValueError('THREE_ACTUAL_GATE_SEEDS_REQUIRED')
    if frozen_plan is not None:
        try:assert_runtime_plan_matches(frozen_plan,plan,verified_gate_seeds=seeds)
        except RuntimeError as exc:raise ValueError('FROZEN_V5_TRIAL_PLAN_MISMATCH:'+str(exc)) from exc
    return dict(runtime_contract_sha256=sha(proof_path),runtime_source_sha256=proof['patched_source_sha256'],
                recorder_source_sha256=proof['recording_source_sha256'],physics_gate_seeds=sorted(seeds),physics_gate_hashes=hashes)


def collision_admission(directory,primary_protocol=PRIMARY_PROTOCOL):
    """Read an independent derived audit; never modify online/raw outcomes."""
    path=directory/'g_collision_audit.json'
    if not path.exists():return dict(status='NOT_VERIFIED',reason='VALIDITY_PENDING_BODY_COLLISION_AUDIT')
    audit=load(path)
    if audit.get('schema')!='round4070-g.body_collision_audit.v1':raise ValueError('COLLISION_AUDIT_SCHEMA_MISMATCH')
    required={'g_robot_collision_geometry.json','g_outcome.json'}
    if primary_protocol==V5_PROTOCOL:
        required|={'g_body_collision_capture.json','g_body_collision_history_index.json','g_body_collision_initial_state.npz',
                   'g_native_geometry_verified.json','g_action_tube_runtime.json','g_trial_metadata.json'}
        capture=load(directory/'g_body_collision_capture.json')
        index=load(directory/'g_body_collision_history_index.json')
        if capture.get('history_layout')!='CHUNKED_NPZ_V2' or capture.get('nominal_hz')!=200:
            raise ValueError('V5_REQUIRES_ACTUAL_200HZ_CHUNK_HISTORY')
        if sha(directory/'g_body_collision_history_index.json')!=capture.get('history_index_sha256'):
            raise ValueError('COLLISION_HISTORY_INDEX_HASH_MISMATCH')
        required|={chunk['path'] for chunk in index['chunks']}
    else:required.add('g_body_collision_history_audit_only.npz')
    sources=audit.get('source_files',{})
    if not required<=set(sources):raise ValueError('COLLISION_AUDIT_SOURCE_HASHES_MISSING')
    for name,digest in sources.items():
        target=(directory/name).resolve()
        if not target.is_relative_to(directory.resolve()):raise ValueError('COLLISION_AUDIT_SOURCE_ESCAPES_EPISODE')
        if sha(target)!=digest:raise ValueError('COLLISION_AUDIT_SOURCE_HASH_MISMATCH:'+name)
    status=audit.get('status')
    if status=='PHYSICS_INVALID':raise ValueError('G1_BODY_COLLISION_PHYSICS_INVALID:'+str(audit.get('reason','')))
    if status=='VERIFIED':
        if not all(audit.get(k) is True for k in ('penetration_free','all_robot_collision_bodies','probe_followup_release_complete','finite')) or int(audit.get('sample_count',0))<1:
            raise ValueError('COLLISION_VERIFIED_WITHOUT_REQUIRED_EVIDENCE')
        if primary_protocol==V5_PROTOCOL:
            from physics_history import verify_history
            independently_verify_wall_evidence(directory,load(directory/'g_robot_collision_geometry.json'),audit)
            if audit.get('initial_state_collision_audited') is not True or audit.get('geometry_checked_state_count')!=audit['sample_count']+1:
                raise ValueError('V5_INITIAL_RESET_STATE_COLLISION_NOT_AUDITED')
            verified=verify_history(directory,capture,load(directory/'g_robot_collision_geometry.json'))
            if verified.get('status')!='VERIFIED_HISTORY' or verified['sample_count']!=audit['sample_count']:
                raise ValueError('COLLISION_AUDIT_ACTUAL_HISTORY_COUNT_MISMATCH')
            if any(sources.get(name)!=digest for name,digest in verified['source_files'].items()):
                raise ValueError('COLLISION_AUDIT_STREAM_SOURCE_HASH_MISMATCH')
    elif status!='NOT_VERIFIED':raise ValueError('UNKNOWN_COLLISION_AUDIT_STATUS')
    return dict(status=status,reason=audit.get('reason',status),source_evidence_path=rel(path),source_evidence_sha256=sha(path),
                sample_count=audit.get('sample_count'),scope=audit.get('scope','ACTUAL_SAMPLED_BODY_COLLISION_AUDIT'),
                candidate_clearance=audit.get('candidate_clearance',{}),action_contract_audit=audit.get('action_contract_audit',{}))


def actual_collider_clearance(collision,scene,action):
    evidence=collision.get('candidate_clearance') or {}
    if evidence.get('status')!='VERIFIED':return False
    expected=float(scene['geometry']['center_x_m'])+.06+float(action['candidate_clear_margin_m'])
    numeric=np.asarray([evidence.get('threshold_x_m',np.nan),evidence.get('min_world_x_m',np.nan),evidence.get('time_s',np.nan)],float)
    if not np.isfinite(numeric).all() or abs(numeric[0]-expected)>1e-9:
        raise ValueError('INDEPENDENT_COLLIDER_CLEARANCE_THRESHOLD_MISMATCH')
    if evidence.get('actual_collision_geometry') is not True or evidence.get('phase_final_sample') is not True or evidence.get('phase')!='responsive_traverse_stand' or evidence.get('comparison')!='>=' or int(evidence.get('sample_index',-1))<0:
        raise ValueError('INDEPENDENT_COLLIDER_CLEARANCE_MISSING_FINAL_HOLD_EVIDENCE')
    # VERIFIED certifies the actual support computation, not that its value
    # crossed the threshold. A valid uncleared execution remains UNKNOWN.
    return bool(numeric[1]>=expected)


def conservative_action_label(label,reason):
    """V1 has no invariant lateral/height tube; a finite wall is not proof.

    Keep original online outcomes separately. Do not infer a counterfactual
    closed-loop failure from a withheld action or a blocked nominal corridor.
    """
    if label=='BLOCKED':
        return 'LABEL_UNKNOWN','FINITE_WALL_NO_CLOSED_LOOP_ACTION_IMPOSSIBILITY_PROOF'
    return label,reason


def original_online_label(outcome,action):
    """Validate historical online semantics without promoting them to truth."""
    revision=outcome.get('online_label_revision')
    if revision not in (None,ONLINE_LABEL_REVISION):raise ValueError('UNSUPPORTED_ORIGINAL_ONLINE_LABEL_REVISION')
    label,reason=assign_label(physics_valid=outcome['physics_valid'],runtime_valid=outcome.get('runtime_valid',False),
        probe_valid=outcome['probe_valid'],recovered=outcome.get('safe_recovery',False),
        clearance=outcome.get('whole_body_candidate_clearance',False),advance_steps=outcome.get('followup_advance_logged_steps',0),
        metadata=outcome.get('scene_evaluation_only',{}),action=action)
    if revision is None and reason=='FINITE_WALL_NO_CLOSED_LOOP_ACTION_IMPOSSIBILITY_PROOF':
        # Only the superseded source contract had this unsupported conclusion.
        # It is retained solely to validate immutable old raw records.
        return 'BLOCKED','INDEPENDENT_FIXED_WALL_COVERS_FORWARD_ACTION_CORRIDOR'
    return label,reason


def randomization_evidence(directory,plan):
    path=directory/'g_reset_randomization.json';record=load(path)
    expected=plan['g1_reset_randomization'];array_path=directory/record['arrays_file']
    if record.get('schema')!='G1_RNG_V1' or record.get('seed')!=plan['seed'] or record.get('spec_sha256')!=expected['spec_sha256']:
        raise ValueError('RNG_ACTUAL_EVIDENCE_METADATA_MISMATCH')
    if not record.get('readback_verified') or not record.get('robot_pose_readback_verified') or record.get('injection_stage')!='AFTER_SIM_RESET_BEFORE_FIRST_PHYSICS_STEP':
        raise ValueError('RNG_INJECTION_NOT_VERIFIED')
    if sha(array_path)!=record['arrays_sha256']:raise ValueError('RNG_ARRAY_FILE_HASH_MISMATCH')
    rng=np.random.default_rng(plan['seed']);draws=rng.uniform(-.001,.001,size=2)
    with np.load(array_path,allow_pickle=False) as arrays:
        applied=arrays['applied_node_velocities_world'];free=arrays['free_vertex_indices'];fixed=arrays['fixed_vertex_indices']
        if not np.array_equal(arrays['robot_draws_y_yaw'],draws):raise ValueError('RNG_ROBOT_DRAW_REPLAY_MISMATCH')
        replay=np.zeros_like(applied);replay[free]=rng.uniform(-.001,.001,size=(len(free),3)).astype(np.float32)
        if not np.array_equal(replay,applied) or np.any(applied[fixed]):raise ValueError('RNG_NODE_DRAW_REPLAY_MISMATCH')
        if not np.allclose(applied,arrays['readback_node_velocities_world'],atol=1e-8,rtol=0):raise ValueError('RNG_READBACK_MISMATCH')
        if hashlib.sha256(applied.astype('<f4').tobytes()).hexdigest()!=record['applied_array_sha256']:raise ValueError('RNG_ARRAY_HASH_MISMATCH')
    for key,expected_key in [('actual_robot_delta_y_m','frozen_robot_delta_y_m'),('actual_robot_delta_yaw_rad','frozen_robot_delta_yaw_rad')]:
        if record[key]!=expected[expected_key]:raise ValueError('RNG_PLAN_DRAW_MISMATCH')
    return dict(source_evidence_path=rel(path),source_evidence_sha256=sha(path),readback_verified=True,robot_pose_readback_verified=True,seed=plan['seed'])


def import_trial(directory,output,primary_protocol=PRIMARY_PROTOCOL):
    meta=load(directory/'g_trial_metadata.json');plan=meta['plan'];scene=meta['scene_config_evaluation_only']
    outcome=load(directory/'g_outcome.json');action=action_spec(plan);planned_trial=plan['trial_id']
    if not planned_trial or Path(planned_trial).name!=planned_trial:raise ValueError('UNSAFE_TRIAL_ID')
    if outcome.get('trial_id')!=planned_trial:raise ValueError('TRIAL_ID_MISMATCH')
    # A retry is a different actual episode, even when it shares a frozen plan.
    trial=planned_trial+'__'+directory.name if directory.name.startswith('attempt') else planned_trial
    if plan.get('physical_config_id')!=physical_hash(scene):raise ValueError('PHYSICAL_GROUP_HASH_MISMATCH')
    if outcome.get('label_contract')!=action['label_contract']:raise ValueError('WRONG_ACTION_LABEL_CONTRACT')
    if not outcome.get('physics_valid'):raise ValueError('PHYSICS_INVALID_OR_UNFINALIZED')
    if not outcome.get('probe_valid'):raise ValueError('PROBE_INVALID_OR_UNFINALIZED')
    provenance=frozen_evidence(directory,meta,plan)
    truth=outcome.get('scene_evaluation_only',{})
    if truth.get('physical_config_id')!=plan['physical_config_id']:raise ValueError('CONSTRUCTED_PHYSICS_METADATA_MISMATCH')
    derived,reason=original_online_label(outcome,action)
    if outcome.get('label')!=derived:raise ValueError('LABEL_RECOMPUTATION_MISMATCH')
    if bool(outcome.get('label_valid'))!=(derived!='LABEL_UNKNOWN'):raise ValueError('LABEL_VALID_FLAG_MISMATCH')
    if derived=='PASSABLE_FOR_ACTION' and truth.get('wall'):
        if not wall_crossing_audited(truth):raise ValueError('WALL_PASSABILITY_REQUIRES_ACTUAL_G1_NONPENETRATION_EVIDENCE')
        wall_audit=truth['g1_body_wall_nonpenetration_audit'];audit_path=Path(wall_audit['source_evidence_path'])
        audit_path=audit_path if audit_path.is_absolute() else ROOT/audit_path
        if sha(audit_path)!=wall_audit['source_evidence_sha256']:raise ValueError('G1_WALL_NONPENETRATION_EVIDENCE_HASH_MISMATCH')
    source=directory/'proprioception_50hz.jsonl'
    if outcome.get('legal_history_sha256')!=sha(source):raise ValueError('HISTORY_HASH_MISMATCH')
    with source.open(encoding='utf-8') as f:rows=[json.loads(s) for s in f if s.strip()]
    seq,t,phase,raw,audit=legal_sequence(rows,action,float(outcome['probe_cutoff_time_s']))
    names,_,_=schema();path=output/'episodes'/f'{trial}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(seq=seq,time=t,phase=phase,feature_names=np.array(names),**raw)
    if path.exists():
        with np.load(path,allow_pickle=False) as previous:
            if set(previous.files)!=set(payload) or any(not np.array_equal(previous[k],v) for k,v in payload.items()):
                raise ValueError('EXISTING_EXPORTED_EPISODE_DIFFERS_USE_NEW_SNAPSHOT')
    else:np.savez_compressed(path,**payload)
    robot=plan.get('robot',plan.get('robot_conditions',{}))
    control=bool(plan.get('is_control',False) or scene['geometry']['kind']=='empty')
    primary=plan.get('primary_protocol')==primary_protocol
    collision=collision_admission(directory,primary_protocol) if primary else dict(status='NOT_VERIFIED',reason='SUPERSEDED_PROTOCOL_NOT_CURRENT_PRIMARY')
    rng_evidence=randomization_evidence(directory,plan) if primary else None
    collision=archive_derived_evidence(collision,output);rng_evidence=archive_derived_evidence(rng_evidence,output)
    label=derived
    if primary and collision['status']=='VERIFIED':
        audited_truth=dict(truth)
        if truth.get('wall'):
            audited_truth['g1_body_wall_nonpenetration_audit']=dict(status='VERIFIED',all_robot_collision_bodies=True,
                probe_followup_release_complete=True,finite=True,penetration_free=True,
                sample_count=collision['sample_count'],source_evidence_path=collision['source_evidence_path'],
                source_evidence_sha256=collision['source_evidence_sha256'])
        hold=(outcome.get('physical_probe_result',{}).get('responsive_traverse') or {}).get('stand') or {}
        hold_stable=bool(hold.get('speed_gate_reached') and hold.get('finite_no_fall')
            and float(hold.get('final_planar_speed_mps',float('inf')))<=.06
            and outcome.get('fall_reason') is None and outcome.get('external_safety_reason') is None)
        label,reason=assign_label(physics_valid=True,runtime_valid=outcome.get('runtime_valid',False),
            probe_valid=True,recovered=outcome.get('safe_recovery',False),clearance=actual_collider_clearance(collision,scene,action) and hold_stable,
            advance_steps=outcome.get('followup_advance_logged_steps',0),metadata=audited_truth,action=action)
    if not control and collision['status']!='VERIFIED' and derived=='PASSABLE_FOR_ACTION':
        label='LABEL_UNKNOWN';reason='VALIDITY_PENDING_BODY_COLLISION_AUDIT'
    action_audit=collision.get('action_contract_audit',{})
    if primary_protocol==V5_PROTOCOL and primary:
        action_verified=action_audit.get('status')=='VERIFIED'
        action_valid=action_verified and action_audit.get('all_required_steps_valid') is True and action_audit.get('guard_violation') is None
        if not action_valid:
            label='LABEL_UNKNOWN';reason='ACTION_TUBE_VIOLATION' if action_audit.get('guard_violation') else 'ACTION_TUBE_AUDIT_NOT_VERIFIED'
        elif (collision['status']=='VERIFIED' and action_audit.get('static_blocking_certificate',{}).get('status')=='VERIFIED'
              and outcome.get('runtime_valid') and outcome.get('probe_valid') and outcome.get('safe_recovery')
              and not truth.get('out_of_probe_range') and truth.get('label_scope')!='DEFER_VISUAL_REOBSERVATION'):
            label='BLOCKED';reason='INDEPENDENT_DISCRETE_ACTION_TUBE_SEPARATION_PROOF'
    else:
        action_verified=True
        label,reason=conservative_action_label(label,reason)
    y={'PASSABLE_FOR_ACTION':1,'BLOCKED':0,'LABEL_UNKNOWN':-1}[label]
    admitted=primary and collision['status']=='VERIFIED' and action_verified
    entry=dict(trial_id=trial,planned_trial_id=planned_trial,attempt_id=directory.name,campaign='4070-G',source='G',npz_path=rel(path),npz_sha256=sha(path),
        y=y,label=label,evaluation_label={1:'PASSABLE',0:'BLOCKED',-1:'UNKNOWN'}[y],is_control=control,
        group_id=plan['physical_config_id'],physical_config_id=plan['physical_config_id'],scene_id=scene['scene_id'],
        scene_aliases=plan.get('scene_aliases',[scene['scene_id']]),
        condition_id=scene['scene_id'],condition_family=scene.get('family'),side=action['probe_hand'],probe_side=action['probe_hand'],
        followup_hand=action['followup_hand'],seed=plan['seed'],action_contract=action['action_id'],label_contract=action['label_contract'],
        followup_action_id=action['action_id'],followup_action=action,runtime_valid=bool(outcome.get('runtime_valid')),
        probe_valid=True,physics_valid=True,label_valid=y>=0,label_reason=reason,
        supervision_eligible=admitted and y>=0,response_history_eligible=admitted,source_path=rel(directory),
        primary_protocol=plan.get('primary_protocol'),primary_admission=admitted,label_validity_version=LABEL_VALIDITY_VERSION,
        action_evidence_revision=V5_ACTION_EVIDENCE_REVISION if primary_protocol==V5_PROTOCOL and primary else ACTION_EVIDENCE_REVISION,
        action_contract_v2=plan.get('action_contract_v2'),action_contract_v2_sha256=plan.get('action_contract_v2_sha256'),
        original_outcome_label=outcome['label'],original_outcome_label_valid=outcome.get('label_valid'),
        original_outcome_label_reason=outcome.get('label_reason'),body_collision_audit=collision,reset_randomization_evidence=rng_evidence,
        admission_status='PRIMARY_VERIFIED' if admitted else 'SUPERSEDED_PILOT' if not primary else 'PENDING_BODY_COLLISION_OR_ACTION_AUDIT',
        source_sha256={p.name:sha(p) for p in (source,directory/'g_outcome.json',directory/'g_trial_metadata.json')},
        observation_cutoff_time_s=float(t[-1]),observation_cutoff_before_followup=True,
        scene_config_evaluation_only=scene,robot_conditions_audit_only=robot,
        repetition_index=plan.get('repetition_index'),experiment_role=plan.get('experiment_role'),posture_variant_id=plan.get('posture_variant_id',robot.get('posture_variant_id')),
        robot_variant=plan.get('posture_variant_id',robot.get('posture_variant_id','nominal')),
        source_stage=plan.get('phase',plan.get('stage','UNSPECIFIED')),**audit,**provenance)
    return entry


def seed_role(e):
    role=e.get('experiment_role')
    if role in {'train','calibration','test'}:return role
    if role=='confirmation':return 'confirmation'
    if role=='heldout_posture':return 'test'
    index=e.get('repetition_index')
    index=int(index)%12 if index is not None else stable_int(e['seed'])%12
    return 'calibration' if index in (1,10) else 'test' if index in (2,4,5,6,7,11) else 'train'


def pose_is_new(e):
    robot=e['robot_conditions_audit_only'];variant=e.get('posture_variant_id')
    if variant is not None:return str(variant).lower() not in {'baseline','nominal','default','0'}
    return (abs(float(robot.get('reset_root_y',-.004))+.004)>1e-9 or
            abs(float(robot.get('reset_root_yaw',-.004))+.004)>1e-9 or
            any(abs(float(v))>1e-9 for v in robot.get('upper_offset_rad',[])))


def make_splits(episodes):
    definitions=[('split1_iid_seed','iid_seed'),('split2_anchor','anchor'),('split3_size_resistance','length_or_resistance'),
                 ('split4_left_to_right','cross_hand'),('split4_right_to_left','cross_hand'),('split5_wall_gap','wall_gap'),
                 ('split6_material','material'),('split7_posture','posture')]
    result=[]
    for sid,kind in definitions:
        roles={r:[] for r in ('train','calibration','test','confirmation')};excluded=[]
        for e in episodes:
            s=e['scene_config_evaluation_only'];g,p,top,w=s['geometry'],s['physics'],s['topology'],s.get('wall')
            independent_confirmation=e.get('experiment_role')=='confirmation'
            if s.get('scope')!='NEAR_FIELD_ACTION_TRAVERSABILITY':
                excluded.append(dict(trial_id=e['trial_id'],reason='DEFER_VISUAL_OUTSIDE_SINGLE_PROBE_CONTRACT'));continue
            if kind=='iid_seed':
                if e.get('experiment_role')=='heldout_posture':
                    excluded.append(dict(trial_id=e['trial_id'],reason='PAIRED_SAME_SEED_POSTURE_VARIANT_RESERVED_FOR_SPLIT7'));continue
                role=seed_role(e)
            elif kind=='cross_hand':
                train_side='left' if sid.endswith('left_to_right') else 'right'
                role='test' if e['side']!=train_side else ('calibration' if seed_role(e)=='calibration' else 'train')
            elif kind=='posture':
                role='test' if pose_is_new(e) else ('calibration' if seed_role(e)=='calibration' else 'train')
            elif kind=='material':
                if not s.get('real_material_validated',False):
                    excluded.append(dict(trial_id=e['trial_id'],reason='UNCALIBRATED_PROXY_NOT_REAL_MATERIAL_VALIDATION'));continue
                material=s.get('material_fidelity','')
                role='test' if any(x in material for x in ('PVC','NET')) else ('calibration' if stable_int(e['group_id'])%5==0 else 'train')
            elif kind=='anchor':
                role='test' if top['kind'] in {'FOUR_CORNERS','THREE_FREE_LEFT','THREE_FREE_RIGHT','PERIMETER'} else 'calibration' if top['kind'] in {'TOP_PLUS_BOTH','PARTIAL_TOP'} else 'train'
            elif kind=='length_or_resistance':
                role='test' if float(g['height_m'])<1.9 or float(g['height_m'])>2.05 or float(p['density_kg_m3'])>400 else 'calibration' if float(g['height_m'])<1.99 or float(p['density_kg_m3'])>300 else 'train'
            else:
                role='test' if w and abs(float(w['initial_gap_m'])-.08)<1e-8 else 'calibration' if stable_int(e['group_id'])%5==0 and not w else 'train'
            # Frozen confirmation seeds are a separate experimental endpoint,
            # never training, calibration, normalization, pretraining or main
            # test data. OOD confirmation must belong to that split's test
            # domain; confirmation from a training-domain configuration cannot
            # quietly become a new training example.
            if independent_confirmation:
                if kind=='iid_seed' or role=='test':roles['confirmation'].append(e['trial_id'])
                else:excluded.append(dict(trial_id=e['trial_id'],reason='INDEPENDENT_CONFIRMATION_OUTSIDE_THIS_SPLIT_TEST_DOMAIN'))
            else:roles[role].append(e['trial_id'])
        byid={e['trial_id']:e for e in episodes};counts={}
        for role,ids in roles.items():
            selected=[byid[i] for i in ids];core=[e for e in selected if not e['is_control']]
            counts[role]=dict(episodes=len(selected),core_P=sum(e['y']==1 for e in core),core_B=sum(e['y']==0 for e in core),
                              core_UNKNOWN=sum(e['y']==-1 for e in core),controls=sum(e['is_control'] for e in selected),
                              physical_groups=len({e['group_id'] for e in selected}))
        reasons=[]
        if kind=='material' and not any(s.get('scene_config_evaluation_only',{}).get('real_material_validated',False) for s in episodes):
            reasons.append('REAL_MATERIAL_MODELS_NOT_VALIDATED')
        for role in ('train','test'):
            if not counts[role]['core_P'] or not counts[role]['core_B']:reasons.append(role.upper()+'_CORE_BOTH_LABELS_NOT_SUPPORTED')
        calibration_status='SUPPORTED' if counts['calibration']['core_P'] and counts['calibration']['core_B'] else 'CALIBRATION_BOTH_CLASSES_NOT_SUPPORTED'
        split=dict(split_id=sid,split_type=kind,**roles,status='NOT_EVALUABLE' if reasons else 'AVAILABLE',reasons=reasons,
                   calibration_status=calibration_status,counts=counts,excluded=excluded,
                   confirmation_policy='Separate heldout confirmation endpoint only; never fit or main-test metrics',
                   confirmation_status='AVAILABLE' if counts['confirmation']['core_P'] and counts['confirmation']['core_B'] else 'NOT_EVALUABLE',
                   confirmation_reasons=[] if counts['confirmation']['core_P'] and counts['confirmation']['core_B'] else ['INDEPENDENT_CONFIRMATION_CORE_BOTH_LABELS_NOT_SUPPORTED'],
                   evaluation_role_reason='Predeclared metadata rule; no label/model-dependent allocation; whole episode atomic')
        if kind=='posture':
            split.update(evaluation_scope='HELDOUT_ROBOT_POSTURE_AND_ACTION_CONDITIONS',
                display_name='姿态/动作条件留出（position/yaw/arm/speed分别报告）',
                causal_limitation='Pooled effects are not a pure posture effect; compare matched seed/config per robot_variant',
                test_robot_variants=sorted({byid[i].get('robot_variant','nominal') for i in roles['test']}))
        for a,b in itertools.combinations(roles,2):
            if set(roles[a])&set(roles[b]):raise AssertionError('EPISODE_LEAKAGE')
            if kind not in {'iid_seed','cross_hand','posture'} and {a,b}!={'test','confirmation'}:
                ga={byid[x]['group_id'] for x in roles[a]};gb={byid[x]['group_id'] for x in roles[b]}
                if ga&gb:raise AssertionError('PHYSICAL_CONFIG_LEAKAGE')
        result.append(split)
    return dict(schema='round4070-g.splits.v1',allocation_frozen_without_labels=True,
                seed_role_rule='Frozen campaign experiment_role first. confirmation has a separate endpoint; heldout_posture reserved from IID. Non-confirmation fallback mod12 calibration1/10,test2/4/5/6/7/11,other train; absent index uses sha256(seed) mod12',
                independent_confirmation='All experiment_role=confirmation excluded from every fit and main test. Separate confirmation IDs only within each predeclared test domain; IID includes all confirmation.',
                group_equivalence='sha256 actual geometry/physics/topology/sheets/wall; aliases, seed, side, pose do not create physical groups',
                response_pretraining_rule='Only train IDs, including train UNKNOWN; calibration/test excluded from every fit',splits=result)


def matched_pairs(episodes,splits):
    byid={e['trial_id']:e for e in episodes};by_split={}
    for split in splits['splits']:
        valid=[byid[i] for i in split['train'] if byid[i]['y']>=0 and not byid[i]['is_control']]
        for episode in valid:
            action=episode['followup_action']
            if (episode['action_contract']!=action['action_id'] or episode['followup_action_id']!=action['action_id']
                    or episode['label_contract']!=action['label_contract']):
                raise ValueError('MATCHED_PAIR_ACTION_METADATA_MISMATCH')
        pairs=[]
        for a,b in itertools.combinations(valid,2):
            if a['group_id']==b['group_id'] or a['followup_action']!=b['followup_action']:continue
            sa,sb=a['scene_config_evaluation_only'],b['scene_config_evaluation_only']
            ga,gb=sa['geometry'],sb['geometry']
            comparable_geometry=all(abs(float(ga[k])-float(gb[k]))<=tol for k,tol in [('height_m',.100001),('width_m',.100001),('bottom_z_m',.030001)])
            same_physics=sa['physics']==sb['physics'] and sa['material_fidelity']==sb['material_fidelity']
            if not comparable_geometry:continue
            if a['y']!=b['y'] and same_physics:kind='hard_negative'
            elif a['y']==b['y']==1:kind='cross_configuration_positive'
            else:continue
            pairs.append(dict(a=a['trial_id'],b=b['trial_id'],kind=kind,evidence=dict(
                actual_labels=[a['label'],b['label']],different_actual_physical_groups=True,
                action_contract=a['action_contract'],label_contract=a['label_contract'],identical_specific_followup_action=True,probe_hand=a['side'],
                geometric_tolerance_m=dict(height=.1,width=.1,bottom=.03),same_physics_parameters=same_physics,
                source_label_evidence=[e['body_collision_audit']['source_evidence_path'] for e in (a,b)],
                source_label_evidence_sha256=[e['body_collision_audit']['source_evidence_sha256'] for e in (a,b)],
                scope='TRAIN_ONLY_REAL_EPISODES_NO_UNKNOWN_NO_MIRROR_LABEL')))
        by_split[split['split_id']]=pairs
    return dict(schema='round4070-g.approved_pairs.v1',by_split=by_split)


def legacy_manifest():
    path=ROOT/'outputs/round4070_learning_traversability/dataset_final/manifest.json'
    if not path.exists():return dict(status='UNAVAILABLE',episodes=[])
    original=load(path);episodes=[]
    for e in original['episodes']:
        e=dict(e);e.update(source='legacy',y=-1,original_label_audit_only=e['label'],label='LABEL_UNKNOWN',
            label_valid=False,supervision_eligible=False,action_contract='LEGACY_E_F_CLEARANCE_NOT_PROVEN_G_ACTION_EQUIVALENT',
            response_history_eligible=True,admission_status='SEPARATE_SCHEMA_NOT_AUTOMATICALLY_ADMITTED_TO_G_TRAINING')
        episodes.append(e)
    return {**original,'schema':'round4070-g.legacy_response_source.v1','source_manifest_sha256':sha(path),'source_manifest':rel(path),
            'episodes':episodes,'labels':dict(LABEL_UNKNOWN=len(episodes)),
            'restriction':'No old P/B supervision. Missing true previous action/full gyro/acceleration are not synthesized. Any future response adapter must exclude G-heldout equivalent physical groups.'}


def reports(episodes,invalid,splits,output,report,pilots=(),pending=(),primary_protocol=PRIMARY_PROTOCOL,launches=None):
    protocol_contract=V2_CONTRACT if primary_protocol==V5_PROTOCOL else CONTRACT
    report.mkdir(parents=True,exist_ok=True)
    counts=[]
    for side in ('left','right'):
        es=[e for e in episodes if e['side']==side]
        counts.append(dict(side=side,episodes=len(es),core_PASSABLE=sum(e['y']==1 and not e['is_control'] for e in es),
                           core_BLOCKED=sum(e['y']==0 and not e['is_control'] for e in es),core_UNKNOWN=sum(e['y']==-1 and not e['is_control'] for e in es),
                           controls=sum(e['is_control'] for e in es),positive_physical_groups=len({e['group_id'] for e in es if e['y']==1 and not e['is_control']})))
    save_csv(report/'LEFT_RIGHT_COVERAGE.csv',counts,list(counts[0]))
    grouped=[]
    for group in sorted({e['group_id'] for e in episodes}):
        es=[e for e in episodes if e['group_id']==group]
        grouped.append(dict(physical_config_id=group,scene_aliases='|'.join(sorted({a for e in es for a in e.get('scene_aliases',[e['scene_id']])})),
                            episodes=len(es),PASSABLE=sum(e['y']==1 for e in es),BLOCKED=sum(e['y']==0 for e in es),
                            UNKNOWN=sum(e['y']==-1 for e in es),controls=sum(e['is_control'] for e in es),
                            left=sum(e['side']=='left' for e in es),right=sum(e['side']=='right' for e in es)))
    save_csv(report/'POSITIVE_NEGATIVE_COVERAGE.csv',grouped,['physical_config_id','scene_aliases','episodes','PASSABLE','BLOCKED','UNKNOWN','controls','left','right'])
    assignments=[];byid={e['trial_id']:e for e in episodes}
    for s in splits['splits']:
        for role in ('train','calibration','test','confirmation'):
            for tid in s[role]:
                e=byid[tid];assignments.append(dict(split_id=s['split_id'],split_type=s['split_type'],status=s['status'],role=role,
                    trial_id=tid,physical_config_id=e['group_id'],scene_id=e['scene_id'],side=e['side'],robot_variant=e.get('robot_variant','nominal'),seed=e['seed'],label=e['label'],is_control=e['is_control']))
    save_csv(report/'DATA_SPLIT_MANIFEST.csv',assignments,['split_id','split_type','status','role','trial_id','physical_config_id','scene_id','side','robot_variant','seed','label','is_control'])
    core=[e for e in episodes if not e['is_control']];labels=Counter(e['label'] for e in core)
    summary=dict(status='PARTIAL_ACTUAL_DATA_ONLY',episodes=len(episodes),core_labels=dict(labels),controls=sum(e['is_control'] for e in episodes),
                 primary_protocol=primary_protocol,label_validity_version=LABEL_VALIDITY_VERSION,
                 action_evidence_revision=V5_ACTION_EVIDENCE_REVISION if primary_protocol==V5_PROTOCOL else ACTION_EVIDENCE_REVISION,
                 superseded_pilot_count=len(pilots),pending_collision_audit_count=len(pending),
                 runtime_valid=sum(e['runtime_valid'] for e in episodes),probe_valid=sum(e['probe_valid'] for e in episodes),
                 probe_valid_but_followup_runtime_invalid=sum(not e['runtime_valid'] for e in episodes),
                 excluded_or_unfinalized=len(invalid),source_raw_unchanged=True,physical_groups=len({e['group_id'] for e in core}),
                 left_right=counts,split_status={s['split_id']:dict(status=s['status'],reasons=s['reasons']) for s in splits['splits']})
    if launches is not None:
        summary.update(actual_launch_count=launches['actual_launch_count'],finalized_outcome_launch_count=launches['finalized_outcome_count'],
            launches_without_metadata=launches['no_metadata_count'],launches_without_final_outcome=launches['no_final_outcome_count'],
            incomplete_launch_count=launches['incomplete_launch_count'],metadata_import_excluded_count=len(invalid),
            excluded_count_scope='Only directories with g_trial_metadata.json attempted by the episode importer; not all launch failures')
    write_json(output/'dataset_status.json',summary)
    (report/'PROBE_DATASET_AUDIT.md').write_text('# Round4070-G actual dataset audit\n\n'+
        f'Current importer view: **{len(episodes)}** primary accepted real G histories, **{len(pilots)}** superseded pilots, **{len(pending)}** pending body collision audit, **{len(invalid)}** excluded/unfinalized source directories. This is a partial dataset status, not benchmark completion.\n\n'+
        (f'Actual launch inventory separately records **{launches["actual_launch_count"]}** starts, including **{launches["no_metadata_count"]}** without episode metadata and **{launches["no_final_outcome_count"]}** without a finalized outcome. The metadata-import excluded count does not imply all starts succeeded. Original attempt paths, terminal/open state and available errors remain in actual_launch_inventory.json; these rows do not create model episodes or labels.\n\n' if launches is not None else '')+
        f'Primary protocol `{primary_protocol}` and validity rule `{LABEL_VALIDITY_VERSION}` require actual replayable seed injection and an independent VERIFIED whole-robot sampled collision audit. Missing/unsupported collision geometry is pending, not usable UNKNOWN response data. Confirmed penetration is physics-invalid and excluded. Superseded protocols remain separate with original outcomes. This instrumentation/validity repair follows the requested physical-validity standard, not observed model performance. V5 adds the exact versioned 200 Hz discrete action tube described in V5_DISCRETE_ACTION_CONTRACT_REVIEW.md; V1/V4 labels are never retroactively reinterpreted as V5.\n\n'+
        'Raw G files remain read-only. The model array contains only robot q/qdot/actually sent q_cmd/applied torque, actual previous control action, current policy action, body-local dual-hand FK and causal displacement, robot IMU/state, known commanded action and true observation intervals. All geometry, material, constraint node indices, contact truth, scene IDs, labels and followup outcomes remain in separate metadata. No absolute world position, map or goal channel is exported. Missing acceleration is explicitly zero-plus-availability-mask, never presented as an observed zero.\n\n'+
        'Physics runs at 200 Hz, policy at 50 Hz, geometry/progress safety at 10 Hz. New robot records use existing 50 Hz control ticks plus actual phase-first ticks; no interpolated observations. Actual per-episode timestamps determine Hz and dt. Only physical_probe/physical_probe_hold before the recorded cutoff and at most 2 s enter seq. All methods must select the same cutoff; .5/1 s prefixes cannot normalize by future final duration.\n\n'+
        'Official joint order comes from falcon_contract.py and is recorded in manifest; left/right arm columns are selected by joint name. q_cmd is the target already sent before the measured state, and previous_control_action is copied immediately before the next policy computation. Consecutive recorded control updates verify previous==preceding current. Arm commanded reference equals the actual q_cmd upper subset.\n\n'+
        'UNKNOWN is y=-1 and never a third supervised class. Physics-invalid or probe-invalid files do not enter valid arrays. Runtime/probe/label/task flags remain independent. Source F117 histories are listed separately with all old labels audit-only; their 10 Hz schema cannot silently fill unavailable G channels.\n\n'+
        'If the probe was independently valid but later followup execution failed, the finite physics-valid probe can remain response-only UNKNOWN; this does not assert runtime validity for the whole episode. Separate runtime/probe counts are recorded in dataset_status.json. Repeated actual attempts have separate IDs, retain the same physical group and frozen split role, and are never selected by best label. Existing exported NPZ arrays are compared and reused, preserving their hashes across partial imports.\n\n'+
        'Splits are fixed metadata rules, not optimized from labels. Every episode is atomic. Group holdouts collapse identical physical hashes across aliases/seeds/hands/poses. Only training UNKNOWN may enter response fitting or normalization. Material OOD remains unavailable unless actual material fidelity has independently been validated.\n\n'+
        'G E0/E1/E2 retain the historical frozen thresholds, but compute progress along the initial robot-body forward axis using the legal FK channels. This is an explicit G coordinate adaptation, not a bit-for-bit replay of the old world-x online rule. Empty-response references are derived separately from each split\'s training same-hand controls only; stall uses real timestamp left integration. Acceleration candidates remain audit-only throughout the frozen v3 model protocol because gravity/frame calibration has not been independently established.\n\n'+
        f'Current source and validity inventory: `{rel(output)}/import_audit.json`. Feature schema and actual splits: `{rel(output)}/manifest.json`, `split_manifest.json`.\n',encoding='utf-8')
    (report/'LABEL_VALIDITY_REPORT.md').write_text('# Round4070-G frozen action label contract\n\n'+
        f'Formal label: `{protocol_contract}`. Probe hand and followup hand are separate fields. The action holds the executed final probe pose with the same hand and requests frozen R1 bounded forward motion: at most 24 segments of 0.12 m toward (1.30, 0), cruise 0.25 m/s, max vx 0.30 m/s, 7 s segment timeout, followed by 1.5 s release, 1.5 s neutralization and 1.5 s stable recovery. The old classifier is audit-only and does not select which cases receive followup. Existing safety vetoes remain active. V5 additionally requires a native-verified fixed interior witness, the frozen tube and bounded adjacent displacement at every actual 200 Hz state. Only a verified V5 discrete separation certificate can establish static BLOCKED; guard violation remains UNKNOWN. See V5_DISCRETE_ACTION_CONTRACT_REVIEW.md.\n\n'+
        'PASSABLE_FOR_ACTION requires an actual recorded followup, actual whole-body collider clearance, finite valid physics/runtime/probe, independent actual-collider body/cloth/wall nonpenetration audit and verified stable release/recovery. The existing visual-subtree bounds/fallback-radius envelope is diagnostic and is not collision proof. Merely opening cloth, safe retraction, detouring or reaching PointGoal is insufficient. Timeout, low progress and controller refusal remain LABEL_UNKNOWN. D07 outside the probe range is always excluded from near-field supervision and deferred to visual reobservation.\n\n'+
        f'Action evidence revision `{ACTION_EVIDENCE_REVISION}` withdraws all finite-wall static BLOCKED conclusions from V1/V4 formal labels. The 1.2 m boards cover the nominal central path but the closed-loop R1 action has no enforced lateral/height tube. D01–D04 followup is withheld, so their original source labels do not establish actual action failure either. Original raw labels are retained as audit-only fields; formal labels are UNKNOWN/FINITE_WALL_NO_CLOSED_LOOP_ACTION_IMPOSSIBILITY_PROOF. Missing negative support is reported, never repaired by changing labels to populate classes. See ACTION_LABEL_GAP_REPORT.md.\n\n'+
        'A phase-level safety stop can leave the probe history valid but the action outcome UNKNOWN. Online deformable failures invalidate physics and prevent all supervised/response admission. New labels are recomputed from evidence at import, not read from material names. Historical E/F clearance-based labels are not automatically equivalent to this intervention-independent G contract.\n\n'+
        f'Validity revision `{LABEL_VALIDITY_VERSION}` applies the same actual-body nonpenetration requirement to cloth and wall scenes. Original C01 source PASSABLE outcomes prove only the previous envelope/recovery condition and remain original_outcome_label audit fields; their supervised interpretation is UNKNOWN/VALIDITY_PENDING_BODY_COLLISION_AUDIT and all V3 pilots are excluded from primary learning. Raw files are unchanged. EMPTY controls retain their original action label but old pilots are still excluded by protocol. A physics-only probe-box gate does not substitute for G1-body collision evidence. Finite-wall corridor obstruction remains an audit fact, not a formal BLOCKED action label.\n\n'+
        'The independent g_collision_audit.json is a derived artifact with source hashes, not a mutation of g_outcome.json. VERIFIED means the supported actual collider geometry and synchronized sampled trajectory passed the recorded audit scope; temporal sampling limitations must be stated and must not be presented as a continuous-time proof. NOT_VERIFIED forbids both supervision and response fitting; PHYSICS_INVALID excludes the episode.\n\n'+
        'Independent PASSABLE clearance uses the actual enabled collision geometry at the exact responsive_traverse_stand phase-final sample, then requires the original finite/stable release and recovery. The unchanged candidate interval is scene.center_x ± 0.06 m, so its back boundary plus the frozen 0.04 m margin is scene.center_x + 0.10 m (0.950 m for the standard 0.850 m plane). The 0.04 m margin is not measured directly from the center plane. Old visual-envelope clearance remains diagnostic and cannot substitute for this actual-collider check.\n\n'+
        f'Current accepted core labels: `{dict(labels)}`; control episodes: {sum(e["is_control"] for e in episodes)}. Left/right counts and unique positive physical groups are exported separately. All unexecuted coverage goals remain pending.\n',encoding='utf-8')
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--trials-root',type=Path,default=OUT/'trials')
    ap.add_argument('--output',type=Path,default=OUT/'dataset');ap.add_argument('--report-dir',type=Path,default=REPORT)
    ap.add_argument('--primary-protocol',choices=[PRIMARY_PROTOCOL,V5_PROTOCOL],default=None)
    ap.add_argument('--no-legacy-index',action='store_true');args=ap.parse_args()
    if not args.output.resolve().is_relative_to(OUT.resolve()):raise ValueError('Output must remain within new G outputs')
    protocol=args.primary_protocol or load(REPORT/'CURRENT_CAMPAIGN.json')['primary_protocol']
    if protocol not in (PRIMARY_PROTOCOL,V5_PROTOCOL):raise ValueError('Unsupported current primary protocol')
    args.output.mkdir(parents=True,exist_ok=True);episodes=[];invalid=[];pilots=[];pending=[]
    for path in sorted(args.trials_root.rglob('g_trial_metadata.json')):
        try:
            episode=import_trial(path.parent,args.output,protocol)
            if episode['primary_admission']:episodes.append(episode)
            elif episode['admission_status']=='SUPERSEDED_PILOT':pilots.append(episode)
            else:pending.append(episode)
        except (ValueError,KeyError,FileNotFoundError,TypeError) as exc:
            invalid.append(dict(source_path=rel(path.parent),reason=str(exc),status='EXCLUDED_OR_UNFINALIZED'))
    if len({e['trial_id'] for e in episodes})!=len(episodes):raise ValueError('DUPLICATE_TRIAL_ID')
    names,blocks,units=schema()
    arms=[i for i,n in enumerate(names) if any(j in n for j in UPPER_JOINTS) or 'hand_' in n]
    manifest=dict(schema='round4070-g.causal_episodes.v3' if protocol==V5_PROTOCOL else 'round4070-g.causal_episodes.v2',primary_protocol=protocol,label_validity_version=LABEL_VALIDITY_VERSION,
        action_evidence_revision=V5_ACTION_EVIDENCE_REVISION if protocol==V5_PROTOCOL else ACTION_EVIDENCE_REVISION,
        path_base='repository_root',action_contract=V2_ACTION if protocol==V5_PROTOCOL else ACTION_ID,label_contract=V2_CONTRACT if protocol==V5_PROTOCOL else CONTRACT,
        feature_names=names,feature_blocks=blocks,feature_units_by_block=units,joint_order=list(JOINTS),arm_indices=arms,
        body_indices=sorted(set(range(len(names)))-set(arms)),episodes=episodes,labels=dict(Counter(e['label'] for e in episodes)),
        same_legal_window='real probe/probe_hold samples, <=2s, before any followup',
        forbidden_learning_inputs=['scene_id','group_id','material','constraint nodes','wall','contact truth','future observations','absolute world pose','map','goal'],
        self_supervision='current train group only; no heldout UNKNOWN',status='PARTIAL_ACTUAL_DATA_ONLY')
    splits=make_splits(episodes);pairs=matched_pairs(episodes,splits)
    launches=actual_launch_inventory(args.trials_root)
    write_json(args.output/'manifest.json',manifest);write_json(args.output/'split_manifest.json',splits)
    write_json(args.output/'matched_pairs.json',pairs);write_json(args.output/'actual_launch_inventory.json',launches)
    write_json(args.output/'import_audit.json',dict(accepted=episodes,excluded=invalid,incomplete_actual_launches=launches['incomplete_launches'],
        excluded_count_scope='Metadata-bearing directories only; all actual starts are independently inventoried'))
    write_json(args.output/'superseded_pilot_manifest.json',dict(schema='round4070-g.superseded_pilots.v1',episodes=pilots,
        restriction='Original raw labels retained; no primary training, normalization, auxiliary response fitting, calibration or test admission. V3 seed was not effectively injected; full robot collision history missing.'))
    write_json(args.output/'pending_collision_audit_manifest.json',dict(schema='round4070-g.pending_collision_audit.v1',episodes=pending,
        restriction='Missing or unsupported actual body collision audit; no supervised or response-learning admission.'))
    if not args.no_legacy_index:write_json(args.output/'legacy_response_manifest.json',legacy_manifest())
    print(json.dumps(reports(episodes,invalid,splits,args.output,args.report_dir,pilots,pending,protocol,launches),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
