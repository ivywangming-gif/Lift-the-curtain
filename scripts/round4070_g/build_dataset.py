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
from g1_recording import ACTION_ID, CONTRACT, PROBE_PHASES, action_spec, assign_label, clean, write_json
from physics_scenes import physical_hash

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


def frozen_evidence(directory,meta,plan):
    proof_path=directory/'g_runner_contract.json';proof=load(proof_path)
    if proof.get('dry_run') or proof.get('followup_policy_uses_classifier') is not False:
        raise ValueError('ACTION_COLLECTION_CONTRACT_NOT_VERIFIED')
    if proof.get('patched_source_sha256')!=meta.get('runtime_source_sha256'):
        raise ValueError('RUNTIME_SOURCE_METADATA_MISMATCH')
    source=OUT/'runtime_sources'/f"{proof['patched_source_sha256']}.py"
    normalized=hashlib.sha256(source.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
    if normalized!=proof['patched_source_sha256']:raise ValueError('EXECUTED_SOURCE_ARCHIVE_HASH_MISMATCH')
    recorder=OUT/'runtime_sources'/f"g1_recording_{proof['recording_source_sha256']}.py"
    if sha(recorder)!=proof['recording_source_sha256']:raise ValueError('RECORDER_SOURCE_ARCHIVE_HASH_MISMATCH')
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
    return dict(runtime_contract_sha256=sha(proof_path),runtime_source_sha256=proof['patched_source_sha256'],
                recorder_source_sha256=proof['recording_source_sha256'],physics_gate_seeds=sorted(seeds),physics_gate_hashes=hashes)


def import_trial(directory,output):
    meta=load(directory/'g_trial_metadata.json');plan=meta['plan'];scene=meta['scene_config_evaluation_only']
    outcome=load(directory/'g_outcome.json');action=action_spec(plan);planned_trial=plan['trial_id']
    if not planned_trial or Path(planned_trial).name!=planned_trial:raise ValueError('UNSAFE_TRIAL_ID')
    if outcome.get('trial_id')!=planned_trial:raise ValueError('TRIAL_ID_MISMATCH')
    # A retry is a different actual episode, even when it shares a frozen plan.
    trial=planned_trial+'__'+directory.name if directory.name.startswith('attempt') else planned_trial
    if plan.get('physical_config_id')!=physical_hash(scene):raise ValueError('PHYSICAL_GROUP_HASH_MISMATCH')
    if outcome.get('label_contract')!=CONTRACT:raise ValueError('WRONG_ACTION_LABEL_CONTRACT')
    if not outcome.get('physics_valid'):raise ValueError('PHYSICS_INVALID_OR_UNFINALIZED')
    if not outcome.get('probe_valid'):raise ValueError('PROBE_INVALID_OR_UNFINALIZED')
    provenance=frozen_evidence(directory,meta,plan)
    truth=outcome.get('scene_evaluation_only',{})
    derived,reason=assign_label(physics_valid=outcome['physics_valid'],runtime_valid=outcome.get('runtime_valid',False),
        probe_valid=outcome['probe_valid'],recovered=outcome.get('safe_recovery',False),
        clearance=outcome.get('whole_body_candidate_clearance',False),advance_steps=outcome.get('followup_advance_logged_steps',0),
        metadata=truth,action=action)
    if outcome.get('label')!=derived:raise ValueError('LABEL_RECOMPUTATION_MISMATCH')
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
    label=derived;y={'PASSABLE_FOR_ACTION':1,'BLOCKED':0,'LABEL_UNKNOWN':-1}[label]
    entry=dict(trial_id=trial,planned_trial_id=planned_trial,attempt_id=directory.name,campaign='4070-G',source='G',npz_path=rel(path),npz_sha256=sha(path),
        y=y,label=label,evaluation_label={1:'PASSABLE',0:'BLOCKED',-1:'UNKNOWN'}[y],is_control=control,
        group_id=plan['physical_config_id'],physical_config_id=plan['physical_config_id'],scene_id=scene['scene_id'],
        condition_id=scene['scene_id'],condition_family=scene.get('family'),side=action['probe_hand'],probe_side=action['probe_hand'],
        followup_hand=action['followup_hand'],seed=plan['seed'],action_contract=ACTION_ID,label_contract=CONTRACT,
        followup_action_id=ACTION_ID,followup_action=action,runtime_valid=bool(outcome.get('runtime_valid')),
        probe_valid=True,physics_valid=True,label_valid=y>=0,label_reason=reason,
        supervision_eligible=y>=0,response_history_eligible=True,source_path=rel(directory),
        source_sha256={p.name:sha(p) for p in (source,directory/'g_outcome.json',directory/'g_trial_metadata.json')},
        observation_cutoff_time_s=float(t[-1]),observation_cutoff_before_followup=True,
        scene_config_evaluation_only=scene,robot_conditions_audit_only=robot,
        repetition_index=plan.get('repetition_index'),experiment_role=plan.get('experiment_role'),posture_variant_id=plan.get('posture_variant_id',robot.get('posture_variant_id')),
        source_stage=plan.get('phase',plan.get('stage','UNSPECIFIED')),**audit,**provenance)
    return entry


def seed_role(e):
    role=e.get('experiment_role')
    if role in {'train','calibration','test'}:return role
    if role in {'confirmation','heldout_posture'}:return 'test'
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
        roles={r:[] for r in ('train','calibration','test')};excluded=[]
        for e in episodes:
            s=e['scene_config_evaluation_only'];g,p,top,w=s['geometry'],s['physics'],s['topology'],s.get('wall')
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
            roles[role].append(e['trial_id'])
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
                   evaluation_role_reason='Predeclared metadata rule; no label/model-dependent allocation; whole episode atomic')
        for a,b in itertools.combinations(roles,2):
            if set(roles[a])&set(roles[b]):raise AssertionError('EPISODE_LEAKAGE')
            if kind not in {'iid_seed','cross_hand','posture'}:
                ga={byid[x]['group_id'] for x in roles[a]};gb={byid[x]['group_id'] for x in roles[b]}
                if ga&gb:raise AssertionError('PHYSICAL_CONFIG_LEAKAGE')
        result.append(split)
    return dict(schema='round4070-g.splits.v1',allocation_frozen_without_labels=True,
                seed_role_rule='Frozen campaign experiment_role first; confirmation/heldout_posture=test. Fallback mod12 calibration1/10,test2/4/5/6/7/11,other train; absent index uses sha256(seed) mod12',
                group_equivalence='sha256 actual geometry/physics/topology/sheets/wall; aliases, seed, side, pose do not create physical groups',
                response_pretraining_rule='Only train IDs, including train UNKNOWN; calibration/test excluded from every fit',splits=result)


def matched_pairs(episodes,splits):
    byid={e['trial_id']:e for e in episodes};by_split={}
    for split in splits['splits']:
        valid=[byid[i] for i in split['train'] if byid[i]['y']>=0 and not byid[i]['is_control']]
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
                action_contract=ACTION_ID,identical_specific_followup_action=True,probe_hand=a['side'],
                geometric_tolerance_m=dict(height=.1,width=.1,bottom=.03),same_physics_parameters=same_physics,
                source_label_evidence=[a['source_path']+'/g_evaluation_truth.json',b['source_path']+'/g_evaluation_truth.json'],
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


def reports(episodes,invalid,splits,output,report):
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
        grouped.append(dict(physical_config_id=group,scene_aliases='|'.join(sorted({e['scene_id'] for e in es})),
                            episodes=len(es),PASSABLE=sum(e['y']==1 for e in es),BLOCKED=sum(e['y']==0 for e in es),
                            UNKNOWN=sum(e['y']==-1 for e in es),controls=sum(e['is_control'] for e in es),
                            left=sum(e['side']=='left' for e in es),right=sum(e['side']=='right' for e in es)))
    save_csv(report/'POSITIVE_NEGATIVE_COVERAGE.csv',grouped,['physical_config_id','scene_aliases','episodes','PASSABLE','BLOCKED','UNKNOWN','controls','left','right'])
    assignments=[];byid={e['trial_id']:e for e in episodes}
    for s in splits['splits']:
        for role in ('train','calibration','test'):
            for tid in s[role]:
                e=byid[tid];assignments.append(dict(split_id=s['split_id'],split_type=s['split_type'],status=s['status'],role=role,
                    trial_id=tid,physical_config_id=e['group_id'],scene_id=e['scene_id'],side=e['side'],seed=e['seed'],label=e['label'],is_control=e['is_control']))
    save_csv(report/'DATA_SPLIT_MANIFEST.csv',assignments,['split_id','split_type','status','role','trial_id','physical_config_id','scene_id','side','seed','label','is_control'])
    core=[e for e in episodes if not e['is_control']];labels=Counter(e['label'] for e in core)
    summary=dict(status='PARTIAL_ACTUAL_DATA_ONLY',episodes=len(episodes),core_labels=dict(labels),controls=sum(e['is_control'] for e in episodes),
                 runtime_valid=sum(e['runtime_valid'] for e in episodes),probe_valid=sum(e['probe_valid'] for e in episodes),
                 probe_valid_but_followup_runtime_invalid=sum(not e['runtime_valid'] for e in episodes),
                 excluded_or_unfinalized=len(invalid),source_raw_unchanged=True,physical_groups=len({e['group_id'] for e in core}),
                 left_right=counts,split_status={s['split_id']:dict(status=s['status'],reasons=s['reasons']) for s in splits['splits']})
    write_json(output/'dataset_status.json',summary)
    (report/'PROBE_DATASET_AUDIT.md').write_text('# Round4070-G actual dataset audit\n\n'+
        f'Current importer view: **{len(episodes)}** accepted real G histories, **{len(invalid)}** excluded/unfinalized source directories. This is a partial dataset status, not benchmark completion.\n\n'+
        'Raw G files remain read-only. The model array contains only robot q/qdot/actually sent q_cmd/applied torque, actual previous control action, current policy action, body-local dual-hand FK and causal displacement, robot IMU/state, known commanded action and true observation intervals. All geometry, material, constraint node indices, contact truth, scene IDs, labels and followup outcomes remain in separate metadata. No absolute world position, map or goal channel is exported. Missing acceleration is explicitly zero-plus-availability-mask, never presented as an observed zero.\n\n'+
        'Physics runs at 200 Hz, policy at 50 Hz, geometry/progress safety at 10 Hz. New robot records use existing 50 Hz control ticks plus actual phase-first ticks; no interpolated observations. Actual per-episode timestamps determine Hz and dt. Only physical_probe/physical_probe_hold before the recorded cutoff and at most 2 s enter seq. All methods must select the same cutoff; .5/1 s prefixes cannot normalize by future final duration.\n\n'+
        'Official joint order comes from falcon_contract.py and is recorded in manifest; left/right arm columns are selected by joint name. q_cmd is the target already sent before the measured state, and previous_control_action is copied immediately before the next policy computation. Consecutive recorded control updates verify previous==preceding current. Arm commanded reference equals the actual q_cmd upper subset.\n\n'+
        'UNKNOWN is y=-1 and never a third supervised class. Physics-invalid or probe-invalid files do not enter valid arrays. Runtime/probe/label/task flags remain independent. Source F117 histories are listed separately with all old labels audit-only; their 10 Hz schema cannot silently fill unavailable G channels.\n\n'+
        'If the probe was independently valid but later followup execution failed, the finite physics-valid probe can remain response-only UNKNOWN; this does not assert runtime validity for the whole episode. Separate runtime/probe counts are recorded in dataset_status.json. Repeated actual attempts have separate IDs, retain the same physical group and frozen split role, and are never selected by best label. Existing exported NPZ arrays are compared and reused, preserving their hashes across partial imports.\n\n'+
        'Splits are fixed metadata rules, not optimized from labels. Every episode is atomic. Group holdouts collapse identical physical hashes across aliases/seeds/hands/poses. Only training UNKNOWN may enter response fitting or normalization. Material OOD remains unavailable unless actual material fidelity has independently been validated.\n\n'+
        'G E0/E1/E2 retain the historical frozen thresholds, but compute progress along the initial robot-body forward axis using the legal FK channels. This is an explicit G coordinate adaptation, not a bit-for-bit replay of the old world-x online rule. Empty-response references are derived separately from each split\'s training same-hand controls only; stall uses real timestamp left integration. Acceleration candidates remain audit-only throughout the frozen v3 model protocol because gravity/frame calibration has not been independently established.\n\n'+
        f'Current source and validity inventory: `{rel(output)}/import_audit.json`. Feature schema and actual splits: `{rel(output)}/manifest.json`, `split_manifest.json`.\n',encoding='utf-8')
    (report/'LABEL_VALIDITY_REPORT.md').write_text('# Round4070-G frozen action label contract\n\n'+
        f'Formal label: `{CONTRACT}`. Probe hand and followup hand are separate fields. V1 holds the executed final probe pose with the same hand and requests frozen R1 bounded forward motion: at most 24 segments of 0.12 m toward (1.30, 0), cruise 0.25 m/s, max vx 0.30 m/s, 7 s segment timeout, followed by 1.5 s release, 1.5 s neutralization and 1.5 s stable recovery. The old classifier is audit-only and does not select which cases receive followup. Existing safety vetoes remain active.\n\n'+
        'PASSABLE_FOR_ACTION requires an actual recorded followup, robot whole-body candidate-envelope clearance, finite valid physics/runtime/probe and verified stable release/recovery. Merely opening cloth, safe retraction, detouring or reaching PointGoal is insufficient. BLOCKED requires an independently constructed fixed full-height/full-width near-field wall covering the frozen forward corridor; timeout, low progress and controller refusal alone remain LABEL_UNKNOWN. Partial coverage does not receive this proof. D07 outside the probe range is always excluded from near-field supervision and deferred to visual reobservation.\n\n'+
        'A phase-level safety stop can leave the probe history valid but the action outcome UNKNOWN. Online deformable failures invalidate physics and prevent all supervised/response admission. New labels are recomputed from evidence at import, not read from material names. Historical E/F clearance-based labels are not automatically equivalent to this intervention-independent G contract.\n\n'+
        f'Current accepted core labels: `{dict(labels)}`; control episodes: {sum(e["is_control"] for e in episodes)}. Left/right counts and unique positive physical groups are exported separately. All unexecuted coverage goals remain pending.\n',encoding='utf-8')
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--trials-root',type=Path,default=OUT/'trials')
    ap.add_argument('--output',type=Path,default=OUT/'dataset');ap.add_argument('--report-dir',type=Path,default=REPORT)
    ap.add_argument('--no-legacy-index',action='store_true');args=ap.parse_args()
    if not args.output.resolve().is_relative_to(OUT.resolve()):raise ValueError('Output must remain within new G outputs')
    args.output.mkdir(parents=True,exist_ok=True);episodes=[];invalid=[]
    for path in sorted(args.trials_root.rglob('g_trial_metadata.json')):
        try:episodes.append(import_trial(path.parent,args.output))
        except (ValueError,KeyError,FileNotFoundError,TypeError) as exc:
            invalid.append(dict(source_path=rel(path.parent),reason=str(exc),status='EXCLUDED_OR_UNFINALIZED'))
    if len({e['trial_id'] for e in episodes})!=len(episodes):raise ValueError('DUPLICATE_TRIAL_ID')
    names,blocks,units=schema()
    arms=[i for i,n in enumerate(names) if any(j in n for j in UPPER_JOINTS) or 'hand_' in n]
    manifest=dict(schema='round4070-g.causal_episodes.v1',path_base='repository_root',action_contract=ACTION_ID,label_contract=CONTRACT,
        feature_names=names,feature_blocks=blocks,feature_units_by_block=units,joint_order=list(JOINTS),arm_indices=arms,
        body_indices=sorted(set(range(len(names)))-set(arms)),episodes=episodes,labels=dict(Counter(e['label'] for e in episodes)),
        same_legal_window='real probe/probe_hold samples, <=2s, before any followup',
        forbidden_learning_inputs=['scene_id','group_id','material','constraint nodes','wall','contact truth','future observations','absolute world pose','map','goal'],
        self_supervision='current train group only; no heldout UNKNOWN',status='PARTIAL_ACTUAL_DATA_ONLY')
    splits=make_splits(episodes);pairs=matched_pairs(episodes,splits)
    write_json(args.output/'manifest.json',manifest);write_json(args.output/'split_manifest.json',splits)
    write_json(args.output/'matched_pairs.json',pairs);write_json(args.output/'import_audit.json',dict(accepted=episodes,excluded=invalid))
    if not args.no_legacy_index:write_json(args.output/'legacy_response_manifest.json',legacy_manifest())
    print(json.dumps(reports(episodes,invalid,splits,args.output,args.report_dir),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
