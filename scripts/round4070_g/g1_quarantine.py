"""Explicit, evidence-bound scheduling quarantine; never scientific admission.

No collector, label, campaign, budget, or raw file is mutated here. Only a
separately authorized exact-plan record can withhold one already observed local
uncertainty while other plans retain their complete ordinary admission checks.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/round4070_comprehensive_traversability'
SCHEMA='round4070-g.exact_plan_quarantine.v1'
CLASSIFICATION='ROOT_REVIEWED_CONSERVATIVE_NATIVE_ENVELOPE_WALL_OVERLAP'
ALLOWED_UNCERTAINTY={'NATIVE_VERTEX_CONSERVATIVE_ENVELOPE_OVERLAP_NOT_VERIFIED','CONSERVATIVE_ENVELOPE_OVERLAP_NOT_VERIFIED'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def classify_local_uncertainty(audit,native):
    """A narrow eligibility check for root review, never an automatic quarantine."""
    if audit.get('schema')!='round4070-g.body_collision_audit.v1' or audit.get('status')!='NOT_VERIFIED' or audit.get('penetration_free') is not None:
        raise ValueError('NOT_THE_AUTHORIZED_PENDING_COLLISION_STATE')
    if audit.get('reason')!='CONSERVATIVE_ENVELOPE_OR_NUMERICAL_OVERLAP_NOT_VERIFIED':
        raise ValueError('UNCLASSIFIED_AUDIT_FAILURE_REQUIRES_GLOBAL_PAUSE')
    if not all(audit.get(k) is True for k in ['all_robot_collision_bodies','probe_followup_release_complete','finite','initial_state_collision_audited']):
        raise ValueError('INCOMPLETE_COLLISION_EVIDENCE_REQUIRES_GLOBAL_PAUSE')
    if audit.get('initial_state_collision_status')!='VERIFIED' or audit.get('geometry_checked_state_count')!=audit.get('sample_count',0)+1:
        raise ValueError('INITIAL_OR_COMPLETE_STATE_AUDIT_MISSING')
    if audit.get('main_strict_penetration_tolerance_m')!=.005 or audit.get('sensitivity_tolerances_m')!=[0.,.001,.005]:
        raise ValueError('FROZEN_PENETRATION_TOLERANCES_CHANGED')
    counts=audit.get('strict_inner_triangle_intersections_by_tolerance_m',{})
    if any(counts.get(k)!=0 for k in ['0.0','0.001','0.005']) or audit.get('penetration_witnesses'):
        raise ValueError('EXACT_PENETRATION_EVIDENCE_CANNOT_BE_QUARANTINED_AS_LOCAL_UNCERTAINTY')
    action=audit.get('action_contract_audit',{})
    if action.get('status')!='VERIFIED' or action.get('all_required_steps_valid') is not True or action.get('guard_violation') is not None:
        raise ValueError('ACTION_OR_GUARD_FAILURE_REQUIRES_GLOBAL_PAUSE')
    history=audit.get('history_verification',{})
    if history.get('status')!='VERIFIED_HISTORY' or history.get('sample_count')!=audit.get('sample_count') or audit.get('nominal_hz')!=200:
        raise ValueError('FULL_200HZ_HISTORY_NOT_VERIFIED')
    if not all(history.get(k) is True for k in ['phase_ledger_verified','execution_anchors_verified','initial_state_verified']):
        raise ValueError('HISTORY_ANCHOR_OR_PHASE_FAILURE_REQUIRES_GLOBAL_PAUSE')
    if native.get('status')!='VERIFIED' or not native.get('geometry_complete') or not native.get('frame_verified'):
        raise ValueError('NATIVE_GEOMETRY_IDENTITY_NOT_VERIFIED')
    query=audit.get('scene_query_configuration_audit',{})
    if query.get('status')!='VERIFIED' or query.get('actual_native_readback_enabled') is not True or query.get('only_query_infrastructure_changed') is not True:
        raise ValueError('NATIVE_QUERY_FACILITY_NOT_VERIFIED')
    if audit.get('actual_wall_entity_audit',{}).get('status')!='VERIFIED':
        raise ValueError('ACTUAL_WALL_NOT_VERIFIED')
    uncertain=audit.get('unverified_geometry_interactions',[])
    if not uncertain:raise ValueError('NO_RECORDED_LOCAL_ENVELOPE_WALL_EVIDENCE')
    part_indices=set()
    for row in uncertain:
        if row.get('reason') not in ALLOWED_UNCERTAINTY or row.get('surface')!='wall':
            raise ValueError('OTHER_UNCLASSIFIED_GEOMETRY_FAILURE_REQUIRES_GLOBAL_PAUSE')
        index=row.get('part_index')
        if not isinstance(index,int) or not 0<=index<len(native['parts']):raise ValueError('UNCERTAINTY_PART_IDENTITY_INVALID')
        part=native['parts'][index]
        if part.get('evidence_mode')!='NATIVE_VERTEX_CONSERVATIVE_ENVELOPE' or part.get('exact_intersection_allowed') is not False:
            raise ValueError('UNCERTAINTY_NOT_BOUND_TO_CONSERVATIVE_NATIVE_PART')
        step=row.get('physics_step')
        if not isinstance(step,int) or not 0<=step<=audit['sample_count']:raise ValueError('UNCERTAINTY_STEP_OUTSIDE_VERIFIED_HISTORY')
        part_indices.add(index)
    return dict(classification=CLASSIFICATION,recorded_uncertainty_count=len(uncertain),
        affected_parts=[dict(part_index=i,body_name=native['parts'][i]['body_name'],evidence_mode=native['parts'][i]['evidence_mode']) for i in sorted(part_indices)],
        exact_5mm_penetration_count=0,history_sample_count=audit['sample_count'],geometry_checked_state_count=audit['geometry_checked_state_count'],
        interpretation='Scheduling-only root classification of this exact audit; conservative overlap remains NOT_VERIFIED under frozen V5',
        diagnostic_limit='Stored uncertainty witnesses are capped; this classification does not assert that the finite witness list exhausts every overlap, and does not promote any label')


def inspect_evidence(run,plan):
    """Hash/read-only verification, no collision replay and no simulator launch."""
    from build_dataset import frozen_evidence,randomization_evidence
    from g1_action_contract_v2 import PRIMARY_PROTOCOL
    from g1_trial import assert_runtime_plan_matches
    from physics_history import verify_history
    from physics_native_bind import bind_trial_geometry
    from physics_wall_evidence import verified_wall_for_audit
    run=Path(run).resolve()
    if not run.is_relative_to((OUT/'trials').resolve()):raise ValueError('QUARANTINE_RUN_OUTSIDE_ACTUAL_G_TRIALS')
    meta=read(run/'g_trial_metadata.json');runtime=meta['plan'];proof=read(run/'g_runner_contract.json')
    if runtime.get('primary_protocol')!=PRIMARY_PROTOCOL:raise ValueError('QUARANTINE_REQUIRES_FROZEN_V5')
    provenance=frozen_evidence(run,meta,runtime)
    campaign=Path(proof['verified_empty_control_admission']['frozen_campaign_path']);manifest=read(campaign)
    frozen=next(p for p in manifest['trial_plans'] if p['trial_id']==runtime['trial_id'])
    candidate=dict(plan);candidate['physics_valid']=True
    assert_runtime_plan_matches(frozen,candidate,verified_gate_seeds=provenance['physics_gate_seeds'])
    if plan['trial_id']!=runtime['trial_id'] or run.parent.name!=plan['trial_id']:raise ValueError('EXACT_PLAN_RUN_IDENTITY_MISMATCH')
    outcome=read(run/'g_outcome.json')
    if outcome.get('trial_id')!=plan['trial_id'] or outcome.get('status')!='G_FIXED_ACTION_FINISHED':raise ValueError('FINALIZED_OUTCOME_IDENTITY_MISSING')
    if not all(outcome.get(k) is True for k in ['runtime_valid','probe_valid','physics_valid','safe_recovery']):raise ValueError('EXECUTION_FAILURE_CANNOT_USE_LOCAL_QUARANTINE')
    if outcome.get('label')!='LABEL_UNKNOWN':raise ValueError('QUARANTINE_MUST_PRESERVE_ORIGINAL_ONLINE_UNKNOWN')
    audit_path=run/'g_collision_audit.json';digest=sha(audit_path);immutable=run/f'g_collision_audit_{digest}.json'
    if not immutable.is_file() or sha(immutable)!=digest:raise ValueError('ORIGINAL_IMMUTABLE_AUDIT_COPY_MISSING')
    audit=read(immutable);sources=dict(audit['source_files'])
    for name,value in sources.items():
        path=(run/name).resolve()
        if not path.is_relative_to(run) or not path.is_file() or sha(path)!=value:raise ValueError('AUDIT_RAW_SOURCE_CHANGED:'+name)
    geometry=read(run/'g_robot_collision_geometry.json');native=read(run/'g_native_geometry_verified.json');capture=read(run/'g_body_collision_capture.json')
    classification=classify_local_uncertainty(audit,native)
    history=verify_history(run,capture,geometry)
    if history.get('status')!='VERIFIED_HISTORY' or history['sample_count']!=audit['sample_count']:raise ValueError('HISTORY_REPLAY_DISAGREEMENT')
    if any(sources.get(k)!=v for k,v in history['source_files'].items()):raise ValueError('HISTORY_SOURCE_BINDING_INCOMPLETE')
    rebound=bind_trial_geometry(run,runtime['action_contract_v2'],write_output=False,archived_query_source_sha256=native['binder_source_sha256'])
    if rebound!=native:raise ValueError('NATIVE_BINDING_REPLAY_DISAGREEMENT')
    wall_sources={};wall=verified_wall_for_audit(run,geometry,wall_sources)
    if wall is None or audit['actual_wall_entity_audit']!=dict(status='VERIFIED',actual_geometry=wall):raise ValueError('ACTUAL_WALL_REPLAY_DISAGREEMENT')
    if any(sources.get(k)!=v for k,v in wall_sources.items()):raise ValueError('WALL_SOURCE_BINDING_INCOMPLETE')
    query=read(run/'g_scene_query_configuration.json');query_sha=sha(run/'g_scene_query_configuration.json')
    binding=dict(path='g_scene_query_configuration.json',sha256=query_sha)
    if capture.get('scene_query_configuration')!=binding or proof.get('scene_query_configuration_evidence',{}).get('sha256')!=query_sha or sources.get(binding['path'])!=query_sha:
        raise ValueError('QUERY_CAPTURE_RUNNER_AUDIT_BINDING_MISMATCH')
    if query.get('actual_native_readback_enabled') is not True or query.get('required_for_actual_wall_evidence') is not True or query.get('changed_configuration_fields')!=['enable_scene_query_support']:
        raise ValueError('ACTUAL_QUERY_CONFIGURATION_INVALID')
    rng=randomization_evidence(run,runtime);reset=read(run/'g_reset_randomization.json')
    for name in ['g_reset_randomization.json',reset['arrays_file'],'proprioception_50hz.jsonl','campaign_launch.json']:
        sources[name]=sha(run/name)
    if outcome.get('legal_history_sha256')!=sources['proprioception_50hz.jsonl']:raise ValueError('LEGAL_HISTORY_HASH_CHANGED')
    archives=proof['runtime_source_archive']
    for name,row in archives.items():
        if sha(row['path'])!=row['sha256']:raise ValueError('RUNTIME_SOURCE_ARCHIVE_CHANGED:'+name)
    return dict(run_path=relative(run),trial_id=plan['trial_id'],attempt_id=run.name,primary_protocol=runtime['primary_protocol'],
        physical_config_id=runtime['physical_config_id'],probe_side=runtime['probe_side'],seed=runtime['seed'],
        frozen_campaign_path=relative(campaign),frozen_campaign_sha256=sha(campaign),frozen_plan_sha256=canonical(frozen),
        immutable_audit_path=relative(immutable),immutable_audit_sha256=digest,auditor_source_sha256=audit['auditor_source_sha256'],
        raw_source_sha256=sources,runtime_source_archive=archives,rng_readback_verified=rng['readback_verified'],
        existing_attempt_directories=sorted(p.name for p in run.parent.glob('attempt*') if p.is_dir()),classification_evidence=classification)


def create_authorized_quarantine(run,plan,authorization,quarantine_root=None):
    if authorization.get('authorized_by')!='/root' or authorization.get('scope')!='EXACT_PLAN' or authorization.get('trial_id')!=plan.get('trial_id') or not authorization.get('instruction'):
        raise ValueError('EXPLICIT_ROOT_EXACT_PLAN_AUTHORIZATION_REQUIRED')
    evidence=inspect_evidence(run,plan)
    record=dict(schema=SCHEMA,status='ACTIVE_WITHHOLD_ONLY',scope='EXACT_PLAN',authorization=authorization,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),evidence=evidence,
        automatic_retry_permitted=False,changes_labels=False,changes_campaign_or_budget=False,
        formal_status_retained='NOT_VERIFIED',supervision_or_response_admission=False,
        release_conditions=['Explicit root reclassification tied to any new raw/audit/source evidence; never automatic expiry',
            'A new exact geometry/protocol version requires separate frozen admission; 5mm support diagnostics cannot upgrade current V5',
            'Any new attempt, changed source hash, identity, RNG, history or audit requires global pause until reviewed'])
    folder=Path(quarantine_root or OUT/'quarantine');folder.mkdir(parents=True,exist_ok=True)
    path=folder/(plan['trial_id']+'.json');digest_path=path.with_suffix('.sha256')
    if path.exists() or digest_path.exists():raise FileExistsError('Never overwrite an authorized quarantine record')
    content=(json.dumps(record,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    with path.open('xb') as f:f.write(content)
    with digest_path.open('x',encoding='ascii') as f:f.write(hashlib.sha256(content).hexdigest()+'\n')
    return record


def quarantine_state(run,plan,quarantine_root=None):
    """Selector: NOT_QUARANTINED never bypasses the collector's ordinary checks."""
    identity=plan.get('trial_id')
    if not isinstance(identity,str) or Path(identity).name!=identity:
        return dict(status='GLOBAL_PAUSE_REQUIRED',withhold=True,reason='UNSAFE_PLAN_IDENTITY')
    path=Path(quarantine_root or OUT/'quarantine')/(identity+'.json');digest_path=path.with_suffix('.sha256')
    if not path.exists() and not digest_path.exists():return dict(status='NOT_QUARANTINED',withhold=False,reason='NO_EXPLICIT_QUARANTINE_RECORD')
    try:
        if not path.is_file() or not digest_path.is_file() or sha(path)!=digest_path.read_text(encoding='ascii').strip():raise ValueError('QUARANTINE_RECORD_HASH_MISMATCH')
        record=read(path);authorization=record.get('authorization',{})
        if record.get('schema')!=SCHEMA or record.get('scope')!='EXACT_PLAN' or record.get('status')!='ACTIVE_WITHHOLD_ONLY':raise ValueError('QUARANTINE_SCOPE_OR_STATUS_MISMATCH')
        if authorization.get('authorized_by')!='/root' or authorization.get('scope')!='EXACT_PLAN' or authorization.get('trial_id')!=identity or not authorization.get('instruction'):raise ValueError('ROOT_AUTHORIZATION_IDENTITY_MISMATCH')
        if record.get('automatic_retry_permitted') is not False or record.get('changes_labels') is not False or record.get('changes_campaign_or_budget') is not False or record.get('supervision_or_response_admission') is not False or record.get('formal_status_retained')!='NOT_VERIFIED':raise ValueError('QUARANTINE_ATTEMPTS_TO_CHANGE_SCIENTIFIC_OR_BUDGET_STATE')
        current=inspect_evidence(run,plan)
        if current!=record.get('evidence'):raise ValueError('QUARANTINE_EVIDENCE_CHANGED_RECLASSIFICATION_REQUIRED')
        return dict(status='VALIDATED_WITHHOLD',withhold=True,reason=CLASSIFICATION,trial_id=identity,
            attempt_path=current['run_path'],quarantine_record_path=relative(path),quarantine_record_sha256=sha(path),
            immutable_audit_sha256=current['immutable_audit_sha256'],automatic_retry_permitted=False,
            formal_status_retained='NOT_VERIFIED',scope='EXACT_PLAN')
    except (ValueError,RuntimeError,KeyError,TypeError,OSError,StopIteration) as exc:
        return dict(status='GLOBAL_PAUSE_REQUIRED',withhold=True,reason=type(exc).__name__+': '+str(exc),trial_id=identity)
