"""Read-only G1 admission from immutable raw and independent audit evidence.

Runtime completion is not scientific admission. No retry selects a better label.
An implementation recapture must have a separate, explicit repair authorization.
"""
from pathlib import Path
import hashlib
import json


def read(path, default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return default


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auditor_source_hashes():
    folder=Path(__file__).parent
    names=['physics_g1_collision_audit.py','collision_geometry.py','collision_clearance.py','physics_history.py','physics_native_bind.py','g1_action_contract_v2.py','physics_wall_evidence.py']
    return {name:sha(folder/name) for name in names if (folder/name).is_file()}


def audit_state(run):
    run=Path(run);audit=read(run/'g_collision_audit.json',{})
    if not audit:return dict(status='NOT_VERIFIED',reason='INDEPENDENT_COLLISION_AUDIT_MISSING')
    if audit.get('schema')!='round4070-g.body_collision_audit.v1' or audit.get('sample_count',0)<=0:
        return dict(status='NOT_VERIFIED',reason='INDEPENDENT_COLLISION_AUDIT_CONTRACT_INVALID')
    history='g_body_collision_history_index.json' if (run/'g_body_collision_history_index.json').exists() else 'g_body_collision_history_audit_only.npz'
    required=['g_robot_collision_geometry.json',history,'g_outcome.json']
    sources=audit.get('source_files',{})
    for name in required:
        path=run/name
        if not path.is_file() or sources.get(name)!=sha(path):
            return dict(status='NOT_VERIFIED',reason='COLLISION_AUDIT_SOURCE_MISMATCH:'+name)
    for name,digest in sources.items():
        path=(run/name).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file() or sha(path)!=digest:
            return dict(status='NOT_VERIFIED',reason='COLLISION_AUDIT_BOUND_SOURCE_MISMATCH:'+name)
    if audit.get('status')=='PHYSICS_INVALID':return dict(status='PHYSICS_INVALID',reason='INDEPENDENT_GEOMETRY_PENETRATION',audit=audit)
    flags=['penetration_free','all_robot_collision_bodies','probe_followup_release_complete','finite']
    if audit.get('status')!='VERIFIED' or not all(audit.get(k) is True for k in flags):
        return dict(status='NOT_VERIFIED',reason=audit.get('reason','COLLISION_AUDIT_NOT_VERIFIED'),audit=audit)
    if history.endswith('index.json'):
        index=read(run/history,{})
        verified=audit.get('history_verification',{})
        complete=(audit.get('initial_state_collision_audited') is True and audit.get('initial_state_collision_status')=='VERIFIED'
                  and audit.get('geometry_checked_state_count')==audit['sample_count']+1
                  and verified.get('status')=='VERIFIED_HISTORY' and verified.get('sample_count')==audit['sample_count']
                  and audit.get('nominal_hz')==200 and index.get('sample_count')==audit['sample_count'])
        if not complete:return dict(status='NOT_VERIFIED',reason='V5_INITIAL_OR_COMPLETE_ACTUAL_STEP_AUDIT_MISSING')
        for name in ['g_body_collision_capture.json','g_native_geometry_verified.json','g_trial_metadata.json','g_action_tube_runtime.json','g_runner_contract.json']:
            if name not in sources:return dict(status='NOT_VERIFIED',reason='V5_REQUIRED_AUDIT_SOURCE_NOT_BOUND:'+name)
        if not verified.get('source_files') or any(sources.get(name)!=digest for name,digest in verified['source_files'].items()):
            return dict(status='NOT_VERIFIED',reason='V5_HISTORY_SOURCE_BINDINGS_INCOMPLETE')
        for entry in index.get('chunks',[]):
            path=(run/entry['path']).resolve()
            if not path.is_relative_to(run.resolve()) or not path.is_file() or sources.get(entry['path'])!=sha(path) or entry['sha256']!=sha(path):
                return dict(status='NOT_VERIFIED',reason='COLLISION_CHUNK_SOURCE_MISMATCH')
    return dict(status='VERIFIED',reason='INDEPENDENT_COLLISION_AUDIT_VERIFIED',audit=audit)


def rng_verified(run,plan):
    row=read(Path(run)/'g_reset_randomization.json',{})
    if not (row.get('schema')=='G1_RNG_V1' and row.get('readback_verified') is True and
            row.get('robot_pose_readback_verified') is True and row.get('seed')==plan.get('seed') and
            row.get('spec_sha256')==plan.get('g1_reset_randomization',{}).get('spec_sha256')):return False
    array=Path(run)/row.get('arrays_file','MISSING')
    return array.is_file() and row.get('arrays_sha256')==sha(array)


def trial_state(run,plan):
    outcome=read(Path(run)/'g_outcome.json',{})
    collision=audit_state(run)
    if collision['status']=='PHYSICS_INVALID':return collision
    if not outcome:return dict(status='EXECUTION_INCOMPLETE',reason='NO_FINAL_OUTCOME')
    if not outcome.get('runtime_valid') or not outcome.get('probe_valid'):
        return dict(status='EXECUTION_INVALID',reason='RUNTIME_OR_PROBE_INVALID')
    if not rng_verified(run,plan):return dict(status='NOT_VERIFIED',reason='RNG_READBACK_NOT_VERIFIED')
    if collision['status']!='VERIFIED':return collision
    if '200HZ' in plan.get('primary_protocol','') and collision['audit'].get('action_contract_audit',{}).get('status')!='VERIFIED':
        return dict(status='NOT_VERIFIED',reason='V5_ACTION_CONTRACT_AUDIT_NOT_VERIFIED')
    return dict(status='COMPLETE',reason='RAW_EXECUTION_RNG_AND_COLLISION_AUDIT_VERIFIED',audit=collision['audit'])


def retry_authorized(run,plan):
    """A failed acquisition may be recaptured only after a source-linked repair.

    Physical invalidity never qualifies. The existing complete raw should be
    re-audited first; recapture authorization explicitly records why that fails.
    """
    run=Path(run);state=trial_state(run,plan)
    if state['status'] in ('COMPLETE','PHYSICS_INVALID'):return False
    repair=read(run/'recapture_authorization.json',{})
    required=(repair.get('schema')=='round4070-g.recapture_authorization.v1' and
        repair.get('raw_reaudit_attempted') is True and repair.get('recapture_required') is True and
        repair.get('maximum_new_attempt')==2 and bool(repair.get('reason_existing_raw_insufficient')) and
        bool(repair.get('archived_repair_evidence')) and bool(repair.get('source_sha256_after_repair')))
    if not required:return False
    repository=Path(__file__).resolve().parents[2]
    for field in ['archived_repair_evidence','source_sha256_after_repair']:
        entries=repair.get(field)
        if not isinstance(entries,dict):return False
        for name,digest in entries.items():
            path=(repository/name).resolve()
            if not path.is_relative_to(repository) or not path.is_file() or sha(path)!=digest:return False
    return True


def variant_key(plan):
    return (plan.get('probe_side'),plan.get('posture_variant_id','nominal'),
            json.dumps(plan.get('robot',{}),sort_keys=True,separators=(',',':')))


def empty_dependency(plan,manifest,out):
    if plan['scene_id']=='EMPTY':return dict(ready=True,reason='EMPTY_SELF_VALIDATION')
    candidates=[p for p in manifest['trial_plans'] if p['scene_id']=='EMPTY' and
                p.get('primary_protocol')==plan.get('primary_protocol') and variant_key(p)==variant_key(plan)]
    inspected=[]
    for candidate in candidates:
        for run in sorted((Path(out)/'trials'/candidate['trial_id']).glob('attempt*')):
            state=trial_state(run,candidate);outcome=read(run/'g_outcome.json',{})
            clearance=state.get('audit',{}).get('candidate_clearance',{})
            threshold=float(manifest['scene_specs']['EMPTY']['geometry']['center_x_m'])+.10
            hold=(outcome.get('physical_probe_result',{}).get('responsive_traverse') or {}).get('stand') or {}
            stable=(hold.get('speed_gate_reached') is True and hold.get('finite_no_fall') is True and
                    float(hold.get('final_planar_speed_mps',float('inf')))<=.06 and
                    outcome.get('fall_reason') is None and outcome.get('external_safety_reason') is None)
            if '200HZ' in candidate.get('primary_protocol',''):
                action_audit=state.get('audit',{}).get('action_contract_audit',{})
                stable=stable and action_audit.get('status')=='VERIFIED' and action_audit.get('all_required_steps_valid') is True and action_audit.get('guard_violation') is None
            clear=(clearance.get('status')=='VERIFIED' and clearance.get('actual_collision_geometry') is True and
                   clearance.get('phase_final_sample') is True and clearance.get('phase')=='responsive_traverse_stand' and
                   clearance.get('comparison')=='>=' and abs(clearance.get('threshold_x_m',float('inf'))-threshold)<1e-9 and
                   clearance.get('min_world_x_m',float('-inf'))>=clearance.get('threshold_x_m',float('inf')))
            if state['status']=='COMPLETE' and outcome.get('safe_recovery') is True and clear and stable:
                return dict(ready=True,reason='MATCHING_EMPTY_VALIDATED',trial_id=candidate['trial_id'],
                            path=str(run),audit_sha256=sha(run/'g_collision_audit.json'))
            inspected.append(dict(trial_id=candidate['trial_id'],status=state['status'],reason=state['reason']))
    return dict(ready=False,reason='MATCHING_HAND_AND_ROBOT_VARIANT_EMPTY_NOT_VALIDATED',
                candidate_count=len(candidates),inspected=inspected)
