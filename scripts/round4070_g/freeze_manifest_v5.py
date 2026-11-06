"""Freeze the reviewed discrete-action/native-geometry amendment, never execute it.

Requires a validated, independently reviewed action sidecar. The existing scene
matrix, resets, split roles and controller limits are preserved. Activation is a
separate explicit command after the running collector has reached a safe boundary.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import csv
import hashlib
import json
import os

from freeze_manifest_v4 import data, exact_new

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/round4070_comprehensive_traversability'
DOC=ROOT/'research/round4070_comprehensive_traversability'
PROTOCOL='G1_RNG_V1_BODY_COLLISION_200HZ_ACTION_TUBE_V2'
ACTION='G_HOLD_PROBE_POSE_FORWARD24_TUBE_V2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def repository_file(name):
    path=(ROOT/name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('Missing repository-local frozen evidence: '+str(name))
    return path


def validate_contract(contract):
    from g1_action_contract_v2 import validate_contract as validate_runtime_contract
    validate_runtime_contract(contract,require_ready=True,verify_files=True)
    required=dict(schema='round4070-g.action-contract.v2',status='VALIDATED_READY_TO_FREEZE',
                  action_id=ACTION,sample_dt_s=.005,max_step_x_m=.020,
                  penetration_tolerance_m=.005,required_witness_interior_radius_m=.006)
    for key,value in required.items():
        if contract.get(key)!=value:
            raise ValueError('Unvalidated or unexpected action field: '+key)
    if contract.get('label_contract')!='PASSABLE_FOR_ACTION:'+ACTION:
        raise ValueError('Wrong label semantics')
    if contract.get('tube_bounds_m')!={'y':[-.35,.35],'z':[.45,1.25]}:
        raise ValueError('Unexpected action tube')
    witness=contract.get('witness',{})
    if witness.get('status')!='VERIFIED' or witness.get('evidence_mode')!='EXACT_NATIVE':
        raise ValueError('An actual exact-shape interior witness is required')
    if witness.get('verified_interior_radius_m',0)<.006:
        raise ValueError('Insufficient witness interior radius')
    if not witness.get('body_name') or not witness.get('native_shape_id'):
        raise ValueError('Missing native witness identity')
    point=witness.get('point_body_m')
    if not isinstance(point,list) or len(point)!=3 or not all(isinstance(x,(float,int)) and abs(x)<10 for x in point):
        raise ValueError('Invalid body-local witness point')
    library=repository_file(witness['native_library_path'])
    if sha(library)!=witness.get('native_library_sha256'):
        raise ValueError('Native library SHA256 mismatch')
    evidence=repository_file(witness['source_evidence_path'])
    if sha(evidence)!=witness.get('source_evidence_sha256'):
        raise ValueError('Native witness proof SHA256 mismatch')
    proof=read(evidence);native=read(library)
    if (proof.get('status')!='VERIFIED' or proof.get('native_library_sha256')!=sha(library)
            or native.get('status')!='VERIFIED_MIXED_EXACT_AND_CONSERVATIVE'
            or not native.get('geometry_complete') or not native.get('frame_verified')):
        raise ValueError('The frozen native proof/library pair is not verified')
    source_witness=proof.get('witness',{})
    for key in ['body_name','body_index','native_shape_id','point_body_m','verified_interior_radius_m','evidence_mode']:
        if witness.get(key)!=source_witness.get(key):
            raise ValueError('Action witness differs from the actual native proof: '+key)
    if source_witness.get('simulation_shape') is not True or witness.get('native_geometry_sha256')!=source_witness.get('native_part_geometry_sha256'):
        raise ValueError('Action witness is not bound to the actual simulation shape')
    return library,evidence


def freeze():
    target=OUT/'campaign_manifest_v5.json'
    if target.exists():
        raise FileExistsError('V5 already exists; never overwrite a frozen protocol')
    previous=OUT/'campaign_manifest_v4.json'
    old=read(previous);old_sha=sha(previous)
    if old_sha!=read(previous.with_suffix('.sha256.json'))['sha256']:
        raise RuntimeError('V4 source SHA256 mismatch')
    contract_path=OUT/'g1_action_contract_v2.json'
    contract=read(contract_path)
    library,witness_proof=validate_contract(contract)
    launches=sorted((OUT/'trials').glob('*/*/campaign_launch.json'))
    prior_primary={p['trial_id'] for p in old['trial_plans']}
    if any(p.parent.parent.name in prior_primary for p in launches):
        raise RuntimeError('V4 primary data already exist; this amendment requires an explicit separate data migration review')
    historical={r['launch_file']:r for r in old.get('prior_actual_g1_launches',[])}
    # V4's original freeze predates the explicit hash ledger. Its known paths
    # and the later actual pilot ledger cannot disappear on a fresh checkout.
    known_legacy=set(old.get('prior_pilots',{}).get('launch_files',[]))
    pilot_ledger=DOC/'PILOT_PROGRESS_LEDGER.csv'
    if pilot_ledger.exists():
        with pilot_ledger.open(encoding='utf-8-sig',newline='') as handle:
            known_legacy.update(str(Path(row['raw_path'])/'campaign_launch.json') for row in csv.DictReader(handle))
    for name in known_legacy:
        path=(ROOT/name).resolve()
        if not path.is_relative_to(OUT/'trials'):
            raise ValueError('Historical pilot path leaves the actual G1 launch directory')
        key=path.relative_to(ROOT).as_posix()
        if key not in historical:
            if not path.is_file():
                raise FileNotFoundError('Known historical G1 launch lacks its original SHA evidence; recover it before freezing: '+key)
            historical[key]=dict(launch_file=key,sha256=sha(path))
    for path in launches:
        key=path.relative_to(ROOT).as_posix()
        row=dict(launch_file=key,sha256=sha(path))
        if key in historical and historical[key]!=row:
            raise RuntimeError('Historical launch record changed')
        historical[key]=row
    if len(historical)+len(old['trial_plans'])>old['max_g1_launches']:
        raise RuntimeError('Insufficient remaining budget for the unchanged primary plan')
    artifacts={p.name:sha(p) for p in [contract_path,library,witness_proof]}
    if len(artifacts)!=3:
        raise ValueError('Evidence artifact basenames must be distinct')
    for path in [contract_path,library,witness_proof]:
        for destination in (DOC,OUT):
            exact_new(destination/path.name,path.read_bytes())
    result=copy.deepcopy(old)
    result.update(schema='round4070-g.campaign.v5',created_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_manifest=previous.name,predecessor_manifest_sha256=old_sha,
        supersedes_manifest_sha256=old_sha,revision_utc=datetime.now(timezone.utc).isoformat(),
        revision_reason='Explicit discrete action contract, actual 200 Hz geometry/phase evidence, and native exact/conservative-separation audit before primary collection.',
        config_directory='configs_v5',plan_directory='plans_v5',g1_primary_protocol=PROTOCOL,
        label_contract='PASSABLE_FOR_ACTION:'+ACTION,
        body_collision_audit_hz=200,
        action_contract_v2=contract,action_contract_v2_sha256=sha(contract_path),frozen_artifacts=artifacts,
        prior_actual_g1_launches=list(historical.values()),
        predecessor_pilot_snapshot=old.get('prior_pilots'),
        prior_pilots=dict(status='SUPERSEDED_DIAGNOSTIC_ONLY_NOT_PRIMARY',actual_launches_at_freeze=len(historical),
            launch_files=list(historical),reason='Old pilots have neither the revised action contract nor complete primary evidence; original raw is retained and every launch consumes budget.'),
        reserved_retry_launches=old['max_g1_launches']-len(historical)-len(old['trial_plans']),
        retry_reserve=old['max_g1_launches']-len(historical)-len(old['trial_plans']),
        total_budget_rule='Count union of frozen historical launch paths and all local actual G1 launch paths; failed attempts and retries count. Missing old raw never restores budget.',
        primary_admission='V5-only explicit seeded reset, complete actual 200 Hz step chain, native exact/conservative-separation geometry audit, real phase ledger, and discrete action tube. Unsupported evidence excluded even from response fitting.',
        amendment='Pre-primary evidence repair: finite-wall V1 labels remain UNKNOWN; new explicit discrete action has a validated exact-shape witness and step bound. Existing physical scenes/controller parameters/split roles unchanged. No formal G model was fitted before this amendment.',
        collision_scope='All actual 200 Hz PhysX states; no continuous-time or real-world guarantee. Native outer-envelope separation suffices; outer-envelope overlap is NOT_VERIFIED, never proof of actual penetration.')
    result['execution']['minimum_output_disk_free_gib']=20
    result['predecessor_source_sha256']=old.get('source_sha256',{})
    result['source_sha256']={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    for plan in result['trial_plans']:
        previous_id=plan['trial_id']
        plan.update(schema='round4070-g.trial.v3',trial_id=previous_id+'_act2',
            predecessor_planned_trial_id=previous_id,primary_protocol=PROTOCOL,
            followup_action_id=ACTION,label_contract='PASSABLE_FOR_ACTION:'+ACTION,
            action_contract_v2=contract,action_contract_v2_sha256=sha(contract_path),
            body_collision_audit_hz=200,body_collision_audit_required=True)
    if result['scene_specs']!=old['scene_specs'] or len({p['trial_id'] for p in result['trial_plans']})!=976:
        raise RuntimeError('The scene matrix or primary plan cardinality changed')
    for sid,spec in result['scene_specs'].items():
        exact_new(OUT/'configs_v5'/(sid+'.json'),data(spec))
    for plan in result['trial_plans']:
        exact_new(OUT/'plans_v5'/(plan['trial_id']+'.json'),data(plan))
    frozen=data(result);digest=hashlib.sha256(frozen).hexdigest()
    for destination in (OUT,DOC):
        exact_new(destination/target.name,frozen)
        exact_new((destination/target.name).with_suffix('.sha256.json'),data(dict(path=target.name,sha256=digest)))
    print(json.dumps(dict(status='FROZEN_NOT_ACTIVATED_NOT_EXECUTED',sha256=digest,
        planned=976,historical_launches=len(historical),retry_reserve=result['reserved_retry_launches'])))


def activate():
    # Do not silently change the manifest under a live collector/workflow.
    for name in ['workflow_lock.json','campaign_lock.json','resource_lock.json']:
        if (OUT/name).exists():
            raise RuntimeError('Reach a verified idle boundary before activation: '+name)
    if not (OUT/'PAUSE_G1.flag').exists():
        raise RuntimeError('Activation must retain the G1 admission pause until the controlled smoke handoff')
    target=OUT/'campaign_manifest_v5.json';manifest=read(target);digest=sha(target)
    if digest!=read(target.with_suffix('.sha256.json'))['sha256'] or (DOC/target.name).read_bytes()!=target.read_bytes():
        raise RuntimeError('V5 frozen bytes mismatch')
    for name,expected in manifest['frozen_artifacts'].items():
        if sha(DOC/name)!=expected or sha(OUT/name)!=expected:
            raise RuntimeError('V5 artifact mismatch: '+name)
    pointer=DOC/'CURRENT_CAMPAIGN.json'
    old=read(pointer)
    if old.get('manifest') not in ['campaign_manifest_v4.json',target.name]:
        raise RuntimeError('Unexpected predecessor activation pointer')
    archived=OUT/'protocol_activation_history'/(old['manifest']+'.pointer.json')
    exact_new(archived,pointer.read_bytes())
    temporary=pointer.with_suffix('.json.tmp')
    with temporary.open('wb') as handle:
        handle.write(data(dict(manifest=target.name,sha256=digest,primary_protocol=PROTOCOL)))
        handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,pointer)
    print('ACTIVATED_WITH_G1_ADMISSION_STILL_PAUSED',target.name,digest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activate',action='store_true')
    arguments=parser.parse_args()
    activate() if arguments.activate else freeze()
