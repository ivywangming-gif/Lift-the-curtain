"""Discrete simulator action contract; environment evidence is never a feature.

This module does not select a witness, freeze a campaign, or prove a label from
an online guard. Independent native geometry and collision auditing are required.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ACTION_ID='G_HOLD_PROBE_POSE_FORWARD24_TUBE_V2'
LABEL_CONTRACT='PASSABLE_FOR_ACTION:'+ACTION_ID
PRIMARY_PROTOCOL='G1_RNG_V1_BODY_COLLISION_200HZ_ACTION_TUBE_V2'
SIDECAR_NAME='g1_action_contract_v2.json'
ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs/round4070_comprehensive_traversability'


def draft_contract():
    return dict(schema='round4070-g.action-contract.v2',status='DRAFT_PENDING_NATIVE_WITNESS',
        action_id=ACTION_ID,label_contract=LABEL_CONTRACT,primary_protocol=PRIMARY_PROTOCOL,
        sample_dt_s=.005,max_step_x_m=.020,tube_bounds_m=dict(y=[-.35,.35],z=[.45,1.25]),
        penetration_tolerance_m=.005,required_witness_interior_radius_m=.006,
        witness=dict(status='PENDING',native_geometry_sha256=None,native_library_path=None,native_library_sha256=None,
            body_name=None,body_index=None,native_shape_id=None,point_body_m=None,verified_interior_radius_m=None,
            evidence_mode='EXACT_NATIVE',source_evidence_path=None,source_evidence_sha256=None),
        completion=dict(candidate_actual_collider_min_x_offset_m=.10,candidate_region_half_extent_x_m=.06,
            existing_clear_margin_m=.04,stable_hold_required=True,safe_recovery_required=True,
            note='Original completion unchanged; conservative envelope minX is a sufficient lower bound'),
        phase_scope=dict(entry='actual final physical_probe_hold state before followup selection',
            guarded_interval='every actual step after entry until the first release/retract step',
            recovery_start_phases=['physical_retract','responsive_release'],
            recovery_exempt=True,violation_label='LABEL_UNKNOWN',violation_latched=True),
        audit_sampling=dict(nominal_hz=200,history_layout='CHUNKED_NPZ_V2',chunk_steps=500,
            initial_reset_state_separate=True,every_actual_physics_step=True,no_interpolation=True,
            arrays=['body_positions_world','body_quaternions_wxyz','nodes_world','node_velocities_world']),
        geometry_evidence_modes=dict(EXACT_NATIVE='Actual native primitive or consistent native convex faces',
            NATIVE_VERTEX_CONSERVATIVE_ENVELOPE='Complete native vertex outer hull: separation sufficient; overlap NOT_VERIFIED; minX conservative'),
        unchanged_control=dict(controller_hz=50,legal_observation_hz=50,physics_hz=200,maximum_forward_segments=24,
            maximum_segment_step_m=.12,segment_timeout_s=7.,cruise_speed_mps=.25,vx_max_mps=.30,
            exit_xy_m=[1.3,0.],release_s=1.5,neutralize_s=1.5,recovery_stand_s=1.5),
        proof_scope='Actual discrete 200Hz PhysX states only; no continuous-time or real-world guarantee',
        static_blocking_requirements=['actual entry x<=eroded wall low x','actual enabled fixed wall covers tube with >5mm margins',
            'max_step_x_m < eroded wall width','unchanged completion implies witness x>eroded wall low x',
            'verified exact witness interior radius>=6mm','valid probe, actual executed steps, and independent physics audit',
            'guard violation or supervisor refusal alone never proves BLOCKED'],
        model_input=False)


def validate_contract(value,*,require_ready=True,verify_files=True):
    reference=draft_contract()
    for key in ('schema','action_id','label_contract','primary_protocol','sample_dt_s','max_step_x_m','tube_bounds_m',
                'penetration_tolerance_m','required_witness_interior_radius_m','completion','phase_scope','audit_sampling','unchanged_control'):
        if value.get(key)!=reference[key]:raise ValueError('Action V2 frozen specification mismatch: '+key)
    if not require_ready:return value
    w=value.get('witness',{})
    if value.get('status')!='VALIDATED_READY_TO_FREEZE' or w.get('status')!='VERIFIED' or w.get('evidence_mode')!='EXACT_NATIVE':
        raise ValueError('Action V2 requires independently verified exact native witness')
    point=np.asarray(w.get('point_body_m'),float)
    if point.shape!=(3,) or not np.isfinite(point).all() or float(w.get('verified_interior_radius_m',0))<.006:
        raise ValueError('Invalid actual witness point/interior evidence')
    if not isinstance(w.get('body_index'),int) or w['body_index']<0 or not w.get('body_name') or not w.get('native_shape_id'):
        raise ValueError('Witness native body/shape mapping missing')
    for name in ('native_geometry_sha256','native_library_sha256','source_evidence_sha256'):
        if not isinstance(w.get(name),str) or len(w[name])!=64:raise ValueError('Missing witness evidence hash: '+name)
    if verify_files:
        documents={}
        for path_key,sha_key in (('native_library_path','native_library_sha256'),('source_evidence_path','source_evidence_sha256')):
            path=Path(w[path_key]);path=path if path.is_absolute() else ROOT/path
            if hashlib.sha256(path.read_bytes()).hexdigest()!=w[sha_key]:raise ValueError('Native witness source hash mismatch: '+path_key)
            documents[path_key]=json.loads(path.read_text(encoding='utf-8-sig'))
        library=documents['native_library_path'];proof=documents['source_evidence_path'];pw=proof.get('witness',{})
        if proof.get('status')!='VERIFIED' or proof.get('native_library_sha256')!=w['native_library_sha256'] or pw.get('status')!='VERIFIED':
            raise ValueError('Native witness proof is not verified/bound to library')
        for key in ('body_name','body_index','point_body_m','native_shape_id','verified_interior_radius_m','evidence_mode'):
            if w.get(key)!=pw.get(key):raise ValueError('Action witness differs from native proof: '+key)
        actual=library['action_witness'];part=library['parts'][actual['part_index']]
        part_sha=hashlib.sha256(json.dumps(part,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if part_sha!=w['native_geometry_sha256'] or pw.get('native_part_geometry_sha256')!=part_sha:
            raise ValueError('Exact witness native part hash mismatch')
        if part.get('evidence_mode')!='EXACT_NATIVE' or part.get('inside_witness_allowed') is not True or part.get('type')!='convex':
            raise ValueError('Witness cannot use a conservative or unsupported shape')
        planes=np.asarray(part['planes_body_m'],float);norms=np.linalg.norm(planes[:,:3],axis=1)
        distances=-(planes[:,:3]@point+planes[:,3])/norms
        if not np.isfinite(distances).all() or float(distances.min())<.006 or abs(float(distances.min())-w['verified_interior_radius_m'])>1e-9:
            raise ValueError('Exact native witness interior radius failed independent replay')
    return value


def load_plan_contract(plan):
    if plan.get('primary_protocol')!=PRIMARY_PROTOCOL:raise ValueError('Action V2 primary protocol mismatch')
    path=OUTPUT/SIDECAR_NAME
    content=path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=plan.get('action_contract_v2_sha256'):raise ValueError('Frozen action sidecar hash missing or mismatched')
    value=validate_contract(json.loads(content))
    if plan.get('action_contract_v2')!=value or plan.get('body_collision_audit_hz')!=200:
        raise ValueError('Embedded frozen action sidecar or audit rate differs')
    if plan.get('followup_action_id')!=ACTION_ID or plan.get('label_contract')!=LABEL_CONTRACT:
        raise ValueError('Frozen action identifiers differ from sidecar')
    return value


def witness_world(sample,witness):
    i=int(witness['body_index']);q=np.asarray(sample['body_quaternions_wxyz'][i],float)
    norm=float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm==0:return np.full(3,np.nan)
    q=q/norm
    p=np.asarray(witness['point_body_m'],float);v=q[1:]
    return np.asarray(sample['body_positions_world'][i],float)+p+2*np.cross(v,np.cross(v,p)+q[0]*p)


class ActionTubeGuard:
    """An evaluation-only supervisor; a violation can only stop and yield U."""
    def __init__(self,contract=None):
        self.contract=contract;self.entry=None;self.previous=None;self.violation=None;self.records=[];self.recovery_started=False

    def begin(self,sample):
        if self.contract is None:return None
        if sample is None or sample['phase']!='physical_probe_hold':
            self.violation='ACTION_TUBE_ENTRY_STATE_UNAVAILABLE';return self.violation
        self.entry=dict(physics_step=sample['physics_step'],time_s=sample['time_s'],
                        witness_world_m=witness_world(sample,self.contract['witness']).tolist())
        self.previous=(int(sample['physics_step']),np.asarray(self.entry['witness_world_m']))
        return self._check(sample,self.previous[1],entry=True)

    def _check(self,sample,p,entry=False):
        if self.violation:return self.violation
        bounds=self.contract['tube_bounds_m']
        if not np.isfinite(p).all():self.violation='ACTION_TUBE_NONFINITE'
        elif not (bounds['y'][0]<=p[1]<=bounds['y'][1] and bounds['z'][0]<=p[2]<=bounds['z'][1]):
            self.violation='ACTION_TUBE_BOUNDS_VIOLATION'
        elif not entry and self.previous is not None:
            step,old=self.previous
            if sample['physics_step']!=step+1:self.violation='ACTION_TUBE_PHYSICS_STEP_GAP'
            elif abs(p[0]-old[0])>self.contract['max_step_x_m']:self.violation='ACTION_TUBE_STEP_X_VIOLATION'
        self.previous=(int(sample['physics_step']),p.copy())
        if self.violation:self.records.append(dict(reason=self.violation,physics_step=sample['physics_step'],witness_world_m=p.tolist()))
        return self.violation

    def observe(self,sample):
        if self.contract is None or self.entry is None:return None
        if sample['phase'] in self.contract['phase_scope']['recovery_start_phases']:self.recovery_started=True
        if self.recovery_started:return None  # Existing recovery remains available even after a latched violation.
        return self._check(sample,witness_world(sample,self.contract['witness']))

    def evidence(self):
        return dict(schema='round4070-g.action_tube_runtime.v2',enabled=self.contract is not None,
            entry=self.entry,violation=self.violation,violations=self.records,model_input=False,
            status='REQUIRES_INDEPENDENT_200HZ_AUDIT' if self.contract else 'NOT_APPLICABLE_V1')


def separation_certificate(contract,wall,*,entry_x,candidate_center_x):
    """Mathematical candidate certificate; physical verification is separate."""
    validate_contract(contract,require_ready=False)
    center=np.asarray(wall.get('center_xyz'),float);size=np.asarray(wall.get('size_xyz'),float)
    if center.shape!=(3,) or size.shape!=(3,) or not np.isfinite(np.r_[center,size,entry_x,candidate_center_x]).all() or np.any(size<=0):
        return dict(mathematically_sufficient=False,reason='INVALID_ACTUAL_WALL_OR_ENTRY')
    low,high=center-size/2,center+size/2;d=contract['penetration_tolerance_m'];tube=contract['tube_bounds_m']
    ilow,ihigh=low[0]+d,high[0]-d
    endpoint_lower=candidate_center_x+contract['completion']['candidate_actual_collider_min_x_offset_m']+contract['required_witness_interior_radius_m']
    checks=dict(fixed=wall.get('fixed') is True,
        tube_strictly_in_wall_y=low[1]+d<tube['y'][0] and tube['y'][1]<high[1]-d,
        tube_strictly_in_wall_z=low[2]+d<tube['z'][0] and tube['z'][1]<high[2]-d,
        step_strictly_less_than_eroded_width=contract['max_step_x_m']<ihigh-ilow,
        actual_entry_at_or_before_eroded_low=entry_x<=ilow,
        completion_strictly_beyond_eroded_low=endpoint_lower>ilow)
    return dict(mathematically_sufficient=all(checks.values()),checks=checks,eroded_x_open_interval_m=[float(ilow),float(ihigh)],
        actual_entry_x_m=float(entry_x),completion_witness_x_lower_bound_m=float(endpoint_lower),
        physical_evidence_verified=False,scope='DISCRETE_STATE_GEOMETRIC_LEMMA_ONLY_NOT_A_LABEL')


if __name__=='__main__':
    path=OUTPUT/SIDECAR_NAME
    if path.exists():raise FileExistsError('Draft or frozen sidecar already exists; do not replace implicitly')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(draft_contract(),indent=2)+'\n',encoding='utf-8')
    print(path)
