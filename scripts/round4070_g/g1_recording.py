"""Simulator-independent G recorder and frozen action-label contract.

Only robot-owned values enter proprioception_50hz.jsonl. Environment truth is
written separately by finish(), after the legal observation cutoff.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

CONTRACT = "PASSABLE_FOR_ACTION:G_HOLD_PROBE_POSE_FORWARD24_V1"
ACTION_ID = "G_HOLD_PROBE_POSE_FORWARD24_V1"
ONLINE_LABEL_REVISION = "G_FINITE_WALL_STATIC_B_GAP_V1"
DT = .005
PROBE_PHASES = {"physical_probe", "physical_probe_hold"}


def clean(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, Path): return str(x)
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)


def rotation(rpy):
    r,p,y = rpy; cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def action_spec(plan):
    from g1_action_contract_v2 import PRIMARY_PROTOCOL as V5_PROTOCOL,ACTION_ID as V2_ACTION,LABEL_CONTRACT as V2_CONTRACT
    robot = plan.get("robot", plan.get("robot_conditions", {}))
    side = plan.get("probe_side", robot.get("probe_side", "right"))
    followup = plan.get("followup_hand", side)
    if followup != side:
        raise ValueError("V1 only validates hold of the just-probed hand; other followup hands require a new action contract")
    v2=plan.get('primary_protocol')==V5_PROTOCOL
    return dict(action_id=V2_ACTION if v2 else ACTION_ID, label_contract=V2_CONTRACT if v2 else CONTRACT, probe_hand=side, followup_hand=followup,
                hold_upper_offset_rad=list(robot.get('upper_offset_rad',[0.]*14)),
                hold="executed final probe upper reference", exit_xy_m=[1.3,0.0],
                maximum_forward_segments=24, maximum_segment_step_m=.12, segment_timeout_s=7.,
                cruise_speed_mps=.25, vx_max_mps=.30, position_tolerance_m=.08,
                candidate_clear_margin_m=.04, release_s=1.5, neutralize_s=1.5, recovery_stand_s=1.5,
                safety="unchanged frozen E joint/effort/balance/progress supervision; safety refusal gives UNKNOWN unless independent geometric proof",
                policy="frozen R1 bounded followup with local probe classifier bypassed for action selection",
                observation_cutoff="end of single physical_probe + physical_probe_hold, before any followup")


def local_forward_corridor_obstructed(metadata, action):
    """Diagnostic obstruction of the nominal corridor, not action impossibility.

    The existing V1 closed-loop action has no invariant lateral/height tube.
    Finite wall coverage cannot prove BLOCKED for that action. This fact may
    justify withholding an unsafe trial, but its formal label stays UNKNOWN.
    """
    wall = metadata.get("wall") or {}
    if not wall.get("fixed", False) or metadata.get("defer_visual_reobserve", False) or metadata.get("out_of_probe_range",False): return False
    center, size = wall.get("center_xyz"), wall.get("size_xyz")
    if center is None or size is None: return False
    center, size = np.asarray(center, float), np.asarray(size, float)
    if center.shape != (3,) or size.shape != (3,) or not np.isfinite(np.r_[center,size]).all() or min(size)<=0: return False
    low, high = center-size/2, center+size/2
    plane = float(metadata.get("candidate_plane_x", .85))
    gap = metadata.get("initial_gap_m")
    near = bool(metadata.get("rigid_candidate", False)) or (gap is not None and 0 <= float(gap) <= .100001)
    return bool(near and low[0] >= plane-.03 and low[0] < action["exit_xy_m"][0] and low[1] <= -.35 and high[1] >= .35 and low[2] <= .05 and high[2] >= 1.6)


def wall_crossing_audited(metadata):
    """A candidate-plane crossing cannot establish wall nonpenetration.

    This evidence must be produced by a separate actual G1 collision audit.
    The current scene builder has no such audit; a physics-only probe-box gate
    is deliberately insufficient. Wall-free scenes do not need this evidence.
    """
    if not metadata.get('wall'):return True
    audit=metadata.get('g1_body_wall_nonpenetration_audit') or {}
    return bool(audit.get('status')=='VERIFIED' and audit.get('all_robot_collision_bodies') is True
                and audit.get('probe_followup_release_complete') is True and audit.get('finite') is True
                and audit.get('penetration_free') is True and int(audit.get('sample_count',0))>0
                and audit.get('source_evidence_path') and audit.get('source_evidence_sha256'))


def assign_label(*, physics_valid, runtime_valid, probe_valid, recovered, clearance, advance_steps, metadata, action):
    if not physics_valid: return "LABEL_UNKNOWN", "PHYSICS_NOT_VALIDATED"
    if metadata.get("out_of_probe_range",False) or metadata.get("label_scope")=="DEFER_VISUAL_REOBSERVATION":
        return "LABEL_UNKNOWN", "DEFER_VISUAL_REOBSERVATION_OUTSIDE_NEAR_FIELD_CONTRACT"
    if not runtime_valid or not probe_valid: return "LABEL_UNKNOWN", "EXECUTION_OR_PROBE_INVALID"
    if not recovered: return "LABEL_UNKNOWN", "SAFE_RECOVERY_NOT_VERIFIED"
    if clearance and advance_steps > 0:
        if not wall_crossing_audited(metadata):
            return "LABEL_UNKNOWN", "WALL_PRESENT_G1_BODY_NONPENETRATION_NOT_VERIFIED"
        return "PASSABLE_FOR_ACTION", "FIXED_ACTION_WHOLE_BODY_CLEARANCE_AND_RECOVERY"
    if local_forward_corridor_obstructed(metadata, action):
        return "LABEL_UNKNOWN", "FINITE_WALL_NO_CLOSED_LOOP_ACTION_IMPOSSIBILITY_PROOF"
    return "LABEL_UNKNOWN", "WALL_PRESENT_G1_BODY_NONPENETRATION_NOT_VERIFIED" if metadata.get('wall') and not wall_crossing_audited(metadata) else "NO_INDEPENDENT_ACTION_PASSABILITY_PROOF"


def online_physics_invalid(telemetry, external_reason):
    if external_reason and any(s in external_reason for s in ('CURTAIN_','DEFORMABLE_','PENETRATION','NONFINITE')):
        return external_reason
    limits={'evaluation_curtain_top_drift_m':.01,'evaluation_curtain_anchor_drift_m':.01,
            'evaluation_curtain_max_neighbor_stretch_ratio':4.,'evaluation_curtain_max_speed_mps':5.,
            'evaluation_curtain_region_displacement_m':1.5}
    for key,limit in limits.items():
        observed=[r[key] for r in telemetry if key in r]
        if observed and (not np.isfinite(np.asarray(observed,float)).all() or max(observed)>limit):
            return 'ONLINE_PHYSICS_INVALID:'+key
    return None


class Recorder:
    def __init__(self, run_root, plan, scene, source_hash):
        self.root = Path(run_root); self.root.mkdir(parents=True, exist_ok=True)
        self.plan, self.scene, self.action = plan, scene, action_spec(plan)
        self.file_path = self.root/"proprioception_50hz.jsonl"
        if self.file_path.exists() and self.file_path.stat().st_size:
            raise FileExistsError("Never overwrite an existing G robot history")
        self.stream = self.file_path.open("x", encoding="utf-8")
        self.previous_action = np.zeros(29); self.control_index = 0; self.control_time = 0.
        self.rows = []; self.first_probe = None; self.last_record_time = None; self.completed = False
        self.source_hash = source_hash
        self.pelvis_index = None
        self.metadata = dict(plan=plan, scene_config_evaluation_only=scene, action_contract=self.action,
                             online_label_revision=ONLINE_LABEL_REVISION,
                             runtime_source_sha256=source_hash, record_nominal_hz=50, physics_hz=200,
                             controller_hz=50, evaluation_geometry_hz=10,
                             previous_action_semantics="copied immediately before compute_target, at recorded control_time; current_action is resulting policy action",
                             effort_semantics="applied_torque simulator actuator proxy in N.m; no contact truth",
                             imu_semantics="robot root RPY/angular velocity/projected gravity; pelvis COM acceleration transformed into body specific-force proxy; not hardware calibration")
        write_json(self.root/"g_trial_metadata.json", self.metadata)

    def before_control(self, adapter, sim_step):
        self.previous_action = np.asarray(adapter.previous_action, float).copy()
        self.control_index += 1; self.control_time = sim_step*DT

    def observe(self, *, phase, local_step, sim_step, q, qdot, q_cmd, effort, after, hands, velocities, rpy,
                adapter, robot, upper, preprobe_upper, target_upper, command, walk_mode):
        if phase not in PROBE_PHASES|{"physical_preprobe","physical_preprobe_hold","precontact_explicit_stand"}: return
        if sim_step % 4 != 0 and local_step != 0: return
        t = (sim_step+1)*DT; R = rotation(rpy)
        positions = {s:np.asarray(hands[s],float) for s in ("left","right")}
        if phase in PROBE_PHASES and self.first_probe is None:
            self.first_probe = (t, {s:x.copy() for s,x in positions.items()}, R.copy())
        d = np.asarray(target_upper,float)-np.asarray(preprobe_upper,float)
        executed_upper = np.asarray(q_cmd,float)[15:]
        progress = float(np.dot(executed_upper-preprobe_upper,d)/max(np.dot(d,d),1e-12))
        accel = None
        try:
            if self.pelvis_index is None: self.pelvis_index = robot.body_names.index("pelvis")
            acc_w = robot.data.body_com_lin_acc_w[0,self.pelvis_index].detach().cpu().numpy()
            accel = R.T@(acc_w-np.array([0.,0.,-9.81]))
        except (AttributeError, ValueError, IndexError):
            pass
        validated_accel=accel if self.plan.get('acceleration_channel_validated',False) else None
        row = dict(phase=phase, phase_step=int(local_step), physics_step=int(sim_step),time_s=t,
                   dt_since_logged_s=0. if self.last_record_time is None else t-self.last_record_time,
                   control_update_index=self.control_index, control_command_time_s=self.control_time,
                   q=q,qdot=qdot,q_cmd=q_cmd,effort=effort,effort_source=str(after.effort_source),
                   previous_control_action=self.previous_action,current_policy_action=np.asarray(adapter.previous_action,float),
                   imu_rpy=np.asarray(rpy,float),imu_angular_velocity_b=np.asarray(after.base_ang_vel_b,float),
                   imu_projected_gravity_b=np.asarray(after.projected_gravity_b,float),imu_specific_force_b=validated_accel,
                   imu_specific_force_candidate_b=accel,
                   imu_acceleration_source="physx_pelvis_COM_acceleration_specific_force_proxy" if validated_accel is not None else "UNAVAILABLE_PENDING_INDEPENDENT_SIGNAL_VALIDATION",
                   base_velocity_b=np.asarray(after.base_lin_vel_b,float),
                   left_hand_fk_b=R.T@(positions["left"]-after.base_pos_w),right_hand_fk_b=R.T@(positions["right"]-after.base_pos_w),
                   left_hand_delta_initial_b=np.zeros(3) if self.first_probe is None else (positions["left"]-self.first_probe[1]["left"])@self.first_probe[2],
                   right_hand_delta_initial_b=np.zeros(3) if self.first_probe is None else (positions["right"]-self.first_probe[1]["right"])@self.first_probe[2],
                   left_hand_velocity_initial_b=np.asarray(velocities['left'],float)@(R if self.first_probe is None else self.first_probe[2]),
                   right_hand_velocity_initial_b=np.asarray(velocities['right'],float)@(R if self.first_probe is None else self.first_probe[2]),
                   commanded_upper_reference=executed_upper,requested_upper_reference=np.asarray(upper,float),command_progress=progress,
                   planned_followup_upper_reference=np.asarray(target_upper,float),
                   commanded_body_velocity=np.asarray(command,float),command_walk_mode=bool(walk_mode),
                   probe_side=self.action["probe_hand"],followup_hand=self.action["followup_hand"],
                   probe_elapsed_s=0. if self.first_probe is None else t-self.first_probe[0])
        row = clean(row)
        self.stream.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
        self.stream.flush(); self.rows.append(row); self.last_record_time=t

    def finish(self, *, probe_result, scene_metadata, physics_valid, source_probe_valid, fall_reason, external_safety_reason, events, telemetry, rule_decision, logger,evaluation_arrays=None):
        probe = [r for r in self.rows if r["phase"] in PROBE_PHASES]
        finite = bool(probe) and all(np.isfinite(np.asarray(r[k],float)).all() for r in probe for k in ("q","qdot","q_cmd","effort","previous_control_action"))
        time_valid = len(probe)>1 and np.all(np.diff([r["time_s"] for r in probe])>0)
        effort_valid = bool(probe) and all(r["effort_source"] not in {"unavailable","zero_filled",""} for r in probe)
        probe_valid = bool(source_probe_valid and finite and time_valid and effort_valid and probe_result.get("execution_command_started"))
        runtime_valid = bool(finite and fall_reason is None and telemetry and all(r.get('finite',False) for r in telemetry))
        online_physics_reason = online_physics_invalid(telemetry,external_safety_reason)
        physics_gate_valid=bool(physics_valid)
        physics_valid = bool(physics_gate_valid and online_physics_reason is None)
        recovery = probe_result.get("post_retract_stand") or {}
        recovered = bool(probe_result.get("retract") and probe_result.get("neutralize") and recovery.get("speed_gate_reached") and recovery.get("actor_stand_flag_ok") and fall_reason is None)
        clearance = bool(probe_result.get("candidate_clearance_verified"))
        advance_steps = sum(str(r.get("phase","")).startswith("round26_cautious_advance") for r in telemetry)
        label, reason = assign_label(physics_valid=physics_valid,runtime_valid=runtime_valid,probe_valid=probe_valid,recovered=recovered,
                                     clearance=clearance,advance_steps=advance_steps,metadata=scene_metadata,action=self.action)
        periods = np.diff([r["time_s"] for r in probe])
        result = dict(trial_id=self.plan["trial_id"],status="G_FIXED_ACTION_FINISHED",runtime_valid=runtime_valid,probe_valid=probe_valid,
                      label=label,label_valid=label!="LABEL_UNKNOWN",label_reason=reason,label_contract=self.action['label_contract'],
                      online_label_revision=ONLINE_LABEL_REVISION,
                      label_scope='ONLINE_ACTION_EVIDENCE_PENDING_INDEPENDENT_COLLISION_AUDIT',
                      safe_recovery=recovered,whole_body_candidate_clearance=clearance,followup_advance_logged_steps=advance_steps,
                      task_success=bool(label=="PASSABLE_FOR_ACTION"),source_rule_decision_audit_only=rule_decision,
                      task_success_scope='FROZEN_CANDIDATE_ACTION_ONLY_NOT_POINTGOAL_NAVIGATION',pointgoal_task_success=None,
                      label_evidence=dict(source=reason,local_forward_corridor_obstructed=local_forward_corridor_obstructed(scene_metadata,self.action),
                          closed_loop_blocking_proof_supported=False,
                          g1_body_wall_nonpenetration_verified=wall_crossing_audited(scene_metadata) if scene_metadata.get('wall') else None,
                          limitation='G1_BODY_WALL_NONPENETRATION_NOT_VERIFIED' if scene_metadata.get('wall') and not wall_crossing_audited(scene_metadata) else None),
                      probe_side=self.action["probe_hand"],followup_hand=self.action["followup_hand"],followup_action=self.action,
                      physics_valid=bool(physics_valid),physics_gate_valid=physics_gate_valid,online_physics_invalid_reason=online_physics_reason,
                      probe_sample_count=len(probe),actual_logged_hz=None if not len(periods) else 1/float(np.median(periods)),
                      probe_cutoff_time_s=None if not probe else probe[-1]["time_s"],observation_cutoff_before_followup=True,
                      nominal_probe_displacement_m=.10,actual_hand_displacement_m=probe_result.get("actual_hand_displacement_m"),
                      probe_stop_reason=probe_result.get("probe_stop_reason"),fall_reason=fall_reason,external_safety_reason=external_safety_reason,
                      physical_probe_result=probe_result,scene_evaluation_only=scene_metadata,
                      source_online_rule_sample_period_s=.005,source_online_rule_history_hz=200,
                      source_online_rule_note="deployment_samples append on every physics step; .005 is correct online. Old CSV-only evaluator used .005 on 10Hz rows and was wrong.",
                      measured_control_budget="original 200Hz physics / 50Hz control / 10Hz geometry safety; added 50Hz robot recording only")
        write_json(self.root/"g_evaluation_truth.json",dict(scene=scene_metadata,physical_probe_result=probe_result,label=label,label_reason=reason,
            online_label_revision=ONLINE_LABEL_REVISION,closed_loop_blocking_proof_supported=False))
        if evaluation_arrays and len(evaluation_arrays.get('nodes_world',[])):
            np.savez_compressed(self.root/'g_environment_audit_only.npz',**{k:np.asarray(v) for k,v in evaluation_arrays.items()})
        write_json(self.root/"events.json", events)
        if telemetry:
            import csv
            with (self.root/"telemetry.csv").open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=list(telemetry[0]));w.writeheader();w.writerows(telemetry)
        self.stream.flush(); self.stream.close(); self.completed=True
        result["legal_history_sha256"]=hashlib.sha256(self.file_path.read_bytes()).hexdigest()
        write_json(self.root/"g_outcome.json",result); write_json(self.root/"summary.json",result)
        logger.finalize(status="PASS" if runtime_valid else "FAIL",details={"G_action_contract":self.action['label_contract'],"label":label,"probe_valid":probe_valid})
        return 0 if runtime_valid else 2

    def close_incomplete(self):
        if not self.stream.closed: self.stream.flush(); self.stream.close()
        if not self.completed:
            write_json(self.root/"g_outcome.json",dict(trial_id=self.plan["trial_id"],status="EXECUTION_INCOMPLETE",runtime_valid=False,
                       probe_valid=False,label="LABEL_UNKNOWN",label_valid=False,label_contract=self.action['label_contract'],
                       online_label_revision=ONLINE_LABEL_REVISION,
                       label_reason="FOLLOWUP_OR_RUNTIME_NOT_FINALIZED",recorded_robot_rows=len(self.rows)))
