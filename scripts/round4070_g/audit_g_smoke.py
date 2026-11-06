"""Read actual completed G1 records and audit logging without fitting anything."""
from collections import Counter
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from build_dataset import OUT, REPORT, frozen_evidence, legal_sequence, load, rel, schema, sha, original_online_label
from g1_recording import PROBE_PHASES, action_spec, assign_label, write_json
from g1_randomization import PRIMARY_PROTOCOL


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--read-only',action='store_true',help='Print QA without writing dataset or reports')
    parser.add_argument('--trial-id',default=None)
    args=parser.parse_args()
    results=[]
    for path in sorted((OUT/'trials').rglob('g_outcome.json')):
        outcome=load(path);directory=path.parent
        if args.trial_id and outcome.get('trial_id')!=args.trial_id:continue
        if not outcome.get('probe_valid'):continue
        meta=load(directory/'g_trial_metadata.json');source=directory/'proprioception_50hz.jsonl'
        rows=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines() if s]
        x,t,phase,raw,audit=legal_sequence(rows,action_spec(meta['plan']),float(outcome['probe_cutoff_time_s']))
        provenance=frozen_evidence(directory,meta,meta['plan'])
        label,reason=original_online_label(outcome,action_spec(meta['plan']))
        truth=[]
        if (directory/'telemetry.csv').exists():
            with (directory/'telemetry.csv').open(encoding='utf-8',newline='') as f:truth=list(csv.DictReader(f))
        future=[float(r['time_s']) for r in truth if r['phase'].startswith(('round26_cautious_advance','physical_retract','responsive_release'))]
        acc=np.array([r['imu_specific_force_candidate_b'] for r in rows if r.get('imu_specific_force_candidate_b') is not None],float)
        if len(acc):
            signal=dict(candidate_rows=len(acc),finite=bool(np.isfinite(acc).all()),component_std_mps2=np.std(acc,axis=0).tolist(),
                norm_min_mps2=float(np.linalg.norm(acc,axis=1).min()),norm_max_mps2=float(np.linalg.norm(acc,axis=1).max()),
                status='SIGNAL_PRESENT_NOT_HARDWARE_CALIBRATED' if np.max(np.std(acc,axis=0))>1e-5 else 'CONSTANT_OR_UNVERIFIED',
                model_admitted_rows=sum(r.get('imu_specific_force_b') is not None for r in rows))
        else:signal=dict(candidate_rows=0,status='UNAVAILABLE',model_admitted_rows=0)
        result=dict(trial_id=meta['plan']['trial_id'],attempt_id=directory.name,label=outcome['label'],runtime_valid=outcome['runtime_valid'],
            label_scope='ORIGINAL_ONLINE_OUTCOME_DIAGNOSTIC_ONLY',primary_protocol=meta['plan'].get('primary_protocol'),
            superseded_pilot=meta['plan'].get('primary_protocol')!=PRIMARY_PROTOCOL,
            probe_valid=outcome['probe_valid'],label_valid=outcome['label_valid'],physics_valid=outcome['physics_valid'],
            phase_counts=dict(Counter(r['phase'] for r in rows)),probe_audit=audit,feature_count=x.shape[1],
            legal_source_sha256=sha(source),model_first_time_s=float(t[0]),model_last_time_s=float(t[-1]),
            first_followup_or_retract_time_s=min(future) if future else None,
            cutoff_precedes_future=bool(not future or t[-1]<min(future)),
            actually_sent_upper_qcmd_matches=bool(np.allclose(raw['commanded_upper_reference'],raw['q_cmd'][:,15:],atol=1e-7)),
            candidate_acceleration=signal,
            rederived_label=label,label_recomputation_matches=label==outcome['label'],label_reason=reason,
            physics_gate_seeds=provenance['physics_gate_seeds'],
            world_or_truth_fields_in_robot_log=sorted(set().union(*(set(r) for r in rows))&{'base_position_w','contact_truth','material','wall','constraint','candidate_clearance'}))
        environment_path=directory/'g_environment_audit_only.npz'
        if environment_path.exists():
            with np.load(environment_path,allow_pickle=False) as d:
                result['environment_audit_only']=dict(positions_shape=list(d['nodes_world'].shape),velocities_shape=list(d['node_velocities_world'].shape),
                    all_finite=bool(np.isfinite(d['nodes_world']).all() and np.isfinite(d['node_velocities_world']).all()),time_monotone=bool(np.all(np.diff(d['time_s'])>0)),
                    fixed_node_count=len(d['requested_fixed_indices']))
        body_path=directory/'g_body_collision_history_audit_only.npz'
        if body_path.exists():
            with np.load(body_path,allow_pickle=False) as d:
                result['body_collision_capture_audit_only']=dict(sample_count=len(d['time_s']),body_count=len(d['body_names']),
                    nominal_actual_hz=1/float(np.median(np.diff(d['time_s']))),
                    time_monotone=bool(np.all(np.diff(d['time_s'])>0)),
                    actual_clock_offset_min_s=float(np.min(d['actual_simulation_time_s']-d['time_s'])),
                    actual_clock_offset_max_s=float(np.max(d['actual_simulation_time_s']-d['time_s'])),
                    final_boundaries=[dict(phase=str(d['phase'][i]),time_s=float(d['time_s'][i])) for i in np.flatnonzero(d['boundary'])],
                    finite=bool(all(np.isfinite(d[k]).all() for k in ('body_positions_world','body_quaternions_wxyz','nodes_world','node_velocities_world'))))
        result['audit_pass']=bool(result['label_recomputation_matches'] and result['cutoff_precedes_future'] and result['actually_sent_upper_qcmd_matches'] and not result['world_or_truth_fields_in_robot_log'])
        results.append(result)
    if args.read_only:
        print(json.dumps(results,ensure_ascii=False,indent=2));return
    write_json(OUT/'G1_SMOKE_DATA_AUDIT.json',dict(status='ACTUAL_COMPLETED_RECORDS_ONLY',episodes=results))
    lines=['# G1 actual logging smoke audit','',f'Completed probe-valid histories inspected: {len(results)}. No model fitting or simulation was started by this audit.','',
        'This report audits raw timing and legal robot channels only. Its displayed labels are original online outcomes, not independently approved primary labels. All V3 pilots are superseded and excluded from primary supervision, normalization, auxiliary response fitting and evaluation. See dataset_status.json and LABEL_VALIDITY_REPORT.md for formal admission.','']
    for r in results:
        a=r['probe_audit'];lines.append(f"- {r['trial_id']} / {r['attempt_id']}: label={r['label']}, runtime={r['runtime_valid']}, probe={r['probe_valid']}, physics={r['physics_valid']}; {a['sample_count']} real probe samples, median dt={a['dt_median_s']:.6f} s ({a['actual_hz']:.1f} Hz), span={a['duration_s']:.3f} s; previous-action updates verified={a['true_previous_action_adjacent_updates_verified']}; causal cutoff={r['cutoff_precedes_future']}, actual upper q_cmd={r['actually_sent_upper_qcmd_matches']}; acceleration={r['candidate_acceleration']['status']}.")
    lines+=['','The acceleration candidate is retained robot state, not hardware IMU calibration. Unless an explicit independent validation flag is present, it remains unavailable to every model with an explicit missingness mask. Primary UNKNOWN response admission additionally requires the frozen V4 RNG protocol and an independent VERIFIED whole-body collision audit. A valid robot probe alone does not establish that physical validity or whole-episode runtime validity.','']
    (REPORT/'G1_SMOKE_DATA_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
