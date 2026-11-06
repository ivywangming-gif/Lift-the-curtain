"""Read actual completed G1 records and audit logging without fitting anything."""
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
from build_dataset import OUT, REPORT, import_trial, legal_sequence, load, rel, schema, sha
from g1_recording import PROBE_PHASES, action_spec, write_json


def main():
    results=[]
    for path in sorted((OUT/'trials').rglob('g_outcome.json')):
        outcome=load(path);directory=path.parent
        if not outcome.get('probe_valid'):continue
        meta=load(directory/'g_trial_metadata.json');source=directory/'proprioception_50hz.jsonl'
        rows=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines() if s]
        x,t,phase,raw,audit=legal_sequence(rows,action_spec(meta['plan']),float(outcome['probe_cutoff_time_s']))
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
            probe_valid=outcome['probe_valid'],label_valid=outcome['label_valid'],physics_valid=outcome['physics_valid'],
            phase_counts=dict(Counter(r['phase'] for r in rows)),probe_audit=audit,feature_count=x.shape[1],
            legal_source_sha256=sha(source),model_first_time_s=float(t[0]),model_last_time_s=float(t[-1]),
            first_followup_or_retract_time_s=min(future) if future else None,
            cutoff_precedes_future=bool(not future or t[-1]<min(future)),
            actually_sent_upper_qcmd_matches=bool(np.allclose(raw['commanded_upper_reference'],raw['q_cmd'][:,15:],atol=1e-7)),
            candidate_acceleration=signal,
            world_or_truth_fields_in_robot_log=sorted(set().union(*(set(r) for r in rows))&{'base_position_w','contact_truth','material','wall','constraint','candidate_clearance'}))
        result['audit_pass']=bool(result['cutoff_precedes_future'] and result['actually_sent_upper_qcmd_matches'] and not result['world_or_truth_fields_in_robot_log'])
        results.append(result)
    write_json(OUT/'G1_SMOKE_DATA_AUDIT.json',dict(status='ACTUAL_COMPLETED_RECORDS_ONLY',episodes=results))
    lines=['# G1 actual logging smoke audit','',f'Completed probe-valid histories inspected: {len(results)}. No model fitting or simulation was started by this audit.','']
    for r in results:
        a=r['probe_audit'];lines.append(f"- {r['trial_id']} / {r['attempt_id']}: label={r['label']}, runtime={r['runtime_valid']}, probe={r['probe_valid']}, physics={r['physics_valid']}; {a['sample_count']} real probe samples, median dt={a['dt_median_s']:.6f} s ({a['actual_hz']:.1f} Hz), span={a['duration_s']:.3f} s; previous-action updates verified={a['true_previous_action_adjacent_updates_verified']}; causal cutoff={r['cutoff_precedes_future']}, actual upper q_cmd={r['actually_sent_upper_qcmd_matches']}; acceleration={r['candidate_acceleration']['status']}.")
    lines+=['','The acceleration candidate is retained robot state, not hardware IMU calibration. Unless an explicit independent validation flag is present, it remains unavailable to every model with an explicit missingness mask. A valid probe followed by an execution failure may remain UNKNOWN response history; this does not make that entire runtime valid.','']
    (REPORT/'G1_SMOKE_DATA_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
