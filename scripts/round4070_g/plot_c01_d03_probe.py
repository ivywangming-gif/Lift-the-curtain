"""Descriptive C01/D03 comparison from four admitted, legal probe histories.

This script only reads the dataset and raw provenance. It does not train a model,
resample observations, or derive labels from the plotted responses.
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/round4070_comprehensive_traversability'
DOC = ROOT / 'research/round4070_comprehensive_traversability'
SCENES = {'C01': (71900, 'PASSABLE_FOR_ACTION', '#2464a6'),
          'D03': (72900, 'BLOCKED', '#cc6436')}
METRICS = ('hand_forward_fk_delta_mm', 'active_arm_tracking_error_rms_rad',
           'active_arm_applied_torque_rms_nm', 'body_pitch_deg')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def root_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_episode(manifest, scene, side):
    seed, label, _ = SCENES[scene]
    planned = f'G_{scene}_{side}_coverage_00_nominal_s{seed}_rng1_act2'
    rows = [e for e in manifest['episodes'] if e['planned_trial_id'] == planned]
    if len(rows) != 1:
        raise ValueError('Exactly one admitted actual episode required: ' + planned)
    row = rows[0]
    if (not row['primary_admission'] or row['is_control'] or row['label'] != label
            or not row['observation_cutoff_before_followup']
            or row['effort_sources'] != ['applied_torque']):
        raise ValueError('Required admitted core legal-window provenance differs')
    path = root_path(row['npz_path'])
    if sha(path) != row['npz_sha256']:
        raise ValueError('Admitted NPZ hash differs')
    evidence = row['body_collision_audit']
    if evidence['status'] != 'VERIFIED':
        raise ValueError('Independent collision admission is not verified')
    audit_path = root_path(evidence['source_evidence_path'])
    if sha(audit_path) != evidence['source_evidence_sha256']:
        raise ValueError('Immutable admitted collision audit hash differs')
    run = root_path(row['source_path'])
    for name, digest in row['source_sha256'].items():
        if sha(run / name) != digest:
            raise ValueError('Original source hash differs: ' + name)
    joint_order = manifest['joint_order']
    arm = [i for i, name in enumerate(joint_order)
           if name.startswith(side + '_') and any(
               part in name for part in ('shoulder', 'elbow', 'wrist'))]
    if len(arm) != 7:
        raise ValueError('Official active-arm mapping must contain seven joints')
    joints = [joint_order[i] for i in arm]
    with np.load(path, allow_pickle=False) as data:
        names = data['feature_names'].tolist()
        if names != manifest['feature_names']:
            raise ValueError('Exported legal feature order differs from manifest')
        seq = data['seq']
        absolute_time = data['time'].copy()
        t = absolute_time - absolute_time[0]
        if (not np.isfinite(seq).all() or not np.isfinite(t).all()
                or np.any(np.diff(t) <= 0) or t[-1] > 2.0 + 1e-9
                or absolute_time[-1] > row['observation_cutoff_time_s'] + 1e-9
                or not set(data['phase'].tolist()) <= {'physical_probe', 'physical_probe_hold'}):
            raise ValueError('Actual legal timestamps/phases/cutoff are invalid')
        selected = {'q': [names.index('q/' + name) for name in joints],
                    'q_cmd': [names.index('q_cmd/' + name) for name in joints],
                    'effort': [names.index('effort/' + name) for name in joints],
                    'fk_x': [names.index(f'{side}_hand_fk_b/x')],
                    'pitch': [names.index('imu_rpy/pitch')]}
        # Values are read exclusively from the legal model-input matrix. Match
        # the retained raw legal channels to detect an accidental index error.
        for key in ('q', 'q_cmd', 'effort'):
            if not np.array_equal(seq[:, selected[key]], data[key][:, arm].astype(seq.dtype)):
                raise ValueError('Named legal/raw joint channel mismatch: ' + key)
        fk = seq[:, selected['fk_x'][0]].astype(np.float64)
        pitch = seq[:, selected['pitch'][0]].astype(np.float64)
        if not np.array_equal(fk.astype(seq.dtype), data[f'{side}_hand_fk_b'][:, 0].astype(seq.dtype)):
            raise ValueError('Named legal/raw FK channel mismatch')
        if not np.array_equal(pitch.astype(seq.dtype), data['imu_rpy'][:, 1].astype(seq.dtype)):
            raise ValueError('Named legal/raw pitch channel mismatch')
        q = seq[:, selected['q']].astype(np.float64)
        q_cmd = seq[:, selected['q_cmd']].astype(np.float64)
        effort = seq[:, selected['effort']].astype(np.float64)
        values = (1000 * (fk - fk[0]), np.sqrt(np.mean((q_cmd - q) ** 2, axis=1)),
                  np.sqrt(np.mean(effort ** 2, axis=1)), np.degrees(pitch))
        phases, counts = np.unique(data['phase'], return_counts=True)
        seq_hash = hashlib.sha256(seq.tobytes()).hexdigest()
    source = dict(scene_id=scene, probe_side=side, seed=seed,
                  trial_id=row['trial_id'], planned_trial_id=planned,
                  physical_config_id=row['physical_config_id'],
                  formal_label_metadata_only=label, primary_admission=True,
                  npz_path=path.relative_to(ROOT).as_posix(), npz_sha256=sha(path),
                  legal_seq_array_sha256=seq_hash,
                  raw_source_base=run.relative_to(ROOT).as_posix(),
                  raw_source_sha256=row['source_sha256'],
                  immutable_audit_path=audit_path.relative_to(ROOT).as_posix(),
                  immutable_audit_sha256=sha(audit_path),
                  active_arm_indices_zero_based=arm, active_arm_joint_names=joints,
                  plotted_seq_feature_indices=selected,
                  plotted_seq_feature_names={k: [names[i] for i in v] for k, v in selected.items()},
                  sample_count=len(t), span_s=float(t[-1]),
                  actual_timestamp_first_s=float(absolute_time[0]),
                  actual_timestamp_last_s=float(absolute_time[-1]),
                  time_dt_min_median_max_s=[float(np.min(np.diff(t))),
                                            float(np.median(np.diff(t))), float(np.max(np.diff(t)))],
                  phases=dict(zip(phases.tolist(), counts.tolist())),
                  metrics={key: dict(min=float(np.min(v)), max=float(np.max(v)),
                                     first=float(v[0]), last=float(v[-1]))
                           for key, v in zip(METRICS, values)})
    return t, values, source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True,
                        help='Explicit repository-relative or absolute manifest path; prefer an immutable dataset snapshot')
    args = parser.parse_args(argv)
    manifest_path = root_path(args.manifest).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes.decode('utf-8-sig'))
    snapshot_path = manifest_path.parent / 'snapshot.json'
    snapshot_binding = None
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding='utf-8-sig'))
        if (snapshot.get('schema') != 'round4070-g.dataset_snapshot.v1'
                or snapshot.get('files', {}).get(manifest_path.name) != manifest_hash
                or snapshot.get('identity') != manifest_path.parent.name):
            raise ValueError('Immutable snapshot identity/manifest hash differs')
        snapshot_binding = dict(identity=snapshot['identity'],
                                path=snapshot_path.relative_to(ROOT).as_posix(),
                                sha256=sha(snapshot_path))
    fig, axes = plt.subplots(4, 2, figsize=(12, 10.8), sharex=True, sharey='row')
    sources = []
    for col, side in enumerate(('left', 'right')):
        for scene, (seed, _, color) in SCENES.items():
            t, values, source = read_episode(manifest, scene, side)
            sources.append(source)
            for ax, value in zip(axes[:, col], values):
                ax.plot(t, value, color=color, lw=1.5, marker='.', markersize=2.6,
                        label=f'{scene} / seed {seed} / n={len(t)}')
        axes[0, col].set_title(side.title() + ' probe hand', fontsize=12, pad=10)
        axes[0, col].legend(frameon=False, fontsize=8.5, loc='upper left')
    labels = ['Hand forward FK change (mm)\nBody frame; from first sample',
              'Active arm tracking error\nRMS q_cmd - q (rad)',
              'Applied actuator torque proxy\nActive arm RMS (N m)',
              'Body pitch (degrees)']
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label, fontsize=9.5)
        for ax in axes[row]:
            ax.set_xlim(0, 2)
            ax.grid(alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('Actual elapsed time from first probe observation (s)', fontsize=9.5)
    fig.suptitle('C01 vs D03: actual legal probe responses', fontsize=16, y=.987)
    fig.text(.5, .953, 'Four admitted core episodes: one trial per configuration and hand',
             ha='center', fontsize=10, color='#4b5563')
    caption = ('Markers are actual pre-followup samples; lines only guide the eye. No resampling or smoothing.\n'
               'Torque is an applied-actuator proxy, not contact force. No world geometry or followup truth is plotted.\n'
               'Each configuration has one seed; C01 and D03 use different seeds, so configuration and seed are confounded.\n'
               'Descriptive traces only: no classification accuracy, statistical significance, repeatability or generalization claim.')
    fig.text(.5, .018, caption, ha='center', va='bottom', fontsize=8.6, color='#4b5563', linespacing=1.5)
    fig.tight_layout(rect=(.012, .112, .998, .935), h_pad=1.45, w_pad=1.7)
    if sha(manifest_path) != manifest_hash:
        raise RuntimeError('Manifest changed during plotting; rerun against a stable snapshot')
    image_path = DOC / 'PRIMARY_C01_D03_PROBE_COMPARISON.png'
    fig.savefig(image_path, dpi=180, facecolor='white')
    plt.close(fig)
    evidence = dict(schema='round4070-g.descriptive_probe_plot.v1',
                    scope='ACTUAL_LEGAL_HISTORY_DIAGNOSTIC_NOT_MODEL_COMPARISON',
                    manifest_path=manifest_path.relative_to(ROOT).as_posix(),
                    manifest_sha256=manifest_hash, immutable_snapshot=snapshot_binding,
                    reproduction_command='C:/conda_envs/isaaclab/python.exe scripts/round4070_g/plot_c01_d03_probe.py --manifest '
                                         + manifest_path.relative_to(ROOT).as_posix(),
                    admitted_count_at_snapshot=len(manifest['episodes']),
                    selected_core_episode_count=4, physical_configurations=2,
                    full_policy_joint_order=manifest['joint_order'], sources=sources,
                    transformations=dict(time='actual_timestamp - first_actual_timestamp',
                        hand_fk='1000 * (active_hand_body_FK_x - first_sample_body_FK_x)',
                        tracking='sqrt(mean((q_cmd - q)^2)) over seven named active-arm joints',
                        effort='sqrt(mean(applied_torque_proxy^2)) over the same seven joints',
                        pitch='radians_to_degrees(legal imu_rpy/pitch)',
                        resampling=False, smoothing=False, fitted_model=False),
                    caption=caption, image_sha256=sha(image_path),
                    plotting_source_sha256=sha(Path(__file__)),
                    library_versions=dict(numpy=np.__version__, matplotlib=matplotlib.__version__))
    evidence_path = DOC / 'PRIMARY_C01_D03_PROBE_COMPARISON.json'
    evidence_path.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(image=str(image_path), evidence=str(evidence_path),
                          plotted_episodes=len(sources), manifest_sha256=manifest_hash)))


if __name__ == '__main__':
    main()
