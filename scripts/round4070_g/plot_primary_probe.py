"""Plot actual legal C01 histories, separately from collision/label truth."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/round4070_comprehensive_traversability'
DOC = ROOT / 'research/round4070_comprehensive_traversability'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = OUT / 'dataset/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    joint_names = manifest['joint_order']
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex=True)
    sources = []
    colors = {'left': '#2464a6', 'right': '#cc6436'}
    for side in ('left', 'right'):
        trial = f'G_C01_{side}_coverage_00_nominal_s71900_rng1_act2'
        row = next(e for e in manifest['episodes'] if e['planned_trial_id'] == trial)
        if row.get('label') != 'PASSABLE_FOR_ACTION' or not row.get('primary_admission'):
            raise RuntimeError('Plot requires independently admitted C01 episode')
        path = OUT / 'dataset/episodes' / f'{trial}__attempt01.npz'
        run = OUT / 'trials' / trial / 'attempt01'
        with np.load(path, allow_pickle=False) as data:
            t = data['time'] - data['time'][0]
            if t[-1] > 2.0 + 1e-9:
                raise RuntimeError('Legal observation exceeds frozen cutoff')
            arm = [i for i, name in enumerate(joint_names)
                   if name.startswith(side + '_') and any(
                       part in name for part in ('shoulder', 'elbow', 'wrist'))]
            if len(arm) != 7:
                raise RuntimeError('Actual policy-order arm mapping not verified')
            fk = data[f'{side}_hand_fk_b']
            values = [1000 * (fk[:, 0] - fk[0, 0]),
                      np.sqrt(np.mean((data['q_cmd'][:, arm] - data['q'][:, arm]) ** 2, axis=1)),
                      np.sqrt(np.mean(data['effort'][:, arm] ** 2, axis=1)),
                      np.degrees(data['imu_rpy'][:, 1])]
            for ax, value in zip(axes.flat, values):
                ax.plot(t, value, color=colors[side], lw=1.6,
                        label=f'{side.title()} ({len(t)} actual samples)')
            sources.append(dict(side=side, trial_id=trial, legal_samples=len(t),
                                span_s=float(t[-1]), npz_path=path.relative_to(ROOT).as_posix(),
                                npz_sha256=sha(path), audit_path=(run / 'g_collision_audit.json').relative_to(ROOT).as_posix(),
                                audit_sha256=sha(run / 'g_collision_audit.json'),
                                active_arm_indices=arm, active_arm_joint_names=[joint_names[i] for i in arm]))
    titles = ['Hand forward FK change in body frame', 'Active arm tracking error',
              'Applied actuator torque proxy', 'Body pitch']
    labels = ['Change from first probe sample (mm)', 'Joint error RMS (rad)',
              'Actuator torque RMS (N m)', 'Pitch (degrees)']
    for ax, title, label in zip(axes.flat, titles, labels):
        ax.set(title=title, ylabel=label, xlim=(0, 2))
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('Actual elapsed probe time (s)')
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('C01: one independently verified trial per hand', fontsize=15, y=.98)
    fig.text(.5, .025, 'Only legal pre-followup observations; no interpolation. '
             'Torque is not measured contact force.\nTwo trials in one physical configuration '
             'do not establish repeatability, classification performance, or generalization.',
             ha='center', fontsize=9, color='#4b5563')
    fig.tight_layout(rect=(0, .095, 1, .94))
    image = DOC / 'PRIMARY_BILATERAL_PROBE_AUDIT.png'
    fig.savefig(image, dpi=180)
    plt.close(fig)
    evidence = dict(scope='ACTUAL_LEGAL_HISTORY_DIAGNOSTIC_NOT_MODEL_COMPARISON',
                    manifest_sha256=sha(manifest_path), sources=sources,
                    image_sha256=sha(image), plotting_source_sha256=sha(Path(__file__)))
    (DOC / 'PRIMARY_BILATERAL_PROBE_AUDIT.json').write_text(
        json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(image=str(image), source_trials=len(sources))))


if __name__ == '__main__':
    main()
