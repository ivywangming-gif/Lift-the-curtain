"""Descriptive real-history figure; no model fitting or passability inference."""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/round4070_comprehensive_traversability'
DOC = ROOT / 'research/round4070_comprehensive_traversability'


def main():
    source = OUT / 'dataset/manifest.json'
    manifest = json.loads(source.read_text(encoding='utf-8'))
    selected = []
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7), sharex=True, constrained_layout=True)
    for row, side in enumerate(('left', 'right')):
        joints = [i for i, name in enumerate(manifest['joint_order'])
                  if name.startswith(side + '_') and any(part in name for part in ('shoulder', 'elbow', 'wrist'))]
        if len(joints) != 7:
            raise ValueError('Expected seven named active-arm joints')
        for scene, color in [('EMPTY', '#61717e'), ('A01', '#006fbc')]:
            candidates = sorted((e for e in manifest['episodes'] if e['scene_id'] == scene and e['side'] == side),
                                key=lambda e: e['trial_id'])
            if not candidates:
                continue
            # First lexicographic ID, never best outcome; record exact selection.
            episode = candidates[0]
            path = ROOT / episode['npz_path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != episode['npz_sha256']:
                raise ValueError('Source NPZ changed')
            with np.load(path, allow_pickle=False) as data:
                t = data['time'] - data['time'][0]
                curves = [data[side + '_hand_delta_initial_b'][:, 0],
                          np.sqrt(np.mean((data['q'][:, joints] - data['q_cmd'][:, joints])**2, axis=1)),
                          np.linalg.norm(data['effort'][:, joints], axis=1)]
                label = 'EMPTY control' if scene == 'EMPTY' else 'A01 cloth (UNKNOWN)'
                for column, values in enumerate(curves):
                    axes[row, column].plot(t, values, color=color, lw=1.7, label=label)
            selected.append({key: episode[key] for key in ('trial_id', 'npz_path', 'npz_sha256', 'label', 'scene_id', 'side', 'sample_count', 'actual_hz')})
        for column, unit in enumerate(('Displacement (m)', 'Tracking RMS (rad)', 'Effort norm (N m)')):
            axes[row, column].set(xlim=(0, 2), xlabel='Real time since probe start (s)', ylabel=unit)
            axes[row, column].grid(alpha=.2)
            axes[row, column].set_title(side.capitalize() + ': ' + ('active-hand forward displacement', 'active-arm tracking error', 'active-arm applied torque proxy')[column], fontsize=10)
        axes[row, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('Actual 50 Hz probe histories: first completed EMPTY and A01 trials per hand\n'
                 'Descriptive audit only; controls are not obstacle positives; UNKNOWN is not BLOCKED', fontsize=12)
    fig.savefig(DOC / 'PROBE_HISTORY_AUDIT.png', dpi=150)
    plt.close(fig)
    (DOC / 'PROBE_HISTORY_AUDIT.json').write_text(json.dumps(dict(
        status='ACTUAL_DESCRIPTIVE_AUDIT_NOT_MODEL_EVALUATION',
        source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        selection='first lexicographic completed ID per side in EMPTY/A01; no label-based selection',
        interpolation=False, histories=selected,
        limitation='One trial per condition/hand; no uncertainty bands or class-discrimination inference'), indent=2), encoding='utf-8')
    print('ACTUAL_HISTORY_FIGURE', len(selected), DOC / 'PROBE_HISTORY_AUDIT.png')


if __name__ == '__main__':
    main()
