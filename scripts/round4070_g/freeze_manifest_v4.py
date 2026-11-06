"""Freeze the documented G1 reset/evidence amendment before primary G1 runs.

V3 raw pilot records and physical gates are preserved. The physical scene
parameters and action contract are unchanged; primary IDs and admission differ.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/round4070_comprehensive_traversability'
DOC = ROOT / 'research/round4070_comprehensive_traversability'


def data(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')


def exact_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f'Never overwrite frozen bytes: {path}')
    else:
        with path.open('xb') as handle:
            handle.write(content)


def main():
    target = OUT / 'campaign_manifest_v4.json'
    if target.exists():
        raise FileExistsError('V4 already frozen; restore/resume it rather than resampling')
    previous = OUT / 'campaign_manifest_v3.json'
    old_bytes = previous.read_bytes()
    old_sha = hashlib.sha256(old_bytes).hexdigest()
    assert old_sha == json.loads(previous.with_suffix('.sha256.json').read_text())['sha256']
    old = json.loads(old_bytes)
    pilot_files = sorted((OUT / 'trials').glob('*/*/campaign_launch.json'))
    assert len(pilot_files) + len(old['trial_plans']) <= old['max_g1_launches'], 'Budget cannot absorb complete revised primary queue'
    assert not any('_rng1' in p.parent.parent.name for p in pilot_files), 'Primary G1 started before this freeze'
    randomization = dict(schema='G1_RNG_V1', seed_generator='numpy.random.default_rng(episode_seed); PCG64',
        draw_order=['robot_delta_y_m', 'robot_delta_yaw_rad', 'free_node_velocity_xyz_global_node_order'],
        robot_y_uniform_m=[-.001, .001], robot_yaw_uniform_rad=[-.001, .001],
        free_node_velocity_uniform_mps=[-.001, .001],
        robot_reference='add to existing frozen robot reset center; same seed means same robot draws for hands and paired variants',
        node_rule='one initial post-reset injection before first physical step; all attachment vertices exactly zero',
        evidence='record actual additive robot offsets, applied full node velocity array and SHA256, node IDs and injection simulation time',
        deployment_input='randomization seed, scene and node metadata are evaluation-only, never model features',
        purpose='repair previously ineffective G1 seed variation; ranges fixed before revised primary runs; not outcome tuning')
    rng_bytes = data(randomization)
    rng_sha = hashlib.sha256(rng_bytes).hexdigest()
    result = copy.deepcopy(old)
    result.update(schema='round4070-g.campaign.v4', created_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_manifest='campaign_manifest_v3.json', predecessor_manifest_sha256=old_sha,
        config_directory='configs_v4', plan_directory='plans_v4',
        g1_reset_randomization=randomization, g1_reset_randomization_sha256=rng_sha,
        g1_primary_protocol='G1_RNG_V1_BODY_COLLISION_AUDIT_V1',
        primary_admission='Only revised _rng1 trials with explicit seeded reset and complete sampled actual-collider body/cloth audit may supply supervised P/B. Unsupported audit remains UNKNOWN or physics-invalid; no truth model inputs.',
        prior_pilots=dict(status='SUPERSEDED_DIAGNOSTIC_ONLY_NOT_PRIMARY', actual_launches_at_freeze=len(pilot_files),
            launch_files=[str(p.relative_to(ROOT)) for p in pilot_files],
            reason='G1 seeds did not explicitly randomize state; old G1 lacks synchronized body collider poses/nonpenetration evidence. Preserve raw labels as historical diagnostics, exclude from primary metrics.'),
        reserved_retry_launches=old['max_g1_launches']-len(pilot_files)-len(old['trial_plans']),
        total_budget_rule='All actual G1 launches across v3 pilots, v4 primary, invalid attempts and retries count toward 1000',
        physics_gate_reuse='Exact scene physical hash, three gate seeds and original summaries unchanged; existing gate free-node velocity amplitude .002m/s exceeds revised G1 .001m/s. Robot reset perturbations require new EMPTY controls, not a claim of material validation.',
        amendment='Protocol/evidence bug repair before any model training. Pilot outcomes retained; no trial deletion or best-result replacement.')
    for plan in result['trial_plans']:
        old_id = plan['trial_id']
        generator = np.random.default_rng(plan['seed'])
        delta_y, delta_yaw = generator.uniform(-.001, .001, size=2)
        plan.update(schema='round4070-g.trial.v2', trial_id=old_id + '_rng1', predecessor_planned_trial_id=old_id,
            primary_protocol=result['g1_primary_protocol'],
            g1_reset_randomization=dict(**randomization, seed=plan['seed'], spec_sha256=rng_sha,
                frozen_robot_delta_y_m=float(delta_y), frozen_robot_delta_yaw_rad=float(delta_yaw)),
            body_collision_audit_required=True)
    assert len({p['trial_id'] for p in result['trial_plans']}) == 976
    assert result['scene_specs'] == old['scene_specs'], 'Physical scenes must not change in this amendment'
    for sid, spec in result['scene_specs'].items():
        exact_new(OUT / 'configs_v4' / (sid + '.json'), data(spec))
    for plan in result['trial_plans']:
        exact_new(OUT / 'plans_v4' / (plan['trial_id'] + '.json'), data(plan))
    frozen = data(result)
    digest = hashlib.sha256(frozen).hexdigest()
    for directory in (OUT, DOC):
        exact_new(directory / 'g1_reset_randomization_v1.json', rng_bytes)
        exact_new(directory / 'campaign_manifest_v4.json', frozen)
        exact_new(directory / 'campaign_manifest_v4.sha256.json', data(dict(path='campaign_manifest_v4.json', sha256=digest)))
    exact_new(DOC / 'CURRENT_CAMPAIGN.json', data(dict(manifest='campaign_manifest_v4.json', sha256=digest,
        primary_protocol=result['g1_primary_protocol'])))
    print(json.dumps(dict(status='FROZEN_NOT_EXECUTED', manifest_sha256=digest,
        primary_planned=976, prior_actual_pilots=len(pilot_files), remaining_retry_reserve=result['reserved_retry_launches'])))


if __name__ == '__main__':
    main()
