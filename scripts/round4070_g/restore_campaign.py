"""Restore the currently bundled frozen campaign exactly, without resampling."""
from pathlib import Path
import hashlib
import json

from freeze_manifest_v4 import data, exact_new

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/round4070_comprehensive_traversability'
DOC = ROOT / 'research/round4070_comprehensive_traversability'


def restore():
    pointer = json.loads((DOC / 'CURRENT_CAMPAIGN.json').read_text(encoding='utf-8'))
    name = pointer['manifest']
    assert Path(name).name == name
    frozen = (DOC / name).read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == pointer['sha256']
    hash_sidecar=(DOC / name).with_suffix('.sha256.json')
    hash_metadata=json.loads(hash_sidecar.read_text(encoding='utf-8-sig'))
    if hash_metadata.get('path')!=name or hash_metadata.get('sha256')!=pointer['sha256']:
        raise RuntimeError('Bundled manifest hash metadata differs from verified pointer/bytes')
    manifest = json.loads(frozen)
    for sid, spec in manifest['scene_specs'].items():
        exact_new(OUT / manifest['config_directory'] / (sid + '.json'), data(spec))
    for plan in manifest['trial_plans']:
        exact_new(OUT / manifest['plan_directory'] / (plan['trial_id'] + '.json'), data(plan))
    rng_name = 'g1_reset_randomization_v1.json'
    rng = (DOC / rng_name).read_bytes()
    assert hashlib.sha256(rng).hexdigest() == manifest['g1_reset_randomization_sha256']
    exact_new(OUT / rng_name, rng)
    for artifact_name, digest in manifest.get('frozen_artifacts', {}).items():
        if Path(artifact_name).name != artifact_name:
            raise ValueError('Frozen artifact must have a local basename')
        artifact = (DOC / artifact_name).read_bytes()
        if hashlib.sha256(artifact).hexdigest() != digest:
            raise RuntimeError('Frozen artifact SHA256 mismatch: ' + artifact_name)
        exact_new(OUT / artifact_name, artifact)
    exact_new(OUT / name, frozen)
    exact_new((OUT / name).with_suffix('.sha256.json'), hash_sidecar.read_bytes())
    print('CURRENT_FROZEN_CAMPAIGN_RESTORED', name, pointer['sha256'], len(manifest['trial_plans']))


if __name__ == '__main__':
    restore()
