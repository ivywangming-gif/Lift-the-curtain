"""Restore the currently bundled frozen campaign exactly, without resampling."""
from pathlib import Path
import hashlib
import json

from freeze_manifest_v4 import data, exact_new
from budget_ledger import launch_ledger_summary

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
    # Known post-freeze launches remain spent even when raw outputs are not
    # included in a checkout. Validate this tracked ledger before any writes.
    budget=launch_ledger_summary(manifest,repo_root=ROOT,ledger_path=DOC/'G1_LAUNCH_LEDGER.json')
    quarantines=[]
    for record_path in sorted((DOC/'quarantine').glob('*.json')):
        record=record_path.read_bytes()
        digest_path=record_path.with_suffix('.sha256')
        digest=digest_path.read_bytes()
        if hashlib.sha256(record).hexdigest()!=digest.decode('ascii').strip():
            raise RuntimeError('Bundled quarantine record SHA256 mismatch: '+record_path.name)
        value=json.loads(record)
        if value.get('schema')!='round4070-g.exact_plan_quarantine.v1' or value.get('scope')!='EXACT_PLAN':
            raise RuntimeError('Bundled quarantine scope/schema mismatch')
        if value.get('evidence',{}).get('trial_id')!=record_path.stem:
            raise RuntimeError('Bundled quarantine filename/plan mismatch')
        quarantines.extend([(record_path.name,record),(digest_path.name,digest)])
    rng_name = 'g1_reset_randomization_v1.json'
    rng = (DOC / rng_name).read_bytes()
    if hashlib.sha256(rng).hexdigest() != manifest['g1_reset_randomization_sha256']:
        raise RuntimeError('Frozen reset randomization SHA256 mismatch')
    artifacts=[]
    for artifact_name, digest in manifest.get('frozen_artifacts', {}).items():
        if Path(artifact_name).name != artifact_name:
            raise ValueError('Frozen artifact must have a local basename')
        artifact = (DOC / artifact_name).read_bytes()
        if hashlib.sha256(artifact).hexdigest() != digest:
            raise RuntimeError('Frozen artifact SHA256 mismatch: ' + artifact_name)
        artifacts.append((artifact_name,artifact))
    # Validate every bundled input before creating even the first output.
    for sid, spec in manifest['scene_specs'].items():
        exact_new(OUT / manifest['config_directory'] / (sid + '.json'), data(spec))
    for plan in manifest['trial_plans']:
        exact_new(OUT / manifest['plan_directory'] / (plan['trial_id'] + '.json'), data(plan))
    exact_new(OUT / rng_name, rng)
    for artifact_name,artifact in artifacts:
        exact_new(OUT / artifact_name, artifact)
    exact_new(OUT / name, frozen)
    exact_new((OUT / name).with_suffix('.sha256.json'), hash_sidecar.read_bytes())
    for quarantine_name,content in quarantines:
        exact_new(OUT/'quarantine'/quarantine_name,content)
    print('CURRENT_FROZEN_CAMPAIGN_RESTORED', name, pointer['sha256'], len(manifest['trial_plans']))
    print('DURABLE_G1_BUDGET_PRESERVED',budget['count'],'MISSING_RAW_WITHHELD',len(budget['missing_raw_launches']))
    print('EXACT_QUARANTINE_ARTIFACTS_RESTORED',len(quarantines))


if __name__ == '__main__':
    restore()
