"""Restore the immutable current campaign on a fresh checkout, without rerandomizing.

Historical physics_freeze/manifest_v2/manifest_v3 scripts are provenance of the
three preregistration versions. This is the operational fresh-output entry.
"""
from pathlib import Path
import hashlib
import json
import os
ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'research/round4070_comprehensive_traversability'
OUT=ROOT/'outputs/round4070_comprehensive_traversability'


def exact_file(path,content):
    if path.exists():
        if path.read_bytes()!=content:raise RuntimeError(f'Refuse to overwrite differing frozen file: {path}')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(content);f.flush();os.fsync(f.fileno())


def restore():
    source=DOC/'campaign_manifest_v3.json';data=source.read_bytes();expected=json.loads((DOC/'campaign_manifest_v3.sha256.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(data).hexdigest()==expected['sha256'],'Bundled manifest hash mismatch'
    m=json.loads(data);assert m['schema']=='round4070-g.campaign.v3'
    assert m['label_contract']=='PASSABLE_FOR_ACTION:G_HOLD_PROBE_POSE_FORWARD24_V1'
    for sid,s in m['scene_specs'].items():
        if sid.startswith('DR'):assert s['family']=='DR_COMBINATION'
        exact_file(OUT/m['config_directory']/(sid+'.json'),json.dumps(s,indent=2,sort_keys=True).encode())
    for plan in m['trial_plans']:exact_file(OUT/'plans_v3'/(plan['trial_id']+'.json'),json.dumps(plan,indent=2,sort_keys=True).encode())
    exact_file(OUT/source.name,data)
    exact_file(OUT/'campaign_manifest_v3.sha256.json',(DOC/'campaign_manifest_v3.sha256.json').read_bytes())
    print('FROZEN_V3_RESTORED',expected['sha256'],len(m['trial_plans']))


if __name__=='__main__':restore()
