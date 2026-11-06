"""Pre-G1 metadata correction: bind the actual frozen action contract."""
import datetime as dt
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/round4070_comprehensive_traversability'
source=OUT/'campaign_manifest_v1.json';target=OUT/'campaign_manifest_v2.json'
if target.exists():raise FileExistsError(target)
assert not list((OUT/'trials').glob('*/*/campaign_launch.json')),'Metadata correction must precede first G1'
m=json.loads(source.read_text(encoding='utf-8'))
m['schema']='round4070-g.campaign.v2'
m['supersedes_manifest_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
m['revision_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
m['revision_reason']='Pre-G1 exact metadata alignment to already implemented and agreed action contract; no physical parameter, seed, trial count, robot protocol or gate change.'
m['label_contract']='PASSABLE_FOR_ACTION:G_HOLD_PROBE_POSE_FORWARD24_V1'
for p in m['trial_plans']:
    p['followup_action_id']='G_HOLD_PROBE_POSE_FORWARD24_V1';p['label_contract']=m['label_contract']
    directory=OUT/'plans_v2';directory.mkdir(exist_ok=True)
    (directory/(p['trial_id']+'.json')).write_text(json.dumps(p,indent=2,sort_keys=True),encoding='utf-8')
for f in ['g1_trial.py','g1_recording.py','physics_campaign.py','resource_lock.py']:
    m['source_sha256'][f]=hashlib.sha256((ROOT/'scripts/round4070_g'/f).read_bytes()).hexdigest()
target.write_text(json.dumps(m,indent=2,sort_keys=True),encoding='utf-8')
sha=hashlib.sha256(target.read_bytes()).hexdigest()
target.with_suffix('.sha256.json').write_text(json.dumps(dict(path=str(target),sha256=sha),indent=2),encoding='utf-8')
print(sha)
