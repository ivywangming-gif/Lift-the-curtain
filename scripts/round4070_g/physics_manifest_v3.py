"""Preserve v1/v2, correct DR family metadata and document apparatus repair."""
from pathlib import Path
import json,hashlib,datetime as dt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/round4070_comprehensive_traversability'
old=OUT/'campaign_manifest_v2.json';path=OUT/'campaign_manifest_v3.json'
if path.exists():raise FileExistsError(path)
assert not list((OUT/'trials').glob('*/*/campaign_launch.json'))
m=json.loads(old.read_text(encoding='utf-8'));m.update(schema='round4070-g.campaign.v3',supersedes_manifest_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),revision_utc=dt.datetime.now(dt.timezone.utc).isoformat(),config_directory='configs_v3',revision_reason='DR family metadata corrected to DR_COMBINATION (physical hash unchanged); apparatus frame-conversion defect repaired before G1; v1 first diagnostic preserved. Contact evidence proximity is explicitly proxy, strict penetration unchanged.')
m['physics_apparatus_revision']=dict(id='v2',force_cap_n=3.,torque_cap_nm=.03,force_frame='current body frame explicitly converted each physics step',contact_evidence='triangle proximity within85mm probe-center box AND induced mesh displacement>20mm AND bounded applied force>.05N, or 3D wall proximity',probe_interior_half_m=.05,wall_interior_tolerance_m=.005,collision_sampling_hz=50,not_contact_force_truth=True,first_failed_attempt='physics/A01/seed40701_attempt01')
for sid,s in m['scene_specs'].items():
    if sid.startswith('DR'):s['family']='DR_COMBINATION'
    d=OUT/'configs_v3';d.mkdir(exist_ok=True);(d/(sid+'.json')).write_text(json.dumps(s,indent=2,sort_keys=True),encoding='utf-8')
for name in m['source_sha256']:m['source_sha256'][name]=hashlib.sha256((ROOT/'scripts/round4070_g'/name).read_bytes()).hexdigest()
path.write_text(json.dumps(m,indent=2,sort_keys=True),encoding='utf-8')
sha=hashlib.sha256(path.read_bytes()).hexdigest();path.with_suffix('.sha256.json').write_text(json.dumps(dict(path=str(path),sha256=sha),indent=2),encoding='utf-8');print(sha)
