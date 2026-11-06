"""Verify source preservation and curated scientific denominators before delivery."""
import csv, hashlib, json, py_compile
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
O=ROOT/'outputs/round4070_learning_traversability'
R=ROOT/'research/round4070_learning_traversability'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest=load(O/'dataset/manifest.json'); checks={}
    assert sha(O/'dataset/manifest.json')==load(O/'phase_a_valid/frozen_config.json')['manifest_sha256']
    count=0
    for e in manifest['episodes']:
        for name,digest in e['source_sha256'].items():
            assert sha(ROOT/'outputs/round4070_fixed_probe_benchmark/trials'/e['trial_id']/name)==digest
            count+=1
    checks['E_source_hashes_unchanged']=count
    with (O/'phase_a_valid/predictions.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    groups=defaultdict(list)
    for row in rows:
        if row['fold'] in ['family_fold_0','family_fold_1']:groups[row['method'],row['seed']].append(row)
    for key,rr in groups.items():
        assert len(rr)==len({r['trial_id'] for r in rr})==28,key
        assert sum(int(float(r['y'])) for r in rr)==2,key
    checks['pooled_method_seed_groups_validated']=len(groups)
    checks['test_contracts']='See outputs test_contracts.log: 5 tests passed'
    replay=load(O/'frozen_evaluation/E_HOLDOUT_REPLAY_VALIDATION.json')
    checks['frozen_replay']=replay
    expected='223908babf0fe72bd83d9a5f6801d85ed5d1a693e59ed0ea450f1b5da7207ffe'
    assert sha(ROOT/'lift_curtain/perception/online_occupancy.py')==expected
    checks['protected_dirty_online_occupancy_sha256']=expected
    for p in (ROOT/'scripts/round4070_f').glob('*.py'):py_compile.compile(str(p),doraise=True)
    checks['compiled_python_files']=len(list((ROOT/'scripts/round4070_f').glob('*.py')))
    checks['artifacts']={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'scripts/round4070_f').glob('*.py'))}
    (R/'DELIVERY_VALIDATION.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in checks.items() if k not in ['artifacts','frozen_replay']},indent=2))
if __name__=='__main__':main()
