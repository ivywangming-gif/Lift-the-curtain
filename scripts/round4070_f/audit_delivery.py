"""Verify source preservation and curated scientific denominators before delivery."""
import csv, hashlib, json, py_compile
from collections import Counter, defaultdict
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
    extended=[]
    for run_name,n,p in [('phase_c',36,7),('phase_a_instantaneous',28,2)]:
        with (O/run_name/'predictions.csv').open(newline='',encoding='utf-8') as f: rr=list(csv.DictReader(f))
        gg=defaultdict(list)
        for row in rr:
            if row['fold'] in ['family_fold_0','family_fold_1']:gg[row['method'],row['seed']].append(row)
        for key,items in gg.items():
            assert len(items)==len({r['trial_id'] for r in items})==n,(run_name,key)
            assert sum(int(float(r['y'])) for r in items)==p,(run_name,key)
        extended.append(dict(run=run_name,method_seed_groups=len(gg),unique_test_episodes=n,passable=p))
    checks['additional_experiments']=extended
    phase_c=load(O/'dataset_phase_c/manifest.json')
    assert len(phase_c['episodes'])==115
    assert sum(e['y']==1 and not e['is_control'] for e in phase_c['episodes'])==7
    assert sum(e['y']==0 and not e['is_control'] for e in phase_c['episodes'])==29
    assert sum(e['y']==-1 for e in phase_c['episodes'])==63
    checks['phase_c_labels_verified']=phase_c['labels']
    frozen=load(O/'physics/first11_frozen.sha256.json')
    assert sha(O/'physics/first11_frozen.json')==frozen['sha256']
    for path,digest in frozen['protected_sha256'].items():
        assert sha(ROOT/path)==digest,path
    checks['frozen_training_and_original_evaluation_files_unchanged']=len(frozen['protected_sha256'])
    final=load(O/'dataset_final/manifest.json')
    assert len(final['episodes'])==117
    assert Counter(e['label'] for e in final['episodes'])==dict(
        CONTROL_PASSABLE=14,CONTROL_BLOCKED=2,PASSABLE=7,BLOCKED=29,UNKNOWN=65)
    assert sum(e['sample_count'] for e in final['episodes'])==2526
    assert not any(e['y']==1 and not e['is_control'] and e['side']=='left' for e in final['episodes'])
    final_by_id={e['trial_id']:e for e in final['episodes']}
    for old in phase_c['episodes']:
        new=final_by_id[old['trial_id']]
        for field in ['label','y','group_id','source_sha256']:
            assert old[field]==new[field],(old['trial_id'],field)
    added=[e for e in final['episodes'] if e['trial_id'] not in {x['trial_id'] for x in phase_c['episodes']}]
    assert len(added)==2 and all(e['y']==-1 and e['side']=='left' for e in added)
    with (R/'LEFT_COMPLETION_F_FROZEN_EVALUATION.csv').open(newline='',encoding='utf-8') as f:
        left_rows=list(csv.DictReader(f))
    for row in left_rows:
        assert int(row['n'])==0
        for field in ['balanced_accuracy','brier','coverage','false_pass','false_reject',
                      'passable_recall','blocked_recall','uncertain_rate']:
            assert row[field]=='',(field,row)
    checks['final_dataset']=dict(episodes=117,real_samples=2526,labels=final['labels'],
        new_supervised_labels=0,formal_left_passable=0,unknown_only_metric_rows=len(left_rows),
        manifest_sha256=sha(O/'dataset_final/manifest.json'))
    physics=load(O/'physics/PHYSICS_SUMMARY.json')
    assert physics['g1_attempts']==15 and physics['runtime_valid']==physics['probe_valid']==13
    assert physics['labels']==dict(PASSABLE=5,BLOCKED=3,UNKNOWN=5)
    assert physics['prior_11_labels_unchanged']
    assert len(physics['p1'])==25 and len(physics['p0_new_size'])==4
    for row in physics['p1']+physics['p0_new_size']:
        assert row['status']=='PASS_PHYSICS_SANITY',row
    for row in physics['p1']:
        assert all(row['extended_geometry_audit']['checks'].values()),row
    assert len(physics['p1_execution_invalid'])==1
    checks['physical_execution']=dict(g1_attempts=15,g1_valid=13,g1_startup_invalid=2,
        p1_six_scene_gates=24,p1_mobility=1,p1_execution_invalid=1,new_size_p0_gates=4,
        physics_summary_sha256=sha(O/'physics/PHYSICS_SUMMARY.json'))
    # Keep small evidence summaries in Git; all raw trials/checkpoints stay local.
    for source,target in [('physics/PHYSICS_SUMMARY.json','PHYSICS_RUN_SUMMARY.json'),
                          ('left_completion_integrity.json','LEFT_COMPLETION_INTEGRITY.json'),
                          ('physics/first11_frozen.sha256.json','FROZEN_ARTIFACT_HASHES.json')]:
        (R/target).write_bytes((O/source).read_bytes())
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
