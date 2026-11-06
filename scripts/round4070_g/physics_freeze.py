"""Freeze the complete G queue before collection; never overwrite a manifest."""
from pathlib import Path
import csv
import copy
import datetime as dt
import hashlib
import json
import numpy as np
from physics_scenes import scene_specs,base,physical_hash

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/round4070_comprehensive_traversability'
DOC=ROOT/'research/round4070_comprehensive_traversability'
CORE=['A01','A02','A06','B02','B03','B04','B05','C03','C04','D02','D03','D04']


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf-8')


def freeze():
    specs=scene_specs();aliases={};representatives={}
    for sid,spec in specs.items():
        key=spec['physical_config_id'];aliases.setdefault(key,[]).append(sid);representatives.setdefault(key,sid)
    empty=base('EMPTY','empty response and safe robot protocol control');empty['geometry']['kind']='empty';empty['material_fidelity']='NO_OBSTACLE_CONTROL';empty['physical_config_id']=physical_hash(empty);specs['EMPTY']=empty
    ranges={'height_m':[1.80,2.00],'width_m':[1.05,1.20],'bottom_z_m':[.02,.08],'center_x_m':[.84,.86],'density_kg_m3':[175.,375.],'thickness_m':[.0018,.0022],'stretch_stiffness':[450.,550.],'shear_stiffness':[225.,275.],'bend_stiffness':[.015,.04],'damping':[.04,.08],'static_friction':[.4,.6],'dynamic_friction':[.3,.5],'pleat_amplitude_m':[0.,.015]}
    rng=np.random.default_rng(4070007);lhs={k:(rng.permutation(12)+rng.uniform(size=12))/12 for k in ranges}
    for i in range(12):
        s=base(f'DR{i+1:02d}','frozen Latin hypercube combination')
        for k,(lo,hi) in ranges.items():
            target=s['geometry'] if k in s['geometry'] else s['physics'];target[k]=float(lo+(hi-lo)*lhs[k][i])
        if i%4==1:s['topology']['kind']='TOP_PLUS_LEFT'
        if i%4==2:s['topology']['kind']='TOP_PLUS_RIGHT'
        if i%4==3:s['physics']['hem_mass_kg']=.05
        s['physical_config_id']=physical_hash(s);specs[s['scene_id']]=s
    plans=[]
    rep=list(representatives.values())
    # Family interleaving makes early coverage broad. Equal configurations share
    # trials with explicit aliases, never fictitious independent repetitions.
    rep.sort(key=lambda x:(int(x[1:]),x[0]))
    def add(sid,side,index,phase,variant='nominal',seed=None):
        spec=specs[sid];robot=dict(reset_root_y=-.004,reset_root_yaw=-.004,upper_offset_rad=[0.]*14,probe_duration_s=1.5)
        if variant=='position':robot['reset_root_y']=.006
        elif variant=='yaw':robot['reset_root_yaw']=.008
        elif variant=='arm':robot['upper_offset_rad'][0 if side=='left' else 7]=.02
        elif variant=='speed':robot['probe_duration_s']=1.65
        seed=seed if seed is not None else 70000+list(specs).index(sid)*100+index
        role=('train' if index in [0,3,8,9] else 'calibration' if index in [1,10] else 'test' if index in [2,11] else 'confirmation')
        if phase=='robot_variation':role='heldout_posture'
        trial_id=f'G_{sid}_{side}_{phase}_{index:02d}_{variant}_s{seed}'
        plans.append(dict(schema='round4070-g.trial.v1',trial_id=trial_id,scene_id=sid,scene_aliases=aliases.get(spec['physical_config_id'],[sid]),physical_config_id=spec['physical_config_id'],seed=seed,probe_side=side,followup_hand=side,followup_action_id='FROZEN_R1_BOUNDED_FORWARD_V1',label_contract='PASSABLE_FOR_ACTION',repetition_index=index,experiment_role=role,phase=phase,posture_variant_id=variant,observation_variant_id='clean',robot=robot,physics_valid=False,withhold_followup_for_geometric_safety=sid in ['D01','D02','D03','D04'],initial_status='PLANNED'))
    # Empty controls come first for both hands and each bounded robot change.
    for i in range(12):
        variant=['position','yaw','arm','speed'][i-4] if 4<=i<8 else 'nominal'
        for side in ['left','right']:add('EMPTY',side,i,'empty_control',variant)
    for i in range(3):
        for sid in rep:
            for side in ['left','right']:add(sid,side,i,'coverage')
    for sid in CORE:
        seed=70000+list(specs).index(sid)*100
        for j,variant in enumerate(['position','yaw','arm','speed']):
            for side in ['left','right']:add(sid,side,12+j,'robot_variation',variant,seed)
    for sid in [s for s in specs if s.startswith('DR')]:
        for i in range(3):
            for side in ['left','right']:add(sid,side,i,'combination')
    for i in range(3,12):
        for sid in rep:
            if i>=8 and sid not in CORE:continue
            for side in ['left','right']:add(sid,side,i,'confirmation' if 4<=i<8 else 'repetition')
    assert len(plans)==976,(len(plans),len(rep))
    source_paths=['physics_scenes.py','physics_scene_builder.py','physics_gate.py','physics_freeze.py']
    source_hashes={p:hashlib.sha256((ROOT/'scripts/round4070_g'/p).read_bytes()).hexdigest() for p in source_paths}
    manifest=dict(schema='round4070-g.campaign.v1',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),label_contract='PASSABLE_FOR_ACTION',max_g1_launches=1000,planned_g1_trials=len(plans),retry_reserve=24,physics_seeds=[40701,40702,40703],physics_seconds=24.,representative_scene_ids=rep,core_scene_ids=CORE,physical_aliases=aliases,scene_specs=specs,trial_plans=plans,source_sha256=source_hashes,dr_seed=4070007,dr_ranges=ranges,dr_range_status='SIMULATION_SENSITIVITY_HYPOTHESES_FROM_NEARBY_VALIDATED_CONFIGS_EVERY_COMBINATION_REQUIRES_NEW_GATE',observation_noise=dict(status='FROZEN_SIMULATION_SENSITIVITY_NOT_HARDWARE_CALIBRATION',joint_encoder_sigma_rad=.001,joint_velocity_sigma_rad_s=.01,effort_proxy_relative_sigma=.02,effort_proxy_bias_relative=.01,imu_angular_sigma_rad_s=.005,imu_orientation_sigma_rad=.001,timestamp_jitter_max_s=.002,observation_delay_max_s=.020,drop_probability=.01,applied='OFFLINE_TRAIN_GROUP_ONLY_NEVER_LABEL_GENERATION'),bounded_repairs=dict(max_revisions_per_scene=1,allowed='Exact construction/runtime API repair only without threshold relaxation; physical parameter changes require distinct config hash and three new gates. Preserve initial failures.'),execution=dict(serial=True,headless=True,no_renderer=True,frozen_falcon_sha256='8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0',training_window='training_window.json',resource_lock='resource_lock.json',minimum_available_ram_mb=2048,minimum_free_vram_mb=4096),status='FROZEN_PENDING_EXECUTION')
    for sid,spec in specs.items():dump(OUT/'configs'/f'{sid}.json',spec)
    for plan in plans:dump(OUT/'plans'/f'{plan["trial_id"]}.json',plan)
    path=OUT/'campaign_manifest_v1.json';dump(path,manifest)
    sha=hashlib.sha256(path.read_bytes()).hexdigest();dump(OUT/'campaign_manifest_v1.sha256.json',dict(path=str(path),sha256=sha))
    DOC.mkdir(parents=True,exist_ok=True)
    rows=[dict(scene_id=sid,physical_config_id=s['physical_config_id'],family=s['family'],material_fidelity=s['material_fidelity'],representative=representatives.get(s['physical_config_id'],sid),physics_status='PENDING',g1_status='PENDING',name=s['name']) for sid,s in specs.items()]
    for filename,items in [('SCENE_COVERAGE_MATRIX.csv',rows),('DOMAIN_RANDOMIZATION_MANIFEST.csv',[r for r in rows if r['scene_id'].startswith('DR')])]:
        with (DOC/filename).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    print(json.dumps(dict(manifest=str(path),sha256=sha,named_scenes=48,unique_representatives=len(rep),combination_scenes=12,planned_g1=len(plans),reserve=24),indent=2))


if __name__=='__main__':freeze()
