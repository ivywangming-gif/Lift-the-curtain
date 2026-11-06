"""Synthetic runtime fingerprints against the real frozen calibration.

These are implementation tests, not extra physics observations or G1 trials.
"""
from pathlib import Path
import copy,json,tempfile,unittest
from physics_native_bind import ROOT,read,sha,bind_trial_geometry,verify_native_mesh_scale

TEMP=ROOT/'outputs/round4070_comprehensive_traversability/implementation_smoke'


def fixture(run):
    contract=read(ROOT/'outputs/round4070_comprehensive_traversability/g1_action_contract_v2.json')
    library=read(ROOT/contract['witness']['native_library_path'])
    source=Path(__file__).parent/'physics_native_bind.py';digest=sha(source)
    archive=run/'archived_native_binder.py';archive.write_bytes(source.read_bytes())
    rows=[]
    for old in library['native_query_template']:
        row=copy.deepcopy(old);row.update(body_index=library['body_names'].index(row['body_name']),finished=True,rigid_body_results=['PhysxPropertyQueryResult.VALID'])
        for collider in row['colliders']:collider['result']='PhysxPropertyQueryResult.VALID'
        rows.append(row)
    geom=dict(schema='round4070-g.robot_collision_geometry.v1',stage_meters_per_unit=1.,complete_robot_prim_traversal_with_instance_proxies=True,
        runtime_physx_binding=library['runtime_physx_binding'],runtime_articulation_data={'status':'CAPTURED','source_sha256':library['articulation_data_source_sha256']},
        source_asset_layers=library['source_asset_layers'],body_names=library['body_names'],colliders=copy.deepcopy(library['usd_inventory_template']),
        native_property_query_inventory=dict(status='CAPTURED',shape_count=39,body_count=39,body_names=library['body_names'],bodies=rows,source_sha256=digest),sheets=[],wall=None)
    for collider in geom['colliders']:collider['body_index']=library['body_names'].index(collider['body_name'])
    (run/'g_robot_collision_geometry.json').write_text(json.dumps(geom))
    (run/'g_runner_contract.json').write_text(json.dumps({'runtime_source_archive':{'physics_native_bind.py':{'path':str(archive),'sha256':digest}}}))
    return contract,geom,library


class NativeBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=TEMP);self.run=Path(self.temp.name)
        self.contract,self.geom,self.library=fixture(self.run)
    def tearDown(self):self.temp.cleanup()
    def save(self):(self.run/'g_robot_collision_geometry.json').write_text(json.dumps(self.geom))
    def test_full_fingerprint_and_independent_replay(self):
        result=bind_trial_geometry(self.run,self.contract);self.assertEqual(result['status'],'VERIFIED')
        native=read(self.run/result['path'])
        self.assertEqual(bind_trial_geometry(self.run,self.contract,write_output=False,archived_query_source_sha256=native['binder_source_sha256']),native)
    def test_body_reordering_is_not_silently_remapped(self):
        self.geom['body_names']=self.geom['body_names'][::-1];self.save()
        with self.assertRaises(ValueError):bind_trial_geometry(self.run,self.contract)
    def test_collider_material_or_simulation_flags_reject(self):
        original=copy.deepcopy(self.geom)
        for change in [lambda c:c['physics_attributes'].__setitem__('physics:collisionEnabled',False),
                       lambda c:c['relationships'].__setitem__('material:binding:physics',['/ChangedMaterial']),
                       lambda c:c['ancestor_physics_attributes'].__setitem__('physics:rigidBodyEnabled',False)]:
            self.geom=copy.deepcopy(original);change(self.geom['colliders'][0]);self.save()
            with self.assertRaises(ValueError):bind_trial_geometry(self.run,self.contract)
    def test_native_fingerprint_missing_or_modified_rejects(self):
        original=copy.deepcopy(self.geom)
        for change in [lambda q:q['bodies'][0].__setitem__('finished',False),lambda q:q['bodies'].append(q['bodies'][0]),
                       lambda q:q.__setitem__('source_sha256','0'*64),
                       lambda q:q['bodies'][1]['colliders'][0]['aabb_local_min'].__setitem__(0,99.)]:
            self.geom=copy.deepcopy(original);change(self.geom['native_property_query_inventory']);self.save()
            with self.assertRaises(ValueError):bind_trial_geometry(self.run,self.contract)
    def test_witness_index_shape_or_hash_cannot_change(self):
        for key,value in [('body_index',0),('native_shape_id','not_actual'),('native_geometry_sha256','0'*64),('verified_interior_radius_m',.1)]:
            contract=copy.deepcopy(self.contract);contract['witness'][key]=value
            with self.assertRaises(ValueError):bind_trial_geometry(self.run,contract)
    def test_native_mesh_scale_guard_uses_actual_handle_chain(self):
        source={'geometry':{'native_geometry_class':'convexmesh_ref'},'native_handle_chain':{
            'native_convex_mesh_handle':10,'native_recorded_mesh_scale':[1,1,1],
            'native_mesh_scale_isotropic':True,'mesh_scale_rotation_evidence':'IRRELEVANT_FOR_ISOTROPIC_SCALE'}}
        verify_native_mesh_scale(source)
        source['native_handle_chain']['native_recorded_mesh_scale']=[1,2,1]
        with self.assertRaises(ValueError):verify_native_mesh_scale(source)


if __name__=='__main__':unittest.main(verbosity=2)
