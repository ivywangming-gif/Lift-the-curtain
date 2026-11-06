import json,tempfile,unittest
from pathlib import Path
import numpy as np
from physics_g1_collision_audit import audit_run,sha,write_audit,audit_intersections
from collision_geometry import prepare_convex


class AuditTests(unittest.TestCase):
    def setUp(self):
        workspace=Path(__file__).resolve().parents[2]
        self.temp=tempfile.TemporaryDirectory(dir=workspace/'outputs/round4070_comprehensive_traversability/implementation_smoke')
        self.root=Path(self.temp.name).resolve();assert self.root.is_relative_to(workspace)
        self.geom=dict(body_names=['torso'],sheets=[dict(global_node_offset=0,vertex_count=3,faces=[[0,1,2]])],wall=None)
        self.write('g_robot_collision_geometry.json',self.geom)
        self.write('g_outcome.json',dict(status='G_FIXED_ACTION_FINISHED'))
        self.write('g_trial_metadata.json',{'scene_config_evaluation_only':{'geometry':{'center_x_m':.85}}})
        self.native=dict(schema='round4070-g.verified_native_robot_geometry.v1',status='VERIFIED',geometry_complete=True,
            frame_verified=True,body_names=['torso'],expected_collision_body_indices=[0],native_collision_shape_count=1,
            runtime_geometry_sha256=sha(self.root/'g_robot_collision_geometry.json'),
            parts=[dict(type='sphere',body_index=0,body_name='torso',radius_m=.1,center_body_m=[0,0,0])])
        self.write('g_native_geometry_verified.json',self.native)
        self.nodes=np.tile(np.array([[0.,-.02,0],[0,.02,0],[0,0,.03]]),(3,1,1))
        self.history()

    def tearDown(self):self.temp.cleanup()
    def write(self,name,value):(self.root/name).write_text(json.dumps(value))
    def history(self,gap=.02):
        np.savez_compressed(self.root/'g_body_collision_history_audit_only.npz',time_s=np.arange(3)*gap+.02,
            actual_simulation_time_s=np.arange(3)*gap+.02,physics_step=np.arange(3)*round(gap/.005)+3,
            body_names=np.array(['torso']),body_positions_world=np.tile(np.array([[[1.1,0,0]]]),(3,1,1)),
            body_quaternions_wxyz=np.tile(np.array([[[1.,0,0,0]]]),(3,1,1)),nodes_world=self.nodes,
            node_velocities_world=np.zeros_like(self.nodes),phase=np.array(['physical_probe','responsive_traverse_stand','responsive_release_stand']),boundary=[False,True,True])
        self.write('g_body_collision_capture.json',dict(episode_finalized=True,synchronized_poststep=True,nominal_hz=50,
            history_sha256=sha(self.root/'g_body_collision_history_audit_only.npz'),geometry={'sha256':sha(self.root/'g_robot_collision_geometry.json')}))
    def test_exact_disjoint_and_final_clearance(self):
        result=audit_run(self.root);self.assertEqual(result['status'],'VERIFIED',result)
        self.assertAlmostEqual(result['candidate_clearance']['min_world_x_m'],1.)
        self.assertEqual(result['candidate_clearance']['sample_index'],1)
    def test_actual_surface_penetration_rejected(self):
        self.nodes[:,:,0]=1.1;self.history();result=audit_run(self.root)
        self.assertEqual(result['status'],'PHYSICS_INVALID',result)
        self.assertTrue(result['penetration_witnesses'])
    def test_missing_native_frame_never_passes(self):
        self.native['frame_verified']=False;self.write('g_native_geometry_verified.json',self.native)
        self.assertEqual(audit_run(self.root)['status'],'NOT_VERIFIED')
    def test_temporal_gap_never_passes(self):
        self.history(gap=.04);self.assertEqual(audit_run(self.root)['status'],'NOT_VERIFIED')
    def test_wall_contains_actual_solid(self):
        self.geom['wall']={'center_xyz':[1.1,0,0],'size_xyz':[1,1,1]};self.write('g_robot_collision_geometry.json',self.geom)
        self.native['runtime_geometry_sha256']=sha(self.root/'g_robot_collision_geometry.json');self.write('g_native_geometry_verified.json',self.native);self.history()
        self.assertEqual(audit_run(self.root)['status'],'PHYSICS_INVALID')
    def test_small_exact_sphere_fully_inside_wall_is_single_sided_intrusion(self):
        self.native['parts'][0]['radius_m']=.002
        self.geom['wall']={'center_xyz':[1.1,0,0],'size_xyz':[.04,1,1]};self.write('g_robot_collision_geometry.json',self.geom)
        self.native['runtime_geometry_sha256']=sha(self.root/'g_robot_collision_geometry.json');self.write('g_native_geometry_verified.json',self.native);self.history()
        result=audit_run(self.root);self.assertEqual(result['status'],'PHYSICS_INVALID')
        self.assertTrue(any(x.get('tolerance_applies_to')=='WALL_INTERIOR_DISTANCE_ONLY' for x in result['penetration_witnesses']))
    def test_immutable_audit_versions(self):
        first=write_audit(self.root,audit_run(self.root));original=first.read_bytes()
        self.nodes[:,:,0]=1.1;self.history();second=write_audit(self.root,audit_run(self.root))
        self.assertNotEqual(first,second);self.assertEqual(first.read_bytes(),original)
    def test_coincident_exact_box_and_conservative_box_have_different_results(self):
        vertices=np.array([[x,y,z] for x in [-.1,.1] for y in [-.1,.1] for z in [-.1,.1]])
        planes=np.c_[np.r_[np.eye(3),-np.eye(3)],np.full(6,-.1)]
        geometry=prepare_convex(planes,vertices=vertices)
        part=dict(type='convex',body_index=0,body_name='box',vertices_body_m=vertices,geometry=geometry)
        args=([part],np.zeros((1,1,3)),np.array([[[1.,0,0,0]]]),np.empty((1,0,3)),np.empty((0,3),int),{'center_xyz':[0,0,0],'size_xyz':[.2,.2,.2]})
        exact=audit_intersections(*args);self.assertIs(exact['penetration_free'],False)
        self.assertTrue(any(x.get('criterion')=='COMMON_INTERIOR_BALL_RADIUS_EXCEEDS_5MM' for x in exact['penetration_witnesses']))
        geometry['exact_intersection_allowed']=False;geometry['evidence_mode']='NATIVE_VERTEX_CONSERVATIVE_ENVELOPE'
        outer=audit_intersections(*args);self.assertIsNone(outer['penetration_free']);self.assertFalse(outer['penetration_witnesses'])


if __name__=='__main__':unittest.main()
