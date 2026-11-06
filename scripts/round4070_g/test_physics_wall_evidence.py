"""Pure actual-wall evidence counterexamples; no USD/Isaac imports or launch."""
import copy,unittest
import numpy as np
from collision_geometry import GeometryNotVerified
from physics_wall_evidence import SCHEMA,validate_wall_snapshot


def fixture():
    expected={'center_xyz':[.85,0,1.],'size_xyz':[.04,1.2,2.],'fixed':True,'candidate_plane_x':.85,'initial_gap_m':0.}
    matrix=np.diag([.02,.6,1.,1.]);matrix[:3,3]=expected['center_xyz'];path='/World/GScene/RigidBarrier/geometry/mesh'
    evidence=dict(schema=SCHEMA,wall_requested=True,barrier_root_exists=True,stage_meters_per_unit=1.,
        collision_prims=[dict(prim_path=path,type='Cube',has_collision_api=True,collision_enabled=True,trigger_api=False,
            enabled_rigid_body_ancestors=[],time_varying_transform_attributes=[],cube_size=2.,local_to_world_matrix4=matrix.tolist())],
        native_presence_query=dict(api='IPhysxSceneQuery.overlap_box',completed=True,any_hit=False,center_world_m=expected['center_xyz'],
            half_extent_m=[.001]*3,returned_count=2,hits=[{'collision':'/OtherCollider'},{'collision':path}]))
    rays=[];center=np.asarray(expected['center_xyz']);half=np.asarray(expected['size_xyz'])/2
    for axis in range(3):
        for side in [-1,1]:
            position=center.copy();position[axis]+=side*half[axis];origin=position.copy();origin[axis]+=side*.01
            normal=np.zeros(3);normal[axis]=side
            rays.append(dict(axis=axis,side=side,api='IPhysxSceneQuery.raycast_all',completed=True,origin_world_m=origin.tolist(),
                unit_direction=(-normal).tolist(),returned_hit=True,callback_count=1,hits=[{'collision':path,'distance_m':.01,'position_world_m':position.tolist(),'normal_world':normal.tolist()}]))
    evidence['native_face_rays']=rays
    return expected,evidence


class ActualWallEvidenceTests(unittest.TestCase):
    def test_true_analytic_static_readback_and_exact_native_path(self):
        expected,evidence=fixture();actual=validate_wall_snapshot(evidence,expected)
        self.assertTrue(actual['fixed']);self.assertTrue(actual['actual_static_analytic_geometry_verified'])
        np.testing.assert_allclose(actual['size_xyz'],expected['size_xyz'])
    def test_missing_disabled_trigger_dynamic_animated_collider_rejected(self):
        expected,original=fixture()
        changes=[lambda e:e.__setitem__('barrier_root_exists',False),lambda e:e.__setitem__('collision_prims',[]),
            lambda e:e['collision_prims'][0].__setitem__('collision_enabled',False),lambda e:e['collision_prims'][0].__setitem__('trigger_api',True),
            lambda e:e['collision_prims'][0].__setitem__('enabled_rigid_body_ancestors',['/World/GScene/RigidBarrier']),
            lambda e:e['collision_prims'][0].__setitem__('time_varying_transform_attributes',['wall.xformOp:translate'])]
        for change in changes:
            evidence=copy.deepcopy(original);change(evidence)
            with self.subTest(change=change),self.assertRaises(GeometryNotVerified):validate_wall_snapshot(evidence,expected)
    def test_visual_or_transformed_geometry_does_not_substitute(self):
        expected,original=fixture()
        for change in [lambda c:c.__setitem__('type','Mesh'),lambda c:c['local_to_world_matrix4'][0].__setitem__(3,.9),
                       lambda c:c.__setitem__('cube_size',1.),lambda c:c['local_to_world_matrix4'][0].__setitem__(1,.1)]:
            evidence=copy.deepcopy(original);change(evidence['collision_prims'][0])
            with self.assertRaises(GeometryNotVerified):validate_wall_snapshot(evidence,expected)
    def test_missing_native_path_or_incomplete_query_rejected(self):
        expected,original=fixture()
        changes=[lambda q:q.__setitem__('hits',[{'collision':'/DifferentActualPath'}]),lambda q:q.__setitem__('completed',False),
                 lambda q:q.__setitem__('any_hit',True),lambda q:q.__setitem__('center_world_m',[0,0,0])]
        for change in changes:
            evidence=copy.deepcopy(original);change(evidence['native_presence_query'])
            with self.assertRaises(GeometryNotVerified):validate_wall_snapshot(evidence,expected)
    def test_no_wall_explicit_absence_only(self):
        value=dict(schema=SCHEMA,wall_requested=False,barrier_root_exists=False)
        self.assertIsNone(validate_wall_snapshot(value,None));value['barrier_root_exists']=True
        with self.assertRaises(GeometryNotVerified):validate_wall_snapshot(value,None)
    def test_center_presence_cannot_replace_six_native_boundary_faces(self):
        expected,original=fixture()
        for change in [lambda e:e['native_face_rays'].pop(),lambda e:e['native_face_rays'][0]['hits'][0]['position_world_m'].__setitem__(0,.7),
                       lambda e:e['native_face_rays'][0]['hits'][0].__setitem__('distance_m',.02),
                       lambda e:e['native_face_rays'][0]['hits'][0].__setitem__('normal_world',[1,0,0]),
                       lambda e:e['native_face_rays'][0]['hits'][0].__setitem__('collision','/AnotherWall')]:
            evidence=copy.deepcopy(original);change(evidence)
            with self.assertRaises(GeometryNotVerified):validate_wall_snapshot(evidence,expected)
    def test_raycast_boolean_return_is_not_mistaken_for_hit_count(self):
        expected,evidence=fixture();ray=evidence['native_face_rays'][0]
        ray['hits'].append(dict(collision='/Ground',distance_m=.02,position_world_m=[0,0,0],normal_world=[0,0,1]))
        ray['callback_count']=2
        self.assertIsNotNone(validate_wall_snapshot(evidence,expected))
        ray['returned_hit']=False
        with self.assertRaises(GeometryNotVerified):validate_wall_snapshot(evidence,expected)


if __name__=='__main__':unittest.main(verbosity=2)
