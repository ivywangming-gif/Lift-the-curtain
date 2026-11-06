"""Synthetic mathematical support-function contracts; no simulator evidence."""
import itertools
import unittest
import numpy as np

from collision_geometry import GeometryNotVerified,prepare_convex,prepare_sphere,prepare_cylinder
from collision_clearance import actual_collider_min_world_x


def cube(half=(.1,.2,.3)):
    half = np.asarray(half,dtype=float)
    vertices = np.asarray(list(itertools.product(*[(-v,v) for v in half])))
    planes = np.column_stack((np.concatenate((np.eye(3),-np.eye(3))),-np.r_[half,half]))
    return prepare_convex(planes,vertices),vertices


class ActualClearanceContracts(unittest.TestCase):
    def test_conservative_native_envelope_is_only_a_lower_bound(self):
        geometry, vertices = cube()
        geometry["evidence_mode"] = "CONSERVATIVE_NATIVE_VERTEX_OUTER_ENVELOPE_SEPARATION_ONLY"
        geometry["exact_intersection_allowed"] = False
        r = actual_collider_min_world_x([dict(body_index=0,geometry=geometry,vertices_body_m=vertices)],[[1,0,0]],[[1,0,0,0]])
        self.assertAlmostEqual(r["min_world_x_m"],.9)
        self.assertEqual(r["min_world_x_evidence"],"CONSERVATIVE_LOWER_BOUND")
        self.assertFalse(r["all_parts_exact_support"])
        self.assertFalse(r["per_part"][0]["witness_is_actual_surface_certified"])

    def test_all_parts_and_multiple_parts_per_body(self):
        geometry,vertices = cube()
        parts = [dict(body_index=0,geometry=geometry,vertices_body_m=vertices),
                 dict(body_index=1,geometry=prepare_sphere(.05,center=[-.2,0,0])),
                 dict(body_index=0,geometry=prepare_cylinder(.05,.2,axis="x",center=[.2,0,0]))]
        r = actual_collider_min_world_x(parts,[[1,0,0],[2,0,0]],[[1,0,0,0]]*2,expected_body_indices=[0,1])
        self.assertAlmostEqual(r["min_world_x_m"],.9)
        self.assertEqual(r["part_count"],3)
        self.assertEqual(len(r["per_part"]),3)
        self.assertTrue(r["all_expected_bodies_covered"])
        parts.append(dict(body_index=0,geometry=prepare_sphere(.05,center=[-.4,0,0])))
        r = actual_collider_min_world_x(parts,[[1,0,0],[2,0,0]],[[1,0,0,0]]*2)
        self.assertAlmostEqual(r["min_world_x_m"],.55)
        self.assertEqual(r["min_part_index"],3)

    def test_convex_rotation_uses_vertices_not_axis_aligned_proxy(self):
        geometry,vertices = cube()
        q = [np.sqrt(.5),0,0,np.sqrt(.5)]
        r = actual_collider_min_world_x([dict(body_index=0,geometry=geometry,vertices_body_m=vertices)],[[1,0,0]],[q])
        self.assertAlmostEqual(r["min_world_x_m"],.8)
        np.testing.assert_allclose(r["witness_world_m"][0],.8,atol=1e-12)

    def test_sphere_center_rotates_radius_does_not(self):
        q = [np.sqrt(.5),0,0,np.sqrt(.5)]
        shape = prepare_sphere(.05,center=[.3,0,0])
        r = actual_collider_min_world_x([dict(body_index=0,geometry=shape)],[[1,2,3]],[q])
        self.assertAlmostEqual(r["min_world_x_m"],.95)
        np.testing.assert_allclose(r["witness_world_m"],[.95,2.3,3],atol=1e-12)

    def test_cylinder_parallel_perpendicular_and_oblique_support(self):
        for axis,expected in (("x",.8),("z",.9),([1,0,1],1-.3/np.sqrt(2))):
            with self.subTest(axis=axis):
                r = actual_collider_min_world_x([dict(body_index=0,geometry=prepare_cylinder(.1,.2,axis=axis))],[[1,0,0]],[[1,0,0,0]])
                self.assertAlmostEqual(r["min_world_x_m"],expected)

    def test_fixed_random_cylinder_pose_analytic_formula_and_samples(self):
        rng = np.random.default_rng(4070)
        for _ in range(60):
            q = rng.normal(size=4); q /= np.linalg.norm(q)
            axis = rng.normal(size=3)
            center,position = rng.normal(size=(2,3))
            geometry = prepare_cylinder(.08,.23,axis=axis,center=center)
            r = actual_collider_min_world_x([dict(body_index=0,geometry=geometry)],position[None],q[None])
            w,x,y,z = q
            row = np.asarray([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)])
            cosine = float(row @ geometry["axis"])
            expected = position[0]+row @ center-.23*abs(cosine)-.08*np.sqrt(max(0.,1-cosine*cosine))
            self.assertAlmostEqual(r["min_world_x_m"],expected,places=12)
            angle = np.linspace(0,2*np.pi,200,endpoint=False)
            ring = np.column_stack((.08*np.cos(angle),.08*np.sin(angle),np.full(200,.23)))
            cloud = np.concatenate((ring,ring*np.array([1,1,-1]))) @ geometry["basis"].T+center
            self.assertLessEqual(r["min_world_x_m"],float(np.min(cloud @ row+position[0]))+1e-12)

    def test_invalid_quaternion_and_pose_arrays(self):
        parts = [dict(body_index=0,geometry=prepare_sphere(.1))]
        for q in ([[2,0,0,0]],[[0,0,0,0]],[[np.nan,0,0,1]]):
            with self.assertRaises(GeometryNotVerified):
                actual_collider_min_world_x(parts,[[0,0,0]],q)
        with self.assertRaises(GeometryNotVerified):
            actual_collider_min_world_x(parts,[[0,0,0],[1,0,0]],[[1,0,0,0]])
        r = actual_collider_min_world_x(parts,[[0,0,0]],[[1+1e-6,0,0,0]])
        self.assertAlmostEqual(r["min_world_x_m"],-.1)
        self.assertGreater(r["max_input_quaternion_norm_error"],0)

    def test_missing_parts_indices_and_unknown_geometry_fail(self):
        with self.assertRaises(GeometryNotVerified):
            actual_collider_min_world_x([],[[0,0,0]],[[1,0,0,0]])
        valid = dict(body_index=0,geometry=prepare_sphere(.1))
        with self.assertRaises(GeometryNotVerified):
            actual_collider_min_world_x([valid],[[0,0,0],[0,0,0]],[[1,0,0,0]]*2,expected_body_indices=[0,1])
        for invalid in (dict(body_index=2,geometry=prepare_sphere(.1)),dict(body_index=0,geometry={"type":"mesh"}),
                        dict(body_index=0,geometry={"type":"sphere","radius":.1,"scale":[1,2,1]})):
            with self.assertRaises(GeometryNotVerified):
                actual_collider_min_world_x([valid,invalid],[[0,0,0]],[[1,0,0,0]])

    def test_convex_requires_actual_vertices_and_frame_consistency(self):
        geometry,vertices = cube()
        for part in (dict(body_index=0,geometry=geometry),dict(body_index=0,geometry=geometry,vertices_body_m=vertices+1.)):
            with self.assertRaises(GeometryNotVerified):
                actual_collider_min_world_x([part],[[0,0,0]],[[1,0,0,0]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
