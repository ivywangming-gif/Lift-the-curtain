"""Pure NumPy boundary contracts; synthetic geometry, never G1 evidence."""
import itertools
import unittest
from unittest import mock
import numpy as np
from collision_geometry import convex_box_solid_relation

from collision_geometry import (GeometryNotVerified,aabb_overlap,prepare_collider,prepare_convex,
    prepare_cylinder,prepare_sphere,triangle_collider_relation,triangle_convex_relation,
    triangle_cylinder_relation,triangle_sphere_relation)


def box(half=(.1,.2,.3)):
    half = np.asarray(half,dtype=float)
    planes = np.column_stack((np.concatenate((np.eye(3),-np.eye(3))),-np.r_[half,half]))
    vertices = np.asarray(list(itertools.product(*[(-v,v) for v in half])))
    return planes,vertices


class CollisionGeometryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planes,cls.vertices = box()
        cls.convex = prepare_convex(cls.planes,cls.vertices)
        cls.cross = np.asarray([[-.3,-.3,0.],[.3,-.3,0.],[0.,.4,0.]])

    def test_convex_face_crossing_with_all_vertices_outside(self):
        self.assertTrue(np.all(np.any(self.cross @ self.planes[:,:3].T+self.planes[:,3] > 0,axis=1)))
        r = triangle_collider_relation(self.cross,self.convex)
        self.assertTrue(r["intersects_closed"])
        self.assertTrue(r["inner_penetration"])
        self.assertGreaterEqual(len(r["inner_clipped_polygon_local"]),3)
        witness = np.asarray(r["inner_witness_local"])
        self.assertGreater(float(np.min(-(self.planes[:,:3] @ witness+self.planes[:,3]))),.005)

    def test_convex_touch_shallow_exact_and_deep(self):
        for height,status in ((.3,"TOUCH_ONLY"),(.298,"INTERSECTION_WITHIN_TOLERANCE"),
                              (.295,"INTERSECTION_WITHIN_TOLERANCE"),(.294,"INNER_PENETRATION")):
            with self.subTest(height=height):
                triangle = self.cross.copy(); triangle[:,2] = height
                self.assertEqual(triangle_collider_relation(triangle,self.convex)["contact_status"],status)

    def test_convex_aabb_is_only_rejection_not_proof(self):
        triangle = np.asarray([[.111,.19,0],[.09,.211,0],[.111,.211,0]])
        self.assertTrue(aabb_overlap(triangle,self.convex["aabb"]))
        r = triangle_collider_relation(triangle,self.convex)
        self.assertFalse(r["intersects_closed"])
        self.assertFalse(r["aabb_rejected"])
        far = triangle_collider_relation(self.cross+10,self.convex)
        self.assertTrue(far["aabb_rejected"])

    def test_convex_normalization_tetrahedron_and_baked_affine(self):
        scaled = self.planes*np.asarray([2,3,4,5,6,7])[:,None]
        self.assertTrue(triangle_convex_relation(self.cross,scaled)["inner_penetration"])
        tetra = np.asarray([[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[1,1,1,-1]],dtype=float)
        self.assertTrue(triangle_convex_relation(np.asarray([[-1,-1,.2],[2,-1,.2],[-1,2,.2]]),tetra)["inner_penetration"])
        angle = .37
        rotation = np.asarray([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        linear = rotation @ np.diag([2.,.7,.5]); translation = np.asarray([.2,-.3,.1])
        transformed_n = self.planes[:,:3] @ np.linalg.inv(linear)
        transformed_planes = np.column_stack((transformed_n,self.planes[:,3]-transformed_n @ translation))
        baked = prepare_convex(transformed_planes,self.vertices @ linear.T+translation)
        self.assertTrue(triangle_collider_relation(self.cross @ linear.T+translation,baked)["inner_penetration"])

    def test_invalid_planes_and_unsafe_aabb_are_not_verified(self):
        for planes in (-self.planes,np.asarray([[1,0,0,-1],[0,1,0,-1],[0,0,1,-1],[1,1,0,-2]],dtype=float)):
            with self.assertRaises(GeometryNotVerified):
                prepare_convex(planes)
        with self.assertRaises(GeometryNotVerified):
            prepare_convex(self.planes,self.vertices+1.)
        with self.assertRaises(GeometryNotVerified):
            prepare_convex(self.planes,aabb=[[-.01,-.01,-.01],[.01,.01,.01]])
        # Incomplete supplied vertices cannot tighten the bounds derived from planes.
        cube_planes,cube_vertices = box((.1,.1,.1))
        prepared = prepare_convex(cube_planes,cube_vertices[[0,3,5,6]])
        self.assertTrue(np.all(prepared["aabb"][0] <= -.1))
        self.assertTrue(np.all(prepared["aabb"][1] >= .1))

    def test_sphere_crossing_touch_and_tolerance(self):
        self.assertTrue(np.all(np.linalg.norm(self.cross,axis=1) > .1))
        self.assertTrue(triangle_sphere_relation(self.cross,.1)["inner_penetration"])
        for height,status in ((.1,"TOUCH_ONLY"),(.098,"INTERSECTION_WITHIN_TOLERANCE"),
                              (.095,"INTERSECTION_WITHIN_TOLERANCE"),(.094,"INNER_PENETRATION")):
            triangle = self.cross.copy(); triangle[:,2] = height
            self.assertEqual(triangle_sphere_relation(triangle,.1)["contact_status"],status)

    def test_five_mm_foot_sphere_empty_erosion_preserves_contact(self):
        r = triangle_sphere_relation(self.cross,.005,penetration_tolerance=.005)
        self.assertTrue(r["intersects_closed"])
        self.assertTrue(r["inner_region_empty"])
        self.assertFalse(r["inner_penetration"])
        self.assertTrue(triangle_sphere_relation(self.cross,.005,penetration_tolerance=.001)["inner_penetration"])
        self.assertTrue(triangle_sphere_relation(self.cross,.005,penetration_tolerance=0.)["inner_penetration"])

    def test_degenerate_triangles_are_not_silently_skipped(self):
        line = np.asarray([[-.3,0,0],[.3,0,0],[.3,0,0]])
        for shape in (self.convex,prepare_sphere(.1),prepare_cylinder(.1,.2)):
            self.assertTrue(triangle_collider_relation(line,shape)["inner_penetration"])
        point = np.repeat([[.1,0,0]],3,axis=0)
        self.assertEqual(triangle_sphere_relation(point,.1)["contact_status"],"TOUCH_ONLY")

    def test_cylinder_interior_crossing_with_outside_vertices(self):
        self.assertTrue(triangle_cylinder_relation(self.cross,.1,.2)["inner_penetration"])
        cap_cross = np.asarray([[-.2,0,-.3],[.2,0,.3],[.2,.1,-.3]])
        r = triangle_cylinder_relation(cap_cross,.1,.2)
        self.assertTrue(r["inner_penetration"])
        witness = np.asarray(r["inner_witness_local"])
        self.assertLess(np.linalg.norm(witness[:2]),.095)
        self.assertLess(abs(witness[2]),.195)

    def test_cylinder_cap_and_side_tangency_and_exact_tolerance(self):
        cap = self.cross.copy(); cap[:,2] = .2
        self.assertEqual(triangle_cylinder_relation(cap,.1,.2)["contact_status"],"TOUCH_ONLY")
        side = np.asarray([[.1,-.2,-.1],[.1,.2,-.1],[.1,0,.1]])
        self.assertEqual(triangle_cylinder_relation(side,.1,.2)["contact_status"],"TOUCH_ONLY")
        side[:,0] = .095
        self.assertFalse(triangle_cylinder_relation(side,.1,.2)["inner_penetration"])
        side[:,0] = .094
        self.assertTrue(triangle_cylinder_relation(side,.1,.2)["inner_penetration"])

    def test_cylinder_aabb_overlap_does_not_substitute_for_round_side(self):
        triangle = np.asarray([[.08,.08,0],[.09,.08,0],[.08,.09,0]])
        r = triangle_cylinder_relation(triangle,.1,.2)
        self.assertFalse(r["aabb_rejected"])
        self.assertFalse(r["intersects_closed"])
        small = triangle_cylinder_relation(self.cross,.005,.2)
        self.assertTrue(small["intersects_closed"])
        self.assertTrue(small["inner_region_empty"])

    def test_arbitrary_axis_and_translated_primitive(self):
        center = np.asarray([.4,-.2,.3])
        sphere = triangle_sphere_relation(self.cross+center,.1,center=center)
        self.assertTrue(sphere["inner_penetration"])
        cylinder = prepare_cylinder(.1,.2,axis=[1,2,3],center=center)
        transformed = self.cross @ cylinder["basis"].T+center
        self.assertTrue(triangle_collider_relation(transformed,cylinder)["inner_penetration"])
        for axis in ("x","y","z"):
            cylinder = prepare_cylinder(.1,.2,axis=axis)
            self.assertTrue(triangle_collider_relation(self.cross @ cylinder["basis"].T,cylinder)["inner_penetration"])

    def test_unsupported_and_unbaked_scale_fail_closed(self):
        for geometry in (dict(type="mesh",radius=1),dict(type="sphere",radius=.1,scale=[1,2,1]),
                         dict(type="mesh",prepared_geometry=True,aabb=[[-1,-1,-1],[1,1,1]]),
                         dict(type="cylinder",radius=.1,half_height=.2,scale=[2,1,1]),
                         dict(type="convex",planes=self.planes,scale=[1,1,2])):
            with self.assertRaises(GeometryNotVerified):
                prepare_collider(geometry)
        with self.assertRaises(GeometryNotVerified):
            triangle_sphere_relation(self.cross,.1,penetration_tolerance=-.001)
        with self.assertRaises(GeometryNotVerified):
            triangle_sphere_relation(np.full((3,3),np.nan),.1)

    def test_fixed_random_samples_and_penetration_witness_certificates(self):
        rng = np.random.default_rng(4070)
        shapes = (self.convex,prepare_sphere(.1),prepare_cylinder(.1,.2))
        for _ in range(100):
            triangle = rng.uniform(-.35,.35,size=(3,3))
            weights = rng.dirichlet([1,1,1],size=30)
            samples = weights @ triangle
            for shape in shapes:
                r = triangle_collider_relation(triangle,shape)
                if shape["type"] == "convex":
                    clearance = np.min(-(samples @ self.planes[:,:3].T+self.planes[:,3]),axis=1)
                elif shape["type"] == "sphere":
                    clearance = .1-np.linalg.norm(samples,axis=1)
                else:
                    clearance = np.minimum(.1-np.linalg.norm(samples[:,:2],axis=1),.2-np.abs(samples[:,2]))
                if np.any(clearance > .00501):
                    self.assertTrue(r["inner_penetration"])
                if r["inner_penetration"]:
                    witness = np.asarray(r["inner_witness_local"])
                    bary,_,_,_ = np.linalg.lstsq(np.vstack((triangle.T,np.ones(3))),np.r_[witness,1.],rcond=None)
                    np.testing.assert_allclose(bary @ triangle,witness,atol=1e-8)
                    self.assertGreaterEqual(float(bary.min()),-1e-8)
                    self.assertGreater(r["inner_witness_clearance_m"],.005)


class SolidConvexBoxContracts(unittest.TestCase):
    def setUp(self):
        p,v=box((.1,.2,.3));self.convex=prepare_convex(p,v)

    def test_coincident_and_containment(self):
        for bounds in ([[-.1,-.2,-.3],[.1,.2,.3]],[[-.5,-.5,-.5],[.5,.5,.5]], [[-.02,-.03,-.04],[.02,.03,.04]]):
            r=convex_box_solid_relation(self.convex,bounds,np.eye(3),np.zeros(3))
            self.assertTrue(r["verified"]);self.assertTrue(r["inner_penetration"])
            self.assertGreater(r["witness_common_clearance_m"],.005)

    def test_touch_tolerance_and_separation(self):
        for low,status in ((.1,"INTERSECTION_WITHIN_TOLERANCE"),(.09,"INTERSECTION_WITHIN_TOLERANCE"),(.089,"INNER_PENETRATION"),(.11,"DISJOINT")):
            r=convex_box_solid_relation(self.convex,[[low,-.2,-.3],[.3,.2,.3]],np.eye(3),np.zeros(3))
            self.assertTrue(r["verified"]);self.assertEqual(r["status"],status)

    def test_rotated_separation_requires_lp(self):
        angle=np.pi/4;rot=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        p,v=box((.1,.1,.1));shape=prepare_convex(p,v)
        r=convex_box_solid_relation(shape,[[.13,.13,-.03],[.15,.15,.03]],rot,np.zeros(3))
        self.assertFalse(r["aabb_rejected"]);self.assertTrue(r["verified"]);self.assertTrue(r["disjoint"])

    def test_conservative_envelope_never_proves_penetration(self):
        shape={**self.convex,"exact_intersection_allowed":False}
        r=convex_box_solid_relation(shape,[[-.1,-.2,-.3],[.1,.2,.3]],np.eye(3),np.zeros(3))
        self.assertFalse(r["verified"]);self.assertFalse(r["inner_penetration"])
        self.assertEqual(r["status"],"CONSERVATIVE_ENVELOPE_OVERLAP_NOT_VERIFIED")

    def test_solver_failure_and_invalid_rotation_fail_closed(self):
        with mock.patch("scipy.optimize.linprog",side_effect=RuntimeError("synthetic numerical failure")):
            r=convex_box_solid_relation(self.convex,[[-.1,-.2,-.3],[.1,.2,.3]],np.eye(3),np.zeros(3))
            self.assertFalse(r["verified"]);self.assertEqual(r["status"],"NUMERICAL_UNVERIFIED")
        with self.assertRaises(GeometryNotVerified):
            convex_box_solid_relation(self.convex,[[-1,-1,-1],[1,1,1]],np.diag([1,1,2]),np.zeros(3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
