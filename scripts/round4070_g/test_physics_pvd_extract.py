"""Native quaternion numerical contracts, independent of USD and simulation."""
import unittest
import numpy as np
from physics_pvd_extract import _native_pose_matrix, ExtractionNotVerified


class NativePoseContracts(unittest.TestCase):
    def test_near_identity_float32_native_pose_stays_near_identity(self):
        raw=np.array([1.2556296e-8,0.,1.4313464e-7,.99999988079,1.,2.,3.])
        original=raw.copy();matrix,norm=_native_pose_matrix(raw)
        self.assertLess(abs(norm-1.),1e-4)
        self.assertLess(np.max(np.abs(matrix[:3,:3]-np.eye(3))),3e-7)
        np.testing.assert_array_equal(raw,original)
        np.testing.assert_allclose(matrix[:3,3],[1,2,3],atol=0)
        np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-14)

    def test_raw_shoulder_cylinder_rotation_maps_native_x_to_body_z(self):
        raw=[0.,-.7071068286895752,0.,.7071068286895752,-.004,.006,-.053]
        matrix,norm=_native_pose_matrix(raw)
        np.testing.assert_allclose(matrix[:3,:3]@np.array([1.,0,0]),[0,0,1],atol=1e-14)
        self.assertLess(abs(norm-1.),1e-4)

    def test_invalid_native_pose_is_rejected(self):
        for raw in ([0,0,0,0,0,0,0],[0,0,0,1.01,0,0,0],[0,0,0,1,np.nan,0,0],[1,2]):
            with self.assertRaises(ExtractionNotVerified):_native_pose_matrix(raw)


if __name__ == "__main__":unittest.main(verbosity=2)
