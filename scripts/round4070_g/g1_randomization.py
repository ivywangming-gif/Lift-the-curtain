"""Frozen, evidenced reset perturbation. Never a deployment/model feature."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from g1_recording import write_json
from g1_action_contract_v2 import PRIMARY_PROTOCOL as V5_PROTOCOL,load_plan_contract

PRIMARY_PROTOCOL='G1_RNG_V1_BODY_COLLISION_AUDIT_V1'
ROOT=Path(__file__).resolve().parents[2]
SPEC_PATH=ROOT/'outputs/round4070_comprehensive_traversability/g1_reset_randomization_v1.json'


class ResetRandomization:
    def __init__(self,plan,run_root):
        self.root=Path(run_root);self.plan=plan;self.spec=plan.get('g1_reset_randomization',{})
        if plan.get('primary_protocol') not in (PRIMARY_PROTOCOL,V5_PROTOCOL) or not plan.get('body_collision_audit_required'):
            raise ValueError('Only frozen V4/V5 primary plans may launch after instrumentation revision')
        if plan['primary_protocol']==V5_PROTOCOL:load_plan_contract(plan)
        frozen=json.loads(SPEC_PATH.read_text(encoding='utf-8-sig'))
        if self.spec.get('spec_sha256')!=hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest():
            raise ValueError('Frozen G1 randomization sidecar hash mismatch')
        if any(self.spec.get(k)!=v for k,v in frozen.items()) or self.spec.get('seed')!=plan['seed']:
            raise ValueError('Frozen G1 randomization specification differs')
        self.rng=np.random.default_rng(int(plan['seed']))
        self.robot_draws=self.rng.uniform(-.001,.001,size=2)
        expected=np.array([self.spec['frozen_robot_delta_y_m'],self.spec['frozen_robot_delta_yaw_rad']])
        if not np.array_equal(self.robot_draws,expected):raise ValueError('Frozen robot RNG draws mismatch')
        self.applied=False

    @staticmethod
    def actual_root_pose(robot):
        # Query the same actual PhysX articulation view used by IsaacLab data,
        # not the authored USD rest transform or the requested reset argument.
        value=robot.root_physx_view.get_root_transforms()
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        raw=np.asarray(value,dtype=float)[0].copy()  # PhysX gives XYZW.
        return np.r_[raw[:3],raw[6],raw[3:6]]

    def apply_nodes(self,deformable,node_count,fixed_indices,*,simulation_time_s,actual_root_pose_wxyz=None):
        if self.applied:raise RuntimeError('Reset randomization may only be injected once')
        fixed=np.unique(np.asarray(fixed_indices,dtype=np.int64))
        if np.any(fixed<0) or np.any(fixed>=node_count):raise ValueError('Invalid attachment vertex index')
        free=np.setdiff1d(np.arange(node_count,dtype=np.int64),fixed)
        velocities=np.zeros((node_count,3),dtype=np.float32)
        velocities[free]=self.rng.uniform(-.001,.001,size=(len(free),3)).astype(np.float32)
        if deformable is not None:
            deformable.set_nodal_velocities(velocities[None])
            observed=deformable.get_nodal_velocities()
            if hasattr(observed,'detach'):observed=observed.detach().cpu().numpy()
            if hasattr(observed,'numpy'):observed=observed.numpy()
            observed=np.asarray(observed,dtype=np.float32)[0]
            if observed.shape!=velocities.shape or not np.allclose(observed,velocities,atol=1e-8,rtol=0):
                raise RuntimeError('Initial node velocity injection readback failed')
        else:observed=velocities.copy()
        self.root.mkdir(parents=True,exist_ok=True)
        path=self.root/'g_reset_randomization_arrays_audit_only.npz'
        if path.exists():raise FileExistsError('Never replace actual RNG evidence')
        np.savez_compressed(path,applied_node_velocities_world=velocities,readback_node_velocities_world=observed,
                            fixed_vertex_indices=fixed,free_vertex_indices=free,robot_draws_y_yaw=self.robot_draws)
        robot=self.plan.get('robot',{})
        expected_y=float(robot.get('reset_root_y',-.004))+float(self.robot_draws[0])
        expected_yaw=float(robot.get('reset_root_yaw',-.004))+float(self.robot_draws[1])
        pose=None if actual_root_pose_wxyz is None else np.asarray(actual_root_pose_wxyz,dtype=float)
        pose_valid=False
        if pose is not None:
            if pose.shape!=(7,) or not np.isfinite(pose).all():raise RuntimeError('Invalid actual reset root pose readback')
            w,x,y,z=pose[3:];actual_yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            pose_valid=abs(pose[1]-expected_y)<1e-6 and abs(actual_yaw-expected_yaw)<1e-6
            if not pose_valid:raise RuntimeError('Actual robot reset randomization readback differs from frozen draws')
        evidence=dict(schema='G1_RNG_V1',primary_protocol=self.plan['primary_protocol'],seed=self.plan['seed'],
            spec_sha256=self.spec['spec_sha256'],generator='numpy.random.PCG64',numpy_version=np.__version__,
            actual_robot_delta_y_m=float(self.robot_draws[0]),actual_robot_delta_yaw_rad=float(self.robot_draws[1]),
            actual_reset_root_y_m=expected_y,actual_reset_root_yaw_rad=expected_yaw,
            actual_root_pose_wxyz=pose,robot_pose_readback_verified=pose_valid,
            robot_pose_source='PhysX articulation root_physx_view.get_root_transforms (XYZW converted to WXYZ)',
            node_draw_order='free vertex indices ascending, xyz, after the two robot draws; fixed vertices exactly zero',
            node_count=node_count,free_node_count=len(free),fixed_node_count=len(fixed),readback_verified=True,
            injection_stage='AFTER_SIM_RESET_BEFORE_FIRST_PHYSICS_STEP',injection_simulation_time_s=float(simulation_time_s),
            array_dtype='float32 little endian C order',applied_array_sha256=hashlib.sha256(velocities.astype('<f4').tobytes()).hexdigest(),
            arrays_file=path.name,arrays_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            model_input=False)
        write_json(self.root/'g_reset_randomization.json',evidence);self.applied=True
        return evidence
