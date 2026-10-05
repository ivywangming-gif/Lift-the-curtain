"""Minimal, simulator-independent FALCON G1 deployment contract.

Constants are transcribed from the pinned FALCON deployment used by the
read-only reference repository and the MIT-licensed upstream checkpoint.
Runtime code maps by names instead of assuming Isaac Lab articulation order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


OFFICIAL_FALCON_COMMIT = "a967a6d8494f57777cf8d266a644ac8e45833301"
EXPECTED_ONNX_SHA256 = "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0"

OFFICIAL_POLICY_JOINT_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# Isaac Lab order observed from the FALCON G1 articulation. This is separate
# from the policy/training order above and is used only at the adapter edge.
ISAACLAB_JOINT_ORDER = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint", "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

OFFICIAL_BODY_ORDER = (
    "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link",
    "left_wrist_pitch_link", "left_wrist_yaw_link", "left_rubber_hand",
    "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
    "right_wrist_pitch_link", "right_wrist_yaw_link", "right_rubber_hand",
)

LOWER_JOINTS = OFFICIAL_POLICY_JOINT_ORDER[:15]
UPPER_JOINTS = OFFICIAL_POLICY_JOINT_ORDER[15:]
DEFAULT_JOINT_POS = np.asarray(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     0.0, 0.0, 0.0] + [0.0] * 14, dtype=np.float32)
JOINT_KP = np.asarray(
    [100, 100, 100, 200, 20, 20, 100, 100, 100, 200, 20, 20, 300, 300, 300,
     90, 60, 20, 60, 4, 4, 4, 90, 60, 20, 60, 4, 4, 4], dtype=np.float32)
JOINT_KD = np.asarray(
    [2.5, 2.5, 2.5, 5, 0.2, 0.1, 2.5, 2.5, 2.5, 5, 0.2, 0.1, 5, 5, 5,
     2, 1, 0.4, 1, 0.2, 0.2, 0.2, 2, 1, 0.4, 1, 0.2, 0.2, 0.2], dtype=np.float32)
JOINT_POS_LOWER = np.asarray(
    [-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618, -2.5307, -2.9671,
     -2.7576, -0.087267, -0.87267, -0.2618, -2.618, -0.52, -0.52, -3.0892,
     -1.5882, -2.618, -1.0472, -1.972222054, -1.61443, -1.61443, -3.0892,
     -2.2515, -2.618, -1.0472, -1.972222054, -1.61443, -1.61443], dtype=np.float32)
JOINT_POS_UPPER = np.asarray(
    [2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618, 2.8798, 0.5236, 2.7576,
     2.8798, 0.5236, 0.2618, 2.618, 0.52, 0.52, 2.6704, 2.2515, 2.618,
     2.0944, 1.972222054, 1.61443, 1.61443, 2.6704, 1.5882, 2.618, 2.0944,
     1.972222054, 1.61443, 1.61443], dtype=np.float32)
JOINT_VELOCITY_LIMIT = np.asarray(
    [32, 32, 32, 20, 37, 37, 32, 32, 32, 20, 37, 37, 32, 37, 37, 37, 37, 37,
     37, 37, 22, 22, 37, 37, 37, 37, 37, 22, 22], dtype=np.float32)
JOINT_EFFORT_LIMIT = np.asarray(
    [88, 88, 88, 139, 50, 50, 88, 88, 88, 139, 50, 50, 88, 50, 50, 25, 25, 25,
     25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5], dtype=np.float32)

ACTION_SCALE = 0.25
ACTION_CLIP = 100.0
PHYSICS_DT = 0.005
DECIMATION = 4
CONTROL_DT = PHYSICS_DT * DECIMATION
HISTORY_LENGTH = 5

OBSERVATION_DIMS = {
    "actions": 29, "base_ang_vel": 3, "command_ang_vel": 1,
    "command_base_height": 1, "command_lin_vel": 2, "command_stand": 1,
    "command_waist_dofs": 3, "dof_pos": 29, "dof_vel": 29,
    "projected_gravity": 3, "ref_upper_dof_pos": 14,
}
OBSERVATION_ORDER = tuple(sorted(OBSERVATION_DIMS))
OBSERVATION_SCALES = {
    "actions": 1.0, "base_ang_vel": 0.25, "command_ang_vel": 1.0,
    "command_base_height": 2.0, "command_lin_vel": 1.0, "command_stand": 1.0,
    "command_waist_dofs": 1.0, "dof_pos": 1.0, "dof_vel": 0.05,
    "projected_gravity": 1.0, "ref_upper_dof_pos": 1.0,
}
SINGLE_FRAME_DIM = sum(OBSERVATION_DIMS.values())
POLICY_OBSERVATION_DIM = SINGLE_FRAME_DIM * HISTORY_LENGTH


def named_permutation(source: Sequence[str], target: Sequence[str]) -> tuple[int, ...]:
    if len(source) != len(set(source)) or len(target) != len(set(target)):
        raise ValueError("names must be unique")
    missing = set(target) - set(source)
    extra = set(source) - set(target)
    if missing or extra:
        raise ValueError(f"name sets differ: missing={sorted(missing)}, extra={sorted(extra)}")
    return tuple(source.index(name) for name in target)


OFFICIAL_TO_ISAACLAB = named_permutation(OFFICIAL_POLICY_JOINT_ORDER, ISAACLAB_JOINT_ORDER)
ISAACLAB_TO_OFFICIAL = named_permutation(ISAACLAB_JOINT_ORDER, OFFICIAL_POLICY_JOINT_ORDER)


def reorder(values: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[-1] != len(permutation):
        raise ValueError(f"last dimension must be {len(permutation)}, got {array.shape}")
    return array[..., tuple(permutation)]


def quat_rotate_inverse_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    v = np.asarray(vector, dtype=np.float32)
    if q.shape != (4,) or v.shape != (3,):
        raise ValueError("quat and vector must have shapes (4,) and (3,)")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1.0e-8:
        raise ValueError("quaternion must be finite and nonzero")
    q = q / norm
    qw, qvec = q[0], q[1:]
    return v * (2.0 * qw * qw - 1.0) - 2.0 * qw * np.cross(qvec, v) + 2.0 * qvec * np.dot(qvec, v)


def build_frame(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    missing = set(OBSERVATION_ORDER) - set(fields)
    extra = set(fields) - set(OBSERVATION_ORDER)
    if missing or extra:
        raise ValueError(f"observation fields differ: missing={sorted(missing)}, extra={sorted(extra)}")
    pieces = []
    for name in OBSERVATION_ORDER:
        value = np.asarray(fields[name], dtype=np.float32)
        expected = (OBSERVATION_DIMS[name],)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape {expected}, got {value.shape}")
        pieces.append(value * OBSERVATION_SCALES[name])
    return np.concatenate(pieces).astype(np.float32, copy=False)


@dataclass
class ObservationHistory:
    frames: np.ndarray

    @classmethod
    def zeros(cls) -> "ObservationHistory":
        return cls(np.zeros((HISTORY_LENGTH, SINGLE_FRAME_DIM), dtype=np.float32))

    def push(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (SINGLE_FRAME_DIM,) or not np.isfinite(frame).all():
            raise ValueError(f"frame must be finite with shape ({SINGLE_FRAME_DIM},)")
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = frame
        return self.flatten()

    def flatten(self) -> np.ndarray:
        return self.frames.reshape(1, POLICY_OBSERVATION_DIM).astype(np.float32, copy=False)


class OnnxReferencePolicy:
    """Use ONNX's reference evaluator without changing the installed environment."""

    def __init__(self, path: Path | str):
        import onnx
        from onnx.reference import ReferenceEvaluator

        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.model = onnx.load(str(self.path), load_external_data=False)
        self.input_name = self.model.graph.input[0].name
        self.output_name = self.model.graph.output[0].name
        self.session = ReferenceEvaluator(self.model)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation)
        if value.dtype != np.float32 or value.shape != (1, POLICY_OBSERVATION_DIM):
            raise ValueError(f"observation must be float32 with shape (1, {POLICY_OBSERVATION_DIM})")
        if not np.isfinite(value).all():
            raise ValueError("observation contains non-finite values")
        action = np.asarray(self.session.run([self.output_name], {self.input_name: value})[0])
        if action.shape != (1, 29) or not np.isfinite(action).all():
            raise RuntimeError(f"invalid policy output: {action.shape}")
        return np.clip(action, -ACTION_CLIP, ACTION_CLIP).astype(np.float32, copy=False)
