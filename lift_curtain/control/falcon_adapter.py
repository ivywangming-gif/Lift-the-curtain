"""Small Isaac Lab adapter for the frozen FALCON G1 policy contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .falcon_contract import (
    ACTION_SCALE, DEFAULT_JOINT_POS, ISAACLAB_JOINT_ORDER, ISAACLAB_TO_OFFICIAL,
    JOINT_POS_LOWER, JOINT_POS_UPPER, OBSERVATION_DIMS, OFFICIAL_TO_ISAACLAB,
    ObservationHistory, OnnxReferencePolicy, build_frame, reorder,
)


@dataclass(frozen=True)
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    base_height: float = 0.75
    waist_dofs: tuple[float, float, float] = (0.0, 0.0, 0.0)
    walk_mode: bool | None = None

    def __post_init__(self) -> None:
        values = np.asarray((self.vx, self.vy, self.wz, self.base_height, *self.waist_dofs), dtype=np.float64)
        if not np.isfinite(values).all() or self.base_height <= 0.0:
            raise ValueError("velocity command fields must be finite and base_height must be positive")
        if self.walk_mode is False and self.moving:
            raise ValueError("explicit stand mode cannot carry a nonzero velocity command")

    @property
    def moving(self) -> bool:
        return bool(np.linalg.norm((self.vx, self.vy, self.wz)) > 1.0e-8)

    @property
    def policy_walk_flag(self) -> float:
        """Return the official command_stand field: 0=stance, 1=walk."""
        if self.walk_mode is None:
            return 1.0 if self.moving else 0.0
        return 1.0 if self.walk_mode else 0.0


@dataclass
class RobotState:
    base_pos_w: np.ndarray
    base_quat_w: np.ndarray
    base_lin_vel_b: np.ndarray
    base_ang_vel_b: np.ndarray
    projected_gravity_b: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_effort: np.ndarray
    effort_source: str = "unavailable"

    def validate(self) -> None:
        expected = {
            "base_pos_w": (3,), "base_quat_w": (4,), "base_lin_vel_b": (3,),
            "base_ang_vel_b": (3,), "projected_gravity_b": (3,),
            "joint_pos": (29,), "joint_vel": (29,), "joint_effort": (29,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}, got {value.shape}")


@dataclass(frozen=True)
class FalconAdapterConfig:
    onnx_path: Path
    upper_reference: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 14)
    action_scale: float = ACTION_SCALE
    base_height: float = 0.75

    def __post_init__(self) -> None:
        if len(self.upper_reference) != 14:
            raise ValueError("upper_reference must contain 14 values")
        if self.action_scale <= 0.0 or self.base_height <= 0.0:
            raise ValueError("action_scale and base_height must be positive")


class FalconAdapter:
    """Map Isaac Lab tensors to FALCON and return Isaac Lab position targets."""

    def __init__(self, config: FalconAdapterConfig,
                 policy: Callable[[np.ndarray], np.ndarray] | None = None):
        self.config = config
        upper = np.asarray(config.upper_reference, dtype=np.float32)
        if not np.isfinite(upper).all():
            raise ValueError("upper_reference contains non-finite values")
        self.upper_reference = upper
        self.policy = policy if policy is not None else OnnxReferencePolicy(config.onnx_path)
        self.reset()

    @staticmethod
    def validate_articulation(robot: Any) -> None:
        joint_names = tuple(robot.joint_names)
        if joint_names != ISAACLAB_JOINT_ORDER:
            raise RuntimeError(
                "FALCON joint-order contract failed: "
                f"expected {ISAACLAB_JOINT_ORDER}, got {joint_names}"
            )
        body_names = set(getattr(robot, "body_names", ()))
        missing_hands = {"left_rubber_hand", "right_rubber_hand"} - body_names
        if missing_hands:
            raise RuntimeError(f"rubber-hand body contract failed: missing {sorted(missing_hands)}")
        if getattr(robot, "is_fixed_base", False):
            raise RuntimeError("FALCON requires a floating-base articulation")

    def reset(self) -> None:
        self.history = ObservationHistory.zeros()
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.last_observation: np.ndarray | None = None
        self.evaluation_count = 0

    def make_state(self, robot: Any, env_index: int = 0) -> RobotState:
        """Extract one environment's public state and best available effort signal."""
        data = robot.data

        def array(name: str) -> np.ndarray:
            value = getattr(data, name)
            if value is None:
                raise AttributeError(name)
            if hasattr(value, "detach"):
                value = value[env_index].detach().cpu().numpy()
            else:
                value = np.asarray(value)[env_index]
            return np.asarray(value, dtype=np.float32).copy()

        effort_source = "unavailable"
        effort_value = None
        for candidate in ("applied_torque", "computed_torque", "joint_effort_target"):
            value = getattr(data, candidate, None)
            if value is not None:
                effort_value = array(candidate)
                effort_source = candidate
                break
        if effort_value is None:
            effort_value = np.zeros(29, dtype=np.float32)
        state = RobotState(
            base_pos_w=array("root_pos_w"), base_quat_w=array("root_quat_w"),
            base_lin_vel_b=array("root_lin_vel_b"), base_ang_vel_b=array("root_ang_vel_b"),
            projected_gravity_b=array("projected_gravity_b"), joint_pos=array("joint_pos"),
            joint_vel=array("joint_vel"), joint_effort=effort_value, effort_source=effort_source,
        )
        state.validate()
        return state

    def _validated_upper_reference(self, upper_reference: np.ndarray | None) -> np.ndarray:
        upper = self.upper_reference if upper_reference is None else np.asarray(upper_reference, dtype=np.float32)
        if upper.shape != (14,) or not np.isfinite(upper).all():
            raise ValueError("upper_reference must be finite with shape (14,)")
        return upper

    def _build_observation(
        self,
        state: RobotState,
        command: VelocityCommand,
        upper_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        upper = self._validated_upper_reference(upper_reference)
        q_official = reorder(state.joint_pos, ISAACLAB_TO_OFFICIAL)
        dq_official = reorder(state.joint_vel, ISAACLAB_TO_OFFICIAL)
        fields = {
            "actions": self.previous_action,
            "base_ang_vel": state.base_ang_vel_b,
            "command_ang_vel": np.asarray([command.wz], dtype=np.float32),
            "command_base_height": np.asarray([command.base_height], dtype=np.float32),
            "command_lin_vel": np.asarray([command.vx, command.vy], dtype=np.float32),
            "command_stand": np.asarray([command.policy_walk_flag], dtype=np.float32),
            "command_waist_dofs": np.asarray(command.waist_dofs, dtype=np.float32),
            "dof_pos": q_official - DEFAULT_JOINT_POS,
            "dof_vel": dq_official,
            "projected_gravity": state.projected_gravity_b,
            "ref_upper_dof_pos": upper,
        }
        if set(fields) != set(OBSERVATION_DIMS):
            raise AssertionError("FALCON observation field contract changed")
        return build_frame(fields)

    def compute_target(
        self,
        state: RobotState,
        command: VelocityCommand,
        upper_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the policy and return a 29-vector in Isaac Lab joint order."""
        upper = self._validated_upper_reference(upper_reference)
        frame = self._build_observation(state, command, upper)
        observation = self.history.push(frame)
        action = np.asarray(self.policy(observation), dtype=np.float32)
        if action.shape == (1, 29):
            action = action[0]
        if action.shape != (29,) or not np.isfinite(action).all():
            raise RuntimeError(f"invalid FALCON action shape/value: {action.shape}")
        action = np.clip(action, -100.0, 100.0)
        action[15:] = 0.0
        target_official = np.clip(
            DEFAULT_JOINT_POS + self.config.action_scale * action,
            JOINT_POS_LOWER, JOINT_POS_UPPER,
        )
        target_official[15:] = np.clip(
            upper, JOINT_POS_LOWER[15:], JOINT_POS_UPPER[15:]
        )
        self.previous_action = action.copy()
        self.last_observation = observation.copy()
        self.evaluation_count += 1
        return reorder(target_official, OFFICIAL_TO_ISAACLAB)
