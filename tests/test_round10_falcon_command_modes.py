from pathlib import Path

import numpy as np
import pytest

from lift_curtain.control.falcon_adapter import (
    FalconAdapter,
    FalconAdapterConfig,
    RobotState,
    VelocityCommand,
)
from lift_curtain.control.falcon_contract import (
    OBSERVATION_DIMS,
    OBSERVATION_ORDER,
    OBSERVATION_SCALES,
    SINGLE_FRAME_DIM,
)


def _state() -> RobotState:
    return RobotState(
        base_pos_w=np.asarray([0.0, 0.0, 0.8], dtype=np.float32),
        base_quat_w=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        projected_gravity_b=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        joint_pos=np.zeros(29, dtype=np.float32),
        joint_vel=np.zeros(29, dtype=np.float32),
        joint_effort=np.zeros(29, dtype=np.float32),
    )


def _adapter() -> FalconAdapter:
    return FalconAdapter(
        FalconAdapterConfig(onnx_path=Path("unused.onnx")),
        policy=lambda observation: np.zeros((1, 29), dtype=np.float32),
    )


def _last_field(adapter: FalconAdapter, name: str) -> np.ndarray:
    frame = np.asarray(adapter.last_observation, dtype=np.float32).reshape(-1)[-SINGLE_FRAME_DIM:]
    offset = 0
    for field_name in OBSERVATION_ORDER:
        size = OBSERVATION_DIMS[field_name]
        if field_name == name:
            return frame[offset : offset + size] / float(OBSERVATION_SCALES[field_name])
        offset += size
    raise AssertionError(name)


def test_default_command_mode_remains_backward_compatible() -> None:
    adapter = _adapter()
    adapter.compute_target(_state(), VelocityCommand())
    assert _last_field(adapter, "command_stand").tolist() == [0.0]

    adapter.compute_target(_state(), VelocityCommand(vx=0.08))
    assert _last_field(adapter, "command_stand").tolist() == [1.0]
    assert _last_field(adapter, "command_lin_vel").tolist() == pytest.approx([0.08, 0.0])


def test_zero_velocity_walk_is_distinct_from_stance() -> None:
    adapter = _adapter()
    adapter.compute_target(_state(), VelocityCommand(walk_mode=True))
    assert _last_field(adapter, "command_stand").tolist() == [1.0]
    assert _last_field(adapter, "command_lin_vel").tolist() == [0.0, 0.0]

    adapter.compute_target(_state(), VelocityCommand(walk_mode=False))
    assert _last_field(adapter, "command_stand").tolist() == [0.0]


def test_explicit_stance_rejects_nonzero_velocity() -> None:
    with pytest.raises(ValueError, match="stand mode"):
        VelocityCommand(vx=0.1, walk_mode=False)
