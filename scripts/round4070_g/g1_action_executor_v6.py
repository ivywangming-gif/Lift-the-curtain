"""Opt-in, fail-closed V6 action execution supervisor.

V5 remains frozen. This module only adds a versioned pre-step decision layer;
it never widens the V5 action tube or changes the label contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import hashlib
import math

import numpy as np

PROTOCOL_ID = "G1_RNG_V1_BODY_COLLISION_200HZ_ACTION_TUBE_V6"
SCHEMA = "round4070-g.action-executor.v6"
EXECUTOR_SOURCE_VERSION = "V6_CAUSAL_PREDICTIVE_STOP_BRAKE_RETRACT_1"
ADVANCE = "ADVANCE"
BRAKE = "BRAKE"
STOP = "STOP"
SAFE_RETRACT = "SAFE_RETRACT"


def _finite(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _inside(point: np.ndarray, bounds: dict[str, list[float]]) -> bool:
    return bool(bounds['y'][0] <= point[1] <= bounds['y'][1] and bounds['z'][0] <= point[2] <= bounds['z'][1])


def _margins(point: np.ndarray, bounds: dict[str, list[float]]) -> dict[str, float]:
    return {
        'y_lower': float(point[1] - bounds['y'][0]),
        'y_upper': float(bounds['y'][1] - point[1]),
        'z_lower': float(point[2] - bounds['z'][0]),
        'z_upper': float(bounds['z'][1] - point[2]),
    }


def _outward_speeds(point: np.ndarray, velocity: np.ndarray, bounds: dict[str, list[float]]) -> dict[str, float]:
    y_mid = (bounds['y'][0] + bounds['y'][1]) / 2.0
    z_mid = (bounds['z'][0] + bounds['z'][1]) / 2.0
    return {
        'y_lower': max(0.0, -float(velocity[1])) if point[1] <= y_mid else 0.0,
        'y_upper': max(0.0, float(velocity[1])) if point[1] >= y_mid else 0.0,
        'z_lower': max(0.0, -float(velocity[2])) if point[2] <= z_mid else 0.0,
        'z_upper': max(0.0, float(velocity[2])) if point[2] >= z_mid else 0.0,
    }


@dataclass(frozen=True)
class TubeContract:
    y_bounds_m: tuple[float, float]
    z_bounds_m: tuple[float, float]
    max_step_x_m: float
    sample_dt_s: float

    @classmethod
    def from_v5(cls, contract: dict[str, Any]) -> 'TubeContract':
        bounds = contract['tube_bounds_m']
        result = cls(
            (float(bounds['y'][0]), float(bounds['y'][1])),
            (float(bounds['z'][0]), float(bounds['z'][1])),
            float(contract['max_step_x_m']), float(contract['sample_dt_s']))
        if result.y_bounds_m != (-0.35, 0.35) or result.z_bounds_m != (0.45, 1.25):
            raise ValueError('V6 may only inherit the frozen V5 y/z bounds')
        if result.max_step_x_m != 0.020 or result.sample_dt_s != 0.005:
            raise ValueError('V6 may only inherit the frozen V5 x-step and clock')
        return result

    @property
    def bounds(self) -> dict[str, list[float]]:
        return {'y': list(self.y_bounds_m), 'z': list(self.z_bounds_m)}


@dataclass
class ExecutionDecision:
    mode: str
    reason: str
    physics_step: int
    current_witness_m: np.ndarray
    predicted_witness_m: np.ndarray | None = None
    command_scale: float = 1.0
    retract_requested: bool = False
    clearance_m: float | None = None
    braking_distance_m: float | None = None
    velocity_mps: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            return value
        return {
            'mode': self.mode, 'reason': self.reason, 'physics_step': int(self.physics_step),
            'current_witness_m': convert(self.current_witness_m),
            'predicted_witness_m': convert(self.predicted_witness_m),
            'command_scale': float(self.command_scale),
            'retract_requested': bool(self.retract_requested),
            'clearance_m': None if self.clearance_m is None else float(self.clearance_m),
            'braking_distance_m': None if self.braking_distance_m is None else float(self.braking_distance_m),
            'velocity_mps': convert(self.velocity_mps),
        }


class V6ActionSupervisor:
    """Causal one-step prediction, optional braking, then zero/retract.

    The braking deceleration is a profile parameter that must be certified by
    representative physics before a V6 run is admitted. Without it, the
    one-step causal prediction and hard stop remain active.
    """

    def __init__(self, contract: dict[str, Any], *, braking_deceleration_mps2: float | None = None,
                 brake_command_scale: float = 0.25) -> None:
        self.contract = contract
        self.tube = TubeContract.from_v5(contract)
        self.bounds = self.tube.bounds
        self.braking_deceleration_mps2 = None if braking_deceleration_mps2 is None else float(braking_deceleration_mps2)
        if self.braking_deceleration_mps2 is not None and (not math.isfinite(self.braking_deceleration_mps2) or self.braking_deceleration_mps2 <= 0):
            raise ValueError('braking_deceleration_mps2 must be positive and finite')
        self.brake_command_scale = float(brake_command_scale)
        if not 0 <= self.brake_command_scale < 1:
            raise ValueError('brake_command_scale must be in [0, 1)')
        self.active = False
        self.entry_step: int | None = None
        self.previous_step: int | None = None
        self.previous_witness: np.ndarray | None = None
        self.older_witness: np.ndarray | None = None
        self.latched_reason: str | None = None
        self.stop_step: int | None = None
        self.stop_witness: np.ndarray | None = None
        self.events: list[dict[str, Any]] = []

    def begin(self, physics_step: int, witness_world_m: Iterable[float]) -> None:
        point = _finite(witness_world_m, (3,), 'entry witness')
        if not _inside(point, self.bounds):
            raise ValueError('V6 entry must be inside the unchanged V5 action tube')
        self.active = True
        self.entry_step = int(physics_step)
        self.previous_step = int(physics_step)
        self.previous_witness = point.copy()
        self.older_witness = None
        self.events.append({'event': 'V6_ENTRY_ACCEPTED', 'physics_step': int(physics_step), 'witness_world_m': point.tolist()})

    def _emit(self, mode: str, reason: str, step: int, current: np.ndarray, predicted: np.ndarray | None = None,
              *, scale: float = 1.0, retract: bool = False, velocity: np.ndarray | None = None,
              braking_distance: float | None = None) -> ExecutionDecision:
        result = ExecutionDecision(mode, reason, int(step), current.copy(), None if predicted is None else predicted.copy(),
                                   float(scale), bool(retract), float(min(_margins(current, self.bounds).values())),
                                   None if braking_distance is None else float(braking_distance),
                                   None if velocity is None else velocity.copy())
        self.events.append({'event': 'V6_DECISION', **result.to_dict()})
        return result

    def _latch(self, step: int, current: np.ndarray, reason: str, *, predicted: np.ndarray | None = None,
               velocity: np.ndarray | None = None) -> ExecutionDecision:
        if self.latched_reason is None:
            self.latched_reason = str(reason)
            self.stop_step = int(step)
            self.stop_witness = current.copy()
        return self._emit(STOP, self.latched_reason, step, current, predicted, scale=0.0, retract=True, velocity=velocity)

    def before_step(self, physics_step: int, current_witness_world_m: Iterable[float], *,
                    predicted_witness_world_m: Iterable[float] | None = None) -> ExecutionDecision:
        current = _finite(current_witness_world_m, (3,), 'current witness')
        step = int(physics_step)
        if not self.active:
            return self._emit(ADVANCE, 'BEFORE_ENTRY', step, current)
        if self.latched_reason is not None:
            return self._emit(SAFE_RETRACT, self.latched_reason, step, current, scale=0.0, retract=True)
        if self.previous_step is None or self.previous_witness is None:
            return self._latch(step, current, 'V6_INTERNAL_HISTORY_UNAVAILABLE')
        if step != self.previous_step + 1:
            return self._latch(step, current, 'V6_PHYSICS_STEP_GAP')
        if not _inside(current, self.bounds):
            return self._latch(step, current, 'V6_CURRENT_STATE_OUTSIDE_V5_TUBE')
        if predicted_witness_world_m is None:
            velocity = np.zeros(3, dtype=np.float64) if self.older_witness is None else (self.previous_witness - self.older_witness) / self.tube.sample_dt_s
            predicted = current + velocity * self.tube.sample_dt_s
        else:
            raw_prediction = np.asarray(predicted_witness_world_m, dtype=np.float64)
            if raw_prediction.shape != (3,) or not np.isfinite(raw_prediction).all():
                return self._latch(step, current, 'V6_PREDICTED_STATE_NONFINITE')
            velocity = np.zeros(3, dtype=np.float64) if self.older_witness is None else (self.previous_witness - self.older_witness) / self.tube.sample_dt_s
            predicted = raw_prediction
        if not np.isfinite(predicted).all():
            return self._latch(step, current, 'V6_PREDICTED_STATE_NONFINITE', predicted=predicted, velocity=velocity)
        if abs(float(predicted[0] - current[0])) > self.tube.max_step_x_m:
            return self._latch(step, current, 'V6_PREDICTED_X_STEP_EXCEEDS_V5_CONTRACT', predicted=predicted, velocity=velocity)
        if not _inside(predicted, self.bounds):
            return self._latch(step, current, 'V6_PREDICTED_NEXT_STATE_OUTSIDE_V5_TUBE', predicted=predicted, velocity=velocity)
        braking_distance = None
        if self.braking_deceleration_mps2 is not None:
            speeds = _outward_speeds(current, velocity, self.bounds)
            braking_distance = max(speed * speed / (2 * self.braking_deceleration_mps2) for speed in speeds.values())
            if braking_distance > min(_margins(current, self.bounds).values()):
                return self._emit(BRAKE, 'V6_BRAKING_DISTANCE_REACHES_UNCHANGED_TUBE_BOUNDARY', step, current, predicted,
                                  scale=self.brake_command_scale, velocity=velocity, braking_distance=braking_distance)
        return self._emit(ADVANCE, 'V6_CAUSAL_PREDICTION_WITHIN_UNCHANGED_TUBE', step, current, predicted,
                          velocity=velocity, braking_distance=braking_distance)

    def after_step(self, physics_step: int, witness_world_m: Iterable[float]) -> ExecutionDecision:
        point = _finite(witness_world_m, (3,), 'post-step witness')
        step = int(physics_step)
        if not self.active:
            return self._emit(ADVANCE, 'AFTER_ENTRY_NOT_STARTED', step, point)
        if self.latched_reason is not None:
            return self._emit(SAFE_RETRACT, self.latched_reason, step, point, scale=0.0, retract=True)
        if self.previous_step is None or self.previous_witness is None or step != self.previous_step + 1:
            return self._latch(step, point, 'V6_POSTSTEP_PHYSICS_STEP_GAP')
        if not _inside(point, self.bounds):
            return self._latch(step, point, 'V6_POSTSTEP_STATE_OUTSIDE_V5_TUBE')
        if abs(float(point[0] - self.previous_witness[0])) > self.tube.max_step_x_m:
            return self._latch(step, point, 'V6_POSTSTEP_X_STEP_EXCEEDS_V5_CONTRACT')
        self.older_witness = self.previous_witness.copy()
        self.previous_step = step
        self.previous_witness = point.copy()
        return self._emit(ADVANCE, 'V6_POSTSTEP_ACCEPTED', step, point)

    def safe_retract(self, physics_step: int) -> ExecutionDecision:
        if self.latched_reason is None:
            raise RuntimeError('safe_retract requires a latched V6 stop')
        current = self.previous_witness if self.previous_witness is not None else np.full(3, np.nan)
        return self._emit(SAFE_RETRACT, self.latched_reason, int(physics_step), current, scale=0.0, retract=True)

    def request_stop(self, physics_step: int, current_witness_world_m: Iterable[float], reason: str) -> ExecutionDecision:
        """Latch a caller-provided fail-closed stop without changing the tube."""
        current = _finite(current_witness_world_m, (3,), 'stop witness')
        if not self.active:
            raise RuntimeError('request_stop requires an active V6 supervisor')
        return self._latch(int(physics_step), current, str(reason))

    @staticmethod
    def apply_command(command: Iterable[float], decision: ExecutionDecision) -> np.ndarray:
        value = np.asarray(command, dtype=np.float64)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError('command must be a finite vector')
        if decision.mode in {STOP, SAFE_RETRACT}:
            return np.zeros_like(value)
        if decision.mode == BRAKE:
            return value * float(decision.command_scale)
        return value.copy()

    def evidence(self) -> dict[str, Any]:
        return {
            'schema': SCHEMA, 'protocol_id': PROTOCOL_ID, 'executor_source_version': EXECUTOR_SOURCE_VERSION,
            'status': 'STOP_LATCHED' if self.latched_reason else 'READY_OR_ACTIVE',
            'entry_step': self.entry_step, 'stop_step': self.stop_step, 'latched_reason': self.latched_reason,
            'unchanged_v5_tube': {'y_m': list(self.tube.y_bounds_m), 'z_m': list(self.tube.z_bounds_m),
                                  'max_step_x_m': self.tube.max_step_x_m, 'sample_dt_s': self.tube.sample_dt_s},
            'braking_deceleration_mps2': self.braking_deceleration_mps2,
            'brake_command_scale': self.brake_command_scale, 'events': self.events,
            'physical_resimulation': False, 'model_input': False,
        }


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


if __name__ == '__main__':
    print({'schema': SCHEMA, 'source_sha256': source_sha256()})
