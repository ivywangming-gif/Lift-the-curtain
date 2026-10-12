"""Durable stage logging for long Isaac processes.

The logger writes and fsyncs each event so a native crash still leaves the
last completed stage. It is independent from Isaac imports and can therefore
record ``BEFORE_APP`` before AppLauncher is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping


ROUND13_STAGES = (
    "BEFORE_APP",
    "AFTER_APP",
    "AFTER_SCENE",
    "AFTER_RESET",
    "FIRST_PHYSICS_STEP",
    "ROLLOUT_PROGRESS",
    "FINALIZE",
)


class StageTimeoutError(RuntimeError):
    """Raised when a completed phase exceeded its wall-time budget."""


@dataclass(frozen=True)
class StageEvent:
    stage: str
    wall_time_s: float
    stage_elapsed_s: float
    pid: int
    status: str
    details: dict[str, Any]


class StageLogger:
    """Append-only JSONL logger with explicit flush and phase budgets."""

    def __init__(
        self,
        run_root: Path | str,
        *,
        phase_timeouts_s: Mapping[str, float] | None = None,
        filename: str = "stage_heartbeat.jsonl",
    ) -> None:
        self.run_root = Path(run_root).resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.path = self.run_root / filename
        self.phase_timeouts_s = {
            str(name): float(value) for name, value in (phase_timeouts_s or {}).items()
        }
        if any(value <= 0.0 for value in self.phase_timeouts_s.values()):
            raise ValueError("phase timeouts must be positive")
        self.started_at = time.monotonic()
        self.stage_started_at = self.started_at
        self.last_stage: str | None = None

    def _append(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def mark(
        self,
        stage: str,
        *,
        status: str = "OK",
        details: Mapping[str, Any] | None = None,
        enforce_previous_timeout: bool = True,
    ) -> StageEvent:
        if stage not in ROUND13_STAGES:
            raise ValueError(f"unknown Round 13 stage: {stage}")
        now = time.monotonic()
        stage_elapsed = now - self.stage_started_at
        previous_stage = self.last_stage
        timeout = self.phase_timeouts_s.get(previous_stage or "")
        effective_status = str(status)
        if enforce_previous_timeout and timeout is not None and stage_elapsed > timeout:
            effective_status = "TIMEOUT"
        event = StageEvent(
            stage=stage,
            wall_time_s=now - self.started_at,
            stage_elapsed_s=stage_elapsed,
            pid=os.getpid(),
            status=effective_status,
            details={
                "previous_stage": previous_stage,
                "previous_stage_timeout_s": timeout,
                **dict(details or {}),
            },
        )
        self._append(
            {
                "stage": event.stage,
                "wall_time_s": event.wall_time_s,
                "stage_elapsed_s": event.stage_elapsed_s,
                "pid": event.pid,
                "status": event.status,
                "details": event.details,
            }
        )
        self.last_stage = stage
        self.stage_started_at = now
        if effective_status == "TIMEOUT":
            raise StageTimeoutError(
                f"phase {previous_stage} exceeded {timeout:.3f}s before {stage} "
                f"({stage_elapsed:.3f}s)"
            )
        return event

    def heartbeat(
        self,
        *,
        progress: float | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> StageEvent:
        payload = dict(details or {})
        if progress is not None:
            value = float(progress)
            if not 0.0 <= value <= 1.0:
                raise ValueError("progress must be within [0, 1]")
            payload["progress"] = value
        return self.mark(
            "ROLLOUT_PROGRESS",
            details=payload,
            enforce_previous_timeout=False,
        )

    def finalize(
        self,
        *,
        status: str,
        details: Mapping[str, Any] | None = None,
    ) -> StageEvent:
        return self.mark(
            "FINALIZE",
            status=status,
            details={"last_valid_stage": self.last_stage, **dict(details or {})},
            enforce_previous_timeout=False,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "pid": os.getpid(),
            "last_stage": self.last_stage,
            "wall_time_s": time.monotonic() - self.started_at,
            "phase_timeouts_s": dict(self.phase_timeouts_s),
        }
