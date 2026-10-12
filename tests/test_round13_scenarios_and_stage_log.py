from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from lift_curtain.envs.round13_scenarios import (
    BottomAnchorMode,
    Round13ScenarioName,
    round13_scenarios,
)
from lift_curtain.experiments.stage_log import StageLogger, StageTimeoutError


def test_round13_scenario_matrix_changes_only_declared_physics_fields() -> None:
    specs = round13_scenarios()
    nominal = specs[Round13ScenarioName.NOMINAL].curtain_config()
    assert nominal is not None
    light = specs[Round13ScenarioName.HALF_DENSITY].curtain_config()
    heavy = specs[Round13ScenarioName.DOUBLE_DENSITY].curtain_config()
    assert light is not None and heavy is not None
    assert light.density_kg_m3 == pytest.approx(0.5 * nominal.density_kg_m3)
    assert heavy.density_kg_m3 == pytest.approx(2.0 * nominal.density_kg_m3)
    for field_name in (
        "width_m",
        "height_m",
        "columns",
        "rows",
        "surface_thickness_m",
        "stretch_stiffness",
        "shear_stiffness",
        "bend_stiffness",
        "damping",
        "support_z_m",
        "translation_m",
        "yaw_rad",
    ):
        assert getattr(light, field_name) == getattr(nominal, field_name)
        assert getattr(heavy, field_name) == getattr(nominal, field_name)


def test_anchor_modes_preserve_full_top_edge_and_derive_bottom_indices() -> None:
    specs = round13_scenarios()
    expected = {
        Round13ScenarioName.BOTTOM_LEFT: BottomAnchorMode.LEFT.value,
        Round13ScenarioName.BOTTOM_RIGHT: BottomAnchorMode.RIGHT.value,
        Round13ScenarioName.BOTH_BOTTOM: BottomAnchorMode.BOTH.value,
    }
    for name, mode in expected.items():
        config = specs[name].curtain_config()
        assert config is not None
        assert config.bottom_anchor_mode == mode
        assert config.top_row_indices == tuple(range(config.columns))
        assert config.bottom_left_index == (config.rows - 1) * config.columns
        assert config.bottom_right_index == config.rows * config.columns - 1


def test_hidden_rigid_is_scene_only_and_not_encoded_as_candidate_result() -> None:
    spec = round13_scenarios()[Round13ScenarioName.HIDDEN_RIGID]
    assert spec.hidden_rigid is not None
    assert spec.scene_object == "curtain"
    assert "passable" not in spec.to_dict()
    assert "H1" in spec.map_assumption and "H2" in spec.map_assumption


def stage_log_test_root() -> Path:
    return Path("reports/artifacts/round13/unit_test_stage_log") / str(uuid4())


def test_stage_logger_flushes_last_valid_stage() -> None:
    logger = StageLogger(stage_log_test_root(), phase_timeouts_s={"BEFORE_APP": 10.0})
    logger.mark("BEFORE_APP", details={"test": True})
    logger.mark("AFTER_APP")
    logger.heartbeat(progress=0.5, details={"step": 5})
    logger.finalize(status="PASS")
    rows = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]
    assert [row["stage"] for row in rows] == [
        "BEFORE_APP",
        "AFTER_APP",
        "ROLLOUT_PROGRESS",
        "FINALIZE",
    ]
    assert rows[-1]["details"]["last_valid_stage"] == "ROLLOUT_PROGRESS"


def test_stage_logger_reports_completed_phase_timeout(monkeypatch) -> None:
    ticks = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr("lift_curtain.experiments.stage_log.time.monotonic", lambda: next(ticks))
    logger = StageLogger(stage_log_test_root(), phase_timeouts_s={"BEFORE_APP": 1.0})
    logger.mark("BEFORE_APP")
    with pytest.raises(StageTimeoutError):
        logger.mark("AFTER_APP")
    row = json.loads(logger.path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["status"] == "TIMEOUT"
