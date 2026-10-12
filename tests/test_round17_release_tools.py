from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from lift_curtain.paths import falcon_onnx_path, falcon_urdf_path, modesuite_curtain_path
from lift_curtain.recording import (
    RecordingSpec,
    classify_deformable_visual_sync,
    parse_resolution,
    write_recording_request,
)


REPO = Path(__file__).resolve().parents[1]


def _test_root(name: str) -> Path:
    root = REPO / "reports" / "release" / f".round17-test-{os.getpid()}-{name}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_external_paths_are_environment_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path("test-falcon-root")
    monkeypatch.setenv("FALCON_UPSTREAM_ROOT", str(root))
    assert falcon_onnx_path() == root / "sim2real" / "models" / "falcon" / "g1_29dof.onnx"
    assert falcon_urdf_path() == root / "humanoidverse" / "data" / "robots" / "g1" / "g1_29dof_fakehand.urdf"

    curtain = _test_root("paths") / "curtain.usd"
    monkeypatch.setenv("MODESUITE_CURTAIN_PATH", str(curtain))
    assert modesuite_curtain_path() == curtain


def test_recording_request_is_explicitly_not_a_verified_video() -> None:
    root = _test_root("request")
    try:
        path = root / "recording_request.json"
        write_recording_request(path, RecordingSpec("curtain", root, width=320, height=240, fps=30))
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == "REQUESTED_NOT_STARTED"
        assert payload["verified_video"] is False
        assert payload["spec"]["resolution"] == "320x240"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_recording_resolution_and_visual_mismatch_contract() -> None:
    assert parse_resolution("640x480") == (640, 480)
    assert parse_resolution("640,480") == (640, 480)
    assert classify_deformable_visual_sync(0.03, 0.0) == "VISUAL_DEFORMABLE_MISMATCH"
    assert classify_deformable_visual_sync(0.001, 0.0) == "CONSISTENT_WITHIN_TOLERANCE"
    assert classify_deformable_visual_sync(None, 0.1) == "NOT_ASSESSABLE"


def test_replay_help_and_dry_run_do_not_import_isaac() -> None:
    script = REPO / "scripts" / "replay_nominal.py"
    help_run = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_run.returncode == 0
    assert "curtain_nearfield" in help_run.stdout

    output_dir = _test_root("dry-run")
    try:
        dry_run = subprocess.run(
            [
                sys.executable,
                str(script),
                "--scenario",
                "curtain",
                "--output_dir",
                str(output_dir),
                "--dry-run",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert dry_run.returncode == 0, dry_run.stderr
        manifest = json.loads((output_dir / "replay_manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "DRY_RUN"
        assert manifest["new_replay"] is False
        assert manifest["historical_artifact"]
        assert "test_round12_oracle_map_episode.py" in manifest["script"]
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_recording_request_does_not_claim_renderer_execution() -> None:
    script = REPO / "scripts" / "replay_nominal.py"
    output_dir = _test_root("recording")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--scenario",
                "curtain",
                "--output_dir",
                str(output_dir),
                "--record_video",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 3
        assert "RECORDING_NOT_IMPLEMENTED_ON_THIS_RUNNER" in result.stderr
        payload = json.loads((output_dir / "replay_manifest.json").read_text(encoding="utf-8"))
        assert payload["recording_status"] == "REQUESTED_NOT_STARTED"
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
