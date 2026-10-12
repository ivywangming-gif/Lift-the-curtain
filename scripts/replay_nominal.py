"""Portable entry point for the existing nominal Isaac replay runners.

The wrapper does not import Isaac Lab until a child process is launched.  It
therefore supports ``--help`` and ``--dry-run`` on machines without Isaac.
Video requests are written as sidecar manifests, but are deliberately rejected
for the current headless runners because no renderer backend is verified here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lift_curtain.paths import (
    GENERATED_ROBOT_ASSET,
    PROPRIO_BASELINE,
    PROPRIO_THRESHOLDS,
    falcon_onnx_path,
    isaaclab_launcher_path,
)
from lift_curtain.recording import RecordingSpec, parse_resolution, write_recording_request


HISTORICAL_ARTIFACTS = {
    "curtain": REPO / "reports" / "artifacts" / "round12" / "episodes" / "curtain_primary" / "summary.json",
    "rigid": REPO / "reports" / "artifacts" / "round12" / "episodes" / "rigid_primary" / "summary.json",
    "curtain_nearfield": REPO / "reports" / "artifacts" / "round10" / "formal_matched_effective_speed_e1" / "curtain" / "summary.json",
}


def _child_script(scenario: str) -> Path:
    if scenario in ("curtain", "rigid"):
        return REPO / "scripts" / "test_round12_oracle_map_episode.py"
    return REPO / "scripts" / "test_round6_near_field_contact_diagnostic.py"


def _append(command: list[str], *values: object) -> None:
    command.extend(str(value) for value in values)


def build_child_args(args: argparse.Namespace) -> list[str]:
    common = [
        "--asset", str(args.asset),
        "--onnx", str(args.onnx),
        "--seed", str(args.seed),
        "--headless",
    ]
    if args.scenario in ("curtain", "rigid"):
        result = [
            "--scene-kind", args.scenario,
            "--run-root", str(args.output_dir),
            "--settle-duration", "0.50",
            "--freeze-duration", "0.60",
            "--segment-timeout", "7.0",
            "--position-tolerance", "0.12",
            "--cruise-speed", "0.25",
            "--vx-max", "0.30",
            "--stand-duration", "1.50",
            "--stop-dwell-duration", "0.30",
            "--reset-root-x", "0.0",
            "--reset-root-y", "0.0",
            "--reset-root-yaw", "0.0",
            "--precontact-offset", "0.18",
            "--precontact-region-speed-max", "0.08",
            "--preprobe-duration", "1.50",
            "--preprobe-hold-duration", "0.40",
            "--probe-duration", "1.50",
            "--probe-hold-duration", "0.50",
            "--retract-duration", "1.50",
            "--neutralize-duration", "1.50",
            "--traverse-exit-x", "1.30",
            "--traverse-exit-y", "0.0",
            "--candidate-clear-margin", "0.04",
            "--goal-yaw-tolerance", "0.15",
            *common,
        ]
        if args.proprio_baseline and args.proprio_thresholds:
            result.extend((
                "--proprio-baseline-csv", str(args.proprio_baseline),
                "--proprio-thresholds-json", str(args.proprio_thresholds),
            ))
        return result

    result = [
        "--condition", "curtain",
        "--scene-profile", "formal_2m",
        "--run-root", str(args.output_dir),
        "--wall-x", "0.85",
        "--reset-root-x", "0.45",
        "--reset-root-y", "0.0",
        "--base-mode", "zero",
        "--do-sweep",
        "--sweep-duration", "2.0",
        "--sweep-amplitude-rad", "0.10",
        "--do-walkthrough",
        "--walkthrough-duration", "8.0",
        "--walkthrough-distance", "0.85",
        "--walkthrough-hand-y-offset-rad", "0.10",
        "--walkthrough-upper-policy", "extended",
        "--walkthrough-vx-max", "0.30",
        "--walkthrough-feedforward-speed", "0.25",
        "--walkthrough-terminal-speed", "0.15",
        "--walkthrough-position-tolerance", "0.08",
        "--walkthrough-stop-duration", "1.50",
        "--walkthrough-stop-speed-threshold", "0.06",
        "--walkthrough-stop-dwell-duration", "0.30",
        *common,
    ]
    if args.proprio_baseline and args.proprio_thresholds:
        result.extend((
            "--gate-walkthrough-on-proprio",
            "--proprio-baseline-csv", str(args.proprio_baseline),
            "--proprio-thresholds-json", str(args.proprio_thresholds),
        ))
    return result


def _launcher_command(launcher: Path, script: Path, child_args: Sequence[str]) -> list[str]:
    command = [str(launcher), "-p", str(script), *child_args]
    if launcher.suffix.lower() == ".bat" and os.name == "nt":
        return ["cmd.exe", "/d", "/c", subprocess.list2cmdline(command)]
    if launcher.suffix.lower() == ".sh" and os.name != "nt":
        return ["bash", *command]
    return command


def _command_for_log(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("curtain", "rigid", "curtain_nearfield"), required=True)
    parser.add_argument("--headless", action="store_true", default=True, help="Required headless execution flag")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--asset", type=Path, default=GENERATED_ROBOT_ASSET)
    parser.add_argument("--onnx", type=Path, default=falcon_onnx_path())
    parser.add_argument("--isaaclab", type=Path, help="IsaacLab root or isaaclab.bat/isaaclab.sh")
    parser.add_argument("--record_video", "--record-video", dest="record_video", action="store_true")
    parser.add_argument("--camera_view", "--camera-view", dest="camera_view", default="front")
    parser.add_argument("--resolution", default="320x240")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--proprio-baseline", type=Path, default=PROPRIO_BASELINE)
    parser.add_argument("--proprio-thresholds", type=Path, default=PROPRIO_THRESHOLDS)
    parser.add_argument("--dry-run", action="store_true", help="Write the command manifest without starting Isaac")
    args = parser.parse_args()

    args.asset = args.asset.expanduser().resolve()
    args.onnx = args.onnx.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.proprio_baseline = args.proprio_baseline.expanduser().resolve() if args.proprio_baseline else None
    args.proprio_thresholds = args.proprio_thresholds.expanduser().resolve() if args.proprio_thresholds else None
    if args.scenario == "curtain_nearfield":
        if (
            args.proprio_baseline is None
            or args.proprio_thresholds is None
            or not args.proprio_baseline.is_file()
            or not args.proprio_thresholds.is_file()
        ):
            args.proprio_baseline = None
            args.proprio_thresholds = None
    width, height = parse_resolution(args.resolution)
    spec = RecordingSpec(args.scenario, args.output_dir, args.camera_view, width, height, args.fps)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    request_path = args.output_dir / "recording_request.json"
    if args.record_video:
        write_recording_request(request_path, spec)

    launcher = isaaclab_launcher_path(args.isaaclab)
    script = _child_script(args.scenario)
    child_args = build_child_args(args)
    command = [] if launcher is None else _launcher_command(launcher, script, child_args)
    manifest = {
        "schema": "lift-the-curtain.nominal-replay.v1",
        "scenario": args.scenario,
        "seed": args.seed,
        "headless": True,
        "output_dir": str(args.output_dir),
        "script": str(script),
        "asset": str(args.asset),
        "onnx": str(args.onnx),
        "historical_artifact": str(HISTORICAL_ARTIFACTS[args.scenario]),
        "historical_artifact_exists": HISTORICAL_ARTIFACTS[args.scenario].is_file(),
        "new_replay": not args.dry_run,
        "recording_request": str(request_path) if args.record_video else None,
        "recording_status": "REQUESTED_NOT_STARTED" if args.record_video else "NOT_REQUESTED",
        "command": command,
        "command_for_log": _command_for_log(command) if command else None,
        "status": "DRY_RUN" if args.dry_run else "READY_TO_START",
        "notes": [
            "Scene kind is used by the child only for construction/offline evaluation; deployment decisions use separated observations.",
            "A historical artifact is not evidence that this new replay ran.",
        ],
    }
    manifest_path = args.output_dir / "replay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    if args.record_video:
        print("RECORDING_NOT_IMPLEMENTED_ON_THIS_RUNNER: recording request saved; no renderer backend was started.", file=sys.stderr)
        manifest["status"] = "RECORDING_NOT_IMPLEMENTED_ON_THIS_RUNNER"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return 3
    if launcher is None:
        print("ISAACLAB_LAUNCHER_MISSING: set ISAACLAB_ROOT/ISAACLAB_LAUNCHER or pass --isaaclab.", file=sys.stderr)
        manifest["status"] = "ISAACLAB_LAUNCHER_MISSING"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return 2
    missing = [path for path in (args.asset, args.onnx, script) if not path.is_file()]
    if args.scenario in ("curtain", "rigid"):
        missing.extend(
            path for path in (args.proprio_baseline, args.proprio_thresholds)
            if path is None or not path.is_file()
        )
    if launcher is not None and not launcher.is_file():
        missing.append(launcher)
    if missing:
        print("MISSING_REPLAY_INPUT: " + ", ".join(str(path) for path in missing), file=sys.stderr)
        manifest["status"] = "MISSING_REPLAY_INPUT"
        manifest["missing"] = [str(path) for path in missing]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return 2

    manifest["status"] = "RUNNING"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    completed = subprocess.run(command, cwd=REPO, check=False)
    manifest["status"] = "PASS" if completed.returncode == 0 else "CHILD_FAILED"
    manifest["returncode"] = completed.returncode
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
