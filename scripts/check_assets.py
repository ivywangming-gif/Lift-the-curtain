"""Audit repository-local and externally supplied replay assets.

This script is intentionally standard-library only.  It never imports Isaac
Lab, opens USD files, copies third-party assets, or starts a simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lift_curtain.control.falcon_contract import EXPECTED_ONNX_SHA256, OFFICIAL_FALCON_COMMIT
from lift_curtain.paths import (
    GENERATED_ROBOT_ASSET,
    falcon_onnx_path,
    falcon_urdf_path,
    isaaclab_launcher_path,
    isaaclab_root,
    modesuite_curtain_path,
    modesuite_root,
    path_record,
    sha256_file,
)


def _file_record(path: Path | None, *, kind: str, required: bool, expected_sha256: str | None = None) -> dict[str, object]:
    record = path_record(path, kind=kind, required=required)
    if record["exists"] and expected_sha256 is not None and path is not None:
        actual = sha256_file(path)
        record["sha256"] = actual
        record["sha256_expected"] = expected_sha256
        record["sha256_matches"] = actual == expected_sha256
        if actual != expected_sha256:
            record["status"] = "HASH_MISMATCH"
    return record


def _directory_record(path: Path | None, *, kind: str, required: bool) -> dict[str, object]:
    resolved = None if path is None else str(path.expanduser().resolve())
    exists = bool(path is not None and path.is_dir())
    return {
        "kind": kind,
        "path": resolved,
        "exists": exists,
        "required": required,
        "status": "PASS" if exists else ("MISSING_REQUIRED" if required else "MISSING_OPTIONAL"),
    }


def build_report(args: argparse.Namespace) -> dict[str, object]:
    allow_external_missing = bool(args.allow_missing_external)
    allow_generated_missing = bool(args.allow_missing_generated)

    onnx = args.onnx or falcon_onnx_path()
    urdf = args.urdf or falcon_urdf_path()
    asset = args.asset or GENERATED_ROBOT_ASSET
    curtain = args.modesuite_root / "asset" / "objects" / "curtain.usd" if args.modesuite_root else modesuite_curtain_path()
    sim_root = args.isaaclab_root or isaaclab_root()
    launcher = isaaclab_launcher_path(sim_root)

    records = [
        _directory_record(REPO, kind="repository", required=True),
        _file_record(asset, kind="generated_robot_usd", required=not allow_generated_missing),
        _file_record(onnx, kind="falcon_onnx", required=not allow_external_missing, expected_sha256=EXPECTED_ONNX_SHA256),
        _file_record(urdf, kind="falcon_urdf", required=not allow_external_missing),
        _file_record(curtain, kind="modesuite_curtain_usd", required=not allow_external_missing),
        _directory_record(sim_root, kind="isaaclab_root", required=not allow_external_missing),
        _file_record(launcher, kind="isaaclab_launcher", required=not allow_external_missing),
    ]

    mode_root = args.modesuite_root or modesuite_root()

    def license_record(root: Path | None, names: tuple[str, ...], note: str) -> dict[str, object]:
        candidates = [] if root is None else [root / name for name in names]
        found = next((path for path in candidates if path.is_file()), None)
        return {
            "status": "FOUND" if found is not None else "NOT_FOUND_OR_NOT_DECLARED",
            "path": None if found is None else str(found.resolve()),
            "note": note,
        }

    failures = [
        record
        for record in records
        if record["required"] and record["status"] != "PASS"
    ]
    return {
        "schema": "lift-the-curtain.asset-audit.v1",
        "repository": str(REPO),
        "falcon_commit": OFFICIAL_FALCON_COMMIT,
        "expected_onnx_sha256": EXPECTED_ONNX_SHA256,
        "records": records,
        "licenses": {
            "isaaclab": license_record(
                sim_root,
                ("LICENSE", "LICENSE.txt", "NOTICE"),
                "IsaacLab is installed externally under its own terms; no binary is redistributed.",
            ),
            "modesuite": license_record(
                mode_root,
                ("LICENSE", "LICENSE.txt", "COPYING", "NOTICE"),
                "MoDeSuite licensing remains unclear when no license file is present; do not redistribute it.",
            ),
        },
        "ok": not failures,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, help="Explicit external FALCON ONNX path")
    parser.add_argument("--urdf", type=Path, help="Explicit external FALCON URDF path")
    parser.add_argument("--asset", type=Path, help="Repository-local generated robot USD")
    parser.add_argument("--modesuite-root", type=Path, help="Read-only MoDeSuite checkout")
    parser.add_argument("--isaaclab-root", type=Path, help="IsaacLab installation root")
    parser.add_argument("--json-out", type=Path, help="Write the audit report to this path")
    parser.add_argument("--allow-missing-external", action="store_true", help="Make external dependencies optional")
    parser.add_argument("--allow-missing-generated", action="store_true", help="Make the generated USD optional")
    args = parser.parse_args()
    report = build_report(args)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
