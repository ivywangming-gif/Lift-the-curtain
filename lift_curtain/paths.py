"""Portable paths for local assets and external simulator installations.

The repository deliberately does not vendor the FALCON checkpoint, the
FALCON URDF, MoDeSuite assets, or Isaac Lab.  Callers can configure them with
explicit environment variables or command-line paths.  The repository-local
fallbacks are intentionally non-existent placeholders so a missing external
dependency fails with an actionable message instead of silently selecting a
machine-specific path.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FALCON_UPSTREAM_COMMIT = "a967a6d8494f57777cf8d266a644ac8e45833301"
FALCON_ONNX_RELATIVE = Path("sim2real") / "models" / "falcon" / "g1_29dof.onnx"
FALCON_URDF_RELATIVE = Path("humanoidverse") / "data" / "robots" / "g1" / "g1_29dof_fakehand.urdf"
MODESUITE_CURTAIN_RELATIVE = Path("asset") / "objects" / "curtain.usd"
GENERATED_ROBOT_ASSET = REPO_ROOT / "assets" / "generated" / "g1_29dof_falcon_rubberhand.usd"
PROPRIO_BASELINE = (
    REPO_ROOT
    / "reports"
    / "artifacts"
    / "round8"
    / "formal_matched_proprio"
    / "reset_x045"
    / "none"
    / "deployment_observation.csv"
)
PROPRIO_THRESHOLDS = REPO_ROOT / "configs" / "proprioceptive_thresholds.json"


def _expanded(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.fspath(value))).expanduser()


def _placeholder(*parts: str) -> Path:
    return REPO_ROOT.joinpath("external", *parts)


def configured_path(name: str, *, default: Path | None = None) -> Path | None:
    """Return a path from an environment variable or a supplied default."""

    value = os.environ.get(name)
    if value:
        return _expanded(value)
    return default


def falcon_upstream_root() -> Path | None:
    return configured_path("FALCON_UPSTREAM_ROOT")


def falcon_onnx_path() -> Path:
    direct = configured_path("FALCON_ONNX_PATH")
    if direct is not None:
        return direct
    root = falcon_upstream_root()
    if root is not None:
        return root / FALCON_ONNX_RELATIVE
    return _placeholder("FALCON-upstream", *FALCON_ONNX_RELATIVE.parts)


def falcon_urdf_path() -> Path:
    direct = configured_path("FALCON_URDF_PATH")
    if direct is not None:
        return direct
    root = falcon_upstream_root()
    if root is not None:
        return root / FALCON_URDF_RELATIVE
    return _placeholder("FALCON-upstream", *FALCON_URDF_RELATIVE.parts)


def modesuite_root() -> Path | None:
    return configured_path("MODESUITE_ROOT")


def modesuite_curtain_path() -> Path:
    direct = configured_path("MODESUITE_CURTAIN_PATH")
    if direct is not None:
        return direct
    root = modesuite_root()
    if root is not None:
        return root / MODESUITE_CURTAIN_RELATIVE
    return _placeholder("MoDeSuite-reference", *MODESUITE_CURTAIN_RELATIVE.parts)


def isaaclab_root() -> Path | None:
    return configured_path("ISAACLAB_ROOT")


def isaaclab_launcher_path(root_or_launcher: Path | None = None) -> Path | None:
    """Resolve an IsaacLab launcher without importing Isaac packages."""

    explicit = root_or_launcher or configured_path("ISAACLAB_LAUNCHER")
    if explicit is not None:
        if explicit.is_file():
            return explicit
        if explicit.is_dir():
            root = explicit
        else:
            return explicit
    else:
        root = isaaclab_root()
        if root is None:
            return None

    candidates = (
        root / "isaaclab.bat",
        root / "isaaclab.sh",
        root / "isaaclab",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if os.name == "nt" else candidates[1]


def resolve_repo_path(value: str | os.PathLike[str] | None, default: Path | None = None) -> Path | None:
    """Resolve an optional CLI path while preserving relative paths."""

    if value is None:
        return default
    return _expanded(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def path_record(path: Path | None, *, kind: str, required: bool) -> dict[str, object]:
    """Create a JSON-safe asset status record for audit manifests."""

    resolved = None if path is None else str(path.expanduser().resolve())
    exists = bool(path is not None and path.is_file())
    return {
        "kind": kind,
        "path": resolved,
        "exists": exists,
        "required": required,
        "status": "PASS" if exists else ("MISSING_REQUIRED" if required else "MISSING_OPTIONAL"),
    }
