"""Convert the external MIT-licensed FALCON rubber-hand URDF to USD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lift_curtain.paths import falcon_urdf_path
from isaaclab.app import AppLauncher


DEFAULT_INPUT = falcon_urdf_path()
DEFAULT_OUTPUT = REPO / "assets" / "generated" / "g1_29dof_falcon_rubberhand.usd"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--merge-fixed-joints", action="store_true", help="Merge fixed links; off by default to preserve hands")
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
APP = AppLauncher(ARGS).app


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    input_path = ARGS.input.expanduser().resolve()
    output_path = ARGS.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import isaaclab.sim as sim_utils
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    config = UrdfConverterCfg(
        asset_path=str(input_path),
        usd_dir=str(output_path.parent),
        usd_file_name=output_path.name,
        fix_base=False,
        merge_fixed_joints=bool(ARGS.merge_fixed_joints),
        force_usd_conversion=True,
        self_collision=True,
        collider_type="convex_hull",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=1.0, damping=0.1),
            target_type="position",
        ),
    )
    converter = UrdfConverter(config)
    generated = Path(converter.usd_path).resolve()
    manifest = {
        "input_urdf": str(input_path),
        "input_sha256": sha256(input_path),
        "output_usd": str(generated),
        "output_sha256": sha256(generated) if generated.is_file() else None,
        "merge_fixed_joints": bool(ARGS.merge_fixed_joints),
        "fix_base": False,
        "source_commit": "a967a6d8494f57777cf8d266a644ac8e45833301",
        "notes": "Converted locally from the external FALCON rubber-hand URDF; not a copied upstream repository.",
    }
    manifest_path = REPO / "reports" / "artifacts" / "m0" / "falcon_asset_conversion.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
