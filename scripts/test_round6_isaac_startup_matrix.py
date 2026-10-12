"""Bounded Isaac Lab official-style empty/offscreen/camera startup diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("empty_headless", "headless_enable_cameras_no_camera", "minimum_camera"), required=True)
parser.add_argument("--run-root", type=Path, default=REPO / "reports/artifacts/round6/renderer_startup")
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
ROOT = (ARGS.run_root / ARGS.mode).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
if not ARGS.headless or ARGS.steps < 1:
    raise ValueError("headless mode and bounded positive steps are required")
if ARGS.mode != "empty_headless" and not ARGS.enable_cameras:
    raise ValueError("renderer conditions require --enable_cameras")
EXPERIENCE = Path(ARGS.experience).resolve()
if not EXPERIENCE.is_file():
    raise ValueError("an explicit existing .kit experience is required")
CONFIG = {
    "time_local": datetime.now().astimezone().isoformat(),
    "mode": ARGS.mode,
    "headless": True,
    "num_envs": 1,
    "enable_cameras": bool(ARGS.enable_cameras),
    "camera_count": 1 if ARGS.mode == "minimum_camera" else 0,
    "experience": str(EXPERIENCE),
    "official_empty_source": "<ISAACLAB_ROOT>/scripts/tutorials/00_sim/create_empty.py",
    "official_camera_source": "<ISAACLAB_ROOT>/scripts/tutorials/04_sensors/run_usd_camera.py",
    "hypothesis": "A minimal official AppLauncher + SimulationContext path isolates renderer startup from G1, cloth, mapping, and camera allocation.",
    "exact_question": "Does the app reach script body and step a bounded empty stage, before any camera is constructed?" if ARGS.mode != "minimum_camera" else "Does one 64x48 RGB-D sensor produce finite positive depth?",
    "changed_variable": "offscreen renderer enablement only" if ARGS.mode == "headless_enable_cameras_no_camera" else "one minimum-resolution sensor" if ARGS.mode == "minimum_camera" else "minimal empty scene instead of task scene",
    "acceptance": "launcher returned, simulation reset and all bounded steps completed" if ARGS.mode != "minimum_camera" else "actual RGB/depth arrays with nonzero valid depth fraction",
    "status": "LAUNCHING",
}
(ROOT / "launch.json").write_text(json.dumps(CONFIG, indent=2) + "\n", encoding="utf-8")
(ROOT / "experience_snapshot.kit").write_text(EXPERIENCE.read_text(encoding="utf-8"), encoding="utf-8")


def marker(stage: str) -> None:
    with (ROOT / "stage_markers.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time_local": datetime.now().astimezone().isoformat(), "stage": stage}) + "\n")
    print(f"ROUND6_STARTUP_STAGE={stage}", flush=True)


marker("BEFORE_APP_LAUNCHER")
LAUNCHER = AppLauncher(ARGS)
APP = LAUNCHER.app
marker("APP_LAUNCHER_RETURNED")


def main() -> int:
    import numpy as np
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationCfg, SimulationContext

    sim = None
    result = dict(CONFIG)
    try:
        sim = SimulationContext(SimulationCfg(dt=0.01, device=ARGS.device))
        marker("SIMULATION_CONTEXT_CREATED")
        camera = None
        if ARGS.mode == "minimum_camera":
            import torch
            from isaaclab.sensors.camera import Camera, CameraCfg

            ground = sim_utils.GroundPlaneCfg()
            ground.func("/World/Ground", ground)
            light = sim_utils.DistantLightCfg(intensity=3000.0)
            light.func("/World/Light", light)
            box = sim_utils.CuboidCfg(size=(0.4, 0.4, 0.4), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.6, 0.2)))
            box.func("/World/Box", box, translation=(0.0, 0.0, 0.2))
            camera = Camera(CameraCfg(
                prim_path="/World/Camera", width=64, height=48, update_period=0,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)),
            ))
            marker("CAMERA_CONSTRUCTED")
        sim.reset()
        marker("SIMULATION_RESET_COMPLETED")
        if camera is not None:
            camera.set_world_poses_from_view(torch.tensor([[2.0, 2.0, 1.5]], device=sim.device), torch.tensor([[0.0, 0.0, 0.2]], device=sim.device))
        for _ in range(ARGS.steps):
            sim.step(render=bool(ARGS.enable_cameras))
            if camera is not None:
                camera.update(dt=0.01)
        marker("BOUNDED_STEPS_COMPLETED")
        if camera is not None:
            rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
            depth = camera.data.output["distance_to_image_plane"][0].detach().cpu().numpy()
            valid = np.isfinite(depth) & (depth > 0.0) & (depth < 20.0)
            np.savez_compressed(ROOT / "camera_arrays.npz", rgb=rgb, depth=depth)
            result.update({"rgb_shape": list(rgb.shape), "depth_shape": list(depth.shape), "valid_depth_fraction": float(valid.mean())})
            if valid.mean() <= 0.0:
                raise RuntimeError("no finite positive camera depth")
        result.update({"status": "PASS", "steps_completed": ARGS.steps, "app_launcher_returned": True})
        return_code = 0
    except Exception as exc:
        traceback.print_exc()
        result.update({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})
        return_code = 2
    finally:
        (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        if sim is not None:
            sim.stop()
        APP.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
