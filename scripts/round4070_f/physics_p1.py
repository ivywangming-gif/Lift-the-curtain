"""Run six explicitly constructed physics-only candidates using frozen E gates.

Only sliding top's permitted rail direction is exempt from top-position drift;
its transverse attachment drift retains the original 10 mm limit.
"""
from pathlib import Path
import sys
import re
import time
import json

# A boundary pause allows offline training between independent P1 scenes.
# It happens before importing/starting Isaac, retains the guard's overall
# watchdog, and never interrupts an active physics scene.
_physics_root = Path(__file__).resolve().parents[2] / 'outputs/round4070_learning_traversability/physics'
_pause = _physics_root / 'TRAINING_PAUSE.flag'
_scene = sys.argv[sys.argv.index('--scene')+1] if '--scene' in sys.argv else ''
if _scene != 'SLIDING_TOP_TRACK' and _pause.exists():
    import os
    (_physics_root/'training_pause_ready.json').write_text(json.dumps(dict(pid=os.getpid(),scene=_scene,isaac_started=False,scope='between independent P1 scenes')),encoding='utf-8')
    _deadline = time.monotonic()+360
    while _pause.exists():
        if time.monotonic()>_deadline:raise TimeoutError('Offline-training boundary pause exceeded 360 s')
        time.sleep(1)

_base = Path(__file__).resolve().parents[1] / 'round4070_fixed_probe_benchmark' / 'validate_p1_flexible_scene.py'
sys.path.insert(0, str(Path(__file__).resolve().parent))
_source = _base.read_text(encoding='utf-8')
_source = _source.replace('from scene_specs_p1 import', 'from physics_p1_specs import')
_source = _source.replace('from structural_curtain_asset import', 'from physics_asset import')
_source = _source.replace('AppLauncher.add_app_launcher_args(parser)', 'parser.add_argument("--rail-mobility", action="store_true")\nAppLauncher.add_app_launcher_args(parser)')
_start = _source.index('        if ARGS.scene == "PVC_INDEPENDENT_STRIPS":')
_end = _source.index('        np.random.seed(ARGS.seed)', _start)
_source = _source[:_start] + '''        report["model_contract"] = spec.mechanism
        report["material_contract"] = "UNCALIBRATED_SURFACE_MODEL_NOT_REAL_PVC_OR_NET"
        report["real_pvc_claim"] = False
        report["real_net_claim"] = False
        report["p1_only"] = True
        report["cannot_invalidate_p0"] = True
        report["p1_eligible_for_g1"] = False
        report["additional_validity_required"] = ["probe-sheet penetration and release contact audit", "material calibration"]
        if ARGS.scene == "SLIDING_TOP_TRACK":
            report["additional_validity_required"] += ["runtime rail bounds and attachment relative to moving support"]
        if ARGS.scene == "NET_MESH":
            report["additional_validity_required"] += ["non-manifold vertex-connected patch mesh validation"]
''' + _source[_end:]
_old = 'current[top_indices[name]] - initial_positions[name][top_indices[name]]'
_source = _source.replace(_old, '(' + _old + ') * np.asarray([1, 0 if ARGS.scene == "SLIDING_TOP_TRACK" else 1, 1])')
_source = _source.replace('        probe = None\n', '''        from physics_p1_audit import rotation_wxyz, audit_snapshots
        from pxr import UsdGeom
        actual_faces = {name: np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(info.prim_path)).GetFaceVertexIndicesAttr().Get(),dtype=int).reshape(-1,3) for name,info in infos.items()}
        (run_root / "mesh_faces_evaluation_only.json").write_text(json.dumps({name:face.tolist() for name,face in actual_faces.items()}),encoding="utf-8")
        rail_bodies = {name: RigidObject(RigidObjectCfg(prim_path=info.support_path, spawn=None)) for name, info in infos.items()} if ARGS.scene == "SLIDING_TOP_TRACK" else {}
        rail_rows = []
        probe = None
''')
_source = _source.replace('        LOGGER.mark("AFTER_RESET",', '''        for body in rail_bodies.values():
            body.reset(); body.update(DT)
        rail_initial = {name: (as_numpy(body.data.root_pos_w)[0].copy(), as_numpy(body.data.root_quat_w)[0].copy()) for name, body in rail_bodies.items()}
        LOGGER.mark("AFTER_RESET",''')
for _indent in (8, 16):
    _key = ' '*_indent + 'sim.step(render=False)\n'
    _source = re.sub(r'(?m)^'+re.escape(_key), lambda match: _key + ' '*_indent + 'for body in rail_bodies.values(): body.update(DT)\n', _source)
_source = _source.replace('probe_positions.append(target.copy())', 'probe_positions.append(as_numpy(probe.data.root_pos_w)[0].copy())')
_source = _source.replace('"probe_x_m": None if target is None else float(target[0])', '"probe_x_m": None if probe is None else float(as_numpy(probe.data.root_pos_w)[0, 0])')
_source = _source.replace('                    sheet_metrics[name] = metrics', '''                    if name in rail_bodies:
                        body = rail_bodies[name]
                        pos = as_numpy(body.data.root_pos_w)[0]
                        quat = as_numpy(body.data.root_quat_w)[0]
                        p0, q0 = rail_initial[name]
                        local_now = (current[top_indices[name]] - pos) @ rotation_wxyz(quat)
                        local_initial = (initial_positions[name][top_indices[name]] - p0) @ rotation_wxyz(q0)
                        rail_rows.append(dict(time_s=step*DT, position_w=pos.tolist(), quaternion_wxyz=quat.tolist(), dy_m=float(pos[1]-p0[1]), transverse_drift_m=float(np.linalg.norm((pos-p0)[[0,2]])), relative_anchor_drift_m=float(np.linalg.norm(local_now-local_initial,axis=1).max()), rotation_error_rad=float(2*np.arccos(np.clip(abs(np.dot(quat,q0)),0,1)))))
                    sheet_metrics[name] = metrics''')
_source = _source.replace('            if step < ARGS.steps:\n', '''            if step < ARGS.steps:
                if ARGS.rail_mobility:
                    for body in rail_bodies.values():
                        force = torch.zeros((1,1,3),device=sim.device)
                        force[0,0,1] = 0.5 if 0.5 <= step*DT < 1.0 else 0.0
                        body.set_external_force_and_torque(force,torch.zeros_like(force),is_global=True)
                        body.write_data_to_sim()
''')
_source = _source.replace('        passed = all(acceptance.values())', '''        extended_audit = audit_snapshots(ARGS.scene, sheet_configs, snapshots, velocity_snapshots, probe_positions, rail_rows, actual_faces, ARGS.sample_every * DT)
        acceptance.update(extended_audit["checks"])
        if ARGS.rail_mobility:
            acceptance["rail_mobility_demonstrated"] = bool(rail_rows and max(abs(r["dy_m"]) for r in rail_rows) >= 0.02)
            extended_audit["stimulus"] = "0.5 N world-Y support force during t=[0.5,1.0) s; physics-only mechanism diagnostic, not hand probing"
            extended_audit["checks"]["rail_mobility_demonstrated"] = acceptance["rail_mobility_demonstrated"]
        report["extended_geometry_audit"] = extended_audit
        report["additional_validity_required"] = ["material calibration", "continuous-time collision bound beyond sampled geometric checks"]
        if ARGS.scene == "NET_MESH": report["additional_validity_required"].append("non-manifold vertex-connected mesh material validation")
        passed = all(acceptance.values())
        report["p1_geometry_gate_passed"] = passed''')
# Preserve REPO lookup from frozen script and new local imports.
exec(compile(_source, str(_base), 'exec'), dict(__name__='__main__', __file__=str(_base)))
