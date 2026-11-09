"""Offline-only Round 4070-G BIMANUAL_V1 scene audit and asset supplement."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "research" / "round4070_comprehensive_traversability"
OUT = ROOT / "outputs" / "round4070_comprehensive_traversability"
CONFIG_V5 = OUT / "configs_v5"
ASSET_DIR = REPORT / "bimanual_v1_assets"
ASSET_SCENE_DIR = ASSET_DIR / "scene_specs"
ASSET_OBJ_DIR = ASSET_DIR / "static_obj_candidates"
VERSION_DIR = REPORT / "bimanual_v1"
AUDIT_INPUTS = {
    "current_campaign": REPORT / "CURRENT_CAMPAIGN.json",
    "progress_checkpoint": REPORT / "PROGRESS_CHECKPOINT.json",
    "scene_coverage_matrix": REPORT / "SCENE_COVERAGE_MATRIX.csv",
    "g1_launch_ledger": REPORT / "G1_LAUNCH_LEDGER.json",
    "learning_delivery_status": REPORT / "LEARNING_DELIVERY_STATUS.json",
    "frozen_if_else_baseline": REPORT / "FROZEN_IF_ELSE_BASELINE.md",
    "v6_go_no_go_review": REPORT / "V6_GO_NO_GO_REVIEW.md",
    "v6_physics_validation_status": REPORT / "V6_PHYSICS_VALIDATION_STATUS.json",
}

def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file(): return default
    return json.loads(path.read_text(encoding="utf-8-sig"))

def sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

def audit_input_sources() -> dict[str, dict[str, Any]]:
    return {name: {"path": str(path.relative_to(ROOT)), "exists": path.is_file(), "sha256": sha256_file(path)} for name, path in AUDIT_INPUTS.items()}

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

def load_scene_registry() -> dict[str, dict[str, Any]]:
    module_path = ROOT / "scripts" / "round4070_g" / "physics_scenes.py"
    spec = importlib.util.spec_from_file_location("round4070_g_scene_registry", module_path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    registry = module.scene_specs()
    if len(registry) != 48 or len({x["physical_config_id"] for x in registry.values()}) != 43:
        raise RuntimeError("Expected legacy scene registry shape 48 scenes / 43 configs")
    return registry

def load_source_configs() -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(CONFIG_V5.glob("*.json")):
        value = read_json(path)
        if isinstance(value, dict) and value.get("scene_id"): result[str(value["scene_id"])] = value
    return result

def load_legacy_status() -> dict[str, dict[str, Any]]:
    value = read_json(OUT / "campaign_progress.json", {}) or {}
    return {str(x.get("scene_id")): x for x in value.get("scene_states", [])}

def load_legacy_matrix() -> dict[str, dict[str, Any]]:
    path = REPORT / "SCENE_COVERAGE_MATRIX.csv"
    if not path.is_file(): return {}
    with path.open(encoding="utf-8-sig", newline="") as handle: return {x["scene_id"]: x for x in csv.DictReader(handle)}

def fixed_indices(kind: str, rows: int = 33, cols: int = 25) -> list[int]:
    top = list(range(cols)); bottom = list(range((rows - 1) * cols, rows * cols))
    right = [r * cols for r in range(rows)]; left = [r * cols + cols - 1 for r in range(rows)]
    corners = [0, cols - 1, (rows - 1) * cols, rows * cols - 1]
    options = {"TOP_EDGE": top, "LEFT_EDGE": left, "RIGHT_EDGE": right,
               "BOTTOM_LEFT_CORNER": [corners[3]], "BOTTOM_RIGHT_CORNER": [corners[2]],
               "BOTTOM_BOTH_CORNERS": [corners[2], corners[3]], "FOUR_CORNERS_ONLY": corners,
               "TOP_PLUS_BOTH_BOTTOM": top + [corners[2], corners[3]], "FULL_PERIMETER": top + bottom + left + right}
    if kind.startswith("THREE_CORNERS_MISSING_"):
        missing = kind.removeprefix("THREE_CORNERS_MISSING_")
        names = ["TOP_RIGHT", "TOP_LEFT", "BOTTOM_RIGHT", "BOTTOM_LEFT"]
        return [index for name, index in zip(names, corners) if name != missing]
    if kind not in options: raise ValueError(f"Unknown topology {kind}")
    return sorted(set(options[kind]))

def explicit_topology(kind: str, **extra: Any) -> dict[str, Any]:
    value = {"kind": "EXPLICIT_VERTEX_SET", "semantic_kind": kind, "suspension": "STATIC", "rail_limit_m": .30,
             "ring_stride": 3, "local_fixed_vertex_indices": fixed_indices(kind)}
    value.update(extra); return value

def physical_payload(scene: dict[str, Any]) -> dict[str, Any]:
    return {key: scene.get(key) for key in ("geometry", "physics", "topology", "sheets", "wall", "door", "initial_state")}

def physical_config_id(scene: dict[str, Any]) -> str:
    return sha256_value(physical_payload(scene))[:16]

def deep_update(target: dict[str, Any], updates: dict[str, Any] | None) -> None:
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(target.get(key), dict): deep_update(target[key], value)
        else: target[key] = copy.deepcopy(value)

def make_candidate(sources, scene_id, category, condition, base_ref, target, *, geometry=None, physics=None,
                   topology=None, sheets=None, wall=None, door=None, initial_state=None, material_fidelity=None,
                   real_material_validated=False, is_control=False, notes=""):
    if base_ref not in sources: raise KeyError(f"Missing source config {base_ref}")
    scene = copy.deepcopy(sources[base_ref])
    scene.update({"scene_id": scene_id, "family": "BI", "name": condition, "scope": "BIMANUAL_V1_CANDIDATE"})
    deep_update(scene["geometry"], geometry); deep_update(scene["physics"], physics)
    if topology is not None: scene["topology"] = copy.deepcopy(topology)
    if sheets is not None: scene["sheets"] = copy.deepcopy(sheets)
    scene["wall"] = copy.deepcopy(wall)
    if door is not None: scene["door"] = copy.deepcopy(door)
    else: scene.pop("door", None)
    if initial_state is not None: scene["initial_state"] = copy.deepcopy(initial_state)
    else: scene.pop("initial_state", None)
    if material_fidelity is not None: scene["material_fidelity"] = material_fidelity
    scene["real_material_validated"] = bool(real_material_validated)
    scene["physical_config_id"] = physical_config_id(scene)
    scene["bimanual_metadata"] = {"base_ref": base_ref, "category": category, "target": target, "is_control": is_control, "notes": notes}
    return scene

def default_sheets(width: float, count: int, overlap: float = 0., x_offset: float = 0.) -> list[dict[str, Any]]:
    return [{"center_y_m": round((i - (count - 1) / 2) * (width + overlap), 6), "width_m": width, "columns": 7,
             "center_x_offset_m": x_offset if i % 2 else 0.} for i in range(count)]

def build_candidates(sources):
    result = []
    def add(*args, **kwargs): result.append(make_candidate(sources, *args, **kwargs))
    add("BI_A01_BASIC_CLOTH", "A_CONTROL", "ordinary top-fixed free-bottom curtain", "A01", 50)
    add("BI_A02_RIGID_BOARD", "A_CONTROL", "fixed rigid board", "D01", 50, material_fidelity="RIGID_GEOMETRY_WOOD_MATERIAL_UNCALIBRATED")
    add("BI_A03_EMPTY_CONTROL", "A_CONTROL", "empty robot response control", "EMPTY", 30, is_control=True, material_fidelity="NO_OBSTACLE_CONTROL")
    for distance, base in [(0.02, "D02"), (0.04, "A01"), (0.06, "A01"), (0.08, "D04")]:
        tag = int(round(distance * 100))
        wall = {"bottom_z_m": 0., "center_y_m": 0., "fixed": True, "height_m": 2., "initial_gap_m": distance, "thickness_m": .04, "width_m": 1.2}
        add(f"BI_B_WALL_{tag:02d}CM", "B_HIDDEN_WALL", f"rear rigid wall gap {tag} cm", base, 50, wall=wall)
    add("BI_B_WALL_NOWALL", "B_HIDDEN_WALL", "matched curtain with no rear wall", "D05", 50, wall=None)
    topology_rows = [
        ("BI_C_LEFT_EDGE_FIXED", "left edge fixed, right edge free", "LEFT_EDGE", "B08"),
        ("BI_C_RIGHT_EDGE_FIXED", "right edge fixed, left edge free", "RIGHT_EDGE", "B09"),
        ("BI_C_LEFT_BOTTOM_CORNER", "left bottom corner fixed", "BOTTOM_LEFT_CORNER", "A01"),
        ("BI_C_RIGHT_BOTTOM_CORNER", "right bottom corner fixed", "BOTTOM_RIGHT_CORNER", "A01"),
        ("BI_C_BOTH_BOTTOM_CORNERS", "both bottom corners fixed", "BOTTOM_BOTH_CORNERS", "A01"),
        ("BI_C_THREE_MISSING_TOP_RIGHT", "three corners fixed, top-right missing", "THREE_CORNERS_MISSING_TOP_RIGHT", "A01"),
        ("BI_C_THREE_MISSING_TOP_LEFT", "three corners fixed, top-left missing", "THREE_CORNERS_MISSING_TOP_LEFT", "A01"),
        ("BI_C_THREE_MISSING_BOTTOM_RIGHT", "three corners fixed, bottom-right missing", "THREE_CORNERS_MISSING_BOTTOM_RIGHT", "A01"),
        ("BI_C_THREE_MISSING_BOTTOM_LEFT", "three corners fixed, bottom-left missing", "THREE_CORNERS_MISSING_BOTTOM_LEFT", "A01"),
        ("BI_C_FOUR_CORNERS", "only four corners fixed", "FOUR_CORNERS_ONLY", "B05"),
        ("BI_C_TOP_PLUS_BOTTOM_CORNERS", "top edge plus both bottom corners", "TOP_PLUS_BOTH_BOTTOM", "B04"),
        ("BI_C_FULL_PERIMETER", "all four edges fixed", "FULL_PERIMETER", "B10"),
    ]
    for scene_id, condition, semantic, base in topology_rows:
        add(scene_id, "C_CONSTRAINT_TOPOLOGY", condition, base, 30, topology=explicit_topology(semantic), notes="Explicit local fixed-vertex set; distinct attachment topology required.")
    parameter_rows = [
        ("BI_D_DENSITY_ORDINARY", "ordinary surface density", "A01", {"density_kg_m3": 250.}, None),
        ("BI_D_DENSITY_HIGH", "higher surface density", "C03", {"density_kg_m3": 375.}, None),
        ("BI_D_DENSITY_VERY_HIGH", "very high surface density", "C05", {"density_kg_m3": 562.5}, None),
        ("BI_D_THICKNESS_HIGH", "increased sheet thickness", "A01", {"thickness_m": .004}, None),
        ("BI_D_BEND_HIGH", "increased bending stiffness", "C08", {"bend_stiffness": .06}, None),
        ("BI_D_STRETCH_HIGH", "increased stretch stiffness", "A01", {"stretch_stiffness": 1000.}, None),
        ("BI_D_DAMPING_HIGH", "increased damping", "C07", {"damping": .10}, None),
        ("BI_D_FRICTION_HIGH", "increased static and dynamic friction", "A01", {"static_friction": .80, "dynamic_friction": .65}, None),
        ("BI_D_HEM_MASS", "weighted lower hem", "C06", {"hem_mass_kg": .20}, None),
        ("BI_D_PLEAT_GENTLE", "gentle pleats", "A07", {"pleat_amplitude_m": .015}, None),
        ("BI_D_PLEAT_LARGE", "larger pleats", "A08", {"pleat_amplitude_m": .045, "pleat_waves": 3}, None),
        ("BI_D_INITIAL_RELAXED", "initially relaxed curtain", "A01", {}, {"mode": "RELAXED", "settle_seconds": 8.}),
        ("BI_D_INITIAL_TENSIONED", "initially tensioned curtain", "A01", {}, {"mode": "TENSIONED", "preload_fraction": .10}),
    ]
    for scene_id, condition, base, physics, initial_state in parameter_rows:
        add(scene_id, "D_PHYSICAL_PARAMETER", condition, base, 30, physics=physics, initial_state=initial_state)
    material_rows = [
        ("BI_E_CLOTH_PROXY", "ordinary cloth surrogate", "A01", {}, None, "UNCALIBRATED_SURFACE_DEFORMABLE_PROXY"),
        ("BI_E_OPEN_MESH_PROXY", "open mesh surrogate", "F02", {}, None, "UNCALIBRATED_PERFORATED_SURFACE_NET_PROXY"),
        ("BI_E_DENSE_MESH_PROXY", "dense mesh surrogate", "F03", {}, None, "UNCALIBRATED_PERFORATED_SURFACE_NET_PROXY"),
        ("BI_E_PVC_INDEPENDENT_PROXY", "independent PVC strip surrogate", "F04", {}, default_sheets(.20, 6, overlap=.02), "UNCALIBRATED_SURFACE_PVC_PROXY"),
        ("BI_E_PVC_OVERLAP_PROXY", "overlapping PVC strip surrogate", "F05", {}, default_sheets(.20, 7, overlap=-.03, x_offset=.025), "UNCALIBRATED_SURFACE_PVC_PROXY"),
        ("BI_E_PVC_WIDTH_VARIANT", "PVC strip width variant surrogate", "A01", {}, default_sheets(.12, 8, overlap=.03, x_offset=.015), "UNCALIBRATED_SURFACE_PVC_PROXY"),
        ("BI_E_PVC_THICKNESS_VARIANT", "PVC strip thickness variant surrogate", "F06", {"thickness_m": .003}, None, "UNCALIBRATED_SURFACE_PVC_PROXY"),
        ("BI_E_PVC_SPACING_VARIANT", "PVC spacing and overlap variant surrogate", "A01", {}, default_sheets(.20, 6, overlap=-.04), "UNCALIBRATED_SURFACE_PVC_PROXY"),
        ("BI_E_DOUBLE_CENTER_OPENING", "double-sheet center opening", "F07", {}, None, "UNCALIBRATED_SURFACE_DEFORMABLE_PROXY"),
        ("BI_E_LEFT_OPEN", "left-side opening structure", "F08", {}, None, "UNCALIBRATED_SURFACE_DEFORMABLE_PROXY"),
        ("BI_E_RIGHT_OPEN", "right-side opening structure", "F09", {}, None, "UNCALIBRATED_SURFACE_DEFORMABLE_PROXY"),
        ("BI_E_REAL_CLOTH", "real cloth material required", "A01", {}, None, "REAL_CLOTH_ASSET_REQUIRED_NOT_PRESENT"),
        ("BI_E_REAL_MESH", "real mesh material required", "F02", {}, None, "REAL_MESH_ASSET_REQUIRED_NOT_PRESENT"),
        ("BI_E_REAL_PVC", "real PVC material required", "F05", {}, None, "REAL_PVC_ASSET_REQUIRED_NOT_PRESENT"),
    ]
    for scene_id, condition, base, physics, sheets, material in material_rows:
        add(scene_id, "E_MATERIAL_STRUCTURE", condition, base, 30, physics=physics, sheets=sheets, material_fidelity=material, notes="Proxy and real-material evidence remain separate.")
    add("BI_F_TOP_RAIL", "F_MOVABLE_STRUCTURE", "sliding top rail", "E02", 30)
    add("BI_F_INDEPENDENT_RINGS", "F_MOVABLE_STRUCTURE", "independent sliding suspension points", "E03", 30)
    add("BI_F_LEFT_OPEN_MOVABLE", "F_MOVABLE_STRUCTURE", "left side opens on independent suspension", "E04", 30)
    add("BI_F_RIGHT_OPEN_MOVABLE", "F_MOVABLE_STRUCTURE", "right side opens on independent suspension", "A01", 30, topology={"kind": "MULTI_RINGS", "suspension": "MULTI_RINGS", "ring_stride": 3, "fixed_outer_ring": "robot_right"})
    add("BI_F_CENTER_SPLIT_MOVABLE", "F_MOVABLE_STRUCTURE", "center-separated double sheet with moving supports", "E05", 30)
    for scene_id, condition, resistance, pushable in [("BI_G_DOOR_ORDINARY", "ordinary pushable hinged door", .20, True), ("BI_G_DOOR_MEDIUM", "medium-resistance pushable hinged door", .60, True), ("BI_G_DOOR_HIGH", "high but in-range pushable hinged door", 1.00, True), ("BI_G_DOOR_NONPUSHABLE", "non-pushable hinged door control", 8.00, False)]:
        add(scene_id, "G_CURTAIN_DOOR_COMPOSITE", condition, "A01", 30, door={"kind": "HINGED_PUSH_DOOR", "pushable": pushable, "width_m": .90, "height_m": 2., "thickness_m": .04, "curtain_to_door_gap_m": .08, "opening_direction": "POSITIVE_Y", "hinge_friction_torque_nm": resistance, "initial_angle_rad": 0., "nominal_open_angle_rad": math.radians(75.), "curtain_layer_label_scope": "CURTAIN_ONLY_FIRST", "composite_route_label_scope": "SECOND_STAGE_ONLY"}, notes="Curtain and composite route labels must remain separate.")
    if len(result) != 56: raise RuntimeError(f"Expected 56 supplemental rows, got {len(result)}")
    return result

def expand_sheets(scene):
    result = []
    for override in scene.get("sheets") or [{}]:
        g, p, t = copy.deepcopy(scene["geometry"]), copy.deepcopy(scene["physics"]), copy.deepcopy(scene["topology"])
        for key, value in override.items():
            if key in g: g[key] = copy.deepcopy(value)
            elif key in p: p[key] = copy.deepcopy(value)
            elif key == "topology": t["kind"] = value
            elif key in t: t[key] = copy.deepcopy(value)
        g["center_x_m"] += float(override.get("center_x_offset_m", 0.))
        result.append({"geometry": g, "physics": p, "topology": t})
    return result

def static_geometry_check(scene):
    errors = []; g = scene["geometry"]
    if float(g.get("width_m", 0)) <= 0 or float(g.get("height_m", 0)) <= 0: errors.append("NON_POSITIVE_CURTAIN_DIMENSION")
    if int(g.get("columns", 0)) < 2 or int(g.get("rows", 0)) < 2: errors.append("INVALID_GRID_RESOLUTION")
    total = int(g.get("columns", 0)) * int(g.get("rows", 0)); indices = scene.get("topology", {}).get("local_fixed_vertex_indices")
    if indices is not None and any(int(i) < 0 or int(i) >= total for i in indices): errors.append("TOPOLOGY_INDEX_OUT_OF_RANGE")
    gap = None
    if scene.get("wall") is not None:
        gap = float(scene["wall"].get("initial_gap_m", -1))
        if gap <= 0: errors.append("NON_POSITIVE_INITIAL_WALL_GAP")
    if scene.get("door") is not None:
        gap = float(scene["door"].get("curtain_to_door_gap_m", -1))
        if gap <= 0: errors.append("NON_POSITIVE_INITIAL_DOOR_GAP")
    return {"status": "PASS_STATIC_ANALYTIC" if not errors else "FAIL_STATIC_ANALYTIC", "errors": errors,
            "initial_clearance_m": gap, "sheet_count": len(expand_sheets(scene)), "topology_vertex_count": len(indices or [])}

def write_static_obj(scene, path):
    vertices, faces = [], []
    def box(center, size):
        x, y, z = center; sx, sy, sz = (v / 2 for v in size); base = len(vertices) + 1
        vertices.extend([(x-sx,y-sy,z-sz),(x+sx,y-sy,z-sz),(x+sx,y+sy,z-sz),(x-sx,y+sy,z-sz),(x-sx,y-sy,z+sz),(x+sx,y-sy,z+sz),(x+sx,y+sy,z+sz),(x-sx,y+sy,z+sz)])
        faces.extend([(base,base+1,base+2),(base,base+2,base+3),(base+4,base+6,base+5),(base+4,base+7,base+6),(base,base+4,base+5),(base,base+5,base+1),(base+1,base+5,base+6),(base+1,base+6,base+2),(base+2,base+6,base+7),(base+2,base+7,base+3),(base+4,base,base+3),(base+4,base+3,base+7)])
    for sheet in expand_sheets(scene):
        g = sheet["geometry"]; start = len(vertices) + 1; cols, rows = int(g["columns"]), int(g["rows"])
        for r in range(rows):
            z = float(g["bottom_z_m"]) + float(g["height_m"]) * r / max(rows - 1, 1)
            for c in range(cols):
                y = float(g["center_y_m"]) + float(g["width_m"]) * (c / max(cols - 1, 1) - .5)
                vertices.append((float(g["center_x_m"]), y, z))
        period = int(g.get("net_hole_period", 0) or 0)
        for r in range(rows - 1):
            for c in range(cols - 1):
                if period and r not in (0, rows-2) and c not in (0, cols-2) and r % period == 0 and c % period == 0: continue
                a = start + r * cols + c; faces.extend([(a,a+1,a+cols+1),(a,a+cols+1,a+cols)])
    if scene.get("wall") is not None:
        w = scene["wall"]; x = float(scene["geometry"]["center_x_m"]) + float(w["initial_gap_m"]) + float(w["thickness_m"]) / 2
        box((x, float(w["center_y_m"]), float(w["height_m"]) / 2), (float(w["thickness_m"]), float(w["width_m"]), float(w["height_m"])))
    if scene["geometry"].get("kind") == "rigid_board": box((float(scene["geometry"]["center_x_m"]) + .02, 0, 1.), (.04, 1.2, 2.))
    if scene.get("door") is not None:
        d = scene["door"]; x = float(scene["geometry"]["center_x_m"]) + float(d["curtain_to_door_gap_m"]) + float(d["thickness_m"]) / 2
        box((x, 0, float(d["height_m"]) / 2), (float(d["thickness_m"]), float(d["width_m"]), float(d["height_m"])))
    lines = ["# BIMANUAL_V1 static candidate; not native USD or physics evidence", f"# scene_id={scene['scene_id']} physical_config_id={scene['physical_config_id']}"]
    lines += [f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in vertices]; lines += [f"f {a} {b} {c}" for a, b, c in faces]
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def candidate_asset_status(scene):
    material = str(scene.get("material_fidelity", ""))
    if material.startswith("REAL_"): return "MISSING_EXTERNAL_MATERIAL_ASSET"
    if scene.get("door") is not None: return "STATIC_DOOR_CANDIDATE_REQUIRES_DOOR_BUILDER"
    return "STATIC_GEOMETRY_CANDIDATE_NOT_RUNTIME_BUILT"

def build_rows(registry, sources, statuses, legacy_matrix, write_assets=True):
    rows, validation, candidates_by_id = [], [], {}
    source_hashes = {"scene_registry": sha256_file(ROOT / "scripts/round4070_g/physics_scenes.py"), "scene_builder": sha256_file(ROOT / "scripts/round4070_g/physics_scene_builder.py"), "manifest": sha256_file(OUT / "campaign_manifest_v5.json"), "current_campaign": sha256_file(REPORT / "CURRENT_CAMPAIGN.json"), "coverage": sha256_file(REPORT / "SCENE_COVERAGE_MATRIX.csv")}
    source_hashes.update({f"audit_input_{name}": value["sha256"] for name, value in audit_input_sources().items()})
    def append(scene, row_id, category, scope, target, legacy, base_ref, asset_path, asset_status, build_status, notes):
        static = static_geometry_check(scene); status = statuses.get(scene["scene_id"], {}); old = legacy_matrix.get(scene["scene_id"], {})
        legacy_physics = status.get("physics_status", old.get("physics_status", "NOT_IN_CURRENT_CAMPAIGN")) if legacy else "NOT_APPLICABLE"
        scene_gate_status = "SCENE_BUILT_LEGACY_RUNTIME_CANDIDATE" if legacy else "NOT_BUILT"
        physics_gate_status = legacy_physics if legacy else "EXECUTION_BLOCKED"
        bimanual_execution_status = "NOT_APPLICABLE_SINGLE_HAND" if legacy else "EXECUTION_BLOCKED"
        if not legacy:
            overall_gate_status = "EXECUTION_BLOCKED"
        elif legacy_physics == "PHYSICS_VALID":
            overall_gate_status = "PHYSICS_VALID"
        elif legacy_physics in ("PHYSICS_GATE_FAILED", "PHYSICS_INVALID"):
            overall_gate_status = "PHYSICS_INVALID"
        else:
            overall_gate_status = "SCENE_BUILT"
        row = {"bimanual_scene_id": row_id, "legacy_scene_id": scene["scene_id"] if legacy else base_ref, "category": category, "condition": scene.get("name", ""), "required_scope": scope, "base_ref": base_ref, "physical_config_id": scene.get("physical_config_id", ""), "config_path": str((CONFIG_V5 / f"{scene['scene_id']}.json").relative_to(ROOT)) if legacy and (CONFIG_V5 / f"{scene['scene_id']}.json").is_file() else "", "asset_path": asset_path, "asset_status": asset_status, "scene_build_status": build_status, "scene_gate_status": scene_gate_status, "physics_gate_status": physics_gate_status, "bimanual_execution_status": bimanual_execution_status, "overall_gate_status": overall_gate_status, "geometry_kind": scene["geometry"].get("kind", ""), "material_fidelity": scene.get("material_fidelity", ""), "real_material_validated": bool(scene.get("real_material_validated", False)), "constraint_topology": scene.get("topology", {}).get("semantic_kind", scene.get("topology", {}).get("kind", "")), "topology_vertex_count": static["topology_vertex_count"], "hidden_wall_distance_m": "" if scene.get("wall") is None else scene["wall"].get("initial_gap_m"), "door_mode": "" if scene.get("door") is None else scene["door"].get("kind", ""), "door_pushable": "" if scene.get("door") is None else scene["door"].get("pushable"), "door_hinge_resistance_nm": "" if scene.get("door") is None else scene["door"].get("hinge_friction_torque_nm"), "initial_geometry_status": static["status"], "initial_geometry_errors": "|".join(static["errors"]), "target_valid_bimanual_episodes": target, "target_independent_parameter_combos": 15 if target >= 50 else (0 if target == 0 else 10), "legacy_physics_status": legacy_physics, "legacy_physics_valid_seed_count": status.get("physics_valid_seeds", old.get("physics_valid_seeds", "0")) if legacy else 0, "legacy_single_hand_episode_count": status.get("g1_completed", old.get("g1_completed", "0")) if legacy else 0, "bimanual_physics_status": "LEGACY_SINGLE_HAND_ONLY" if legacy else "NOT_RUN_PAUSED", "bimanual_physics_valid_seed_count": 0, "bimanual_episode_count": 0, "formal_admission_status": "REFERENCE_ONLY_LEGACY_SINGLE_HAND" if legacy else "BLOCKED_UNTIL_BIMANUAL_PHYSICS_AND_SAFETY_GATES", "exclusion_reason": "old episode is single-hand and cannot enter formal bimanual benchmark" if legacy else "PAUSE_G1 and V6 NO-GO remain active; no bimanual physics run", "source_sha256": sha256_value({"scene": scene, "source_hashes": source_hashes}), "notes": notes}
        rows.append(row); validation.append({"bimanual_scene_id": row_id, "physical_config_id": row["physical_config_id"], "scene_gate_status": row["scene_gate_status"], "physics_gate_status": row["physics_gate_status"], "bimanual_execution_status": row["bimanual_execution_status"], "overall_gate_status": row["overall_gate_status"], "required_physics_seeds": 3, "legacy_physics_status": row["legacy_physics_status"], "legacy_valid_seed_count": row["legacy_physics_valid_seed_count"], "bimanual_physics_status": row["bimanual_physics_status"], "bimanual_valid_seed_count": 0, "static_geometry_status": row["initial_geometry_status"], "contact_status": "NOT_RUN", "release_status": "NOT_RUN", "recovery_status": "NOT_RUN", "penetration_status": "NOT_RUN", "evidence_scope": "legacy single-hand gate is not bimanual evidence" if legacy else "offline candidate only; no Isaac launch", "blocking_reason": row["exclusion_reason"]})
    for scene_id in sorted(registry):
        append(copy.deepcopy(registry[scene_id]), f"LEGACY_{scene_id}", "LEGACY_REFERENCE", "AUDIT_REFERENCE", 30, True, scene_id, str((CONFIG_V5 / f"{scene_id}.json").relative_to(ROOT)), "RUNTIME_CONFIG_AND_BUILDER_SOURCE_NO_STATIC_SCENE_ASSET" if (CONFIG_V5 / f"{scene_id}.json").is_file() else "MISSING_LEGACY_CONFIG", "RUNTIME_BUILDER_CONFIG_PRESENT_NOT_BIMANUAL_BUILT", "Legacy A-F scene; preserve as single-hand reference and revalidate under BIMANUAL_V1.")
    for scene in build_candidates(sources):
        candidates_by_id[scene["scene_id"]] = scene; static_path = ASSET_SCENE_DIR / f"{scene['scene_id']}.json"; obj_path = ASSET_OBJ_DIR / f"{scene['scene_id']}.obj"; static = static_geometry_check(scene); asset_status = candidate_asset_status(scene)
        if write_assets:
            write_json(static_path, {"schema": "round4070-g.bimanual.scene.asset.v1", "scene_id": scene["scene_id"], "physical_config_id": scene["physical_config_id"], "source_base_ref": scene["bimanual_metadata"]["base_ref"], "category": scene["bimanual_metadata"]["category"], "scene_config": scene, "static_geometry_check": static, "static_obj_candidate": str(obj_path.relative_to(ROOT)), "asset_status": asset_status, "scene_build_status": "NOT_BUILT", "overall_gate_status": "EXECUTION_BLOCKED", "physics_status": "NOT_RUN_PAUSED", "formal_use": "PROHIBITED_UNTIL_NATIVE_PHYSICS_VALIDATION_AND_BIMANUAL_SAFETY_CERTIFICATION"}); write_static_obj(scene, obj_path)
        append(scene, scene["scene_id"], scene["bimanual_metadata"]["category"], "FORMAL_REQUIRED_CANDIDATE", int(scene["bimanual_metadata"]["target"]), False, scene["bimanual_metadata"]["base_ref"], str(static_path.relative_to(ROOT)), asset_status, "STATIC_CANDIDATE_ASSET_WRITTEN_NOT_RUNTIME_BUILT", scene["bimanual_metadata"].get("notes", ""))
    return rows, validation, candidates_by_id

def load_current_campaign():
    candidates = [REPORT / "CURRENT_CAMPAIGN.json", OUT / "CURRENT_CAMPAIGN.json"]
    for path in candidates:
        value = read_json(path)
        if isinstance(value, dict):
            manifest = str(value.get("manifest", ""))
            expected = sha256_file(OUT / manifest) if manifest else None
            return {"path": str(path.relative_to(ROOT)), "exists": True, "manifest": manifest, "sha256": value.get("sha256"), "manifest_sha256": expected, "manifest_matches": bool(expected and value.get("sha256") == expected)}
    return {"path": None, "exists": False, "manifest": None, "sha256": None, "manifest_sha256": None, "manifest_matches": False}

def current_counts():
    checkpoint = read_json(REPORT / "PROGRESS_CHECKPOINT.json", {}) or {}; split_path = REPORT / "DATA_SPLIT_MANIFEST.csv"
    if split_path.is_file():
        with split_path.open(encoding="utf-8-sig", newline="") as handle:
            split_rows = list(csv.DictReader(handle))
    else:
        split_rows = []
    scope = checkpoint.get("e1_fixed_baseline", {}).get("input_scope", {})
    labels = checkpoint.get("e1_fixed_baseline", {}).get("dataset_label_counts", {})
    imported = checkpoint.get("imported_core_labels", {})
    core_metrics = checkpoint.get("e1_fixed_baseline", {}).get("core_metrics", {})
    known = int(checkpoint.get("g1_label_valid", sum(int(labels.get(k, 0)) for k in ("PASSABLE_FOR_ACTION", "BLOCKED"))))
    return {"legal_history_count": int(scope.get("history_count", int(checkpoint.get("imported_history_count", 0)))), "admitted_history_count": int(scope.get("accepted_count", int(checkpoint.get("imported_history_count", 0)))), "pending_history_count": int(scope.get("pending_count", 0)), "known_label_count_including_controls": known, "core_label_count_excluding_empty_controls": int(core_metrics.get("n", sum(int(imported.get(k, 0)) for k in ("PASSABLE_FOR_ACTION", "BLOCKED")))), "core_passable_count": int(imported.get("PASSABLE_FOR_ACTION", 0)), "core_blocked_count": int(imported.get("BLOCKED", 0)), "legal_unknown_count": int(labels.get("LABEL_UNKNOWN", 0)), "unique_g1_trials_in_split_manifest": len({x.get("trial_id") for x in split_rows if x.get("trial_id")}), "unique_g1_physical_configs_in_split_manifest": len({x.get("physical_config_id") for x in split_rows if x.get("physical_config_id")}), "bimanual_episode_count": 0, "left_right_synchronized_bimanual_episode_count": 0, "current_campaign": load_current_campaign(), "input_sources": audit_input_sources()}

def protocols():
    probe = {"schema": "round4070-g.bimanual_probe.v1", "version": "BIMANUAL_PROBE_V1", "status": "OFFLINE_PROTOCOL_DRAFT_PENDING_INDEPENDENT_SAFETY_CERTIFICATION", "parent_protocols": ["V5_FROZEN", "V6_OFFLINE_REPAIR_NO_GO"], "protected_invariants": {"v5_v6_unchanged": True, "e1_fixed_unchanged": True, "pause_flags_must_remain_present": True, "max_nominal_probe_m_per_hand": .10, "y_z_action_tube_not_relaxed": True, "no_unilateral_continuation_after_one_hand_stop": True}, "clock": {"raw_physics_hz": 200, "control_hz": 50, "timestamp_source": "actual_monotonic_physics_timestamp", "shared_episode_time_axis": True, "interpolation": "forbidden_for_raw_evidence"}, "coordinate_frames": {"body_forward_axis": "initial_robot_body_forward_x", "hand_action_axis": "precontact_curtain_local_normal", "left_and_right_contact_poses": "frozen_per_scene_and_logged_in_trial_config"}, "staging_region": {"status": "TO_BE_CERTIFIED", "must_check": ["biped_stability", "torso_clearance", "bilateral_reachability", "natural_arm_extension", "retract_space", "self_collision", "environment_collision"], "legacy_y_minus_0_35_m": "retain_until_independent_action_space_audit_resolves_its_physical_meaning"}, "action": {"precontact_pose_required": True, "max_nominal_probe_displacement_m": .10, "actual_displacement_can_be_less_than_nominal": True, "valid_early_stop_displacements_m": [.02, .05, .08, .10], "record": ["commanded_probe_displacement", "actual_left_fk_displacement", "actual_right_fk_displacement", "curtain_surface_displacement", "first_contact_time_left", "first_contact_time_right", "final_hand_positions", "stop_reason"]}, "coordinated_stop_and_retract": {"trigger": "either_hand_verified_safety_stop", "stop_latch": "both_hands_stop_forward_progress_at_next_legal_control_boundary", "unilateral_forward_motion": "forbidden", "retract": "only_after_physics_validated_coordinated_retract_is_admitted", "unvalidated_retract": "hold_safe_state_and_end_trial_as_LABEL_UNKNOWN", "safety_stop_is_not_automatically_blocked": True, "required_logs": ["left_stop_time", "right_stop_time", "left_stop_position", "right_stop_position", "left_retract_trajectory", "right_retract_trajectory"]}, "episode_contract": {"one_overall_label_per_episode": True, "left_and_right_are_not_independent_samples": True, "labels": ["PASSABLE_FOR_ACTION", "BLOCKED", "LABEL_UNKNOWN"], "curtain_layer_label_separate_from_door_route_label": True, "door_second_stage_must_not_leak_into_first_stage_model_input": True}, "launch_gate": {"current_decision": "NO_GO", "required_before_any_physical_launch": ["B10_V6_action_tube_safety_resolution", "native_exact_hand_collider_certification", "representative_stop_retract_physics_only_validation", "independent_safety_approval", "budget_addendum_approval_if_needed"]}}
    schema = {"schema": "round4070-g.bimanual_proprioception.v1", "version": "BIMANUAL_PROPRIOCEPTION_SCHEMA", "status": "OFFLINE_SCHEMA_FROZEN_PENDING_EXECUTION_CERTIFICATION", "time_axis": {"shape": "[T,C]", "timestamp_field": "t_s", "sampling": "raw_200_hz_with_actual_timestamps", "one_episode_only": True, "left_right_body_alignment": "same_row_timestamp_or_explicit_mask"}, "episode_tensor": "X_t = [X_t^L, X_t^R, X_t^Body]", "left_hand": {"required_groups": ["q", "qdot", "q_cmd", "effort_proxy", "fk_body_frame", "velocity_body_frame", "actual_progress", "contact_time", "stop_state", "retract_state"]}, "right_hand": {"required_groups": ["q", "qdot", "q_cmd", "effort_proxy", "fk_body_frame", "velocity_body_frame", "actual_progress", "contact_time", "stop_state", "retract_state"]}, "body": {"required_groups": ["imu_orientation", "imu_angular_velocity", "imu_linear_acceleration", "torso_pose", "root_velocity", "probe_relevant_joint_feedback"]}, "episode_metadata": {"required": ["episode_id", "scene_id", "physical_config_id", "physics_seed", "action_protocol_version", "observation_cutoff_s", "left_contact_pose_id", "right_contact_pose_id", "stop_reason", "curtain_layer_label", "composite_route_label"], "forbidden_model_inputs": ["scene_id", "physical_config_id", "material_truth", "constraint_truth", "hidden_wall_distance", "door_truth", "future_observations", "contact_truth_from_simulator"]}, "mask_contract": {"every_group_has_validity_mask": True, "missing_value_policy": "retain_raw_placeholder_plus_reason_and_mask", "allowed_reasons": ["sensor_missing", "runtime_drop", "not_applicable", "not_yet_recorded", "safety_stop_before_sample"]}, "label_contract": {"one_overall_label_per_episode": True, "allowed_overall_labels": ["PASSABLE_FOR_ACTION", "BLOCKED", "LABEL_UNKNOWN"], "hand_side_not_a_sample_axis": True}}
    return probe, schema

def budget_for(rows):
    candidates = [r for r in rows if r["required_scope"] == "FORMAL_REQUIRED_CANDIDATE" and int(r["target_valid_bimanual_episodes"]) > 0]; row_target = sum(int(r["target_valid_bimanual_episodes"]) for r in candidates); combos = sum(int(r["target_independent_parameter_combos"]) for r in candidates); clusters = defaultdict(int)
    for row in candidates: clusters[row["physical_config_id"]] = max(clusters[row["physical_config_id"]], int(row["target_valid_bimanual_episodes"]))
    checkpoint = read_json(REPORT / "PROGRESS_CHECKPOINT.json", {}) or {}; campaign = read_json(OUT / "campaign_manifest_v5.json", {}) or {}; launched, maximum = int(checkpoint.get("g1_attempts_launched", 36)), int(campaign.get("max_g1_launches", 1000)); checks = combos * 3
    return {"formal_candidate_row_count": len(candidates), "row_cell_valid_episode_target": row_target, "exact_config_cluster_lower_bound": sum(clusters.values()), "independent_parameter_combo_target": combos, "minimum_physics_seed_checks": checks, "existing_v5_max_g1_launches": maximum, "existing_g1_attempts_launched": launched, "existing_remaining_g1_launch_budget": maximum-launched, "lower_bound_budget_deficit_against_remaining": max(0, row_target-(maximum-launched)), "nominal_physics_gate_seconds_at_24s_each": checks*24., "nominal_g1_attempts_at_80_percent_valid_yield": math.ceil(row_target/.80), "assumption_note": "Row total is a conservative no-reuse planning bound; cluster total is an optimistic lower bound."}

def write_reports(rows, validation, counts, budget, probe, schema):
    campaign = counts["current_campaign"]
    matrix_fields = ["bimanual_scene_id", "legacy_scene_id", "category", "condition", "required_scope", "base_ref", "physical_config_id", "config_path", "asset_path", "asset_status", "scene_build_status", "scene_gate_status", "physics_gate_status", "bimanual_execution_status", "overall_gate_status", "geometry_kind", "material_fidelity", "real_material_validated", "constraint_topology", "topology_vertex_count", "hidden_wall_distance_m", "door_mode", "door_pushable", "door_hinge_resistance_nm", "initial_geometry_status", "initial_geometry_errors", "target_valid_bimanual_episodes", "target_independent_parameter_combos", "legacy_physics_status", "legacy_physics_valid_seed_count", "legacy_single_hand_episode_count", "bimanual_physics_status", "bimanual_physics_valid_seed_count", "bimanual_episode_count", "formal_admission_status", "exclusion_reason", "source_sha256", "notes"]
    write_csv(REPORT / "BIMANUAL_V1_SCENE_MATRIX.csv", rows, matrix_fields)
    write_csv(REPORT / "BIMANUAL_PHYSICS_VALIDATION.csv", validation, ["bimanual_scene_id", "physical_config_id", "scene_gate_status", "physics_gate_status", "bimanual_execution_status", "overall_gate_status", "required_physics_seeds", "legacy_physics_status", "legacy_valid_seed_count", "bimanual_physics_status", "bimanual_valid_seed_count", "static_geometry_status", "contact_status", "release_status", "recovery_status", "penetration_status", "evidence_scope", "blocking_reason"])
    write_json(REPORT / "BIMANUAL_PROBE_V1.json", probe); write_json(REPORT / "BIMANUAL_PROPRIOCEPTION_SCHEMA.json", schema)
    candidates = [r for r in rows if r["required_scope"] == "FORMAL_REQUIRED_CANDIDATE"]
    write_json(REPORT / "BIMANUAL_ASSET_CATALOG_V1.json", {"schema": "round4070-g.bimanual_asset_catalog.v1", "status": "STATIC_CANDIDATES_ONLY", "scene_count": len(candidates), "asset_root": "research/round4070_comprehensive_traversability/bimanual_v1_assets", "scene_spec_paths": [r["asset_path"] for r in candidates], "static_obj_root": "research/round4070_comprehensive_traversability/bimanual_v1_assets/static_obj_candidates", "asset_status_counts": dict(Counter(r["asset_status"] for r in candidates)), "overall_gate_status_counts": dict(Counter(r["overall_gate_status"] for r in candidates)), "warning": "JSON and OBJ files are candidate geometry/configuration only, not native USD, PhysX binding, or physics evidence."})
    (REPORT / "BIMANUAL_SAMPLE_SIZE_PLAN.md").write_text(f"""# BIMANUAL_V1 样本量与预算计划

状态：STAGE_1_ONLY_NO_PHYSICAL_LAUNCH。暂停标志和 V6 NO-GO 保持不变。

- 场景矩阵：{len(rows)} 行（旧参考 48 + 新候选 {len(rows)-48}）。
- 有效双手 episode 保守下限：{budget['row_cell_valid_episode_target']}。
- 完全相同物理配置可复用的乐观 cluster 下限：{budget['exact_config_cluster_lower_bound']}；采集前不得自动假定可复用。
- 独立参数组合目标：{budget['independent_parameter_combo_target']}；每组至少 3 个 physics seed。
- 最小 physics seed 检查：{budget['minimum_physics_seed_checks']}。

现有冻结 V5 最大 G1 启动数为 {budget['existing_v5_max_g1_launches']}，已启动 {budget['existing_g1_attempts_launched']}，剩余 {budget['existing_remaining_g1_launch_budget']}。仅按有效 episode 下限，最低额度缺口为 {budget['lower_bound_budget_deficit_against_remaining']}；按 80% 有效率的名义尝试数为 {budget['nominal_g1_attempts_at_80_percent_valid_yield']}。

physics gate 名义计算时间下限为 {budget['nominal_physics_gate_seconds_at_24s_each']/3600:.2f} 小时，不含启动、重试和双手动作安全验证。这是预算估算，不是批准。

统计以完整物理配置为 cluster；左右手和身体属于同一 episode；训练、校准、测试和 confirmation 按完整物理配置隔离。seed 不是独立物理样本。

当前没有合法双手 episode，因此不得产生正式方法排名。
""", encoding="utf-8")
    old_physics = Counter(r["legacy_physics_status"] for r in rows if r["required_scope"] == "AUDIT_REFERENCE")
    physics_counts = Counter(r["bimanual_physics_status"] for r in rows); overall_counts = Counter(r["overall_gate_status"] for r in rows); asset_counts = Counter(r["asset_status"] for r in rows)
    audit = f"""# BIMANUAL_V1 数据与场景审计

状态：STAGE_1_ONLY。未启动 Isaac，未修改 manifest、V5/V6、E1_FIXED 或两个暂停标志。

## 旧状态

- 旧 A-F 命名场景：48；独立物理配置：43。
- 旧场景 physics gate：{dict(old_physics)}。
- 当前合法 G1 历史：{counts['legal_history_count']}（已接纳 {counts['admitted_history_count']} + collision pending 3）。
- 已知标签（含 EMPTY 控制）：{counts['known_label_count_including_controls']}。
- 排除 EMPTY 控制的核心标签：{counts['core_label_count_excluding_empty_controls']}，PASSABLE {counts['core_passable_count']}，BLOCKED {counts['core_blocked_count']}。
- 合法 LABEL_UNKNOWN：{counts['legal_unknown_count']}。
- 当前 split manifest 的独立物理配置：{counts['unique_g1_physical_configs_in_split_manifest']}。
- 合法双手同步 episode：0。

当前活动描述文件：{campaign['path'] or 'NOT_FOUND'}；manifest：{campaign['manifest'] or 'NOT_FOUND'}；descriptor 与冻结 manifest SHA 匹配：{campaign['manifest_matches']}。输出目录镜像不存在不影响研究目录 descriptor；审计没有伪造缺失文件。
审计输入文件已登记 {sum(1 for item in counts['input_sources'].values() if item['exists'])}/{len(counts['input_sources'])} 个，并保存路径与 SHA256 到 BIMANUAL_PROGRESS_CHECKPOINT.json。

## 新矩阵与资产

- 总行数：{len(rows)}（旧参考 48 + 新候选 {len(rows)-48}）。
- 新候选物理配置 ID 数：{len({r['physical_config_id'] for r in rows if r['required_scope']=='FORMAL_REQUIRED_CANDIDATE'})}。
- 资产状态：{dict(asset_counts)}。
- 双手 physics 状态：{dict(physics_counts)}。
- 按项目规定枚举的总门禁状态：{dict(overall_counts)}；新候选统一为 EXECUTION_BLOCKED，不能视为已建模或已通过 physics gate。

新增 JSON/OBJ 是可检查的静态候选几何和配置描述，不是 native PhysX scene asset，也不构成 contact、release、recovery 或 penetration 证据。

## 关键缺口

1. 4 cm、6 cm 隐藏墙和底角/三角固定等新条件没有旧物理证据。
2. PVC 和网状场景仍是 articulated/surface proxy；不能宣称真实材料验证。
3. 门场景需要独立 hinged-door builder、铰链动力学和 curtain/composite 双标签记录，当前不存在。
4. 旧 gate 是单手证据，不能转化为双手同步 gate。
5. B10/V6 native collider、STOP/retract physics-only 验证和独立安全审批仍未通过。
6. 预算下限超过当前冻结 V5 剩余额度，需要正式预算 addendum 和批准。

正式 benchmark 只有在所有必需场景完成 3-seed physics gate、双手 STOP/retract 验证、B10/V6 安全审批和预算审批后才可启动。
"""
    (REPORT / "BIMANUAL_DATASET_AUDIT.md").write_text(audit, encoding="utf-8")
    checkpoint = {"schema": "round4070-g.bimanual_progress.v1", "round": "4070-G", "stage": "STAGE_1_SCENE_AUDIT_AND_ASSET_SUPPLEMENT", "status": "IN_PROGRESS_NOT_COMPLETE", "safety": {"global_pause_present": True, "g1_pause_present": True, "v6_decision": "NO_GO_FOR_FORMAL_G1_COLLECTION", "flags_modified": False, "manifest_reset": False, "e1_fixed_modified": False}, "input_sources": counts["input_sources"], "legacy_audit": counts, "matrix": {"row_count": len(rows), "legacy_reference_rows": 48, "supplemental_candidate_rows": len(rows)-48, "overall_gate_status_counts": dict(overall_counts), "physics_status_counts": dict(physics_counts), "asset_status_counts": dict(asset_counts)}, "budget": budget, "not_run": ["bimanual physics gate", "bimanual G1 collection", "E1_BI evaluation", "M1-M6 training", "formal model comparison"], "next_gates": ["resolve V6/native safety NO-GO", "approve budget addendum", "implement and validate bimanual runtime builder", "run representative physics-only bimanual stop/retract"]}
    write_json(REPORT / "BIMANUAL_PROGRESS_CHECKPOINT.json", checkpoint); VERSION_DIR.mkdir(parents=True, exist_ok=True); write_json(VERSION_DIR / "PROGRESS_CHECKPOINT.json", checkpoint)
    (VERSION_DIR / "RUN_INSTRUCTIONS.md").write_text("""# BIMANUAL_V1 运行说明

当前仅允许离线 Stage 1 审计。

    Set-Location -LiteralPath 'C:\\Robotics\\Lift-the-curtain'
    & 'C:\\conda_envs\\isaaclab\\python.exe' scripts/round4070_g/audit_bimanual_scene_matrix.py

该脚本不会启动 Isaac，不会修改暂停标志，不会重置 manifest，也不会创建 G1 attempt。正式物理命令在安全审批、预算审批和 Stage 3 双手动作认证完成前不具备准入条件。
""", encoding="utf-8")
    (VERSION_DIR / "GIT_DELIVERY_REPORT.md").write_text("""# BIMANUAL_V1 Git 交付状态

- 状态：SCOPED_ARTIFACTS_GENERATED_COMMIT_PENDING
- 允许提交：本脚本、BIMANUAL_V1 报告/协议、候选资产规格和静态 OBJ。
- 禁止提交：outputs 原始 trial、checkpoint、模型权重和大型运行日志。
- 未执行：force-push、manifest 重置、暂停标志修改、物理采集。

根目录既有 PROGRESS_CHECKPOINT.json、RUN_INSTRUCTIONS.md、GIT_DELIVERY_REPORT.md 不覆盖；版本化文件位于 bimanual_v1/，并保留 BIMANUAL_ 前缀产物。
""", encoding="utf-8")

def write_placeholder_results(budget):
    write_csv(REPORT / "E1_BI_BASELINE_RESULTS.csv", [{"method": "E1_BI_ADAPTER_V1", "status": "NOT_RUN_NO_BIMANUAL_EPISODES", "n_episode": 0, "false_pass": "", "false_reject": "", "passable_recall": "", "blocked_recall": "", "balanced_accuracy": "", "decision_coverage": "", "uncertain_rate": "", "calibration_status": "NOT_AVAILABLE"}], ["method", "status", "n_episode", "false_pass", "false_reject", "passable_recall", "blocked_recall", "balanced_accuracy", "decision_coverage", "uncertain_rate", "calibration_status"])
    methods = [("M0_E1_BI", "frozen engineering baseline"), ("M1_BIMANUAL_GRU", "supervised sequence baseline"), ("M2_TS2VEC_CNN_FINETUNE", "TS2Vec-style CNN"), ("M3_GRU_TS2VEC_STYLE", "GRU temporal contrast adaptation"), ("M4_GRU_TS2VEC_MATCHED_SUPCON", "primary candidate"), ("M5_CNN_MATCHED_SUPCON", "encoder comparison"), ("M6_GRU_ORDINARY_SUPCON", "contrastive ablation")]
    write_csv(REPORT / "BIMANUAL_MODEL_COMPARISON.csv", [{"method": m, "purpose": p, "status": "NOT_RUN_NO_BIMANUAL_EPISODES", "seed_count": 0, "formal_claim": "PROHIBITED"} for m, p in methods], ["method", "purpose", "status", "seed_count", "formal_claim"])
    write_csv(REPORT / "PER_SCENE_RESULTS.csv", [], ["bimanual_scene_id", "physical_config_id", "n_episode", "passable_n", "blocked_n", "unknown_n", "status"])
    ablations = [("A", "M1_vs_M3", "pretraining contribution"), ("B", "M3_vs_M4", "matched SupCon contribution"), ("C", "M4_vs_M3_no_supcon", "remove matched SupCon"), ("D", "M4_vs_M6", "matched vs ordinary SupCon"), ("E", "M4_order_vs_no_order", "time-order preservation"), ("F", "left_right_bimanual_body", "input ablation")]
    write_csv(REPORT / "BIMANUAL_ABLATION_RESULTS.csv", [{"ablation_id": a, "comparison": b, "purpose": c, "status": "NOT_RUN_NO_BIMANUAL_EPISODES"} for a, b, c in ablations], ["ablation_id", "comparison", "purpose", "status"])
    (REPORT / "FINAL_BIMANUAL_RESEARCH_REPORT_ZH.md").write_text("""# Round 4070-G 双手场景 benchmark 阶段报告

结论：DATA_INSUFFICIENT / STAGE_1_ONLY。已完成旧场景审计、双手候选矩阵、静态候选资产描述、协议/schema 和预算估算；尚未完成双手 physics gate、双手 G1 采集或模型训练。

当前不能回答双手融合是否优于单手、TS2Vec 是否优于 GRU、Matched SupCon 是否改善泛化，也不能宣布任何模型胜出。旧单手 pilot 与新双手协议严格分开。

B10/V6 仍为 NO-GO；两个暂停标志未修改。新增资产为静态 JSON/OBJ，不是 native PhysX 证据。
""", encoding="utf-8")
    (REPORT / "BIMANUAL_V1_BUDGET_ADDENDUM.md").write_text(f"""# BIMANUAL_V1 预算 Addendum（待审批）

状态：DRAFT_PENDING_USER_APPROVAL。此文件只记录 Stage 1 估算，不改变冻结 V5/V6 预算，也不授权启动。

- 正式候选 named-cell：{budget['formal_candidate_row_count']}
- 有效双手 episode 保守下限：{budget['row_cell_valid_episode_target']}
- 物理配置 cluster 乐观下限：{budget['exact_config_cluster_lower_bound']}
- 独立参数组合目标：{budget['independent_parameter_combo_target']}
- 最小 physics seed 检查：{budget['minimum_physics_seed_checks']}
- 当前冻结 V5 最大 G1 启动：{budget['existing_v5_max_g1_launches']}
- 已启动 G1：{budget['existing_g1_attempts_launched']}
- 当前剩余 G1 启动额度：{budget['existing_remaining_g1_launch_budget']}
- 相对有效 episode 下限的最低额度缺口：{budget['lower_bound_budget_deficit_against_remaining']}
- 80% 有效率的名义 G1 尝试数：{budget['nominal_g1_attempts_at_80_percent_valid_yield']}
- physics gate 名义计算时间：{budget['nominal_physics_gate_seconds_at_24s_each']/3600:.2f} 小时，不含启动和重试

这是保守的 no-reuse row-cell 估算；cluster 数只是乐观下限。正式批准至少需要明确新增 physics attempts、最大 G1 attempts、GPU/磁盘/墙钟预算、失败重试规则和独立安全负责人。批准前不得超出冻结 manifest 的启动上限。
""", encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--check-only", action="store_true"); args = parser.parse_args()
    registry = load_scene_registry(); sources = load_source_configs(); sources.update(registry); statuses, legacy_matrix = load_legacy_status(), load_legacy_matrix(); rows, validation, _ = build_rows(registry, sources, statuses, legacy_matrix, write_assets=not args.check_only); counts, budget = current_counts(), budget_for(rows); probe, schema = protocols()
    if args.check_only:
        print(json.dumps({"rows": len(rows), "legacy": 48, "supplemental": len(rows)-48, "budget": budget}, ensure_ascii=False, indent=2)); return 0
    write_reports(rows, validation, counts, budget, probe, schema); write_placeholder_results(budget)
    print(json.dumps({"status": "STAGE_1_ONLY", "matrix_rows": len(rows), "supplemental_rows": len(rows)-48, "pause_flags_untouched": True, "isaac_started": False, "budget": budget}, ensure_ascii=False, indent=2)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
