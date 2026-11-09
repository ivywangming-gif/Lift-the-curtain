"""Generate BIMANUAL_V1 dynamic build plans and offline readiness reports.

The command is intentionally Kit-free. It updates only versioned research
metadata and generated dynamic plan files; it never launches Isaac, changes a
manifest, or removes a safety pause.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import re
import site
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "research" / "round4070_comprehensive_traversability"
OUT = ROOT / "outputs" / "round4070_comprehensive_traversability"
MATRIX = REPORT / "BIMANUAL_V1_SCENE_MATRIX.csv"
CHECKPOINT = REPORT / "BIMANUAL_PROGRESS_CHECKPOINT.json"
VERSION_CHECKPOINT = REPORT / "bimanual_v1" / "PROGRESS_CHECKPOINT.json"
ASSET_CATALOG = REPORT / "BIMANUAL_ASSET_CATALOG_V1.json"
ENV_JSON = REPORT / "BIMANUAL_PHYSICS_ENVIRONMENT_COMPATIBILITY.json"
ENV_MD = REPORT / "BIMANUAL_PHYSICS_ENVIRONMENT_COMPATIBILITY.md"
DYNAMIC_CATALOG = REPORT / "BIMANUAL_DYNAMIC_BUILD_CATALOG_V1.json"
DYNAMIC_REPORT = REPORT / "BIMANUAL_DYNAMIC_BUILD_REPORT.md"
READINESS_CSV = REPORT / "BIMANUAL_DYNAMIC_SCENE_READINESS.csv"
BUDGET_MD = REPORT / "BIMANUAL_BUDGET_AUDIT_V2.md"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimanual_dynamic_scene_builder import (  # noqa: E402
    BimanualSceneComposer,
    BUILDER_READY,
    NOT_BUILT,
    PLAN_DIR,
    REQUIRES_PHYSICS_VALIDATION,
    SCHEMA,
    build_all_offline,
)


def sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_text(value: str) -> str:
    """Normalize generated Markdown templates that use explicit newline tokens."""
    return value.replace("\\n", "\n")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_text(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _physx_extension_snapshot() -> dict[str, Any]:
    roots: list[Path] = []
    for root in [Path(sys.prefix), *[Path(item) for item in site.getsitepackages()]]:
        for candidate in (
            root / "Lib" / "site-packages" / "isaacsim" / "extscache",
            root / "lib" / "python3.11" / "site-packages" / "isaacsim" / "extscache",
        ):
            if candidate.is_dir():
                roots.append(candidate)
    fallback = Path("C:/conda_envs/isaaclab/Lib/site-packages/isaacsim/extscache")
    if fallback.is_dir():
        roots.append(fallback)
    matches = sorted({path for root in roots for pattern in ("omni.physx-*", "omni.physics.physx-*") for path in root.glob(pattern) if path.is_dir()})
    records: list[dict[str, Any]] = []
    for extension in matches[:4]:
        config = extension / "config" / "extension.toml"
        text = config.read_text(encoding="utf-8") if config.is_file() else ""
        version_match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
        description = next((line.split("=", 1)[1].strip().strip('"') for line in text.splitlines() if line.startswith("description")), None)
        records.append({
            "path": str(extension),
            "extension_toml": str(config),
            "version": version_match.group(1) if version_match else None,
            "description": description,
            "config_has_physx5_description": "PhysX 5" in text,
        })
    core = next((item for item in records if item["path"].lower().find("omni.physx-") >= 0 and "physics.physx" not in item["path"].lower()), None)
    backend = next((item for item in records if "omni.physics.physx" in item["path"].lower()), None)
    return {"extensions": records, "selected_version": core["version"] if core else None, "backend_version": backend["version"] if backend else None}


def environment_snapshot() -> dict[str, Any]:
    torch_info: dict[str, Any] = {"installed": False}
    try:
        import torch  # type: ignore

        torch_info = {
            "installed": True,
            "version": getattr(torch, "__version__", None),
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            torch_info.update({
                "device_name": str(device.name),
                "compute_capability": [int(device.major), int(device.minor)],
                "total_memory_bytes": int(device.total_memory),
                "total_memory_gib": round(float(device.total_memory) / (1024 ** 3), 3),
            })
    except Exception as exc:  # pragma: no cover - environment dependent
        torch_info = {"installed": False, "error": type(exc).__name__}
    nvidia = _run_text(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    nvidia_full = _run_text(["nvidia-smi"])
    isaacsim = _package_version("isaacsim") or _package_version("isaacsim-core")
    isaaclab = _package_version("isaaclab")
    extension = _physx_extension_snapshot()
    source_builder = ROOT / "scripts" / "round4070_g" / "physics_scene_builder.py"
    source_asset = ROOT / "scripts" / "round4070_fixed_probe_benchmark" / "structural_curtain_asset.py"
    source_text = (source_builder.read_text(encoding="utf-8") if source_builder.is_file() else "") + (source_asset.read_text(encoding="utf-8") if source_asset.is_file() else "")
    native_surface_evidence = all(token in source_text for token in (
        "DeformablePrim", "set_physics_surface_deformable_body", "add_surface_deformable_material",
        "create_auto_deformable_attachment", "UsdPhysics.CollisionAPI",
    ))
    hand_cert = read_json(REPORT / "NATIVE_HAND_COLLIDER_CERTIFICATION_V6.json", {}) or {}
    windows = {"release": platform.release(), "version": platform.version(), "system": platform.system()}
    try:
        import winreg  # type: ignore

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion") as key:
            for name in ("ProductName", "DisplayVersion", "CurrentBuild", "UBR"):
                try:
                    windows[name] = winreg.QueryValueEx(key, name)[0]
                except OSError:
                    pass
    except (ImportError, OSError):
        pass
    return {
        "schema": "round4070-g.bimanual.physics_environment.v1",
        "collected_by": "scripts/round4070_g/prepare_bimanual_dynamic_scenes.py",
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "windows": windows,
        "packages": {"isaac_sim": isaacsim, "isaac_lab": isaaclab, "torch": torch_info.get("version"), "torchvision": _package_version("torchvision"), "warp": _package_version("warp-lang")},
        "torch_cuda": torch_info,
        "nvidia_smi": nvidia,
        "nvidia_smi_full": nvidia_full,
        "physx_extension": extension,
        "native_api_evidence": {
            "surface_deformable_source_path": str(source_asset.relative_to(ROOT)) if source_asset.is_file() else None,
            "runtime_scene_builder_path": str(source_builder.relative_to(ROOT)) if source_builder.is_file() else None,
            "surface_deformable_api_present_in_installed_source": native_surface_evidence,
            "surface_deformable_api": "AVAILABLE_BY_INSTALLED_SOURCE_AND_EXTENSION" if native_surface_evidence and extension.get("selected_version") else "NOT_CONFIRMED",
            "g1_and_surface_share_backend": "ARCHITECTURALLY_SUPPORTED_NOT_RUNTIME_CERTIFIED",
            "runtime_validation": "NOT_RUN_PAUSED",
            "kit_launch_performed": False,
            "environment_upgrade_performed": False,
        },
        "native_hand_collider": {
            "certification_file": "research/round4070_comprehensive_traversability/NATIVE_HAND_COLLIDER_CERTIFICATION_V6.json",
            "status": hand_cert.get("status", "NOT_FOUND"),
            "formal_exact_native_collider_certified": bool(hand_cert.get("formal_exact_native_collider_certified", False)),
            "formal_collision_admission": bool(hand_cert.get("formal_collision_admission", False)),
        },
        "compatibility_judgement": {
            "rtx_4070_memory_is_sufficient_for_staged_offline_and_single_scene_validation": True,
            "rtx_4070_memory_is_not_evidence_that_56_scenes_can_run_concurrently": True,
            "same_physx_backend_for_g1_and_cloth": "POSSIBLE_IN_ONE_KIT_STAGE; CONTACT_RELIABILITY_REQUIRES_PHYSICS_ONLY_VALIDATION",
            "decision": "OFFLINE_BUILDER_COMPATIBLE; PHYSICS_RUNTIME_NOT_ADMITTED",
        },
    }


def budget_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("required_scope") == "FORMAL_REQUIRED_CANDIDATE"]
    by_physical: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_stratum: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_physical[row["physical_config_id"]].append(row)
        by_stratum[(row["physical_config_id"], row.get("material_fidelity", ""))].append(row)
    raw_rows = len(candidates)
    raw_episodes = sum(int(row["target_valid_bimanual_episodes"]) for row in candidates)
    raw_combos = sum(int(row["target_independent_parameter_combos"]) for row in candidates)
    physical_lower = sum(max(int(row["target_valid_bimanual_episodes"]) for row in group) for group in by_physical.values())
    physical_combos = sum(max(int(row["target_independent_parameter_combos"]) for row in group) for group in by_physical.values())
    material_lower = sum(max(int(row["target_valid_bimanual_episodes"]) for row in group) for group in by_stratum.values())
    material_combos = sum(max(int(row["target_independent_parameter_combos"]) for row in group) for group in by_stratum.values())
    duplicates = [
        {"execution_stratum": list(key), "scene_ids": [row["bimanual_scene_id"] for row in group]}
        for key, group in by_stratum.items() if len(group) > 1
    ]
    physical_duplicates = [
        {
            "physical_config_id": key,
            "scene_ids": [row["bimanual_scene_id"] for row in group],
            "material_fidelities": sorted({row.get("material_fidelity", "") for row in group}),
            "safe_to_merge": len({row.get("material_fidelity", "") for row in group}) == 1,
        }
        for key, group in by_physical.items() if len(group) > 1
    ]
    checkpoint = read_json(REPORT / "PROGRESS_CHECKPOINT.json", {}) or {}
    manifest = read_json(OUT / "campaign_manifest_v5.json", {}) or {}
    launched = int(checkpoint.get("g1_attempts_launched", 36))
    maximum = int(manifest.get("max_g1_launches", 1000))
    remaining = max(0, maximum - launched)
    minimal_unique_proxy_configs = len({row["physical_config_id"] for row in candidates if not row.get("material_fidelity", "").startswith("REAL_")})
    minimal_physics_checks = minimal_unique_proxy_configs * 3
    minimal_valid_episodes = minimal_unique_proxy_configs
    balanced_valid_episodes = minimal_unique_proxy_configs * 15
    return {
        "registered": {
            "candidate_rows": raw_rows,
            "valid_episode_target": raw_episodes,
            "independent_parameter_combos": raw_combos,
            "physics_seed_checks_at_three_seeds": raw_combos * 3,
        },
        "deduplication": {
            "physical_config_id_count": len(by_physical),
            "physical_config_lower_bound_not_material_safe": physical_lower,
            "physical_config_combo_lower_bound_not_material_safe": physical_combos,
            "material_aware_execution_strata": len(by_stratum),
            "material_aware_episode_lower_bound": material_lower,
            "material_aware_combo_lower_bound": material_combos,
            "material_aware_seed_checks_at_three_seeds": material_combos * 3,
            "duplicate_groups": duplicates,
            "physical_duplicate_groups": physical_duplicates,
            "safe_episode_reduction_from_exact_material_aware_duplicates": raw_episodes - material_lower,
            "safe_combo_reduction_from_exact_material_aware_duplicates": raw_combos - material_combos,
            "warning": "physical_config_id excludes material_fidelity; real-material rows must not be silently merged with proxy rows",
        },
        "current_v5_budget": {
            "max_g1_launches": maximum,
            "g1_attempts_launched": launched,
            "remaining_g1_launches": remaining,
            "registered_episode_deficit_at_100_percent_yield": max(0, raw_episodes - remaining),
            "registered_nominal_attempt_deficit_at_80_percent_yield": max(0, math.ceil(raw_episodes / 0.8) - remaining),
            "max_valid_episodes_at_80_percent_yield": math.floor(remaining * 0.8),
        },
        "minimal_feasible_after_all_safety_approvals": {
            "scope": "one representative valid episode per unique proxy physical configuration",
            "unique_proxy_configs": minimal_unique_proxy_configs,
            "physics_seed_checks": minimal_physics_checks,
            "valid_bimanual_episodes": minimal_valid_episodes,
            "nominal_g1_attempts_at_80_percent_yield": math.ceil(minimal_valid_episodes / 0.8),
            "formal_ranking": False,
        },
        "within_remaining_budget_balanced_plan_not_authorization": {
            "scope": "15 valid episodes per unique proxy physical configuration",
            "valid_bimanual_episodes": balanced_valid_episodes,
            "nominal_g1_attempts_at_80_percent_yield": math.ceil(balanced_valid_episodes / 0.8),
            "remaining_nominal_headroom": remaining - math.ceil(balanced_valid_episodes / 0.8),
            "formal_ranking": False,
        },
        "assumptions": [
            "left and right hands remain one synchronized episode, never two samples",
            "physics seeds are validation checks, not independent physical scenes",
            "a valid episode cannot be claimed before B10/V6, hand collider, stop/retract and independent approval gates",
            "no task is launched by this audit",
        ],
    }


def update_matrix(rows: list[dict[str, str]], plans: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[str]]:
    by_id = {plan["scene_id"]: plan for plan in plans}
    new_fields = [
        "execution_stratum_key",
        "dynamic_builder_version",
        "dynamic_scene_status",
        "dynamic_builder_status",
        "dynamic_asset_status",
        "dynamic_physics_status",
        "dynamic_plan_path",
        "dynamic_missing_reason",
        "dynamic_component_statuses",
        "dynamic_offline_check_status",
    ]
    fields = list(rows[0].keys()) if rows else []
    for field in new_fields:
        if field not in fields:
            fields.append(field)
    for row in rows:
        plan = by_id.get(row.get("bimanual_scene_id"))
        if plan is None:
            row.update({
                "execution_stratum_key": "",
                "dynamic_builder_version": "NOT_APPLICABLE_LEGACY_REFERENCE",
                "dynamic_scene_status": "NOT_APPLICABLE_LEGACY_REFERENCE",
                "dynamic_builder_status": "NOT_APPLICABLE_LEGACY_REFERENCE",
                "dynamic_asset_status": "NOT_APPLICABLE_LEGACY_REFERENCE",
                "dynamic_physics_status": "NOT_APPLICABLE_LEGACY_REFERENCE",
                "dynamic_plan_path": "",
                "dynamic_missing_reason": "",
                "dynamic_component_statuses": "",
                "dynamic_offline_check_status": "",
            })
            continue
        components = "|".join(f"{item['component_type']}={item['status']}" for item in plan["components"])
        checks = "|".join(f"{key}={value}" for key, value in plan["offline_checks"].items())
        row.update({
            "execution_stratum_key": hashlib.sha256(f"{row.get('physical_config_id','')}|{row.get('material_fidelity','')}".encode("utf-8")).hexdigest()[:16],
            "dynamic_builder_version": "BIMANUAL_DYNAMIC_BUILDER_V1",
            "dynamic_scene_status": plan["status"],
            "dynamic_builder_status": plan["dynamic_builder_status"],
            "dynamic_asset_status": plan["dynamic_asset_status"],
            "dynamic_physics_status": plan["physics_validation_status"],
            "dynamic_plan_path": str((PLAN_DIR / f"{plan['scene_id']}.json").relative_to(ROOT)),
            "dynamic_missing_reason": "|".join(plan["missing_or_blocking_reasons"]),
            "dynamic_component_statuses": components,
            "dynamic_offline_check_status": checks,
        })
    return rows, fields


def write_environment_report(environment: dict[str, Any]) -> None:
    write_json(ENV_JSON, environment)
    torch = environment["torch_cuda"]
    packages = environment["packages"]
    extensions = environment["physx_extension"].get("extensions", [])
    extension_lines = "\\n".join(f"- {item['version']}: {item['path']} ({item['description']})" for item in extensions) or "- not found"
    ENV_MD.write_text(markdown_text(
        "# BIMANUAL_V1 Physics Environment Compatibility\\n\\n"
        "Status: OFFLINE_CAPABILITY_AUDIT_ONLY. No Kit, Isaac, physics gate, or G1 process was started.\\n\\n"
        "## Installed environment\\n\\n"
        f"- Windows: {environment['windows'].get('ProductName', environment['windows']['release'])}; DisplayVersion {environment['windows'].get('DisplayVersion', 'unknown')}; build {environment['windows'].get('CurrentBuild', environment['windows']['version'])}; UBR {environment['windows'].get('UBR', 'unknown')}\\n"
        f"- Python: {environment['python']['version']} ({environment['python']['executable']})\\n"
        f"- Isaac Sim package: {packages['isaac_sim']}\\n"
        f"- Isaac Lab package: {packages['isaac_lab']}\\n"
        f"- PyTorch: {packages['torch']} (bundled CUDA runtime {torch.get('cuda_runtime')})\\n"
        f"- CUDA available: {torch.get('cuda_available')}; device: {torch.get('device_name')}; VRAM reported by torch: {torch.get('total_memory_gib')} GiB\\n"
        f"- nvidia-smi: {environment.get('nvidia_smi') or 'unavailable'}\\n"
        "- PhysX extensions:\\n"
        f"{extension_lines}\\n"
        f"- PhysX backend extension version: {environment['physx_extension'].get('backend_version')}\\n\\n"
        "## Compatibility judgement\\n\\n"
        "- The installed source and PhysX extension expose the Isaac Sim 5.1 surface-deformable path: UsdGeom.Mesh, set_physics_surface_deformable_body, add_surface_deformable_material, automatic deformable attachments, and UsdPhysics.CollisionAPI.\\n"
        "- A G1 rigid articulation and a surface-deformable curtain can be authored in one Kit/PhysX stage in principle. The existing runtime builder uses the same stage for deformable sheets, rigid supports, and barriers.\\n"
        "- This is API/source compatibility evidence only. Contact stability, attachment resolution, hinged-door limits, native hand binding, and coordinated bimanual behavior still require a permitted physics-only validation.\\n"
        "- The RTX 4070 Laptop GPU has enough reported memory for staged offline work and one-scene validation; this does not justify concurrent execution of 56 scenes or the registered seed matrix.\\n"
        "- No environment upgrade was performed.\\n\\n"
        "## Safety boundary\\n\\n"
        "B10/V6 remains NO-GO. The native hand certification file still reports no formal exact-collider admission. Dynamic plans therefore remain unverified and cannot authorize a launch.\\n"
    ),
        encoding="utf-8",
    )


def write_dynamic_reports(rows: list[dict[str, str]], plans: list[dict[str, Any]], budget: dict[str, Any]) -> None:
    status_counts = Counter(plan["status"] for plan in plans)
    builder_counts = Counter(plan["dynamic_builder_status"] for plan in plans)
    asset_counts = Counter(plan["dynamic_asset_status"] for plan in plans)
    physics_counts = Counter(plan["physics_validation_status"] for plan in plans)
    missing_counts = Counter(reason for plan in plans for reason in plan["missing_or_blocking_reasons"])
    component_counts = Counter(item["component_type"] for plan in plans for item in plan["components"])
    dynamic_catalog = {
        "schema": "round4070-g.bimanual.dynamic_build_catalog.v1",
        "builder_schema": SCHEMA,
        "status": "OFFLINE_DYNAMIC_PLANS_ONLY",
        "scene_count": len(plans),
        "plan_root": str(PLAN_DIR.relative_to(ROOT)),
        "native_usd_assets_generated": False,
        "kit_launch_performed": False,
        "physics_gate_performed": False,
        "status_counts": dict(status_counts),
        "builder_status_counts": dict(builder_counts),
        "dynamic_asset_status_counts": dict(asset_counts),
        "physics_status_counts": dict(physics_counts),
        "missing_reason_counts": dict(missing_counts),
        "component_counts": dict(component_counts),
        "scene_plans": [
            {
                "scene_id": plan["scene_id"],
                "physical_config_id": plan["physical_config_id"],
                "status": plan["status"],
                "dynamic_builder_status": plan["dynamic_builder_status"],
                "dynamic_asset_status": plan["dynamic_asset_status"],
                "physics_validation_status": plan["physics_validation_status"],
                "plan_path": str((PLAN_DIR / f"{plan['scene_id']}.json").relative_to(ROOT)),
                "missing_or_blocking_reasons": plan["missing_or_blocking_reasons"],
            }
            for plan in plans
        ],
    }
    write_json(DYNAMIC_CATALOG, dynamic_catalog)
    readiness_rows = []
    for plan in plans:
        readiness_rows.append({
            "scene_id": plan["scene_id"],
            "physical_config_id": plan["physical_config_id"],
            "status": plan["status"],
            "dynamic_builder_status": plan["dynamic_builder_status"],
            "dynamic_asset_status": plan["dynamic_asset_status"],
            "physics_validation_status": plan["physics_validation_status"],
            "missing_or_blocking_reasons": "|".join(plan["missing_or_blocking_reasons"]),
            "offline_mesh": plan["offline_checks"]["mesh"],
            "offline_fixed_nodes": plan["offline_checks"]["fixed_nodes"],
            "offline_initial_spacing": plan["offline_checks"]["initial_spacing"],
            "hand_collider_status": plan["hand_collider"]["status"],
            "plan_path": str((PLAN_DIR / f"{plan['scene_id']}.json").relative_to(ROOT)),
        })
    write_csv(READINESS_CSV, readiness_rows, list(readiness_rows[0].keys()))

    rows_by_status = defaultdict(list)
    for plan in plans:
        rows_by_status[plan["status"]].append(plan["scene_id"])
    report_lines = [
        "# BIMANUAL_V1 Dynamic Scene Build Report",
        "",
        "Status: OFFLINE_DYNAMIC_BUILDER_PREPARATION_ONLY.",
        "",
        f"- Candidate scenes: {len(plans)}.",
        f"- REQUIRES_PHYSICS_VALIDATION: {status_counts[REQUIRES_PHYSICS_VALIDATION]}.",
        f"- NOT_BUILT: {status_counts[NOT_BUILT]} (all are missing external real-material assets).",
        f"- ASSET_READY: {status_counts['ASSET_READY']}; no native USD was generated because Kit launch is prohibited under the current pause.",
        f"- Builder-ready components: {builder_counts[BUILDER_READY]}; unbuildable material dependencies: {builder_counts[NOT_BUILT]}.",
        "",
        "## Component coverage",
        "",
        "- Fabric curtain: reusable surface-deformable mesh/attachment recipe for every non-control surface scene.",
        "- Rigid obstacle: hidden wall and rigid-board cuboid recipe with metric dimensions and collision schema mapping.",
        "- PVC/mesh: explicit proxy representation and calibration warning; real material rows are not silently substituted.",
        "- Hinged door: four reusable UsdPhysics.RevoluteJoint plans with hinge axis, gap, resistance, pushability and label scope.",
        "- Hand collider: referenced certification remains non-admitting; no plan upgrades B10/V6 evidence.",
        "",
        "## Per-scene status",
        "",
    ]
    for status in (REQUIRES_PHYSICS_VALIDATION, NOT_BUILT):
        report_lines.append(f"### {status}")
        report_lines.extend(f"- {scene_id}" for scene_id in sorted(rows_by_status[status]))
        report_lines.append("")
    report_lines.extend([
        "## Interpretation",
        "",
        "REQUIRES_PHYSICS_VALIDATION means the dynamic recipe is complete and mapped to the installed native API, but no USD stage was authored and no contact/release/recovery/penetration result exists. It is not a physics pass.",
        "",
        "NOT_BUILT means a required real-material source is absent. The corresponding proxy may be represented separately, but it cannot stand in for the requested material condition.",
        "",
        "The complete machine-readable per-scene table is BIMANUAL_DYNAMIC_SCENE_READINESS.csv; each scene plan is under bimanual_v1_assets/dynamic_scene_plans/.",
    ])
    DYNAMIC_REPORT.write_text(markdown_text("\\n".join(report_lines) + "\\n"), encoding="utf-8")

    d = budget["deduplication"]
    r = budget["registered"]
    current = budget["current_v5_budget"]
    minimal = budget["minimal_feasible_after_all_safety_approvals"]
    balanced = budget["within_remaining_budget_balanced_plan_not_authorization"]
    duplicate_lines = [f"- {', '.join(item['scene_ids'])}: same physical/material execution stratum" for item in d["duplicate_groups"]]
    physical_duplicate_lines = [
        f"- {', '.join(item['scene_ids'])}: material-aware merge={'allowed' if item['safe_to_merge'] else 'forbidden'}; fidelities={', '.join(item['material_fidelities'])}"
        for item in d["physical_duplicate_groups"]
    ]
    BUDGET_MD.write_text(markdown_text(
        "# BIMANUAL_V1 Budget Audit V2\\n\\n"
        "Status: OFFLINE AUDIT; no budget or manifest was changed.\\n\\n"
        "## Relationship of the registered numbers\\n\\n"
        f"- 56 candidate rows are named benchmark cells. Their conservative target is {r['valid_episode_target']} valid synchronized bimanual episodes.\\n"
        f"- The rows request {r['independent_parameter_combos']} independent parameter combinations. At three physics seeds this is {r['physics_seed_checks_at_three_seeds']} physics seed checks.\\n"
        "- A physics seed check is not a G1 episode, and left/right hands are not two samples. A passed physics configuration can support multiple episodes only after the protocol explicitly permits that reuse.\\n\\n"
        "## Duplicate audit\\n\\n"
        f"- Existing structural physical_config_id count: {d['physical_config_id_count']}.\\n"
        f"- Material-aware execution strata: {d['material_aware_execution_strata']}.\\n"
        f"- Material-aware lower bound after exact duplicate removal: {d['material_aware_episode_lower_bound']} valid episodes, {d['material_aware_combo_lower_bound']} combinations, {d['material_aware_seed_checks_at_three_seeds']} seed checks.\\n"
        f"- The tempting structural-ID lower bound is {d['physical_config_lower_bound_not_material_safe']} episodes and {d['physical_config_combo_lower_bound_not_material_safe']} combinations, but it is not material-safe because the current hash excludes material_fidelity.\\n"
        "- Duplicate groups that are safe to share only within the same material fidelity:\\n"
        + ("\\n".join(duplicate_lines) if duplicate_lines else "- none") + "\\n\\n"
        "- All structural duplicate groups, including unsafe material aliases:\\n"
        + ("\\n".join(physical_duplicate_lines) if physical_duplicate_lines else "- none") + "\\n\\n"
        "## Remaining frozen V5 budget\\n\\n"
        f"- Maximum G1 launches: {current['max_g1_launches']}; already launched: {current['g1_attempts_launched']}; remaining: {current['remaining_g1_launches']}.\\n"
        f"- Registered target deficit even at 100% valid yield: {current['registered_episode_deficit_at_100_percent_yield']} valid episodes.\\n"
        f"- Registered nominal deficit at 80% yield: {current['registered_nominal_attempt_deficit_at_80_percent_yield']} launches.\\n"
        f"- 964 remaining launches would yield at most about {current['max_valid_episodes_at_80_percent_yield']} valid episodes at 80% yield.\\n\\n"
        "## Minimum and balanced options after approval\\n\\n"
        f"- Minimum feasibility check: {minimal['unique_proxy_configs']} unique proxy physical configurations x 3 seeds = {minimal['physics_seed_checks']} physics checks, then one valid episode per configuration ({minimal['valid_bimanual_episodes']} episodes; about {minimal['nominal_g1_attempts_at_80_percent_yield']} nominal launches at 80% yield). This is a coverage smoke benchmark, not a formal ranking.\\n"
        f"- A balanced plan within the remaining launch ceiling would target {balanced['valid_bimanual_episodes']} valid episodes (15 per unique proxy configuration), requiring about {balanced['nominal_g1_attempts_at_80_percent_yield']} nominal launches at 80% yield. It still cannot satisfy the registered full target and is not an authorization.\\n\\n"
        "No Cartesian-product expansion was launched. A future approved design should use the 50 unique proxy configurations as strata, a documented partial-factorial/LHS selection for parameter combinations, and separate confirmation configurations.\\n"
    ),
        encoding="utf-8",
    )


def update_asset_catalog(plans: list[dict[str, Any]]) -> None:
    catalog = read_json(ASSET_CATALOG, {}) or {}
    catalog["status"] = "STATIC_CANDIDATES_PLUS_OFFLINE_DYNAMIC_PLANS"
    catalog["dynamic_builder_schema"] = SCHEMA
    catalog["dynamic_plan_root"] = str(PLAN_DIR.relative_to(ROOT))
    catalog["native_usd_assets_generated"] = False
    catalog["dynamic_status_counts"] = dict(Counter(plan["status"] for plan in plans))
    catalog["dynamic_builder_status_counts"] = dict(Counter(plan["dynamic_builder_status"] for plan in plans))
    catalog["dynamic_physics_status_counts"] = dict(Counter(plan["physics_validation_status"] for plan in plans))
    catalog["dynamic_missing_reason_counts"] = dict(Counter(reason for plan in plans for reason in plan["missing_or_blocking_reasons"]))
    catalog["warning"] = "JSON/OBJ files and offline dynamic plans are not native USD, PhysX binding, or physics evidence."
    write_json(ASSET_CATALOG, catalog)


def update_checkpoint(plans: list[dict[str, Any]], budget: dict[str, Any], environment: dict[str, Any], pause_hashes: dict[str, str | None], manifest_hash: str | None) -> None:
    checkpoint = read_json(CHECKPOINT, {}) or {}
    if not isinstance(checkpoint, dict):
        raise RuntimeError("BIMANUAL_PROGRESS_CHECKPOINT.json is not an object")
    status_counts = Counter(plan["status"] for plan in plans)
    builder_counts = Counter(plan["dynamic_builder_status"] for plan in plans)
    checkpoint["stage"] = "STAGE_2_OFFLINE_DYNAMIC_BUILDER_PREPARATION"
    checkpoint["status"] = "IN_PROGRESS_NOT_COMPLETE"
    checkpoint["dynamic_builder_preparation"] = {
        "schema": SCHEMA,
        "builder_version": "BIMANUAL_DYNAMIC_BUILDER_V1",
        "candidate_scene_count": len(plans),
        "status_counts": dict(status_counts),
        "builder_status_counts": dict(builder_counts),
        "native_usd_assets_generated": False,
        "kit_launch_performed": False,
        "physics_gate_performed": False,
        "g1_launch_performed": False,
        "plan_root": str(PLAN_DIR.relative_to(ROOT)),
        "environment_report": str(ENV_JSON.relative_to(ROOT)),
        "readiness_report": str(READINESS_CSV.relative_to(ROOT)),
    }
    checkpoint["environment_compatibility"] = {
        "report": str(ENV_JSON.relative_to(ROOT)),
        "isaac_sim": environment["packages"].get("isaac_sim"),
        "isaac_lab": environment["packages"].get("isaac_lab"),
        "python": environment["python"].get("version"),
        "torch": environment["packages"].get("torch"),
        "physx_extension": environment["physx_extension"].get("selected_version"),
        "decision": environment["compatibility_judgement"]["decision"],
    }
    checkpoint["budget_v2"] = budget
    checkpoint["safety"] = dict(checkpoint.get("safety", {}))
    checkpoint["safety"].update({
        "global_pause_present": all(value is not None for value in pause_hashes.values()),
        "g1_pause_present": pause_hashes.get("PAUSE_G1.flag") is not None,
        "pause_flags_modified": False,
        "manifest_reset": False,
        "e1_fixed_modified": False,
        "manifest_sha256_at_dynamic_preparation": manifest_hash,
        "pause_hashes_at_dynamic_preparation": pause_hashes,
    })
    not_run = set(checkpoint.get("not_run", []))
    not_run.update({"native USD authoring", "bimanual physics gate", "bimanual G1 collection", "Kit launch"})
    checkpoint["not_run"] = sorted(not_run)
    checkpoint["next_gates"] = [
        "resolve B10/V6 action and exact native hand collider safety NO-GO",
        "obtain independent safety approval for representative stop/retract physics-only validation",
        "supply and calibrate the three missing real-material assets before using those rows",
        "approve a material-aware budget and partial-factorial/LHS design",
    ]
    write_json(CHECKPOINT, checkpoint)
    if VERSION_CHECKPOINT.parent.is_dir():
        write_json(VERSION_CHECKPOINT, checkpoint)


def main() -> int:
    pause_paths = {
        "PAUSE.flag": OUT / "PAUSE.flag",
        "PAUSE_G1.flag": OUT / "PAUSE_G1.flag",
    }
    pause_hashes = {name: sha256_file(path) for name, path in pause_paths.items()}
    if any(value is None for value in pause_hashes.values()):
        raise RuntimeError("both required pause flags must exist before offline preparation")
    manifest_path = OUT / "campaign_manifest_v5.json"
    manifest_hash_before = sha256_file(manifest_path)
    rows, fields = read_csv(MATRIX)
    plans = build_all_offline()
    if len(plans) != 56:
        raise RuntimeError(f"expected 56 candidate plans, got {len(plans)}")
    plan_ids = {plan["scene_id"] for plan in plans}
    matrix_ids = {row["bimanual_scene_id"] for row in rows if row.get("required_scope") == "FORMAL_REQUIRED_CANDIDATE"}
    if plan_ids != matrix_ids:
        raise RuntimeError("dynamic plan IDs do not match the existing 56 candidate matrix IDs")
    environment = environment_snapshot()
    budget = budget_audit(rows)
    rows, fields = update_matrix(rows, plans)
    write_csv(MATRIX, rows, fields)
    write_environment_report(environment)
    write_dynamic_reports(rows, plans, budget)
    update_asset_catalog(plans)
    update_checkpoint(plans, budget, environment, pause_hashes, manifest_hash_before)
    if {name: sha256_file(path) for name, path in pause_paths.items()} != pause_hashes:
        raise RuntimeError("pause flag hash changed during offline preparation")
    if sha256_file(manifest_path) != manifest_hash_before:
        raise RuntimeError("frozen campaign manifest changed during offline preparation")
    print(json.dumps({
        "status": "OFFLINE_DYNAMIC_BUILDER_PREPARATION_COMPLETE",
        "candidate_scene_count": len(plans),
        "scene_status_counts": dict(Counter(plan["status"] for plan in plans)),
        "builder_status_counts": dict(Counter(plan["dynamic_builder_status"] for plan in plans)),
        "native_usd_assets_generated": False,
        "physics_gate_performed": False,
        "g1_launch_performed": False,
        "pause_flags_unchanged": True,
        "manifest_unchanged": True,
        "reports": {
            "environment": str(ENV_MD.relative_to(ROOT)),
            "dynamic_catalog": str(DYNAMIC_CATALOG.relative_to(ROOT)),
            "readiness_csv": str(READINESS_CSV.relative_to(ROOT)),
            "budget": str(BUDGET_MD.relative_to(ROOT)),
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
