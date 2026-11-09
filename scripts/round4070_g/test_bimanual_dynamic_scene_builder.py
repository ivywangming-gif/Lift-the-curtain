"""Offline regression tests for BIMANUAL_V1 dynamic scene planning."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bimanual_dynamic_scene_builder as builder


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BimanualDynamicBuilderTests(unittest.TestCase):
    def test_import_is_kit_free(self) -> None:
        source = (Path(builder.__file__)).read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                top_level_imports.append(node.module or "")
        self.assertFalse(any(name.startswith(("isaacsim", "isaaclab", "omni", "pxr")) for name in top_level_imports))
        self.assertNotIn("isaacsim", sys.modules)
        self.assertTrue(callable(builder.BimanualSceneComposer().build_native))

    def test_all_56_scene_plans_are_unique_and_statused(self) -> None:
        plans = builder.BimanualSceneComposer().compose_all()
        self.assertEqual(len(plans), 56)
        self.assertEqual(len({plan["scene_id"] for plan in plans}), 56)
        self.assertTrue(all(plan["status"] in builder.ALLOWED_SCENE_STATUSES for plan in plans))
        self.assertEqual(sum(plan["status"] == builder.REQUIRES_PHYSICS_VALIDATION for plan in plans), 53)
        self.assertEqual(sum(plan["status"] == builder.NOT_BUILT for plan in plans), 3)
        self.assertTrue(all(plan["dynamic_asset_status"] == builder.NOT_BUILT for plan in plans))
        self.assertTrue(all(plan["physics_validation_status"] == "NOT_RUN_PAUSED" for plan in plans))

    def test_real_materials_have_explicit_blocking_reason(self) -> None:
        plans = builder.BimanualSceneComposer().compose_all()
        blocked = [plan for plan in plans if "MISSING_EXTERNAL_MATERIAL_ASSET" in plan["missing_or_blocking_reasons"]]
        self.assertEqual({plan["scene_id"] for plan in blocked}, {"BI_E_REAL_CLOTH", "BI_E_REAL_MESH", "BI_E_REAL_PVC"})
        for plan in blocked:
            self.assertEqual(plan["status"], builder.NOT_BUILT)
            self.assertTrue(any(component["component_type"] == "MaterialAssetDependency" for component in plan["components"]))

    def test_pvc_mesh_and_door_plans_are_native_api_mapped(self) -> None:
        plans = builder.BimanualSceneComposer().compose_all()
        by_id = {plan["scene_id"]: plan for plan in plans}
        for scene_id in ("BI_E_OPEN_MESH_PROXY", "BI_E_DENSE_MESH_PROXY", "BI_E_PVC_INDEPENDENT_PROXY", "BI_E_PVC_OVERLAP_PROXY"):
            component = next(item for item in by_id[scene_id]["components"] if item["component_type"] == "PvcMeshBuilder")
            self.assertTrue(component["plan"]["enabled"])
            self.assertTrue(component["plan"]["calibration_required_before_formal_use"])
        for scene_id in ("BI_G_DOOR_ORDINARY", "BI_G_DOOR_MEDIUM", "BI_G_DOOR_HIGH", "BI_G_DOOR_NONPUSHABLE"):
            component = next(item for item in by_id[scene_id]["components"] if item["component_type"] == "HingedDoorBuilder")
            self.assertEqual(component["status"], builder.BUILDER_READY)
            self.assertEqual(component["plan"]["hinge"]["axis"], "Z")
            self.assertEqual(component["plan"]["native_schema"]["joint"], "UsdPhysics.RevoluteJoint")
            self.assertEqual(component["plan"]["resistance_model"]["kind"], "VISCOUS_ANGULAR_DRIVE_PROXY")
            self.assertEqual(component["plan"]["resistance_model"]["calibration_status"], "REQUIRES_PHYSICS_VALIDATION")
            self.assertGreater(component["plan"]["resistance_model"]["max_force_nm"], 0.0)

    def test_mesh_fixed_nodes_and_clearances_are_offline_valid(self) -> None:
        plans = builder.BimanualSceneComposer().compose_all()
        for plan in plans:
            checks = plan["offline_checks"]
            self.assertEqual(checks["units"], "PASS_METRIC_M_KG_S")
            self.assertNotEqual(checks["mesh"], "FAIL_OFFLINE_ANALYTIC")
            self.assertNotEqual(checks["fixed_nodes"], "FAIL_INDEX_RANGE")
            self.assertNotEqual(checks["initial_spacing"], "FAIL_CLEARANCE_OR_GRID")
            self.assertEqual(checks["physics_gate"], "NOT_RUN_PAUSED")
            self.assertEqual(plan["hand_collider"]["exact_native_collider_admission"], False)

    def test_pause_flags_and_manifest_are_not_modified_by_planner(self) -> None:
        pause = ROOT / "outputs" / "round4070_comprehensive_traversability" / "PAUSE.flag"
        pause_g1 = ROOT / "outputs" / "round4070_comprehensive_traversability" / "PAUSE_G1.flag"
        manifest = ROOT / "outputs" / "round4070_comprehensive_traversability" / "campaign_manifest_v5.json"
        before = {path: file_hash(path) for path in (pause, pause_g1, manifest) if path.is_file()}
        builder.build_all_offline()
        after = {path: file_hash(path) for path in (pause, pause_g1, manifest) if path.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
