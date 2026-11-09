"""CPU-only Stage 2A native-worker contract tests; no Kit is imported/launched."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = Path(__file__).with_name("bimanual_stage2_kit_physics_worker.py")
RUNNER_PATH = Path(__file__).with_name("run_bimanual_kit_physics_only.py")
SPEC = importlib.util.spec_from_file_location("bimanual_stage2_worker", WORKER_PATH)
worker = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(worker)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BimanualStage2WorkerTests(unittest.TestCase):
    def test_worker_import_is_kit_free(self):
        tree = ast.parse(WORKER_PATH.read_text(encoding="utf-8"))
        imported = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(any(name.startswith(("isaacsim", "isaaclab", "omni", "pxr")) for name in imported))

    def test_kit_launch_is_refused_when_bundle_is_pending(self):
        argv = [
            "--plan", str(ROOT / "research/round4070_comprehensive_traversability/bimanual_v1_assets/dynamic_scene_plans/BI_A01_BASIC_CLOTH.json"),
            "--scene-asset", str(ROOT / "research/round4070_comprehensive_traversability/bimanual_v1_assets/scene_specs/BI_A01_BASIC_CLOTH.json"),
            "--scene-id", "BI_A01_BASIC_CLOTH", "--seed", "4070701",
            "--run-id", "BIV1A_BI_A01_BASIC_CLOTH_seed4070701_attempt01",
            "--run-root", str(ROOT / "outputs/round4070_comprehensive_traversability/physics_diagnostics/bimanual_v1_kit_physics_stage2_20261109/runs/BIV1A_BI_A01_BASIC_CLOTH_seed4070701_attempt01"),
            "--usd-path", str(ROOT / "outputs/round4070_comprehensive_traversability/physics_diagnostics/bimanual_v1_kit_physics_stage2_20261109/usd/BI_A01_BASIC_CLOTH_0d1efd8f4594.usd"),
            "--author-usd", "--headless",
        ]
        with patch.object(worker, "require_supervisor_authority", side_effect=RuntimeError("pending approvals")):
            result = worker.main(argv)
        self.assertEqual(result, 3)

    def test_door_contract_is_exact_and_scene_bound(self):
        contract_path = ROOT / "research/round4070_comprehensive_traversability/BIMANUAL_STAGE2_DOOR_ACCEPTANCE_CONTRACT_20261109.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertEqual(contract["status"], "PREREGISTERED")
        self.assertEqual(contract["scene_asset_sha256"], file_hash(ROOT / contract["scene_asset_path"]))
        self.assertEqual(contract["scene_plan_sha256"], file_hash(ROOT / contract["scene_plan_path"]))
        self.assertEqual(tuple(contract["required_checks"]), worker.REQUIRED_DOOR_CHECKS)
        self.assertGreater(contract["probe_lateral_displacement_m"], 0.0)
        self.assertGreater(contract["minimum_open_response_rad"], 0.0)
        self.assertGreater(contract["release_tail_start_s"], contract["push_window_s"][1])

    def test_worker_result_schema_has_runtime_and_physics_separate(self):
        summary = {
            "schema": worker.SCHEMA,
            "runtime_valid": True,
            "physics_valid": False,
            "usd_saved_and_reopened": True,
            "usd_schema_bindings_verified": True,
            "native_runtime_bindings_verified": True,
            "seeded_initial_conditions_verified": True,
            "usd_file_immutable_during_physics": True,
            "g1_loaded": False,
            "controller_or_action_created": False,
            "formal_episode_created": False,
        }
        self.assertEqual(summary["schema"], "round4070-g.bimanual.stage2a.run.v1")
        self.assertNotEqual(summary["runtime_valid"], summary["physics_valid"])
        self.assertFalse(summary["g1_loaded"] or summary["controller_or_action_created"] or summary["formal_episode_created"])

    def test_worker_does_not_call_campaign_or_v5_runner(self):
        source = WORKER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("run_round4070_g.ps1", source)
        self.assertNotIn("campaign_manifest_v5.json", source)
        self.assertNotIn("g1_trial", source)
        self.assertIn("require_supervisor_authority", source)
        self.assertIn("PAUSE.flag", source)
        self.assertIn("PAUSE_G1.flag", source)

    def test_box_penetration_tolerance_applies_only_to_probe_half_extent(self):
        identity = lambda _q: np.eye(3)
        self.assertTrue(worker.oriented_boxes_overlap(
            np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.full(3, 0.06),
            np.array([0.104, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), np.full(3, 0.05),
            identity, margin_a=-0.005))
        self.assertFalse(worker.oriented_boxes_overlap(
            np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.full(3, 0.06),
            np.array([0.106, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), np.full(3, 0.05),
            identity, margin_a=-0.005))
        wall = {"center_xyz": [0.092, 0.0, 0.0], "size_xyz": [0.1, 0.1, 0.1]}
        self.assertTrue(worker.oriented_box_overlap(
            np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 0.05, wall, identity, margin=-0.005))

    def test_door_contract_is_required_for_non_door_scenes_without_binding_it_to_them(self):
        master = worker.read_json(worker.MASTER_PLAN)
        for scene_id in ("BI_A01_BASIC_CLOTH", "BI_G_DOOR_ORDINARY"):
            row = next(item for item in master["run_matrix"] if item["scene_id"] == scene_id)
            scene_plan = ROOT / row["plan_path"]
            plan = worker.read_json(scene_plan)
            asset_path = ROOT / plan["source_scene_asset"]
            run_id = f"BIV1A_{scene_id}_seed4070701_attempt01"
            output_root = ROOT / master["scope"]["output_root"]
            args = worker.base_parser().parse_args([
                "--plan", str(scene_plan), "--scene-asset", str(asset_path), "--scene-id", scene_id,
                "--seed", "4070701", "--run-id", run_id,
                "--run-root", str(output_root / "runs" / run_id),
                "--usd-path", str(output_root / "usd" / f"{scene_id}_{row['plan_sha256'][:12]}.usd"),
                "--author-usd",
            ])
            context = worker.validate_invocation(args)
            self.assertEqual(context["row"]["scene_id"], scene_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
