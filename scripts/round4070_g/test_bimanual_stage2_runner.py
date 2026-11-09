"""CPU-only Stage 2A supervisor tests; no Kit or physics process is launched."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("run_bimanual_kit_physics_only.py")
SPEC = importlib.util.spec_from_file_location("bimanual_stage2_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BimanualStage2RunnerTests(unittest.TestCase):
    def test_import_is_kit_free_and_preflight_has_no_side_effects(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertFalse(any(name.startswith(("isaacsim", "isaaclab", "omni", "pxr")) for name in imports))
        pause = ROOT / "outputs/round4070_comprehensive_traversability/PAUSE.flag"
        pause_g1 = ROOT / "outputs/round4070_comprehensive_traversability/PAUSE_G1.flag"
        manifest = ROOT / "outputs/round4070_comprehensive_traversability/campaign_manifest_v5.json"
        matrix = ROOT / "research/round4070_comprehensive_traversability/BIMANUAL_V1_SCENE_MATRIX.csv"
        before = {path: file_hash(path) for path in (pause, pause_g1, manifest, matrix)}
        result = runner.preflight(ROOT)
        after = {path: file_hash(path) for path in before}
        self.assertEqual(before, after)
        self.assertFalse(result["ready"])
        self.assertIn("PROJECT_OWNER_APPROVAL_MISSING", result["blockers"])
        self.assertIn("INDEPENDENT_REVIEWER_APPROVAL_MISSING", result["blockers"])
        self.assertIn("STAGE_2A_NOT_APPROVED", result["blockers"])
        self.assertNotIn("DOOR_ACCEPTANCE_CONTRACT_NOT_PREREGISTERED", result["blockers"])
        self.assertEqual(result["checks"]["planned_runs"], 24)
        self.assertFalse(result["checks"]["kit_launch_performed"])

    def test_current_frozen_inputs_match_the_plan(self):
        result = runner.preflight(ROOT)
        source_mismatches = [item for item in result["blockers"] if item.startswith("PINNED_SOURCE_MISMATCH:")]
        protected_mismatches = [item for item in result["blockers"] if item.startswith("PROTECTED_HASH_MISMATCH:")]
        self.assertEqual(source_mismatches, [])
        self.assertEqual(protected_mismatches, [])
        self.assertEqual(result["checks"]["existing_gate_capability"]["native_usd_save_reopen"], False)
        self.assertEqual(result["checks"]["existing_gate_capability"]["hinged_door_contact_and_joint_acceptance"], False)

    def test_path_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            runner._inside_root(ROOT, "../outside.json")

    def test_live_guard_detects_pause_or_manifest_mutation(self):
        import shutil

        plan = json.loads((ROOT / "research/round4070_comprehensive_traversability/BIMANUAL_STAGE2_KIT_PHYSICS_PLAN_20261109.json").read_text())
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            for relative in (
                "outputs/round4070_comprehensive_traversability/PAUSE.flag",
                "outputs/round4070_comprehensive_traversability/PAUSE_G1.flag",
                "outputs/round4070_comprehensive_traversability/campaign_manifest_v5.json",
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            runner._assert_live_invariants(root, plan)
            (root / "outputs/round4070_comprehensive_traversability/PAUSE.flag").write_text("changed")
            with self.assertRaisesRegex(runner.PreflightError, "PROTECTED_HASH_CHANGED_DURING_RUN:pause_flag"):
                runner._assert_live_invariants(root, plan)

    def test_worker_summary_requires_bound_usd_seed_and_all_frozen_checks(self):
        plan = json.loads((ROOT / "research/round4070_comprehensive_traversability/BIMANUAL_STAGE2_KIT_PHYSICS_PLAN_20261109.json").read_text())
        row = plan["run_matrix"][0]
        run_id = f"BIV1A_{row['scene_id']}_seed4070701_attempt01"
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            usd = Path(directory) / "scene.usd"
            usd.write_bytes(b"offline-fixture-only")
            summary = {
                "schema": "round4070-g.bimanual.stage2a.run.v1",
                "run_id": run_id,
                "scene_id": row["scene_id"],
                "physical_config_id": row["physical_config_id"],
                "seed": 4070701,
                "usd_sha256": runner.sha256(usd),
                "usd_saved_and_reopened": True,
                "usd_schema_bindings_verified": True,
                "native_runtime_bindings_verified": True,
                "resource_telemetry_complete": True,
                "usd_file_immutable_during_physics": True,
                "seeded_initial_conditions_verified": True,
                "seeded_initial_conditions": {
                    "verified": True,
                    "python_seed": 4070701,
                    "numpy_pcg64_seed": 4070701,
                    "torch_seed": 4070701,
                    "physx_internal_rng_seed_controlled": False,
                    "physx_rng_seed_verified": False,
                    "seed_scope": "deterministic initial-condition perturbations only; no PhysX internal RNG API is used",
                    "probe_draw": [0.001, 0.002],
                    "velocity_draw_sha256": "a" * 64,
                    "initial_condition_signature_sha256": runner._canonical_sha256({
                        "probe_draw": [0.001, 0.002], "velocity_draw_sha256": "a" * 64,
                    }),
                },
                "acceptance_contract_sha256": runner._canonical_sha256(plan["acceptance_contract"]),
                "door_acceptance_contract_sha256": plan["source_hashes"]["door_acceptance_contract"],
                "physics_checks": {name: True for name in (
                    "finite", "node_speed_bounded", "neighbor_stretch", "fixed_or_support_drift", "nodes_below_ground",
                    "sampled_interior_penetrations", "contact_demonstrated", "release_clear", "late_node_speed",
                    "initial_hanging_stable", "rail_bounds_and_rotation", "reset_reproducible", "duration_completed",
                )},
                "contact_event_count": 0,
                "contact_evidence": {"frozen_gate_geometric_contact_predicate_seen": True},
                "runtime_valid": True,
                "physics_valid": False,
                "failure_category": "SEED_CONTROL_UNVERIFIED",
            }
            runner._validate_worker_summary(summary, run_id=run_id, scene_id=row["scene_id"],
                                           physical_config_id=row["physical_config_id"], seed=4070701,
                                           usd_path=usd, plan=plan)
            summary["seeded_initial_conditions_verified"] = False
            summary["physics_valid"] = True
            summary["failure_category"] = None
            with self.assertRaisesRegex(runner.PreflightError, "UNVERIFIED_INITIAL_CONDITION_SEED_CANNOT_PASS_GATE"):
                runner._validate_worker_summary(summary, run_id=run_id, scene_id=row["scene_id"],
                                               physical_config_id=row["physical_config_id"], seed=4070701,
                                               usd_path=usd, plan=plan)

    def test_owner_and_independent_reviewer_must_bind_exact_bundle(self):
        plan_path = ROOT / "research/round4070_comprehensive_traversability/BIMANUAL_STAGE2_KIT_PHYSICS_PLAN_20261109.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        runner_hash = file_hash(SCRIPT)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            report_rel = Path("research/round4070_comprehensive_traversability")
            worker_rel = Path(runner.WORKER_REL)
            contract_rel = Path(runner.DOOR_CONTRACT_REL)
            for relative, payload in (
                (worker_rel, (ROOT / worker_rel).read_bytes()),
                (contract_rel, json.dumps({
                    "schema": "round4070-g.bimanual.door-acceptance-contract.v1",
                    "status": "PREREGISTERED",
                    "scene_id": "BI_G_DOOR_ORDINARY",
                    "required_checks": list(runner.REQUIRED_DOOR_CHECKS),
                    "thresholds": {"minimum_open_response_rad": 0.02},
                }).encode("utf-8")),
                (Path("approvals/owner.json"), b"owner evidence"),
                (Path("approvals/reviewer.json"), b"independent reviewer evidence"),
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            worker_hash = file_hash(root / worker_rel)
            contract_hash = file_hash(root / contract_rel)
            plan_hash = file_hash(plan_path)
            (root / report_rel).mkdir(parents=True, exist_ok=True)
            (root / report_rel / plan_path.name).write_bytes(plan_path.read_bytes())
            scene_plan_rel = Path("research/round4070_comprehensive_traversability/bimanual_v1_assets/dynamic_scene_plans/BI_A01_BASIC_CLOTH.json")
            (root / scene_plan_rel).parent.mkdir(parents=True, exist_ok=True)
            (root / scene_plan_rel).write_bytes((ROOT / scene_plan_rel).read_bytes())
            owner_evidence = Path("approvals/owner.json")
            reviewer_evidence = Path("approvals/reviewer.json")
            limits = {
                "max_physics_runs": plan["scope"]["max_physics_runs"],
                "max_concurrent_kit_processes": plan["resource_limits"]["concurrent_kit_processes"],
                "max_loaded_scenes": plan["scope"]["max_loaded_scenes"],
                "headless": True,
                "single_gpu": plan["scope"]["gpu"],
                "max_wall_seconds_per_run": plan["resource_limits"]["max_wall_seconds_per_run"],
                "max_total_wall_seconds": plan["resource_limits"]["max_total_wall_seconds"],
                "max_output_gib": plan["resource_limits"]["max_output_gib"],
                "stop_if_gpu_memory_used_mib_reaches": plan["resource_limits"]["stop_if_gpu_memory_used_mib_reaches"],
                "automatic_retries": plan["resource_limits"]["automatic_retries"],
                "output_root": plan["scope"]["output_root"],
                "pause_flag_removal_or_mutation": False,
                "manifest_or_g1_queue_access": False,
                "g1_usd_or_controller": False,
                "falcon_or_e1_execution": False,
                "training_or_episode_creation": False,
                "formal_g1_collection": False,
            }
            approval = {
                "status": "APPROVED_FOR_STAGE_2A_ONLY",
                "implementation_author_id": "implementer-3",
                "plan_path": (report_rel / plan_path.name).as_posix(),
                "plan_sha256": plan_hash,
                "runner_sha256": runner_hash,
                "worker_path": worker_rel.as_posix(),
                "worker_sha256": worker_hash,
                "door_acceptance_contract_path": contract_rel.as_posix(),
                "door_acceptance_contract_sha256": contract_hash,
                "required_hashes": {
                    "pause_flag": plan["protected_hashes"]["pause_flag"],
                    "pause_g1_flag": plan["protected_hashes"]["pause_g1_flag"],
                    "v5_manifest": plan["protected_hashes"]["v5_manifest"],
                },
                "limits": limits,
                "stage_2b_g1_native_hand_and_stop_retract": {"included": False, "requires_separate_approval": True},
                "approvals": {
                    "project_owner": {
                        "status": "APPROVED", "name_or_id": "owner-1", "approved_scope": "STAGE_2A_ONLY_PHYSICS_ONLY_NO_G1",
                        "approved_plan_sha256": plan_hash,
                        "approved_runner_sha256": runner_hash, "approved_worker_sha256": worker_hash,
                        "approval_reference": "fixture-owner-signature", "signed_at": "2026-11-09T20:00:00+00:00",
                        "approval_evidence_path": owner_evidence.as_posix(),
                        "approval_evidence_sha256": file_hash(root / owner_evidence),
                    },
                    "independent_safety_reviewer": {
                        "status": "APPROVED", "name_or_id": "reviewer-2", "independent_of_implementation": True,
                        "reviewed_plan_sha256": plan_hash, "reviewed_runner_sha256": runner_hash,
                        "reviewed_worker_sha256": worker_hash,
                        "explicitly_authorizes_kit_while_pause_flags_remain": True,
                        "approves_scoped_addendum_to_global_hold": True,
                        "approves_stage_2a_only_no_g1_usd_or_actions": True,
                        "approves_24_runs_and_resource_scope": True, "door_thresholds_preregistered": True,
                        "acknowledges_physx_seed_unverified_runs_remain_unresolved": True,
                        "approval_reference": "fixture-review-signature", "signed_at": "2026-11-09T20:01:00+00:00",
                        "approval_evidence_path": reviewer_evidence.as_posix(),
                        "approval_evidence_sha256": file_hash(root / reviewer_evidence),
                    },
                },
            }
            blockers = []
            runner._validate_approvals(root, approval, plan_hash, runner_hash, blockers)
            self.assertEqual(blockers, [])
            approval["approvals"]["independent_safety_reviewer"]["name_or_id"] = "owner-1"
            blockers = []
            runner._validate_approvals(root, approval, plan_hash, runner_hash, blockers)
            self.assertIn("REVIEWER_MUST_BE_DIFFERENT_FROM_PROJECT_OWNER", blockers)

    def test_seeded_trials_do_not_claim_physx_internal_rng_control(self):
        plan = json.loads((ROOT / "research/round4070_comprehensive_traversability/BIMANUAL_STAGE2_KIT_PHYSICS_PLAN_20261109.json").read_text())
        row = plan["run_matrix"][0]
        run_id = f"BIV1A_{row['scene_id']}_seed4070701_attempt01"
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            usd = Path(directory) / "scene.usd"
            usd.write_bytes(b"offline-fixture-only")
            summary = self.valid_summary(plan, row, run_id, usd)
            summary["seeded_initial_conditions"]["physx_internal_rng_seed_controlled"] = True
            summary["physics_valid"] = True
            summary["failure_category"] = None
            with self.assertRaisesRegex(runner.PreflightError, "UNVERIFIED_PHYSX_SEED_CANNOT_PASS_GATE"):
                runner._validate_worker_summary(summary, run_id=run_id, scene_id=row["scene_id"],
                                               physical_config_id=row["physical_config_id"], seed=4070701,
                                               usd_path=usd, plan=plan)

    @staticmethod
    def valid_summary(plan, row, run_id, usd):
        seed_record = {
            "verified": True,
            "python_seed": 4070701,
            "numpy_pcg64_seed": 4070701,
            "torch_seed": 4070701,
            "physx_internal_rng_seed_controlled": False,
            "physx_rng_seed_verified": False,
            "seed_scope": "deterministic initial-condition perturbations only; no PhysX internal RNG API is used",
            "probe_draw": [0.001, 0.002],
            "velocity_draw_sha256": "a" * 64,
        }
        seed_record["initial_condition_signature_sha256"] = runner._canonical_sha256({
            "probe_draw": seed_record["probe_draw"],
            "velocity_draw_sha256": seed_record["velocity_draw_sha256"],
        })
        return {
            "schema": "round4070-g.bimanual.stage2a.run.v1",
            "run_id": run_id,
            "scene_id": row["scene_id"],
            "physical_config_id": row["physical_config_id"],
            "seed": 4070701,
            "usd_sha256": runner.sha256(usd),
            "usd_saved_and_reopened": True,
            "usd_schema_bindings_verified": True,
            "native_runtime_bindings_verified": True,
            "resource_telemetry_complete": True,
            "usd_file_immutable_during_physics": True,
            "seeded_initial_conditions_verified": True,
            "seeded_initial_conditions": seed_record,
            "acceptance_contract_sha256": runner._canonical_sha256(plan["acceptance_contract"]),
            "door_acceptance_contract_sha256": plan["source_hashes"]["door_acceptance_contract"],
            "physics_checks": {name: True for name in (
                "finite", "node_speed_bounded", "neighbor_stretch", "fixed_or_support_drift", "nodes_below_ground",
                "sampled_interior_penetrations", "contact_demonstrated", "release_clear", "late_node_speed",
                "initial_hanging_stable", "rail_bounds_and_rotation", "reset_reproducible", "duration_completed",
            )},
            "contact_event_count": 0,
            "contact_evidence": {"frozen_gate_geometric_contact_predicate_seen": True},
            "runtime_valid": True,
            "physics_valid": False,
            "failure_category": "SEED_CONTROL_UNVERIFIED",
        }

    def test_cli_execute_refuses_without_launching_or_creating_output(self):
        output = ROOT / "outputs/round4070_comprehensive_traversability/physics_diagnostics/bimanual_v1_kit_physics_stage2_20261109"
        self.assertFalse(output.exists())
        code = runner.main(["--execute", "--root", str(ROOT)])
        self.assertEqual(code, 3)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
