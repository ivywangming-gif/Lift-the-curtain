"""Offline contract checks for the BIMANUAL_V1 Stage 1 audit."""
import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_bimanual_scene_matrix as audit  # noqa: E402


class BimanualSceneAudit(unittest.TestCase):
    def test_source_counts_and_campaign_descriptor(self):
        counts = audit.current_counts()
        self.assertEqual(counts["legal_history_count"], 21)
        self.assertEqual(counts["admitted_history_count"], 18)
        self.assertEqual(counts["pending_history_count"], 3)
        self.assertEqual(counts["known_label_count_including_controls"], 14)
        self.assertEqual(counts["core_label_count_excluding_empty_controls"], 12)
        self.assertEqual(counts["legal_unknown_count"], 7)
        self.assertTrue(counts["current_campaign"]["exists"])
        self.assertTrue(counts["current_campaign"]["manifest_matches"])
        self.assertTrue(all(item["exists"] for item in counts["input_sources"].values()))

    def test_legacy_shape_and_candidate_shape(self):
        registry = audit.load_scene_registry()
        sources = audit.load_source_configs(); sources.update(registry)
        rows, validation, _ = audit.build_rows(registry, sources, audit.load_legacy_status(), audit.load_legacy_matrix(), write_assets=False)
        self.assertEqual(len(registry), 48)
        self.assertEqual(len({x["physical_config_id"] for x in registry.values()}), 43)
        self.assertEqual(len(rows), 104)
        self.assertEqual(len(validation), 104)
        self.assertEqual(sum(x["required_scope"] == "AUDIT_REFERENCE" for x in rows), 48)
        self.assertEqual(sum(x["required_scope"] == "FORMAL_REQUIRED_CANDIDATE" for x in rows), 56)

    def test_required_scene_cells_are_explicit(self):
        path = audit.REPORT / "BIMANUAL_V1_SCENE_MATRIX.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        candidates = [x for x in rows if x["required_scope"] == "FORMAL_REQUIRED_CANDIDATE"]
        self.assertEqual({x["category"] for x in candidates}, {"A_CONTROL", "B_HIDDEN_WALL", "C_CONSTRAINT_TOPOLOGY", "D_PHYSICAL_PARAMETER", "E_MATERIAL_STRUCTURE", "F_MOVABLE_STRUCTURE", "G_CURTAIN_DOOR_COMPOSITE"})
        walls = {x["hidden_wall_distance_m"] for x in candidates if x["category"] == "B_HIDDEN_WALL"}
        self.assertEqual(walls, {"0.02", "0.04", "0.06", "0.08", ""})
        topologies = {x["constraint_topology"] for x in candidates if x["category"] == "C_CONSTRAINT_TOPOLOGY"}
        self.assertIn("THREE_CORNERS_MISSING_TOP_RIGHT", topologies)
        self.assertIn("THREE_CORNERS_MISSING_TOP_LEFT", topologies)
        self.assertIn("THREE_CORNERS_MISSING_BOTTOM_RIGHT", topologies)
        self.assertIn("THREE_CORNERS_MISSING_BOTTOM_LEFT", topologies)
        self.assertIn("FULL_PERIMETER", topologies)
        materials = {x["material_fidelity"] for x in candidates if x["category"] == "E_MATERIAL_STRUCTURE"}
        self.assertTrue(any(x.startswith("UNCALIBRATED_PERFORATED") for x in materials))
        self.assertTrue(any(x.startswith("REAL_") for x in materials))
        self.assertEqual(sum(x["category"] == "G_CURTAIN_DOOR_COMPOSITE" for x in candidates), 4)

    def test_new_rows_fail_closed(self):
        path = audit.REPORT / "BIMANUAL_V1_SCENE_MATRIX.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        candidates = [x for x in rows if x["required_scope"] == "FORMAL_REQUIRED_CANDIDATE"]
        legacy = [x for x in rows if x["required_scope"] == "AUDIT_REFERENCE"]
        self.assertEqual({x["overall_gate_status"] for x in candidates}, {"EXECUTION_BLOCKED"})
        self.assertEqual({x["overall_gate_status"] for x in legacy}, {"PHYSICS_VALID", "PHYSICS_INVALID"})
        self.assertEqual({x["scene_gate_status"] for x in candidates}, {"NOT_BUILT"})
        self.assertEqual({x["physics_gate_status"] for x in candidates}, {"EXECUTION_BLOCKED"})
        self.assertEqual({x["bimanual_execution_status"] for x in candidates}, {"EXECUTION_BLOCKED"})
        self.assertEqual({x["bimanual_execution_status"] for x in legacy}, {"NOT_APPLICABLE_SINGLE_HAND"})
        self.assertTrue(all(x["bimanual_physics_status"] == "NOT_RUN_PAUSED" for x in candidates))
        self.assertTrue(all(x["bimanual_physics_valid_seed_count"] == "0" for x in candidates))
        self.assertTrue(all(x["bimanual_episode_count"] == "0" for x in candidates))
        self.assertTrue(all(x["initial_geometry_status"] == "PASS_STATIC_ANALYTIC" for x in candidates))
        asset = json.loads((audit.ROOT / candidates[0]["asset_path"]).read_text(encoding="utf-8"))
        self.assertEqual(asset["overall_gate_status"], "EXECUTION_BLOCKED")
        self.assertEqual(asset["scene_build_status"], "NOT_BUILT")
        self.assertEqual(asset["physics_status"], "NOT_RUN_PAUSED")

    def test_protocol_and_pause_contract(self):
        probe = json.loads((audit.REPORT / "BIMANUAL_PROBE_V1.json").read_text(encoding="utf-8"))
        self.assertEqual(probe["version"], "BIMANUAL_PROBE_V1")
        self.assertEqual(probe["protected_invariants"]["max_nominal_probe_m_per_hand"], 0.1)
        self.assertEqual(probe["coordinated_stop_and_retract"]["unilateral_forward_motion"], "forbidden")
        self.assertEqual(probe["launch_gate"]["current_decision"], "NO_GO")
        for name, expected in {
            "PAUSE.flag": "3231DE7B0F0C282178D2B92FB44B0B37EC57B16364E07CD873CCFE8FAFFFC5AE",
            "PAUSE_G1.flag": "B44CCEFFE2C948E9F13BBF10C1C1A63ADF6D76629CF5CD8DCD774720FA8B6F4F",
        }.items():
            actual = hashlib.sha256((audit.OUT / name).read_bytes()).hexdigest().upper()
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
