"""Opt-in V6 source adapter for offline integration validation.

The frozen V5 runner is read and patched in memory. The original file, V5
outputs, manifest, E1_FIXED, and pause flags are never rewritten. Non-dry-run
execution is deliberately rejected until the independent V6 safety gates and
approval record exist.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from g1_action_contract_v2 import PRIMARY_PROTOCOL as V5_PROTOCOL, load_plan_contract, validate_contract, witness_world
from g1_action_executor_v6 import PROTOCOL_ID as V6_PROTOCOL_ID, V6ActionSupervisor
from g1_trial import SOURCE, patch_source as patch_v5_source

V6_SCHEMA = "round4070-g.v6-trial-adapter.v1"
V6_EXECUTION_STATUS = "OFFLINE_DEVELOPMENT_NOT_ADMITTED"
RECOVERY_PHASE_PREFIXES = ("physical_retract", "physical_neutralize", "physical_post_retract")


def should_preempt_physics_step(mode: str, phase: str) -> bool:
    if mode == "STOP":
        return True
    return mode == "SAFE_RETRACT" and not phase.startswith(RECOVERY_PHASE_PREFIXES)


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"V6 runner hook changed ({count} matches): {old[:120]}")
    return source.replace(old, new, 1)


def replace_once_after(source: str, marker: str, old: str, new: str) -> str:
    start = source.find(marker)
    if start < 0:
        raise RuntimeError(f"V6 marker missing: {marker[:120]}")
    offset = source.find(old, start)
    if offset < 0:
        raise RuntimeError(f"V6 hook missing after marker: {old[:120]}")
    if source.find(old, offset + len(old)) >= 0 and source.find(old, offset + len(old)) < source.find("        def ", offset + len(old)):
        raise RuntimeError(f"V6 hook is ambiguous after marker: {old[:120]}")
    return source[:offset] + new + source[offset + len(old):]


def build_v6_runtime_bindings(contract: dict[str, object], *, braking_deceleration_mps2: float | None = None) -> dict[str, object]:
    """Build the explicit namespace entries required by the in-memory patch.

    The frozen V5 runner owns the surrounding namespace.  V6 adds only these
    bindings; keeping them explicit prevents an AST-only dry-run from hiding a
    missing runtime symbol.  The non-dry-run path remains admission-gated.
    """
    supervisor = V6ActionSupervisor(
        contract,
        braking_deceleration_mps2=braking_deceleration_mps2,
    )
    decisions: list[dict[str, object]] = []

    def witness(sample: dict[str, object]):
        return witness_world(sample, contract["witness"])

    def record(decision):
        decisions.append(decision.to_dict())

    return {
        "_G_V6_SUPERVISOR": supervisor,
        "_G_V6_WITNESS": witness,
        "_G_V6_RECORD": record,
        "_G_V6_APPLY": V6ActionSupervisor.apply_command,
        "_G_V6_SHOULD_PREEMPT": should_preempt_physics_step,
        "_G_V6_DECISIONS": decisions,
        "_G_V6_CONTRACT": contract,
    }


def patch_source_v6(source: str) -> str:
    """Build a V6 source string without touching the frozen V5 source file."""
    source = patch_v5_source(source)
    source = replace_once(
        source,
        "            nonlocal last_rollout_heartbeat_wall\n",
        "            nonlocal last_rollout_heartbeat_wall\n"
        "            nonlocal conservative_stop\n",
    )
    source = replace_once_after(
        source,
        "        def execute_step(\n",
        "            state = adapter.make_state(robot)\n",
        """            state = adapter.make_state(robot)
            _g_v6_sample = _G_AUDIT.last_sample
            if _g_v6_sample is None:
                raise RuntimeError("V6 requires an actual 200 Hz sample before every physics step")
            _g_v6_next_step = int(_g_v6_sample["physics_step"]) + 1
            _g_v6_current_witness = _G_V6_WITNESS(_g_v6_sample)
            _g_v6_decision = _G_V6_SUPERVISOR.before_step(_g_v6_next_step, _g_v6_current_witness)
            _G_V6_RECORD(_g_v6_decision)
            if _G_V6_SHOULD_PREEMPT(_g_v6_decision.mode, phase):
                external_safety_reason = "V6_" + _g_v6_decision.reason
                conservative_stop = True
                _g_v6_target_distance = float(np.linalg.norm(
                    np.asarray(target_xy, dtype=np.float64)
                    - np.asarray(state.base_pos_w[:2], dtype=np.float64)
                ))
                return state, _g_v6_target_distance
            _g_v6_control_boundary = bool(local_step == 0 or sim_step % DECIMATION == 0)
            if _g_v6_decision.mode == "BRAKE":
                if not _g_v6_control_boundary:
                    _g_v6_decision = _G_V6_SUPERVISOR.request_stop(
                        _g_v6_next_step,
                        _g_v6_current_witness,
                        "V6_BRAKE_REQUESTED_BETWEEN_CONTROL_UPDATES",
                    )
                    _G_V6_RECORD(_g_v6_decision)
                    external_safety_reason = "V6_" + _g_v6_decision.reason
                    conservative_stop = True
                    _g_v6_target_distance = float(np.linalg.norm(
                        np.asarray(target_xy, dtype=np.float64)
                        - np.asarray(state.base_pos_w[:2], dtype=np.float64)
                    ))
                    return state, _g_v6_target_distance
                command = _G_V6_APPLY(command, _g_v6_decision)
""",
    )
    source = replace_once(
        source,
        "        _g_tube_entry_reason = _G_ACTION_GUARD.begin(_G_AUDIT.last_sample)\n"
        "        if _g_tube_entry_reason is not None and external_safety_reason is None:\n"
        "            external_safety_reason = _g_tube_entry_reason\n",
        """        _g_tube_entry_reason = _G_ACTION_GUARD.begin(_G_AUDIT.last_sample)
        if _g_tube_entry_reason is not None and external_safety_reason is None:
            external_safety_reason = _g_tube_entry_reason
        if not _G_V6_SUPERVISOR.active:
            if _G_AUDIT.last_sample is None:
                raise RuntimeError("V6 entry requires an actual boundary sample")
            _G_V6_SUPERVISOR.begin(
                int(_G_AUDIT.last_sample["physics_step"]),
                _G_V6_WITNESS(_G_AUDIT.last_sample),
            )
""",
    )
    source = replace_once(
        source,
        "                _g_tube_reason = _G_ACTION_GUARD.observe(_G_AUDIT.last_sample)\n"
        "                if _g_tube_reason is not None and external_safety_reason is None:\n"
        "                    external_safety_reason = _g_tube_reason\n",
        """                _g_tube_reason = _G_ACTION_GUARD.observe(_G_AUDIT.last_sample)
                if _g_tube_reason is not None and external_safety_reason is None:
                    external_safety_reason = _g_tube_reason
                if _G_V6_SUPERVISOR.active:
                    _g_v6_after = _G_V6_SUPERVISOR.after_step(
                        int(_G_AUDIT.last_sample["physics_step"]),
                        _G_V6_WITNESS(_G_AUDIT.last_sample),
                    )
                    _G_V6_RECORD(_g_v6_after)
                    if _g_v6_after.mode in {"STOP", "SAFE_RETRACT"} and external_safety_reason is None:
                        external_safety_reason = "V6_" + _g_v6_after.reason
                        conservative_stop = True
""",
    )
    source = replace_once(
        source,
        "        _G_WRITE_JSON(_G_RUN_ROOT/'g_action_tube_runtime.json',_G_ACTION_GUARD.evidence())\n",
        "        _G_WRITE_JSON(_G_RUN_ROOT/'g_action_tube_runtime.json',_G_ACTION_GUARD.evidence())\n"
        "        _G_WRITE_JSON(_G_RUN_ROOT/'g_action_executor_v6.json',_G_V6_SUPERVISOR.evidence())\n",
    )
    ast.parse(source)
    return source


def source_sha256(path: Path = SOURCE) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dry_run_proof(source: str) -> dict[str, object]:
    patched = patch_source_v6(source)
    contract = json.loads((ROOT / "outputs/round4070_comprehensive_traversability/g1_action_contract_v2.json").read_text(encoding="utf-8-sig"))
    validate_contract(contract)
    bindings = build_v6_runtime_bindings(contract)
    return {
        "schema": V6_SCHEMA,
        "protocol_id": V6_PROTOCOL_ID,
        "status": V6_EXECUTION_STATUS,
        "original_source": str(SOURCE),
        "original_source_sha256": source_sha256(),
        "patched_source_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
        "source_edit_mode": "in_memory_only",
        "v5_source_file_unchanged": source_sha256() == hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "unchanged_contract": {
            "tube_bounds_y_m": [-0.35, 0.35],
            "tube_bounds_z_m": [0.45, 1.25],
            "max_step_x_m": 0.020,
            "sample_dt_s": 0.005,
        },
        "hard_stop": True,
        "safe_retract_request": True,
        "braking_profile": "DRAFT_ONLY_REQUIRES_REPRESENTATIVE_PHYSICS_CERTIFICATION",
        "runtime_binding_keys": sorted(bindings),
        "runtime_binding_contract_protocol": contract["primary_protocol"],
        "non_dry_run_admission": "REJECTED_UNTIL_INDEPENDENT_APPROVAL_RECORD",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = SOURCE.read_text(encoding="utf-8-sig")
    proof = build_dry_run_proof(source)
    if not args.dry_run:
        raise SystemExit(
            "V6_COLLECTION_NOT_ADMITTED: use --dry-run only until native contact-plane "
            "certification, representative physics validation, and independent approval are recorded"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proof, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
