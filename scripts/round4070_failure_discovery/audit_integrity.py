"""Audit the frozen Round 4070-B manifest and generated evidence tree."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "configs" / "round4070_failure_discovery" / "PREREGISTERED_MANIFEST.json"
DEFAULT_RESEARCH = REPO / "research" / "round4070_failure_discovery"
DEFAULT_TRIAL_ROOT = REPO / "outputs" / "round4070_failure_discovery" / "trials"
DEFAULT_LEDGER = REPO / "outputs" / "round4070_failure_discovery" / "launch_ledger.json"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def check(condition: bool, message: str, failures: list[str], warnings: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--research", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--trial-root", type=Path, default=DEFAULT_TRIAL_ROOT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []
    manifest = read_json(args.manifest, {}) or {}
    plans = list(manifest.get("trials", []))
    plan_ids = [str(plan.get("trial_id")) for plan in plans]
    plan_id_set = set(plan_ids)
    check(args.manifest.is_file(), "manifest is missing", failures, warnings)
    check(len(plans) == int(manifest.get("trial_count_frozen", -1)), "manifest trial count disagrees with frozen count", failures, warnings)
    check(len(plans) == int(manifest.get("new_g1_trial_hard_limit", -1)), "manifest trial count disagrees with hard budget", failures, warnings)
    check(len(plan_ids) == len(plan_id_set), "manifest contains duplicate trial IDs", failures, warnings)

    trial_dirs = {path.name for path in args.trial_root.iterdir() if path.is_dir()} if args.trial_root.is_dir() else set()
    extra_dirs = sorted(trial_dirs - plan_id_set)
    check(not extra_dirs, f"trial directories outside manifest: {extra_dirs}", failures, warnings)

    ledger = read_json(args.ledger, {}) or {}
    ledger_ids = [str(value) for value in ledger.get("trial_ids", [])]
    check(len(ledger_ids) == len(set(ledger_ids)), "launch ledger contains duplicate IDs", failures, warnings)
    check(set(ledger_ids) <= plan_id_set, "launch ledger contains an ID outside the manifest", failures, warnings)
    check(int(ledger.get("launched_count", -1)) == len(ledger_ids), "launch ledger count disagrees with IDs", failures, warnings)
    check(len(ledger_ids) <= int(manifest.get("new_g1_trial_hard_limit", 0)), "launch ledger exceeds hard G1 budget", failures, warnings)

    launched_ids: list[str] = []
    valid_ids: list[str] = []
    incomplete_ids: list[str] = []
    invalid_ids: list[str] = []
    required_common = (
        "command.json",
        "trial_metadata.json",
        "watchdog_summary.json",
        "resource_summary.json",
        "summary.json",
        "events.json",
        "validation_result.json",
    )
    for trial_id in plan_ids:
        path = args.trial_root / trial_id
        launched = (path / "command.json").is_file()
        if not launched:
            continue
        launched_ids.append(trial_id)
        missing_common = [name for name in required_common if not (path / name).is_file()]
        check(not missing_common, f"{trial_id} missing common evidence: {missing_common}", failures, warnings)
        summary = read_json(path / "summary.json")
        validation = read_json(path / "validation_result.json")
        metadata = read_json(path / "trial_metadata.json", {}) or {}
        check(metadata.get("exit_code") is not None, f"{trial_id} missing recorded exit code", failures, warnings)
        check(isinstance(validation, dict), f"{trial_id} missing validation result", failures, warnings)
        if not summary:
            incomplete_ids.append(trial_id)
            continue
        runtime_valid = summary.get("runtime_status") == "RUNTIME_PASS_FINALIZE_REACHED"
        if runtime_valid:
            valid_ids.append(trial_id)
            missing_valid = [name for name in ("events.json", "deployment_observation.csv", "telemetry.csv") if not (path / name).is_file()]
            check(not missing_valid, f"{trial_id} runtime-pass summary missing evidence: {missing_valid}", failures, warnings)
        else:
            invalid_ids.append(trial_id)
            derived_events = read_json(path / "events.json", {}) or {}
            check(
                derived_events.get("schema") == "lift-the-curtain.round4070-derived-execution-events.v1",
                f"{trial_id} invalid trial lacks explicit derived execution-events ledger",
                failures,
                warnings,
            )

    check(set(launched_ids) == set(ledger_ids), "launch ledger and command directories disagree", failures, warnings)
    if incomplete_ids:
        warnings.append(f"launched directories without summary.json: {incomplete_ids}")

    physics_path = args.research / "PHYSICS_GATE_RESULTS.csv"
    physics_rows: list[dict[str, str]] = []
    if physics_path.is_file():
        import csv

        with physics_path.open(encoding="utf-8-sig", newline="") as handle:
            physics_rows = list(csv.DictReader(handle))
    else:
        failures.append("PHYSICS_GATE_RESULTS.csv is missing")
    gate_ids = [str(row.get("gate_id")) for row in physics_rows]
    check(len(gate_ids) == len(set(gate_ids)), "physics gate results contain duplicate gate IDs", failures, warnings)
    check(len(physics_rows) == 14, "physics gate result row count is not 14", failures, warnings)
    failed_gates = sorted(str(row.get("gate_id")) for row in physics_rows if row.get("gate_status") != "PASS_PHYSICS_GATE")
    check(failed_gates == ["S1D_PARTIAL"], f"unexpected physics gate failures: {failed_gates}", failures, warnings)

    required_outputs = (
        "ANALYSIS_RULES.json",
        "CAMPAIGN_ANALYSIS.json",
        "EVIDENCE_MATRIX.csv",
        "MATCHED_COUNTEREXAMPLES.csv",
        "LEFT_RIGHT_COMPARISON.csv",
        "JUDGMENT_ERROR_CASES.csv",
        "RESOURCE_AUDIT.json",
        "TRIAL_RECORDS.json",
    )
    missing_outputs = [name for name in required_outputs if not (args.research / name).is_file()]
    check(not missing_outputs, f"analysis outputs missing: {missing_outputs}", failures, warnings)

    result = {
        "schema": "lift-the-curtain.round4070-failure-discovery-integrity-audit.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "manifest_path": str(args.manifest),
        "trial_root": str(args.trial_root),
        "manifest_trial_count": len(plans),
        "manifest_unique_trial_count": len(plan_id_set),
        "ledger_launched_count": len(ledger_ids),
        "command_directory_count": len(launched_ids),
        "runtime_valid_count": len(valid_ids),
        "runtime_invalid_count": len(invalid_ids),
        "incomplete_launched_count": len(incomplete_ids),
        "not_started_count": len(plans) - len(launched_ids),
        "physics_gate_count": len(physics_rows),
        "physics_failed_gate_ids": failed_gates,
        "required_output_missing": missing_outputs,
        "warnings": warnings,
        "failures": failures,
        "integrity_pass": not failures,
    }
    out_json = args.research / "INTEGRITY_AUDIT.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md = args.research / "INTEGRITY_AUDIT.md"
    out_md.write_text(
        "# Round 4070-B integrity audit\n\n"
        f"- Result: {'PASS' if not failures else 'FAIL'}\n"
        f"- Manifest: {len(plans)} rows, {len(launched_ids)} launched, {len(valid_ids)} runtime-valid, {len(invalid_ids)} runtime-invalid, {len(plans) - len(launched_ids)} not started\n"
        f"- Physics gates: {len(physics_rows)} rows; failed gates: {', '.join(failed_gates) if failed_gates else 'none'}\n"
        f"- Warnings: {len(warnings)}\n"
        f"- Failures: {len(failures)}\n\n"
        "The audit distinguishes an unlaunched manifest row from a launched runtime-invalid row. "
        "Runtime-invalid trials are excluded from physical judgment-error denominators.\n",
        encoding="utf-8",
    )
    print(json.dumps({"integrity_pass": not failures, "launched": len(launched_ids), "valid": len(valid_ids), "invalid": len(invalid_ids), "not_started": len(plans) - len(launched_ids)}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
