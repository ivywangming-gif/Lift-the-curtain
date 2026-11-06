"""Durable G1 launch accounting; metadata only, no process or simulator launch.

Frozen history, this tracked ledger, and local launch files form one union.
The path identifies an attempt and its original SHA may never change. A known
attempt with absent launch bytes still costs budget and withholds its plan.
The collector must already hold its simulation resource lock before appending
the just-written launch JSON, and must append before starting the guard.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import uuid

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "research/round4070_comprehensive_traversability/G1_LAUNCH_LEDGER.json"
LAUNCH_PREFIX = "outputs/round4070_comprehensive_traversability/trials"
SCHEMA = "round4070-g.g1_launch_ledger.v1"
LEGACY_PROTOCOL = "LEGACY_FROZEN_PRIOR_UNSPECIFIED"
UNSPECIFIED_PROTOCOL = "LOCAL_G1_PROTOCOL_UNSPECIFIED_NOT_CURRENT_PLAN"
RULE = "Union frozen historical, durable, and local attempt paths; same path requires identical original SHA; missing raw never refunds budget or permits replay."
_ENTRY_KEYS = {"launch_file", "sha256", "primary_protocol", "trial_id", "attempt_id", "launch_utc", "evidence_origin"}


class BudgetLedgerError(RuntimeError):
    pass


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise BudgetLedgerError("LEDGER_OR_LAUNCH_UNREADABLE: " + str(path)) from exc


def _relative_launch(path, repo_root, *, allow_absolute=False):
    """Canonical slash spelling; Windows identity is case-insensitive."""
    root = Path(repo_root).resolve()
    raw = str(path).replace("\\", "/")
    if allow_absolute and Path(path).is_absolute():
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(root):
            raise BudgetLedgerError("LAUNCH_PATH_ESCAPES_REPOSITORY")
        raw = resolved.relative_to(root).as_posix()
    parts = PurePosixPath(raw).parts
    if (not raw or raw.startswith("/") or ":" in raw or ".." in parts
            or len(parts) != 6 or "/".join(parts[:3]).casefold() != LAUNCH_PREFIX.casefold()
            or parts[-1].casefold() != "campaign_launch.json"
            or not re.fullmatch(r"attempt[0-9]{2,}", parts[-2], re.IGNORECASE)
            or int(parts[-2][7:]) < 1):
        raise BudgetLedgerError("INVALID_REPOSITORY_RELATIVE_LAUNCH_PATH: " + raw)
    canonical = "/".join(parts)
    # A symlink/junction is never allowed to turn an in-repository identity
    # into an external file, including when the target is currently missing.
    if not (root / canonical).resolve().is_relative_to(root):
        raise BudgetLedgerError("LAUNCH_PATH_RESOLVES_OUTSIDE_REPOSITORY")
    return canonical


def _key(entry):
    return entry["launch_file"].casefold()


def _validate_entry(entry, repo_root):
    if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
        raise BudgetLedgerError("INVALID_DURABLE_ENTRY_FIELDS")
    relative = _relative_launch(entry["launch_file"], repo_root)
    if relative != entry["launch_file"]:
        raise BudgetLedgerError("DURABLE_ENTRY_PATH_NOT_CANONICAL")
    parts = PurePosixPath(relative).parts
    if entry["trial_id"] != parts[-3] or entry["attempt_id"] != parts[-2]:
        raise BudgetLedgerError("ENTRY_TRIAL_OR_ATTEMPT_PATH_MISMATCH")
    if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
        raise BudgetLedgerError("INVALID_LAUNCH_SHA256")
    if not isinstance(entry["primary_protocol"], str) or not entry["primary_protocol"]:
        raise BudgetLedgerError("ENTRY_PROTOCOL_MISSING")
    if entry["evidence_origin"] not in {"LOCAL_LAUNCH_JSON", "FROZEN_HISTORY_WITHOUT_LOCAL_RAW"}:
        raise BudgetLedgerError("INVALID_ENTRY_EVIDENCE_ORIGIN")
    if entry["launch_utc"] is not None:
        try:
            timestamp = datetime.fromisoformat(entry["launch_utc"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError("timezone missing")
        except (AttributeError, TypeError, ValueError) as exc:
            raise BudgetLedgerError("INVALID_LAUNCH_TIMESTAMP") from exc
    return dict(entry)


def load_launch_ledger(ledger_path=DEFAULT_LEDGER, *, repo_root=ROOT, allow_missing=False):
    path = Path(ledger_path)
    if not path.exists():
        if allow_missing:
            return dict(schema=SCHEMA, budget_rule=RULE, entries=[], entry_count=0, entries_sha256=_digest([]))
        raise BudgetLedgerError("DURABLE_LAUNCH_LEDGER_MISSING: " + str(path))
    value = _read(path)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("budget_rule") != RULE:
        raise BudgetLedgerError("DURABLE_LAUNCH_LEDGER_SCHEMA_MISMATCH")
    entries = value.get("entries")
    if not isinstance(entries, list) or value.get("entry_count") != len(entries) or value.get("entries_sha256") != _digest(entries):
        raise BudgetLedgerError("DURABLE_LAUNCH_LEDGER_COUNT_OR_DIGEST_MISMATCH")
    seen = set()
    for entry in entries:
        _validate_entry(entry, repo_root)
        if _key(entry) in seen:
            raise BudgetLedgerError("DUPLICATE_DURABLE_ATTEMPT_PATH")
        seen.add(_key(entry))
    return value


def _historical_entries(manifest, repo_root):
    result = []
    for item in manifest.get("prior_actual_g1_launches", []):
        path = _relative_launch(item["launch_file"], repo_root)
        parts = PurePosixPath(path).parts
        result.append(_validate_entry(dict(launch_file=path, sha256=item["sha256"],
            primary_protocol=LEGACY_PROTOCOL, trial_id=parts[-3], attempt_id=parts[-2],
            launch_utc=None, evidence_origin="FROZEN_HISTORY_WITHOUT_LOCAL_RAW"), repo_root))
    return result


def _local_entry(path, manifest, repo_root):
    relative = _relative_launch(path, repo_root, allow_absolute=True)
    actual = Path(repo_root) / relative
    content = actual.read_bytes()
    try:
        payload = json.loads(content.decode("utf-8-sig"))
    except (UnicodeError, ValueError) as exc:
        raise BudgetLedgerError("LOCAL_LAUNCH_JSON_INVALID: " + relative) from exc
    parts = PurePosixPath(relative).parts
    if payload.get("kind") != "G1" or payload.get("trial_id") != parts[-3]:
        raise BudgetLedgerError("LOCAL_G1_KIND_OR_TRIAL_PATH_MISMATCH: " + relative)
    current = {p["trial_id"]: p for p in manifest.get("trial_plans", [])}
    prior = {_key(e) for e in _historical_entries(manifest, repo_root)}
    if parts[-3] in current:
        declared = [current[parts[-3]].get("primary_protocol"), manifest.get("g1_primary_protocol"), manifest.get("primary_protocol")]
        declared = [value for value in declared if value is not None]
        if not declared or any(not isinstance(value, str) or not value for value in declared):
            raise BudgetLedgerError("CURRENT_MANIFEST_PROTOCOL_MISSING")
        if len(set(declared)) != 1:
            raise BudgetLedgerError("CURRENT_PLAN_AND_MANIFEST_PROTOCOL_CONFLICT")
        protocol = declared[0]
        if payload.get("primary_protocol", protocol) != protocol:
            raise BudgetLedgerError("LOCAL_LAUNCH_PROTOCOL_CONFLICT")
    elif relative.casefold() in prior:
        protocol = LEGACY_PROTOCOL
    else:
        protocol = payload.get("primary_protocol") or UNSPECIFIED_PROTOCOL
    return _validate_entry(dict(launch_file=relative, sha256=hashlib.sha256(content).hexdigest(),
        primary_protocol=protocol, trial_id=parts[-3], attempt_id=parts[-2],
        launch_utc=payload.get("utc"), evidence_origin="LOCAL_LAUNCH_JSON"), repo_root)


def _merge(existing, incoming):
    """Preserve original entry bytes/metadata; only unseen paths append."""
    result = [dict(e) for e in existing]
    by_path = {_key(e): e for e in result}
    for entry in incoming:
        old = by_path.get(_key(entry))
        if old is not None:
            if old["sha256"] != entry["sha256"]:
                raise BudgetLedgerError("SAME_ATTEMPT_PATH_DIFFERENT_LAUNCH_SHA256: " + entry["launch_file"])
            if (old["trial_id"].casefold(), old["attempt_id"].casefold()) != (entry["trial_id"].casefold(), entry["attempt_id"].casefold()):
                raise BudgetLedgerError("SAME_ATTEMPT_IDENTITY_METADATA_CONFLICT")
            unspecified = {LEGACY_PROTOCOL, UNSPECIFIED_PROTOCOL}
            if old["primary_protocol"] != entry["primary_protocol"] and not ({old["primary_protocol"], entry["primary_protocol"]} & unspecified):
                raise BudgetLedgerError("SAME_ATTEMPT_PROTOCOL_CONFLICT")
            continue
        result.append(dict(entry)); by_path[_key(entry)] = result[-1]
    return result


def launch_ledger_summary(manifest, *, repo_root=ROOT, ledger_path=DEFAULT_LEDGER,
                          local_launch_files=None, allow_missing_ledger=False):
    """Read-only union. Missing ledger is fatal except explicit bootstrap."""
    root = Path(repo_root).resolve()
    ledger = load_launch_ledger(ledger_path, repo_root=root, allow_missing=allow_missing_ledger)
    files = list(local_launch_files) if local_launch_files is not None else sorted((root / LAUNCH_PREFIX).glob("*/*/campaign_launch.json"))
    local = [_local_entry(path, manifest, root) for path in files]
    entries = _merge(ledger["entries"], _historical_entries(manifest, root))
    entries = _merge(entries, sorted(local, key=_key))
    local_by_key = {_key(e): e for e in local}
    missing = []
    for entry in entries:
        path = root / entry["launch_file"]
        if path.is_file():
            if _sha(path) != entry["sha256"]:
                raise BudgetLedgerError("KNOWN_LAUNCH_FILE_SHA256_CHANGED: " + entry["launch_file"])
        elif _key(entry) not in local_by_key:
            missing.append(entry)
    maximum = manifest.get("max_g1_launches")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 0:
        raise BudgetLedgerError("INVALID_FROZEN_LAUNCH_BUDGET")
    return dict(schema="round4070-g.g1_launch_budget_summary.v1", count=len(entries), entries=entries,
        max_g1_launches=maximum, remaining=max(0, maximum-len(entries)), budget_exhausted=len(entries) >= maximum,
        budget_overrun=len(entries) > maximum, local_launch_file_count=len(local_by_key),
        durable_entry_count=len(ledger["entries"]), missing_raw_launches=missing,
        withheld_trial_ids=sorted({e["trial_id"] for e in missing}),
        missing_raw_policy="WITHHOLD_SAME_TRIAL_PLAN_NO_REPLAY_OR_BUDGET_REFUND",
        raw_scope="Presence and SHA of campaign_launch.json only; not trajectory or outcome validity.")


def trial_budget_state(trial_id, summary):
    entries = [e for e in summary["entries"] if e["trial_id"].casefold() == str(trial_id).casefold()]
    missing = [e for e in summary["missing_raw_launches"] if e["trial_id"].casefold() == str(trial_id).casefold()]
    return dict(trial_id=trial_id, known_attempts=len(entries), withhold=bool(missing),
        reason="KNOWN_ATTEMPT_RAW_MISSING_RESTORE_OR_REVIEW_REQUIRED" if missing else None,
        minimum_next_attempt_number=1+max((int(e["attempt_id"][7:]) for e in entries), default=0))


@contextmanager
def _ledger_lock(path):
    """A stale lock is never deleted speculatively; fail closed for review."""
    path = Path(str(path) + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise BudgetLedgerError("DURABLE_LEDGER_LOCK_BUSY: " + str(path)) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(pid=os.getpid(), token=token), handle); handle.flush(); os.fsync(handle.fileno())
        yield
    finally:
        current = _read(path)
        if current.get("token") != token or current.get("pid") != os.getpid():
            raise BudgetLedgerError("DURABLE_LEDGER_LOCK_OWNERSHIP_CHANGED")
        path.unlink()


def _write_preserving_entries(path, value, previous):
    if value["entries"][:len(previous["entries"])] != previous["entries"]:
        raise BudgetLedgerError("EXISTING_DURABLE_ENTRY_MUTATION_FORBIDDEN")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False); handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def sync_launch_ledger(manifest, *, repo_root=ROOT, ledger_path=DEFAULT_LEDGER,
                       local_launch_files=None, initialize=False):
    """Append observed/frozen attempts, never remove or revise an old entry.

    `initialize=True` is only for the explicit first bootstrap. Runtime calls
    must leave it false so accidental deletion cannot silently refund budget.
    Persist this tracked file through the normal commit/push handoff; an
    uncommitted filesystem update cannot protect a separate older checkout.
    """
    if local_launch_files is not None:
        local_launch_files = list(local_launch_files)
    with _ledger_lock(ledger_path):
        previous = load_launch_ledger(ledger_path, repo_root=repo_root, allow_missing=initialize)
        summary = launch_ledger_summary(manifest, repo_root=repo_root, ledger_path=ledger_path,
            local_launch_files=local_launch_files, allow_missing_ledger=initialize)
        # When both sources exist during first bootstrap, retain local UTC and
        # metadata while checking frozen identities against the exact same SHA.
        entries = summary["entries"]
        if not previous["entries"]:
            local_lookup = {}
            files = list(local_launch_files) if local_launch_files is not None else sorted((Path(repo_root) / LAUNCH_PREFIX).glob("*/*/campaign_launch.json"))
            for path in files:
                entry = _local_entry(path, manifest, repo_root); local_lookup[_key(entry)] = entry
            entries = [local_lookup.get(_key(e), e) for e in entries]
        value = dict(schema=SCHEMA, budget_rule=RULE, entries=entries, entry_count=len(entries), entries_sha256=_digest(entries))
        if value != previous:
            _write_preserving_entries(ledger_path, value, previous)
        elif not Path(ledger_path).exists():
            _write_preserving_entries(ledger_path, value, previous)
        return launch_ledger_summary(manifest, repo_root=repo_root, ledger_path=ledger_path, local_launch_files=local_launch_files)


def append_actual_launch(launch_file, *, manifest, repo_root=ROOT, ledger_path=DEFAULT_LEDGER):
    """Called with caller's resource lock held, after JSON and before guard.

    No process is started here. If append fails, the guard must not start.
    A crash after the launch file but before this append remains recoverable
    by the local-file union; a crash after append conservatively costs budget.
    """
    expected = _local_entry(launch_file, manifest, repo_root)
    summary = sync_launch_ledger(manifest, repo_root=repo_root, ledger_path=ledger_path)
    stored = next((e for e in summary["entries"] if _key(e) == _key(expected)), None)
    if stored is None or stored["sha256"] != expected["sha256"]:
        raise BudgetLedgerError("ACTUAL_LAUNCH_NOT_DURABLY_RECORDED")
    if summary["budget_overrun"]:
        raise BudgetLedgerError("LAUNCH_BUDGET_OVERRUN_RECORDED_GUARD_MUST_NOT_START")
    return summary
