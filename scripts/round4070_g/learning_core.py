"""Simulator-free G learning contracts, causal preprocessing and evaluation.

Metadata is used for exclusion, splitting, action contracts and approved training
pairs, never concatenated to model features. LABEL_UNKNOWN is never a third class.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ACTION_CONTRACT = "G_HOLD_PROBE_POSE_FORWARD24_TUBE_V2"
PRIMARY_PROTOCOL = "G1_RNG_V1_BODY_COLLISION_200HZ_ACTION_TUBE_V2"
LABEL_VALIDITY_VERSION = "G_BODY_COLLISION_VALIDITY_V1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted(set().union(*(row.keys() for row in rows)))
    if not fields:
        raise ValueError("Empty CSV needs an explicit schema, not invented results")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_primary_manifest(manifest, verify_sources=True, synthetic_test=False):
    """Never admit old pilots/pending collision cases as supervised or response data."""
    synthetic = manifest.get("data_scope") == "SYNTHETIC_CONTRACT_TEST"
    if synthetic_test:
        if not synthetic or verify_sources:
            raise ValueError("Synthetic contracts require an explicit marker and disabled real-source verification")
    elif synthetic or any(e.get("synthetic") for e in manifest.get("episodes", [])):
        raise ValueError("Synthetic fixtures cannot be admitted as a real G benchmark")
    expected = dict(schema="round4070-g.causal_episodes.v3",primary_protocol=PRIMARY_PROTOCOL,label_validity_version=LABEL_VALIDITY_VERSION)
    if any(manifest.get(key) != value for key,value in expected.items()):
        raise ValueError("Primary G manifest requires V5 effective RNG, every-step native collision audit and frozen action tube")
    for episode in manifest["episodes"]:
        y = int(episode["y"])
        if (episode.get("primary_protocol") != PRIMARY_PROTOCOL or episode.get("label_validity_version") != LABEL_VALIDITY_VERSION or
            episode.get("primary_admission") is not True or episode.get("admission_status") != "PRIMARY_VERIFIED" or
            episode.get("response_history_eligible") is not True or episode.get("supervision_eligible") is not (y >= 0) or
            episode.get("action_contract") != ACTION_CONTRACT):
            raise ValueError(f"Episode is not admitted by the primary G contract: {episode['trial_id']}")
        collision = episode.get("body_collision_audit", {})
        reset = episode.get("reset_randomization_evidence", {})
        contract = episode.get("action_contract_v2")
        contract_hash = episode.get("action_contract_v2_sha256", "")
        if not isinstance(contract, dict) or len(contract_hash) != 64:
            raise ValueError("V5 frozen action contract object/hash missing")
        from g1_action_contract_v2 import validate_contract
        validate_contract(contract, verify_files=verify_sources)
        action_audit = collision.get("action_contract_audit", {})
        if action_audit.get("status") != "VERIFIED":
            raise ValueError("V5 action-contract audit incomplete; UNKNOWN is not a bypass")
        if y >= 0 and (action_audit.get("all_required_steps_valid") is not True or action_audit.get("guard_violation") is not None):
            raise ValueError("Action-tube violations cannot receive supervised PASSABLE/BLOCKED labels")
        if y == 0 and action_audit.get("static_blocking_certificate", {}).get("status") != "VERIFIED":
            raise ValueError("BLOCKED requires an independently verified discrete action impossibility certificate")
        if collision.get("status") != "VERIFIED" or int(collision.get("sample_count", 0)) < 1:
            raise ValueError("Missing VERIFIED sampled actual-body collision evidence; UNKNOWN is not an admission bypass")
        if reset.get("readback_verified") is not True or reset.get("seed") != episode.get("seed"):
            raise ValueError("Missing actual replayable reset RNG readback evidence")
        for record,kind in ((collision,"collision"),(reset,"reset RNG")):
            digest,relative = record.get("source_evidence_sha256", ""),record.get("source_evidence_path", "")
            if not relative or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
                raise ValueError(f"Missing {kind} source evidence path/hash")
            if verify_sources:
                source = (ROOT/relative).resolve()
                if not source.is_relative_to(ROOT) or sha(source) != digest:
                    raise ValueError(f"Changed/invalid {kind} source evidence")
                actual = read_json(source)
                if kind == "collision":
                    if actual.get("schema") != "round4070-g.body_collision_audit.v1" or actual.get("status") != "VERIFIED":
                        raise ValueError("Source collision audit does not support admission")
                    if not all(actual.get(k) is True for k in ("penetration_free","all_robot_collision_bodies","probe_followup_release_complete","finite")):
                        raise ValueError("Source collision audit lacks required validity evidence")
                    if actual.get("action_contract_audit") != action_audit:
                        raise ValueError("Manifest action audit differs from source collision evidence")
                    if (actual.get("initial_state_collision_audited") is not True or actual.get("initial_state_collision_status") != "VERIFIED"
                        or actual.get("geometry_checked_state_count") != actual.get("sample_count", -2)+1):
                        raise ValueError("V5 collision audit must geometrically test reset state plus every executed step")
                    base = (ROOT/record.get("source_files_base", "")).resolve()
                    if not record.get("source_files_base") or not base.is_relative_to(ROOT):
                        raise ValueError("Collision source-file base missing or outside repository")
                    source_files = actual.get("source_files", {})
                    required = {"g_body_collision_capture.json", "g_body_collision_history_index.json", "g_body_collision_initial_state.npz",
                                "g_robot_collision_geometry.json", "g_native_geometry_verified.json", "g_action_tube_runtime.json",
                                "g_trial_metadata.json", "g_outcome.json", "g_runner_contract.json"}
                    if not required <= set(source_files):
                        raise ValueError("V5 native geometry/history/action source hashes incomplete")
                    for name, digest in source_files.items():
                        target = (base/name).resolve()
                        if not target.is_relative_to(base) or sha(target) != digest:
                            raise ValueError("Collision source changed or escaped trial: " + name)
                    from physics_history import verify_history
                    history = verify_history(base, read_json(base/"g_body_collision_capture.json"), read_json(base/"g_robot_collision_geometry.json"))
                    if history["sample_count"] != actual.get("sample_count") or any(source_files.get(n) != h for n,h in history["source_files"].items()):
                        raise ValueError("V5 collision audit does not bind complete step/phase/chunk history")
                    plan = read_json(base/"g_trial_metadata.json")["plan"]
                    if plan.get("primary_protocol") != PRIMARY_PROTOCOL or plan.get("action_contract_v2") != contract or plan.get("action_contract_v2_sha256") != contract_hash:
                        raise ValueError("V5 episode action object/hash differs from executed frozen plan")
                elif actual.get("schema") != "G1_RNG_V1" or actual.get("readback_verified") is not True or actual.get("seed") != episode.get("seed"):
                    raise ValueError("Source RNG audit does not support admission")
                else:
                    base = (ROOT/record.get("source_files_base", "")).resolve()
                    array = (base/actual.get("arrays_file", "MISSING")).resolve()
                    if (not record.get("source_files_base") or not base.is_relative_to(ROOT) or not array.is_relative_to(base)
                        or actual.get("robot_pose_readback_verified") is not True or sha(array) != actual.get("arrays_sha256")
                        or actual.get("spec_sha256") != plan.get("g1_reset_randomization", {}).get("spec_sha256")):
                        raise ValueError("RNG raw reset/readback arrays or executed specification are not bound to evidence")


def load_dataset(path, legacy=False):
    manifest = read_json(path)
    if not legacy:
        validate_primary_manifest(manifest)
    elif manifest.get("action_contract") == ACTION_CONTRACT or any(e.get("primary_protocol") or e.get("action_contract") == ACTION_CONTRACT for e in manifest["episodes"]):
        raise ValueError("--legacy is a separate F-contract diagnostic; G pilots/primary cannot bypass admission through it")
    episodes = manifest["episodes"]
    names = list(manifest["feature_names"])
    forbidden = ("material", "anchor", "wall", "cloth_node", "contact_truth", "clearance", "audit_only", "map/", "goal/")
    if len(names) != len(set(names)) or any(any(f in n.lower() for f in forbidden) for n in names):
        raise ValueError("Duplicate or forbidden deployment feature")
    if len({e["trial_id"] for e in episodes}) != len(episodes):
        raise ValueError("Duplicate episodes")
    raw, times, hashes = [], [], {}
    for e in episodes:
        y = int(e["y"])
        if y not in (-1, 0, 1):
            raise ValueError("Unknown labels must be -1")
        if not legacy and y >= 0 and e.get("action_contract") != ACTION_CONTRACT:
            raise ValueError(f"Incompatible supervised action contract: {e['trial_id']}")
        if not legacy and y >= 0 and not e.get("followup_action_id"):
            raise ValueError("Supervision requires the specified subsequent action")
        p = Path(e["npz_path"])
        p = p if p.is_absolute() else ROOT / p
        with np.load(p, allow_pickle=False) as d:
            ns = d["feature_names"].astype(str).tolist()
            x, t = d["seq"].astype(np.float32), d["time"].astype(np.float64)
        if ns != names or x.shape != (len(t), len(names)) or len(t) < 2:
            raise ValueError(f"Invalid sensor schema: {p}")
        if not np.isfinite(x).all() or not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
            raise ValueError(f"Nonfinite/nonmonotone observations: {p}")
        raw.append(x)
        times.append(t - t[0])
        hashes[e["trial_id"]] = sha(p)
    return manifest, episodes, names, raw, times, hashes


def split_indices(split, episodes):
    lookup = {e["trial_id"]: i for i, e in enumerate(episodes)}
    roles = {role: [lookup[t] for t in split.get(role, [])] for role in ("train", "calibration", "test", "confirmation")}
    for a, b in (("train", "calibration"), ("train", "test"), ("calibration", "test"),
                 ("train", "confirmation"), ("calibration", "confirmation"), ("test", "confirmation")):
        if set(roles[a]) & set(roles[b]):
            raise ValueError("An episode crosses split roles")
    if any(episodes[i].get("experiment_role") == "confirmation" for r in ("train", "calibration", "test") for i in roles[r]):
        raise ValueError("Reserved confirmation cannot enter train/calibration/main test")
    kind = split.get("split_type", split.get("kind", "configuration"))
    # IID new seeds, hand and posture diagnostics may share a physical setting.
    if kind not in ("iid_seed", "cross_hand", "posture", "legacy_side"):
        groups = {r: {episodes[i]["group_id"] for i in ix} for r, ix in roles.items()}
        if groups["train"] & (groups["calibration"] | groups["test"] | groups["confirmation"]) or groups["calibration"] & (groups["test"] | groups["confirmation"]):
            raise ValueError("Configuration holdout contains shared physical groups")
    return roles


def frozen_label_budget(episodes, train, split_id, fraction):
    """Nested label-stratified fixed-hash subsets; model seeds never select IDs."""
    if fraction not in (.25, .5, 1.):
        raise ValueError("Only preregistered label fractions .25/.5/1 are allowed")
    selected = []
    for cls in (0, 1):
        candidates = [episodes[i]["trial_id"] for i in train if int(episodes[i]["y"]) == cls]
        ranked = sorted(candidates, key=lambda tid: hashlib.sha256(f"ROUND4070_G_LABEL_BUDGET_V1|{split_id}|{tid}".encode()).hexdigest())
        selected.extend(ranked[:math.floor(fraction*len(ranked))])
    selected = sorted(selected)
    original = {episodes[i]["trial_id"]: int(episodes[i]["y"]) for i in train if int(episodes[i]["y"]) >= 0}
    core_ids = {episodes[i]["trial_id"] for i in train if not episodes[i].get("is_control")}
    return dict(label_fraction=fraction, selected_train_label_ids=selected,
                hidden_train_label_ids=sorted(set(original)-set(selected)), original_train_labels_audit_only=original,
                original_labelled_train_n=len(original), selected_labelled_train_n=len(selected),
                selected_class_counts={str(c): sum(original[tid] == c for tid in selected) for c in (0, 1)},
                selected_core_class_counts={str(c): sum(original[tid] == c and tid in core_ids for tid in selected) for c in (0, 1)},
                scope="all original known train labels, including controls; floor fraction separately per class; train histories retained",
                selection="nested deterministic SHA256 rank within train label strata; no model seed, test, calibration, or model predictions")


def apply_label_budget(episodes, train, budget):
    selected = set(budget["selected_train_label_ids"])
    original = {episodes[i]["trial_id"]: int(episodes[i]["y"]) for i in train if int(episodes[i]["y"]) >= 0}
    if original != budget["original_train_labels_audit_only"] or not selected.issubset(original):
        raise ValueError("Frozen label budget no longer matches this training split")
    result = [dict(e) for e in episodes]
    for i in train:
        if result[i]["trial_id"] in original and result[i]["trial_id"] not in selected:
            result[i].update(y=-1, label="LABEL_UNKNOWN", evaluation_label="UNKNOWN", supervision_eligible=False)
    return result


def causal_rows(x, t, cutoff=2.0, max_steps=128):
    """Select actual observations, never synthesize measurements or use final length."""
    idx = np.flatnonzero(t <= cutoff + 1e-9)
    if len(idx) < 2:
        raise ValueError("Fewer than two real observations before fixed cutoff")
    if len(idx) > max_steps:
        grid = np.linspace(0, cutoff, max_steps)
        idx = np.unique(np.clip(np.searchsorted(t, grid, side="right") - 1, 0, idx[-1]))
    return x[idx].copy(), t[idx].copy()


def time_or_progress_channel(name):
    block = name.lower().split("/")[0]
    return block in {"elapsed", "elapsed_s", "time_since_probe", "dt", "real_dt", "dt_s", "command_progress", "command_progress_m", "probe_command_progress", "probe_progress"}


def history_summary_channel(name):
    value = name.lower()
    return time_or_progress_channel(name) or any(token in value for token in ("hand_delta_initial", "stall", "command_path", "cumulative", "elapsed", "command_progress"))


def temporal_extras(x, t, names):
    """Causal commanded progress, real dt, tracking and measured stall (no truth)."""
    cols = [t[:, None], np.diff(t, prepend=t[0])[:, None]]
    extra = ["derived/elapsed_s", "derived/dt_s"]
    lookup = {n: i for i, n in enumerate(names)}
    q_names = [n for n in names if n.startswith("q/") and "q_cmd/" + n[2:] in lookup]
    if q_names:
        qi = [lookup[n] for n in q_names]
        ci = [lookup["q_cmd/" + n[2:]] for n in q_names]
        cols.append(x[:, ci] - x[:, qi])
        extra += ["derived/tracking/" + n[2:] for n in q_names]
        step = np.linalg.norm(np.diff(x[:, ci], axis=0, prepend=x[:1, ci]), axis=1)
        cols.append(np.cumsum(step)[:, None])
        extra.append("derived/command_path_rad")
    for side in ("left", "right"):
        key = side + "_hand_delta_initial_b/x"
        if key in lookup:
            progress = x[:, lookup[key]]
            dt = np.diff(t)
            velocity = np.divide(np.diff(progress), dt, out=np.zeros_like(dt), where=dt > 0)
            stall = np.r_[0.0, np.cumsum(dt * (np.abs(velocity) <= .004))]
            cols.append(stall[:, None])
            extra.append("derived/" + side + "_stall_s")
    return np.concatenate(cols, axis=1).astype(np.float32), extra


def remap_active_hand(x, names, side):
    """Exact name-based active/opposite permutation, NOT a geometric mirror.

    Coordinates and signed joint values stay in their true body/axis convention.
    The side indicator remains available. No mirrored labels are generated.
    """
    if side not in ("left", "right"):
        raise ValueError("Unknown probe hand")
    if side == "right":
        return x.copy()
    lookup = {n: i for i, n in enumerate(names)}
    permutation = []
    for name in names:
        counterpart = name.replace("left_", "__other_").replace("right_", "left_").replace("__other_", "right_")
        if counterpart != name and counterpart not in lookup:
            raise ValueError(f"Incomplete bilateral name mapping: {name}")
        permutation.append(lookup.get(counterpart, lookup[name]))
    return x[:, permutation].copy()


def fit_residual(raw, times, train, names, episodes):
    targets = [i for i, n in enumerate(names) if n.startswith(("q/", "effort/"))]
    state = [i for i, n in enumerate(names) if n.startswith(("q/", "qdot/"))]
    cmd = [i for i, n in enumerate(names) if n.startswith("q_cmd/")]
    normal = [i for i in train if episodes[i].get("is_control") and int(episodes[i]["y"]) == 1]
    if not normal or not targets or not state or not cmd:
        raise ValueError("RESIDUAL_REQUIRES_TRAINING_EMPTY_CONTROLS_AND_STATE_COMMAND_CHANNELS")
    def inputs(x, t):
        idx = np.maximum(0, np.searchsorted(t, t - .4, side="right") - 1)
        return np.concatenate((x[idx][:, state], x[:, cmd]), axis=1)
    valid = [t >= .4 for t in times]
    aa = [inputs(raw[i], times[i])[valid[i]] for i in normal]
    bb = [raw[i][valid[i]][:, targets] for i in normal]
    if not sum(map(len, aa)):
        raise ValueError("No normal-response observations after state exclusion")
    a, b = np.concatenate(aa), np.concatenate(bb)
    mean, scale = a.mean(0), np.maximum(a.std(0), 1e-4)
    z = np.column_stack(((a - mean) / scale, np.ones(len(a))))
    penalty = np.eye(z.shape[1]) * 10.0
    penalty[-1, -1] = 1e-6
    weight = np.linalg.solve(z.T @ z + penalty, z.T @ b)
    expanded = []
    for i, x in enumerate(raw):
        prediction = np.column_stack(((inputs(x, times[i]) - mean) / scale, np.ones(len(x)))) @ weight
        residual = x[:, targets] - prediction
        residual[~valid[i]] = 0.0
        expanded.append(np.concatenate((x, residual), 1).astype(np.float32))
    audit = dict(normal_train_ids=[episodes[i]["trial_id"] for i in normal], mean=mean.tolist(), scale=scale.tolist(),
                 weight=weight.tolist(), targets=targets, state=state, command=cmd, state_exclusion_s=.4,
                 warmup_residual="zero; warmup excluded from fit", indirect_whole_body_information=True)
    return expanded, names + ["residual/" + names[i] for i in targets], audit


def prepare_features(raw, times, names, episodes, train, method, max_steps=128, cutoff=2., normalization_override=None):
    duration = .5 if method == "GRUPrefix05" else 1.0 if method == "GRUPrefix10" else cutoff
    selected = [causal_rows(x, t, duration, max_steps) for x, t in zip(raw, times)]
    xs, ts = [a for a, b in selected], [b for a, b in selected]
    if method in ("GRU10Hz", "GRU20Hz"):
        rate = 10 if method == "GRU10Hz" else 20
        downsampled = []
        for x, t in zip(xs, ts):
            grid = np.arange(0., duration+1e-9, 1./rate)
            # Unique source indices prevent repeats being called new information.
            ix = np.unique(np.maximum(0, np.searchsorted(t, grid, side="right")-1))
            downsampled.append((x[ix], t[ix]))
        xs, ts = [a for a, b in downsampled], [b for a, b in downsampled]
    # Plain temporal comparators have identical legal sensor inputs, but remove
    # explicit time/process channels. Derived variants add these causally below.
    progress_columns = [j for j, n in enumerate(names) if n.lower().split("/")[0] in {"command_progress", "command_progress_m", "probe_command_progress", "probe_progress"}]
    progress_values = [x[:, progress_columns].copy() for x in xs]
    progress_names = [names[j] for j in progress_columns]
    keep = [j for j, n in enumerate(names) if not time_or_progress_channel(n)]
    names = [names[j] for j in keep]
    xs = [x[:, keep] for x in xs]
    residual_audit = None
    if method in ("ResidualGRU", "OldM2"):
        xs, names, residual_audit = fit_residual(xs, ts, train, names, episodes)
    if method in ("GRUCommand", "GRUTimeProgress"):
        enriched = [temporal_extras(x, t, names) for x, t in zip(xs, ts)]
        if method == "GRUCommand":
            add = [j for j, n in enumerate(enriched[0][1]) if n.startswith("derived/tracking/") or n == "derived/command_path_rad"]
        else:
            add = list(range(len(enriched[0][1])))
        new_names = [enriched[0][1][j] for j in add]
        xs = [np.concatenate((x, value[:, add]), 1) for x, (value, _) in zip(xs, enriched)]
        names += new_names
        if progress_names:
            xs = [np.concatenate((x, progress), 1) for x, progress in zip(xs, progress_values)]
            names += progress_names
    if method == "GRUHandRemap":
        xs = [remap_active_hand(x, names, e.get("probe_side", e.get("side"))) for x, e in zip(xs, episodes)]
    arm = [j for j, n in enumerate(names) if any(s in n for s in ("shoulder", "elbow", "wrist", "hand_", "probe_side")) or n.startswith("followup_")]
    body = [j for j in range(len(names)) if j not in arm]
    if method in ("GRUArm", "ResidualGRU"):
        xs, names = [x[:, arm] for x in xs], [names[j] for j in arm]
        arm, body = list(range(len(names))), []
    if method == "Instantaneous":
        keep = [j for j, n in enumerate(names) if not history_summary_channel(n)]
        xs, names = [x[-1:, keep] for x in xs], [names[j] for j in keep]
        ts = [t[-1:] for t in ts]
    # All preprocessing fitting uses only raw valid rows from training episodes.
    observations = np.concatenate([xs[i] for i in train])
    mean, scale = observations.mean(0), np.maximum(observations.std(0), 1e-3)
    if normalization_override is not None:
        if names != normalization_override["feature_names"]:
            raise ValueError("Frozen preprocessing feature names differ")
        mean = np.asarray(normalization_override["mean"], dtype=np.float32)
        scale = np.asarray(normalization_override["scale"], dtype=np.float32)
    norm = [np.clip((x - mean) / scale, -10, 10).astype(np.float32) for x in xs]
    lengths = np.asarray([len(x) for x in norm], dtype=np.int64)
    padded = np.zeros((len(norm), max_steps, len(names)), dtype=np.float32)
    real_time = np.zeros((len(norm), max_steps), dtype=np.float32)
    for i, (x, t) in enumerate(zip(norm, ts)):
        padded[i, :len(x)] = x
        real_time[i, :len(t)] = t
    audit = dict(feature_names=names, normalization_train_ids=[episodes[i]["trial_id"] for i in train],
                 mean=mean.tolist(), scale=scale.tolist(), cutoff_s=duration, max_steps=max_steps,
                 resampling="actual causal observations; unique past-index decimation only if > max_steps",
                 rate_ablation_hz=10 if method == "GRU10Hz" else 20 if method == "GRU20Hz" else None,
                 removed_command_progress_channels_count=len(progress_columns),
                 acquired_sample_counts=[len(t) for t in ts], observed_durations_s=[float(t[-1]) for t in ts],
                 early_stop_note="valid shortened histories retain real length; length/duration can be a legal safety-policy shortcut",
                 residual=residual_audit, arm=arm, body=body,
                 hand_remapping="active/opposite name permutation; no signed mirror; side retained" if method == "GRUHandRemap" else None)
    return padded, lengths, real_time, audit


def statistics_features(x, lengths):
    return np.stack([np.concatenate((v[:n].mean(0), v[:n].std(0), v[:n].min(0), v[:n].max(0))) for v, n in zip(x, lengths)])


def approved_pairs(document, split_id, train, episodes):
    by_id = {e["trial_id"]: i for i, e in enumerate(episodes)}
    pairs = document.get("by_split", {}).get(split_id, [])
    out = []
    for p in pairs:
        a, b = by_id[p["a"]], by_id[p["b"]]
        ea, eb = episodes[a], episodes[b]
        if a not in train or b not in train or a == b or min(int(ea["y"]), int(eb["y"])) < 0:
            raise ValueError("Matched pair includes holdout/UNKNOWN/self")
        if ea["group_id"] == eb["group_id"] or not p.get("evidence"):
            raise ValueError("Physical matching requires distinct configurations and evidence")
        if ea.get("action_contract") != eb.get("action_contract"):
            raise ValueError("A matched pair mixes label contracts")
        if ea.get("followup_action_id") != eb.get("followup_action_id") and not p["evidence"].get("followup_action_comparability_verified", False):
            raise ValueError("Different subsequent actions need explicit comparability evidence")
        kind = p["kind"]
        if kind == "hard_negative" and ea["y"] == eb["y"]:
            raise ValueError("Hard negative must have opposite reliable labels")
        if kind == "cross_configuration_positive" and ea["y"] != eb["y"]:
            raise ValueError("Positive pair must have equal reliable labels")
        if kind not in ("hard_negative", "cross_configuration_positive"):
            raise ValueError("Unknown matched-pair kind")
        out.append((a, b, kind))
    return out


def wilson(k, n, z=1.959963984540054):
    if not n:
        return None, None
    p = k / n
    c = (p + z*z/(2*n)) / (1+z*z/n)
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return max(0., c-h), min(1., c+h)


def metrics(y, p, low=.5, high=.5, decisions=None):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if np.any(y < 0):
        raise ValueError("UNKNOWN cannot enter supervised metrics")
    d = np.where(p >= high, 1, np.where(p < low, 0, -1)) if decisions is None else np.asarray(decisions)
    pos, neg = y == 1, y == 0
    P, N, n = int(pos.sum()), int(neg.sum()), len(y)
    tp, tn = int(((d == 1) & pos).sum()), int(((d == 0) & neg).sum())
    fp, fn, un = int(((d == 1) & neg).sum()), int(((d == 0) & pos).sum()), int((d < 0).sum())
    pr, br = tp/P if P else None, tn/N if N else None
    low_ci, high_ci = wilson(fp, N)
    result = dict(n=n, passable_n=P, blocked_n=N, true_pass_n=tp, true_block_n=tn,
                  false_pass_n=fp, false_reject_n=fn, uncertain_n=un,
                  passable_uncertain_n=int(((d < 0) & pos).sum()), blocked_uncertain_n=int(((d < 0) & neg).sum()),
                  false_pass=fp/N if N else None, false_reject=fn/P if P else None,
                  passable_recall=pr, blocked_recall=br, balanced_accuracy=(pr+br)/2 if P and N else None,
                  coverage=1-un/n if n else None, uncertain_rate=un/n if n else None,
                  selective_risk=(fp+fn)/(n-un) if n > un else None,
                  brier=float(np.mean((p-y)**2)) if n else None,
                  false_pass_episode_binomial_wilson_low=low_ci, false_pass_episode_binomial_wilson_high=high_ci,
                  interval_scope="descriptive episode-binomial IID assumption; not configuration-cluster inference",
                  auroc=None, auprc=None, ece=None)
    if P and N:
        result["auroc"] = float(np.mean((p[pos, None] > p[None, neg]) + .5*(p[pos, None] == p[None, neg])))
        # Threshold-grouped average precision correctly handles tied scores.
        order = np.argsort(-p, kind="stable")
        pp, yy = p[order], y[order]
        boundaries = np.r_[np.flatnonzero(np.diff(pp) != 0), len(pp)-1]
        ctp = np.cumsum(yy)[boundaries]
        result["auprc"] = float(np.sum(np.diff(np.r_[0, ctp])/P * ctp/(boundaries+1)))
    if n:
        bins = np.minimum((p * 10).astype(int), 9)
        result["ece"] = float(sum(np.mean(bins == b) * abs(p[bins == b].mean()-y[bins == b].mean()) for b in range(10) if np.any(bins == b)))
    return result


def select_thresholds(y, p, target_risk=.05, max_false_reject=.10):
    """Fit an empirical calibration-only rule, explicitly not a safety certificate."""
    y, p = np.asarray(y), np.asarray(p)
    if len(set(y.tolist())) != 2:
        return dict(status="UNAVAILABLE_CALIBRATION_REQUIRES_BOTH_CLASSES", low=None, high=None)
    # Fixed probability grid, not a test-adaptive or unbounded quadratic search.
    choices = np.r_[np.linspace(0., 1., 41), 1.000001]
    best = None
    for lo in choices:
        for hi in choices[choices >= lo]:
            d = np.where(p >= hi, 1, np.where(p < lo, 0, -1))
            fp = float(np.mean(d[y == 0] == 1))
            fn = float(np.mean(d[y == 1] == 0))
            if fp <= target_risk and fn <= max_false_reject:
                score = (float(np.mean(d >= 0)), float(np.mean(d[y == 1] == 1)), -fp, -fn)
                if best is None or score > best[0]:
                    best = score, float(lo), float(hi)
    return dict(status="EMPIRICAL_CALIBRATION_ONLY_NO_DEPLOYMENT_GUARANTEE", low=best[1], high=best[2],
                target_false_pass=target_risk, max_false_reject=max_false_reject, calibration_metrics=metrics(y, p, best[1], best[2]))


def conformal_fit(y, p, alpha=.10):
    y, p = np.asarray(y), np.asarray(p)
    if not np.any(y == 0) or not np.any(y == 1):
        return dict(status="UNAVAILABLE_CALIBRATION_REQUIRES_BOTH_CLASSES", alpha=alpha)
    q = []
    counts = []
    for cls in (0, 1):
        probs = p[y == cls]
        scores = probs if cls == 0 else 1-probs
        k = int(math.ceil((len(scores)+1)*(1-alpha)))
        q.append(float(np.sort(scores)[k-1]) if k <= len(scores) else 1.0)
        counts.append(len(scores))
    return dict(status="EXPLORATORY_LABEL_CONDITIONAL_CONFORMAL_EXCHANGEABILITY_UNVERIFIED", alpha=alpha, q=q, class_n=counts)


def conformal_decisions(p, model):
    p = np.asarray(p)
    included0, included1 = p <= model["q"][0], 1-p <= model["q"][1]
    return np.where(included1 & ~included0, 1, np.where(included0 & ~included1, 0, -1))
