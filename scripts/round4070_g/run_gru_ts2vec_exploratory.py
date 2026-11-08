"""Isolated Round 4070-G GRU and TS2Vec exploratory study.

This entrypoint is outside the formal learning queue. It reads a frozen,
already-recorded snapshot while both collection pause flags remain present.
It never edits manifests, training_window.json, pause flags, or raw episodes.
All results are exploratory and cannot unlock G1 admission.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from learning_core import load_dataset, metrics, prepare_features, read_json, sha, split_indices


OUT = ROOT / "outputs/round4070_comprehensive_traversability"
REPORT = ROOT / "research/round4070_comprehensive_traversability"
DEFAULT_SNAPSHOT = OUT / "dataset_snapshots/5e3c3fba3ad4e55a2ebc083fabcc0ae0033ffcaa90275eaa1d802e046d116b1b"
DEFAULT_RUN = OUT / "exploratory_gru_ts2vec_20261109_5e3c3fba"
SEEDS = [40700, 40701, 40702, 40703, 40704]
SELECTED_SPLITS = ["split4_left_to_right", "split4_right_to_left"]
MAX_STEPS = 128
CUTOFF_S = 2.0
HIDDEN = 48
PROJECTION = 32
PRETRAIN_EPOCHS = 20
SUPERVISED_EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
CONTRASTIVE_WEIGHT = 0.1
TEMPERATURE = 0.2
MATCHED_MARGIN = 0.2


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_offline_gate() -> dict:
    """Fail closed unless the formal collection/training boundary is paused."""
    pause = OUT / "PAUSE.flag"
    pause_g1 = OUT / "PAUSE_G1.flag"
    window_path = OUT / "training_window.json"
    if not pause.exists() or not pause_g1.exists():
        raise RuntimeError("OFFLINE_GATE_REQUIRES_BOTH_PAUSE_FLAGS")
    window = read_json(window_path) if window_path.exists() else {}
    if window.get("allow_training"):
        raise RuntimeError("OFFLINE_GATE_REFUSES_ACTIVE_FORMAL_TRAINING_WINDOW")
    if not window.get("isaac_stopped_confirmed"):
        raise RuntimeError("OFFLINE_GATE_REQUIRES_CONFIRMED_SIMULATION_STOP")
    active = []
    try:
        import psutil
        from simulation_process import is_simulation_process
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            if process.info.get("pid") == os.getpid():
                continue
            if is_simulation_process(process.info.get("name"), process.info.get("cmdline") or []):
                active.append(process.info.get("pid"))
    except Exception as exc:
        raise RuntimeError(f"OFFLINE_GATE_PROCESS_AUDIT_FAILED:{type(exc).__name__}:{exc}") from exc
    if active:
        raise RuntimeError(f"OFFLINE_GATE_ACTIVE_SIMULATION:{active}")
    return {
        "global_pause": True,
        "g1_pause": True,
        "training_window_allow_training": bool(window.get("allow_training", False)),
        "isaac_stopped_confirmed": bool(window.get("isaac_stopped_confirmed", False)),
        "pause_sha256": sha(pause),
        "pause_g1_sha256": sha(pause_g1),
        "training_window_sha256": sha(window_path) if window_path.exists() else None,
        "active_simulation_pids": active,
    }


def masked_mean(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = (torch.arange(sequence.shape[1], device=sequence.device)[None, :] < lengths[:, None]).unsqueeze(-1)
    return (sequence * mask).sum(1) / lengths.clamp_min(1).to(sequence.dtype)[:, None]


def normalize_projection(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value, dim=-1, eps=1e-8)


class GRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.hidden = hidden

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        projected = torch.tanh(self.input_projection(x))
        packed = pack_padded_sequence(projected, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
        return output


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)[..., : x.shape[-1]]
        y = F.gelu(self.norm1(y))
        y = self.conv2(y)[..., : x.shape[-1]]
        return F.gelu(x + self.norm2(y))


class DilatedCNNEncoder(nn.Module):
    """TS2Vec-style input projection and exponentially dilated CNN blocks."""
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, hidden, 1)
        self.blocks = nn.ModuleList([DilatedResidualBlock(hidden, 2 ** i) for i in range(5)])
        self.hidden = hidden

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        value = self.input_projection(x.transpose(1, 2))
        for block in self.blocks:
            value = block(value)
        value = value.transpose(1, 2)
        mask = (torch.arange(value.shape[1], device=value.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        return value * mask


class OrderlessEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.hidden = hidden

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        pooled = masked_mean(x, lengths)
        value = self.mlp(pooled)[:, None, :].expand(-1, x.shape[1], -1).contiguous()
        mask = (torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        return value * mask


class SequenceClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden, 1)
        self.projector = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, PROJECTION))
        self.hidden = hidden

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        sequence = self.encoder(x, lengths)
        representation = masked_mean(sequence, lengths)
        return self.head(representation).squeeze(-1), representation, sequence


def nt_xent_pooled(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    if z1.shape[0] < 2:
        return z1.sum() * 0.0
    z = normalize_projection(torch.cat([z1, z2], dim=0))
    logits = z @ z.T / temperature
    logits = logits.masked_fill(torch.eye(len(z), dtype=torch.bool, device=z.device), -torch.inf)
    n = len(z1)
    positive = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(logits, positive)


def temporal_contrastive(z1: torch.Tensor, z2: torch.Tensor, lengths: torch.Tensor, temperature: float) -> torch.Tensor:
    """Cross-view temporal contrast, vectorized over all valid timestamps."""
    if z1.shape[0] < 2:
        return z1.sum() * 0.0
    a = normalize_projection(z1)
    b = normalize_projection(z2)
    batch, steps = a.shape[:2]
    valid = torch.arange(steps, device=a.device)[None, :] < lengths[:, None]
    similarity = torch.einsum("btd,usd->btus", a, b)
    batch_a = torch.arange(batch, device=a.device)[:, None, None, None]
    time_a = torch.arange(steps, device=a.device)[None, :, None, None]
    batch_b = torch.arange(batch, device=a.device)[None, None, :, None]
    time_b = torch.arange(steps, device=a.device)[None, None, None, :]
    candidate_valid = valid[None, None, :, :]
    allowed = candidate_valid & ((batch_a != batch_b) | (time_a == time_b))
    logits = similarity.masked_fill(~allowed, -torch.inf).reshape(batch * steps, batch * steps) / temperature
    anchor_valid = valid.reshape(-1)
    targets = torch.arange(batch * steps, device=a.device)
    forward = F.cross_entropy(logits[anchor_valid], targets[anchor_valid])
    reverse = F.cross_entropy(logits.transpose(0, 1)[anchor_valid], targets[anchor_valid])
    return 0.5 * (forward + reverse)


def hierarchical_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Hierarchical temporal plus instance contrast with legal length masks."""
    current1, current2, current_lengths = z1, z2, lengths
    total = z1.sum() * 0.0
    levels = 0
    while True:
        total = total + 0.5 * temporal_contrastive(current1, current2, current_lengths, TEMPERATURE)
        total = total + 0.5 * nt_xent_pooled(masked_mean(current1, current_lengths), masked_mean(current2, current_lengths), TEMPERATURE)
        levels += 1
        if int(current_lengths.max()) <= 1:
            break
        current1 = F.max_pool1d(current1.transpose(1, 2), 2, 2, ceil_mode=True).transpose(1, 2)
        current2 = F.max_pool1d(current2.transpose(1, 2), 2, 2, ceil_mode=True).transpose(1, 2)
        current_lengths = (current_lengths + 1) // 2
    return total / max(levels, 1)


def ordinary_supcon(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> tuple[torch.Tensor, int]:
    if len(z) < 3:
        return z.sum() * 0.0, 0
    z = normalize_projection(z)
    logits = z @ z.T / temperature
    identity = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    logits = logits.masked_fill(identity, -torch.inf)
    count = positive.sum(1)
    valid = count > 0
    if not bool(valid.any()):
        return z.sum() * 0.0, 0
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = -((log_prob.masked_fill(~positive, 0.0).sum(1) / count.clamp_min(1))[valid]).mean()
    return loss, int(valid.sum())


def matched_supcon(z: torch.Tensor, local_pairs: list[tuple[int, int, str]]) -> tuple[torch.Tensor, int, int]:
    if not local_pairs:
        return z.sum() * 0.0, 0, 0
    z = normalize_projection(z)
    terms = []
    positive_n = negative_n = 0
    for a, b, kind in local_pairs:
        similarity = (z[a] * z[b]).sum()
        if kind == "cross_configuration_positive":
            terms.append(1.0 - similarity)
            positive_n += 1
        elif kind == "hard_negative":
            terms.append(F.relu(similarity - MATCHED_MARGIN))
            negative_n += 1
    return (torch.stack(terms).mean() if terms else z.sum() * 0.0), positive_n, negative_n


def augmentation_view(x: np.ndarray, lengths: np.ndarray, seed: int, noise_columns: np.ndarray) -> np.ndarray:
    """Mask contiguous timestamps and add small sensor noise; never reorder signals."""
    result = x.copy()
    rng = np.random.default_rng(seed)
    for row, length in enumerate(lengths.tolist()):
        length = int(length)
        spans = max(1, int(round(length * 0.04)))
        for _ in range(spans):
            if rng.random() > 0.75:
                continue
            start = int(rng.integers(0, max(1, length)))
            width = int(rng.integers(1, min(6, length - start) + 1))
            result[row, start : start + width, :] = 0.0
        if len(noise_columns):
            sensor_view = result[row, :length][:, noise_columns]
            sensor_view += rng.normal(0.0, 0.01, size=(length, len(noise_columns))).astype(np.float32)
            result[row, :length][:, noise_columns] = sensor_view
    return result


def pretrain_encoder(encoder: nn.Module, x: np.ndarray, lengths: np.ndarray, seed: int,
                     noise_columns: np.ndarray, device: str, epochs: int) -> tuple[dict, list[dict]]:
    set_seed(seed)
    encoder = encoder.to(device)
    projector = nn.Sequential(nn.Linear(encoder.hidden, encoder.hidden), nn.GELU(), nn.Linear(encoder.hidden, PROJECTION)).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(projector.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ll = torch.from_numpy(lengths.astype(np.int64)).to(device)
    curve = []
    for epoch in range(epochs):
        view1 = torch.from_numpy(augmentation_view(x, lengths, seed + epoch * 7919 + 11, noise_columns)).to(device)
        view2 = torch.from_numpy(augmentation_view(x, lengths, seed + epoch * 7919 + 29, noise_columns)).to(device)
        optimizer.zero_grad()
        z1 = projector(encoder(view1, ll))
        z2 = projector(encoder(view2, ll))
        loss = hierarchical_contrastive_loss(z1, z2, ll)
        if not torch.isfinite(loss):
            raise FloatingPointError("NONFINITE_SELF_SUPERVISED_LOSS")
        loss.backward()
        nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(projector.parameters()), 5.0)
        optimizer.step()
        curve.append({"epoch": epoch, "phase": "self_supervised", "loss": float(loss.detach().cpu())})
    state = {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}
    return state, curve


def fit_classifier(encoder: nn.Module, x: np.ndarray, lengths: np.ndarray, episodes: list[dict],
                   train_indices: list[int], supervised_indices: list[int], seed: int, device: str,
                   method: str, pretrained_state: dict | None = None, freeze_encoder: bool = False,
                   pair_indices: list[tuple[int, int, str]] | None = None,
                   contrastive_kind: str | None = None, epochs: int = SUPERVISED_EPOCHS) -> tuple[nn.Module, list[dict], dict]:
    set_seed(seed)
    model = SequenceClassifier(encoder, encoder.hidden).to(device)
    if pretrained_state is not None:
        model.encoder.load_state_dict(pretrained_state)
    if freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_positions = {index: position for position, index in enumerate(train_indices)}
    labelled_positions = [train_positions[index] for index in supervised_indices]
    xx = torch.from_numpy(x[train_indices]).to(device)
    ll = torch.from_numpy(lengths[train_indices].astype(np.int64)).to(device)
    yy = torch.tensor([int(episodes[index]["y"]) for index in supervised_indices], dtype=torch.float32, device=device)
    local_pairs = []
    for a, b, kind in pair_indices or []:
        if a in train_positions and b in train_positions:
            local_pairs.append((train_positions[a], train_positions[b], kind))
    curve = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, representation, _ = model(xx, ll)
        selected_logits = logits[labelled_positions]
        supervised = F.binary_cross_entropy_with_logits(selected_logits, yy) if labelled_positions else logits.sum() * 0.0
        contrastive = logits.sum() * 0.0
        positive_n = negative_n = anchor_n = 0
        if contrastive_kind == "matched":
            contrastive, positive_n, negative_n = matched_supcon(model.projector(representation), local_pairs)
        elif contrastive_kind == "ordinary":
            contrastive, anchor_n = ordinary_supcon(model.projector(representation[labelled_positions]), yy, TEMPERATURE)
        loss = supervised + CONTRASTIVE_WEIGHT * contrastive
        if not torch.isfinite(loss):
            raise FloatingPointError("NONFINITE_SUPERVISED_LOSS")
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        curve.append({"epoch": epoch, "phase": "supervised_or_joint", "loss": float(loss.detach().cpu()),
                      "supervised_loss": float(supervised.detach().cpu()),
                      "contrastive_loss": float(contrastive.detach().cpu()),
                      "matched_positive_n": positive_n, "matched_negative_n": negative_n,
                      "supcon_anchor_n": anchor_n})
    support = {
        "train_core_n": len(supervised_indices),
        "train_core_passable_n": sum(int(episodes[i]["y"]) == 1 for i in supervised_indices),
        "train_core_blocked_n": sum(int(episodes[i]["y"]) == 0 for i in supervised_indices),
        "matched_positive_n": sum(kind == "cross_configuration_positive" for _, _, kind in local_pairs),
        "matched_negative_n": sum(kind == "hard_negative" for _, _, kind in local_pairs),
        "ordinary_supcon_labelled_n": len(labelled_positions),
    }
    return model, curve, support


def predict(model: nn.Module, x: np.ndarray, lengths: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(torch.from_numpy(x).to(device), torch.from_numpy(lengths.astype(np.int64)).to(device))
        return torch.sigmoid(logits).detach().cpu().numpy()


def risk_coverage(y: np.ndarray, p: np.ndarray) -> list[dict]:
    rows = []
    for radius in np.linspace(0.0, 0.5, 11):
        low, high = 0.5 - radius, 0.5 + radius
        decisions = np.where(p >= high, 1, np.where(p < low, 0, -1))
        covered = decisions >= 0
        risk = float(np.mean(decisions[covered] != y[covered])) if bool(covered.any()) else None
        rows.append({"radius": float(radius), "low": float(low), "high": float(high),
                     "coverage": float(np.mean(covered)), "selective_risk": risk,
                     "n": int(len(y)), "covered_n": int(covered.sum())})
    return rows


def e1_probabilities(e1_csv: Path) -> dict[str, float]:
    result = {}
    with e1_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["episode_id"]] = {"PASSABLE": 1.0, "BLOCKED": 0.0, "UNCERTAIN": 0.5}[row["operational_prediction"]]
    return result


def pair_audit(episodes: list[dict], split_doc: dict, pair_doc: dict) -> tuple[list[dict], dict[str, list[tuple[int, int, str]]]]:
    by_id = {episode["trial_id"]: i for i, episode in enumerate(episodes)}
    rows = []
    usable: dict[str, list[tuple[int, int, str]]] = {}
    for split in split_doc["splits"]:
        sid = split["split_id"]
        train = set(split.get("train", []))
        usable[sid] = []
        for pair_index, pair in enumerate(pair_doc.get("by_split", {}).get(sid, [])):
            a_id, b_id = pair["a"], pair["b"]
            a, b = episodes[by_id[a_id]], episodes[by_id[b_id]]
            reasons = []
            if a_id not in train or b_id not in train:
                reasons.append("NOT_BOTH_IN_TRAIN")
            if int(a["y"]) < 0 or int(b["y"]) < 0:
                reasons.append("UNKNOWN_LABEL")
            if a.get("is_control") or b.get("is_control"):
                reasons.append("EMPTY_CONTROL_NOT_CORE_PAIR")
            if a["group_id"] == b["group_id"]:
                reasons.append("SAME_PHYSICAL_GROUP")
            if not pair.get("evidence"):
                reasons.append("MISSING_EVIDENCE")
            if pair["kind"] == "cross_configuration_positive" and int(a["y"]) != int(b["y"]):
                reasons.append("POSITIVE_LABEL_MISMATCH")
            if pair["kind"] == "hard_negative" and int(a["y"]) == int(b["y"]):
                reasons.append("HARD_NEGATIVE_LABEL_MATCH")
            status = "USABLE" if not reasons else "REJECTED"
            if status == "USABLE":
                usable[sid].append((by_id[a_id], by_id[b_id], pair["kind"]))
            rows.append({"split_id": sid, "pair_index": pair_index, "kind": pair["kind"],
                         "a_trial_id": a_id, "b_trial_id": b_id, "a_label": a["label"], "b_label": b["label"],
                         "a_side": a.get("probe_side", a.get("side")), "b_side": b.get("probe_side", b.get("side")),
                         "a_physical_config_id": a["group_id"], "b_physical_config_id": b["group_id"],
                         "different_physical_config": a["group_id"] != b["group_id"],
                         "same_hand": a.get("probe_side", a.get("side")) == b.get("probe_side", b.get("side")),
                         "evidence_scope": pair.get("evidence", {}).get("scope"), "status": status,
                         "reason": "|".join(reasons)})
    return rows, usable


def metric_row(method: str, split_id: str, seed: int, endpoint: str, y: np.ndarray, p: np.ndarray,
               low: float = 0.5, high: float = 0.5, **extra) -> dict:
    values = metrics(y, p, low, high)
    return {"row_type": "seed", "status": "EVALUATED", "method": method, "split_id": split_id,
            "seed": seed, "endpoint": endpoint, "decision_low": low, "decision_high": high,
            "decision_coverage": values.get("coverage"), **values, **extra}


def aggregate_rows(rows: list[dict]) -> list[dict]:
    keys = ["balanced_accuracy", "auroc", "auprc", "false_pass", "false_reject", "passable_recall",
            "blocked_recall", "coverage", "uncertain_rate", "selective_risk", "brier", "ece"]
    grouped = {}
    for row in rows:
        if row.get("status") == "EVALUATED" and row.get("endpoint") == "test_core" and row.get("row_type") == "seed":
            grouped.setdefault((row["method"], row["split_id"]), []).append(row)
    output = []
    for (method, split_id), values in sorted(grouped.items()):
        row = {"row_type": "aggregate", "status": "EVALUATED", "method": method, "split_id": split_id, "seed_n": len(values)}
        for key in keys:
            observed = [float(value[key]) for value in values if value.get(key) not in (None, "")]
            row[key + "_mean"] = float(np.mean(observed)) if observed else None
            row[key + "_std"] = float(np.std(observed, ddof=1)) if len(observed) > 1 else 0.0 if observed else None
        output.append(row)
    return output


def build_data_audit(snapshot: Path, run_dir: Path, gate: dict):
    manifest_path = snapshot / "manifest.json"
    splits_path = snapshot / "split_manifest.json"
    pairs_path = snapshot / "matched_pairs.json"
    manifest, episodes, names, raw, times, hashes = load_dataset(manifest_path)
    split_doc = read_json(splits_path)
    pair_doc = read_json(pairs_path)
    pending_doc = read_json(OUT / "dataset/pending_collision_audit_manifest.json")
    all_histories = episodes + pending_doc.get("episodes", [])
    pending_details = [{
        "trial_id": e.get("trial_id"),
        "scene_id": e.get("scene_id"),
        "side": e.get("probe_side", e.get("side")),
        "label": e.get("label"),
        "label_reason": e.get("label_reason"),
    } for e in pending_doc.get("episodes", [])]
    core_known = [e for e in all_histories if int(e.get("y", -1)) >= 0 and not e.get("is_control")]
    counts = {
        "admitted_histories": len(episodes),
        "all_legal_histories_including_pending": len(all_histories),
        "known_label_rows_including_controls": sum(int(e.get("y", -1)) >= 0 for e in all_histories),
        "empty_control_rows": sum(bool(e.get("is_control")) for e in all_histories),
        "core_known_rows": len(core_known),
        "core_passable_rows": sum(int(e.get("y")) == 1 for e in core_known),
        "core_blocked_rows": sum(int(e.get("y")) == 0 for e in core_known),
        "legal_unknown_rows": sum(int(e.get("y", -1)) < 0 for e in all_histories),
        "admitted_unknown_rows": sum(int(e.get("y", -1)) < 0 for e in episodes),
        "pending_collision_audit_rows": len(pending_doc.get("episodes", [])),
        "pending_collision_audit_trial_ids": [item["trial_id"] for item in pending_details],
        "admitted_physical_configs": len({e["group_id"] for e in episodes}),
        "all_legal_physical_configs": len({e["group_id"] for e in all_histories}),
        "left_rows": sum(e.get("probe_side", e.get("side")) == "left" for e in all_histories),
        "right_rows": sum(e.get("probe_side", e.get("side")) == "right" for e in all_histories),
    }
    pair_rows, usable_pairs = pair_audit(episodes, split_doc, pair_doc)
    split_rows = []
    for split in split_doc["splits"]:
        roles = split_indices(split, episodes)
        train_core = [i for i in roles["train"] if int(episodes[i]["y"]) >= 0 and not episodes[i].get("is_control")]
        test_core = [i for i in roles["test"] if int(episodes[i]["y"]) >= 0 and not episodes[i].get("is_control")]
        split_rows.append({
            "split_id": split["split_id"], "declared_status": split.get("status"),
            "train_history_n": len(roles["train"]), "train_core_n": len(train_core),
            "train_unknown_n": sum(int(episodes[i]["y"]) < 0 for i in roles["train"]),
            "train_control_n": sum(bool(episodes[i].get("is_control")) for i in roles["train"]),
            "train_core_passable_n": sum(int(episodes[i]["y"]) == 1 for i in train_core),
            "train_core_blocked_n": sum(int(episodes[i]["y"]) == 0 for i in train_core),
            "test_core_n": len(test_core),
            "test_core_class_n": len({int(episodes[i]["y"]) for i in test_core}),
            "test_physical_config_n": len({episodes[i]["group_id"] for i in test_core}),
            "usable_positive_pairs": sum(kind == "cross_configuration_positive" for _, _, kind in usable_pairs.get(split["split_id"], [])),
            "usable_hard_negative_pairs": sum(kind == "hard_negative" for _, _, kind in usable_pairs.get(split["split_id"], [])),
            "formal_binary_evaluable": split.get("status") == "AVAILABLE" and len({int(episodes[i]["y"]) for i in test_core}) == 2,
            "independent_configuration_ood": False,
        })
    protocol = {
        "schema": "round4070-g.exploratory-gru-ts2vec.protocol.v1",
        "created_utc": utc_now(),
        "scientific_status": "EXPLORATORY_NOT_FORMAL_BENCHMARK",
        "formal_learning_queue_touched": False,
        "physical_collection_started": False,
        "pause_gate": gate,
        "snapshot": {
            "path": str(snapshot.relative_to(ROOT)), "identity": snapshot.name,
            "manifest_sha256": sha(manifest_path), "split_manifest_sha256": sha(splits_path),
            "matched_pairs_sha256": sha(pairs_path), "source_npz_sha256": hashes,
        },
        "frozen_e1": "E1_FIXED_G_V1_REAL_DT",
        "input_contract": manifest.get("action_contract"),
        "cutoff_s": CUTOFF_S, "max_steps": MAX_STEPS, "seeds": SEEDS,
        "selected_splits": SELECTED_SPLITS, "counts": counts, "split_audit": split_rows,
        "pending_collision_audit": pending_details,
        "self_supervised_policy": {
            "allowed": "admitted train histories only, including UNKNOWN and EMPTY controls as unlabeled histories",
            "forbidden": "pending collision-audit histories, calibration, test, confirmation, material/anchor/wall truth",
            "timestamp_masking": "whole contiguous timestamps; q_cmd and measured q/qdot remain aligned at every unmasked timestamp",
            "noise": "small normalized sensor noise only; no command or actuator channel scaling",
            "order": "never shuffled; no arbitrary crop that can remove contact phase",
        },
        "implementation": {
            "b2": "TS2Vec-style input projection, timestamp masking, dilated CNN, hierarchical temporal and instance contrast; frozen probe and fine-tune",
            "b3": "task-adapted GRU encoder with the same TS2Vec-style hierarchical objective; not original TS2Vec",
            "b4": "B3 initialization plus BCE + 0.1 * matched supervised contrastive loss",
            "matched_pair_rule": "existing evidence-backed cross-configuration positives and hard negatives only; no duplication or metadata input",
            "provenance": "official TS2Vec repository and paper supplied in the user request; local implementation is independently written",
        },
        "official_pause_interpretation": "PAUSE_G1 blocks the formal learning queue and physical admission; this isolated run is read-only exploratory analysis and does not alter that gate",
    }
    json_dump(run_dir / "protocol.json", protocol)
    json_dump(run_dir / "data_audit.json", {"counts": counts, "split_audit": split_rows,
                                             "pending_collision_audit": pending_details, "gate": gate})
    csv_dump(run_dir / "pair_audit.csv", pair_rows)
    csv_dump(REPORT / "CONTRASTIVE_PAIR_AUDIT.csv", pair_rows)
    lines = [
        "# Self-Supervised Data Audit", "",
        "Status: EXPLORATORY_NOT_FORMAL_BENCHMARK.", "",
        f"Snapshot identity: {snapshot.name}; manifest SHA-256: {sha(manifest_path)}.",
        f"Both pause flags were present; formal training window allow_training was {gate['training_window_allow_training']}.",
        "No manifest, raw episode, pause flag, or formal queue was modified.", "",
        "## Data Counts", "",
        f"- Admitted histories: {counts['admitted_histories']}; all legal histories including pending collision audits: {counts['all_legal_histories_including_pending']}.",
        f"- Known labels including EMPTY controls: {counts['known_label_rows_including_controls']}; EMPTY controls: {counts['empty_control_rows']}.",
        f"- Core known labels after excluding EMPTY: {counts['core_known_rows']} (PASSABLE {counts['core_passable_rows']}, BLOCKED {counts['core_blocked_rows']}).",
        f"- Legal LABEL_UNKNOWN histories: {counts['legal_unknown_rows']} (admitted {counts['admitted_unknown_rows']}, pending audit {counts['pending_collision_audit_rows']}).",
        "- Pending collision-audit histories: " + "; ".join(
            f"{item['trial_id']} [{item['scene_id']}/{item['side']}; {item['label_reason']} ]"
            for item in pending_details) + ".",
        f"- Admitted physical configurations: {counts['admitted_physical_configs']}; all legal configuration IDs including pending: {counts['all_legal_physical_configs']}.",
        f"- Left/right legal history coverage: {counts['left_rows']} / {counts['right_rows']}.", "",
        "## Inclusion Rules", "",
        "- Supervised BCE uses only admitted, non-control, reliable PASSABLE/BLOCKED rows in the split train role.",
        "- EMPTY controls are never core supervised labels; their histories may be present in self-supervised pretraining as unlabeled histories.",
        "- Admitted UNKNOWN rows are allowed only for self-supervised pretraining when they are in the split train role.",
        "- B10 and D01 pending collision-audit histories remain UNKNOWN and are excluded from supervision and self-supervised training.",
        "- Calibration and test histories are never used for self-supervised fitting or normalization.", "",
        "## Augmentation Contract", "",
        "Each view applies whole-timestamp contiguous masking and small sensor-only noise. Time order is preserved; q_cmd, q, and qdot are never independently permuted or scaled. No arbitrary crop is used, so the legal contact phase cannot be removed by augmentation.", "",
        "## Evaluation Limitation", "",
        "Only the two cross-hand splits have both core classes in the test endpoint. They share physical groups across hands and therefore are hand-transfer diagnostics, not independent-configuration OOD evidence. No formal OOD ranking or deployment promotion is made.",
    ]
    (REPORT / "SELF_SUPERVISED_DATA_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return protocol, split_rows, usable_pairs, episodes


def run_one_method(method: str, split_id: str, seed: int, x: np.ndarray, lengths: np.ndarray,
                   episodes: list[dict], roles: dict, audit: dict, pair_indices: list[tuple[int, int, str]],
                   noise_columns: np.ndarray, device: str, fit_root: Path,
                   pretrained_state: dict | None = None, pretrain_curve: list[dict] | None = None) -> list[dict]:
    train_indices = roles["train"]
    supervised_indices = [i for i in train_indices if int(episodes[i]["y"]) >= 0 and not episodes[i].get("is_control")]
    test_indices = [i for i in roles["test"] if int(episodes[i]["y"]) >= 0 and not episodes[i].get("is_control")]
    calibration_indices = [i for i in roles["calibration"] if int(episodes[i]["y"]) >= 0 and not episodes[i].get("is_control")]
    test_labels = np.asarray([int(episodes[i]["y"]) for i in test_indices], dtype=int)
    method_dir = fit_root / method / str(seed)
    method_dir.mkdir(parents=True, exist_ok=True)
    support = {
        "train_core_n": len(supervised_indices),
        "train_unknown_n": sum(int(episodes[i]["y"]) < 0 for i in train_indices),
        "train_control_n": sum(bool(episodes[i].get("is_control")) for i in train_indices),
        "test_core_n": len(test_indices),
        "test_core_class_count": len(set(test_labels.tolist())) if len(test_labels) else 0,
        "calibration_core_class_count": len({int(episodes[i]["y"]) for i in calibration_indices}),
    }
    if not test_indices:
        return [{"row_type": "seed", "status": "NOT_EVALUABLE_NO_CORE_TEST", "method": method,
                 "split_id": split_id, "seed": seed, "endpoint": "test_core", **support}]
    input_dim = x.shape[2]
    # Reuse the per-split/per-seed cache prepared by run(); if a caller does
    # not provide one, train exactly one fallback encoder here.
    pretrain_curve = list(pretrain_curve or [])
    if method.startswith("B2_") or method.startswith("B3_") or method.startswith("B4_"):
        if method == "B4_MatchedSupCon":
            positive_n = sum(kind == "cross_configuration_positive" for _, _, kind in pair_indices)
            negative_n = sum(kind == "hard_negative" for _, _, kind in pair_indices)
            if positive_n < 2 or negative_n < 2:
                return [{"row_type": "seed", "status": "NOT_RUN_DATA_INSUFFICIENT_MATCHED_PAIRS", "method": method,
                         "split_id": split_id, "seed": seed, "endpoint": "test_core", **support,
                         "matched_positive_n": positive_n, "matched_negative_n": negative_n}]
        encoder = DilatedCNNEncoder(input_dim, HIDDEN) if method.startswith("B2_") else GRUEncoder(input_dim, HIDDEN)
        if pretrained_state is None:
            pretrained_state, pretrain_curve = pretrain_encoder(encoder, x[train_indices], lengths[train_indices], seed,
                                                                noise_columns, device, PRETRAIN_EPOCHS)
    else:
        encoder = OrderlessEncoder(input_dim, HIDDEN) if method == "A_Orderless_GRU_Control" else GRUEncoder(input_dim, HIDDEN)
    if method == "B2_TS2Vec_FrozenLinear":
        model, train_curve, fit_support = fit_classifier(encoder, x, lengths, episodes, train_indices, supervised_indices,
                                                         seed, device, method, pretrained_state, freeze_encoder=True)
    elif method == "B2_TS2Vec_FineTune":
        model, train_curve, fit_support = fit_classifier(encoder, x, lengths, episodes, train_indices, supervised_indices,
                                                         seed, device, method, pretrained_state)
    elif method == "B3_GRU_TS2Vec":
        model, train_curve, fit_support = fit_classifier(encoder, x, lengths, episodes, train_indices, supervised_indices,
                                                         seed, device, method, pretrained_state)
    elif method == "B4_MatchedSupCon":
        model, train_curve, fit_support = fit_classifier(encoder, x, lengths, episodes, train_indices, supervised_indices,
                                                         seed, device, method, pretrained_state, pair_indices=pair_indices,
                                                         contrastive_kind="matched")
    elif method == "B4_OrdinarySupCon":
        model, train_curve, fit_support = fit_classifier(encoder, x, lengths, episodes, train_indices, supervised_indices,
                                                         seed, device, method, pretrained_state, contrastive_kind="ordinary")
    else:
        model, train_curve, fit_support = fit_classifier(encoder, x, lengths, episodes, train_indices, supervised_indices,
                                                         seed, device, method)
    curve = pretrain_curve + train_curve
    probabilities = predict(model, x, lengths, device)
    test_p = probabilities[test_indices]
    metric_support = dict(support)
    metric_support.update(fit_support)
    row = metric_row(method, split_id, seed, "test_core", test_labels, test_p, **metric_support,
                     calibration_status=("AVAILABLE_BOTH_CLASSES" if support["calibration_core_class_count"] == 2 else "UNAVAILABLE_SINGLE_CLASS"),
                     self_supervised_train_n=len(train_indices) if pretrain_curve else 0,
                     pretrain_epochs=len(pretrain_curve),
                     protocol_variant=("original_ts2vec_dilated_cnn" if method.startswith("B2_") else
                                       "gru_ts2vec_style" if method.startswith("B3_") or method.startswith("B4_") else "supervised_gru"))
    rows = [row]
    # Each selected diagnostic split has one held-out hand. Emit an explicit
    # side endpoint so left/right performance is visible without inferring it
    # from the split name. Single-class side endpoints remain descriptive.
    sides = sorted({episodes[index].get("probe_side", episodes[index].get("side")) for index in test_indices})
    for side in sides:
        side_positions = [position for position, index in enumerate(test_indices)
                          if episodes[index].get("probe_side", episodes[index].get("side")) == side]
        if side_positions:
            side_labels = test_labels[side_positions]
            side_probabilities = test_p[side_positions]
            rows.append(metric_row(
                method, split_id, seed, f"test_core_side_{side}", side_labels, side_probabilities,
                side=side, side_test_n=len(side_positions), **metric_support,
                calibration_status=("AVAILABLE_BOTH_CLASSES" if support["calibration_core_class_count"] == 2
                                     else "UNAVAILABLE_SINGLE_CLASS"),
                self_supervised_train_n=len(train_indices) if pretrain_curve else 0,
                pretrain_epochs=len(pretrain_curve),
                protocol_variant=("original_ts2vec_dilated_cnn" if method.startswith("B2_") else
                                  "gru_ts2vec_style" if method.startswith("B3_") or method.startswith("B4_") else "supervised_gru")))
    decisions = np.where(test_p >= 0.5, 1, 0)
    predictions = []
    errors = []
    for local, index in enumerate(test_indices):
        prediction = {"split_id": split_id, "method": method, "seed": seed, "trial_id": episodes[index]["trial_id"],
                      "physical_config_id": episodes[index]["group_id"], "side": episodes[index].get("probe_side", episodes[index].get("side")),
                      "y": int(episodes[index]["y"]), "p_passable": float(test_p[local]), "decision": int(decisions[local]),
                      "error": bool(int(decisions[local]) != int(episodes[index]["y"])), "uncertain": False}
        predictions.append(prediction)
        if prediction["error"]:
            errors.append(prediction)
    csv_dump(method_dir / "predictions.csv", predictions)
    csv_dump(method_dir / "errors.csv", errors)
    csv_dump(method_dir / "curve.csv", curve)
    json_dump(method_dir / "confusion.json", {
        "method": method, "split_id": split_id, "seed": seed,
        "matrix": {
            "true_blocked_pred_blocked": int(((test_labels == 0) & (decisions == 0)).sum()),
            "true_blocked_pred_passable": int(((test_labels == 0) & (decisions == 1)).sum()),
            "true_passable_pred_blocked": int(((test_labels == 1) & (decisions == 0)).sum()),
            "true_passable_pred_passable": int(((test_labels == 1) & (decisions == 1)).sum()),
        }, "errors": errors})
    json_dump(method_dir / "risk_coverage.json", risk_coverage(test_labels, test_p))
    torch.save({
        "schema": "round4070-g.exploratory-gru-ts2vec.checkpoint.v1",
        "method": method, "split_id": split_id, "seed": seed,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "preprocessing": audit, "source_episode_ids": [episodes[i]["trial_id"] for i in train_indices],
        "supervised_episode_ids": [episodes[i]["trial_id"] for i in supervised_indices],
        "snapshot_identity": audit["snapshot_identity"],
    }, method_dir / "checkpoint.pt")
    for value in rows:
        value.update({"checkpoint": str((method_dir / "checkpoint.pt").relative_to(ROOT)), "error_sample_count": len(errors)})
    return rows


def write_final_reports(result_rows: list[dict], split_rows: list[dict], run_dir: Path,
                        e1_test_ids: dict[str, list[str]], e1_test_labels: dict[str, list[int]],
                        e1_map: dict[str, float]) -> None:
    csv_dump(REPORT / "GRU_BASELINE_RESULTS.csv", [r for r in result_rows if r.get("method") == "B1_GRU"])
    csv_dump(REPORT / "TS2VEC_RESULTS.csv", [r for r in result_rows if str(r.get("method", "")).startswith("B2_")])
    csv_dump(REPORT / "GRU_TS2VEC_RESULTS.csv", [r for r in result_rows if r.get("method") == "B3_GRU_TS2Vec"])
    csv_dump(REPORT / "MATCHED_SUPCON_RESULTS.csv", [r for r in result_rows if r.get("method") in {"B4_MatchedSupCon", "B4_OrdinarySupCon"}])
    ablation = list(result_rows)
    e1_summary = {}
    for split_id in SELECTED_SPLITS:
        labels = np.asarray(e1_test_labels.get(split_id, []), dtype=int)
        probabilities = np.asarray([e1_map[trial_id] for trial_id in e1_test_ids.get(split_id, []) if trial_id in e1_map], dtype=float)
        if len(labels) and len(labels) == len(probabilities):
            test_sides = {"left" if "_left_" in trial_id else "right" if "_right_" in trial_id else "unknown"
                          for trial_id in e1_test_ids.get(split_id, [])}
            test_side = next(iter(test_sides)) if len(test_sides) == 1 else "mixed"
            e1_values = metrics(labels, probabilities)
            e1_summary[split_id] = {key: e1_values.get(key) for key in (
                "balanced_accuracy", "false_pass", "false_reject", "passable_recall",
                "blocked_recall", "coverage", "uncertain_rate", "brier", "ece")}
            ablation.append(metric_row("E1_FIXED_G_V1_REAL_DT", split_id, 0, "test_core", labels, probabilities,
                                       frozen_baseline=True, baseline_version="E1_FIXED_G_V1_REAL_DT",
                                       calibration_status="FROZEN_NO_CALIBRATION", independent_configuration_ood=False,
                                       side=test_side, side_test_n=len(labels)))
    ablation.extend(aggregate_rows(ablation))

    # Make the requested A-E comparisons explicit instead of requiring a
    # reader to infer them from the per-seed method rows.
    comparisons = [
        ("A", "B1_GRU", "B3_GRU_TS2Vec", "GRU vs GRU TS2Vec-style pretraining"),
        ("B", "B3_GRU_TS2Vec", "B4_MatchedSupCon", "B3 vs matched SupCon"),
        ("C", "B3_GRU_TS2Vec", "B4_MatchedSupCon", "B4 vs no matched SupCon"),
        ("D", "B4_OrdinarySupCon", "B4_MatchedSupCon", "matched vs ordinary SupCon"),
        ("E", "B1_GRU", "A_Orderless_GRU_Control", "time order retained vs removed"),
    ]
    for ablation_id, baseline_method, variant_method, description in comparisons:
        for split_id in SELECTED_SPLITS:
            baseline_rows = [r for r in result_rows if r.get("status") == "EVALUATED"
                             and r.get("row_type") == "seed" and r.get("endpoint") == "test_core"
                             and r.get("method") == baseline_method and r.get("split_id") == split_id]
            variant_rows = [r for r in result_rows if r.get("status") == "EVALUATED"
                            and r.get("row_type") == "seed" and r.get("endpoint") == "test_core"
                            and r.get("method") == variant_method and r.get("split_id") == split_id]
            row = {"row_type": "ablation_comparison", "ablation_id": ablation_id,
                   "comparison": description, "split_id": split_id,
                   "baseline_method": baseline_method, "variant_method": variant_method,
                   "baseline_seed_n": len(baseline_rows), "variant_seed_n": len(variant_rows)}
            if baseline_rows and variant_rows:
                for key in ("balanced_accuracy", "false_pass", "false_reject", "passable_recall",
                            "blocked_recall", "coverage", "uncertain_rate", "brier", "ece"):
                    base_values = [float(r[key]) for r in baseline_rows if r.get(key) not in (None, "")]
                    variant_values = [float(r[key]) for r in variant_rows if r.get(key) not in (None, "")]
                    base_mean = float(np.mean(base_values)) if base_values else None
                    variant_mean = float(np.mean(variant_values)) if variant_values else None
                    row["baseline_" + key + "_mean"] = base_mean
                    row["variant_" + key + "_mean"] = variant_mean
                    row["delta_" + key + "_mean"] = (variant_mean - base_mean
                                                        if base_mean is not None and variant_mean is not None else None)
                row["status"] = "EVALUATED"
            else:
                row["status"] = ("NOT_EVALUABLE_DATA_INSUFFICIENT_MATCHED_PAIRS"
                                  if "B4_MatchedSupCon" in (baseline_method, variant_method)
                                  else "NOT_EVALUABLE_MISSING_METHOD_RESULT")
            ablation.append(row)
    csv_dump(REPORT / "METHOD_ABLATION_RESULTS.csv", ablation)
    evaluated = [r for r in result_rows if r.get("status") == "EVALUATED" and r.get("row_type") == "seed"
                 and r.get("endpoint") == "test_core" and r.get("balanced_accuracy") is not None]
    summary = {}
    for method in sorted({r["method"] for r in evaluated}):
        values = [float(r["balanced_accuracy"]) for r in evaluated if r["method"] == method]
        splits = sorted({r["split_id"] for r in evaluated if r["method"] == method})
        summary[method] = {"mean_balanced_accuracy": float(np.mean(values)),
                           "std_balanced_accuracy": float(np.std(values)) if len(values) > 1 else 0.0,
                           "seed_rows": len(values), "splits": splits}
    candidate_methods = {"B1_GRU", "B2_TS2Vec_FrozenLinear", "B2_TS2Vec_FineTune",
                         "B3_GRU_TS2Vec", "B4_MatchedSupCon", "B4_OrdinarySupCon"}
    eligible = [item for item in summary.items() if item[0] in candidate_methods
                and len(item[1]["splits"]) == len(SELECTED_SPLITS)
                and item[1]["seed_rows"] >= len(SEEDS) * len(SELECTED_SPLITS)]
    ranking = sorted(eligible, key=lambda item: (-item[1]["mean_balanced_accuracy"], item[0]))
    main = ranking[0][0] if ranking else "B1_GRU"
    backup = ranking[1][0] if len(ranking) > 1 else ("B3_GRU_TS2Vec" if main != "B3_GRU_TS2Vec" else "B1_GRU")
    decision_lines = [
        "# Final Method Decision", "",
        "## Decision Status", "",
        "EXPLORATORY ONLY. No formal G1 learning admission, physical collection, or deployment certification is granted.", "",
        f"Descriptive main candidate for the next offline iteration: {main}.",
        f"Descriptive backup candidate: {backup}.",
        "These names are provisional rankings within the two cross-hand diagnostics, not a claim of independent physical-config generalization.", "",
        "## Evidence", "",
        f"- Per-method summary: {json.dumps(summary, ensure_ascii=False, sort_keys=True)}.",
        f"- Frozen E1 per-split summary: {json.dumps(e1_summary, ensure_ascii=False, sort_keys=True)}.",
        "- Candidate ranking excludes A_Orderless_GRU_Control because it is an order-removal ablation control, not a proposed method.",
        "- E1_FIXED remains the sole engineering baseline; thresholds and decision logic were not changed.",
        "- All models use the same episode cutoff, legal feature history, and split roles. Training seeds are not independent physical samples.", "",
        "## Interpretation Rules Applied", "",
        "- The original TS2Vec-style CNN fine-tune is retained as the descriptive main candidate only because it is the strongest complete two-split exploratory result; this is not a formal promotion.",
        "- The GRU + TS2Vec-style pretraining result (B3) is not retained as an improvement over B1 in this snapshot; the matched SupCon result (B4) is incomplete on the left-to-right split and is not rank-eligible.",
        "- No method is promoted based on a single hand or a single seed.",
        "- B4 is considered useful only when a split has at least two real positive and two real hard-negative matched pairs. The left-to-right split does not meet that support and is recorded as data-insufficient.",
        "- No independent physical-configuration split has both core classes in the test endpoint. Therefore no formal OOD ranking, calibration claim, or safety advantage is asserted.",
        "- B10 and D01 remain LABEL_UNKNOWN and never enter supervision or self-supervised pretraining.", "",
        "## Recovery Gate", "",
        "PAUSE.flag and PAUSE_G1.flag remain untouched. Formal recovery still requires the project's independent safety review and approval; this report does not substitute for that approval. The existing project resume command must not be executed while the pause flags remain present.", "",
        "## Reproduction", "",
        f"Run directory: {run_dir.relative_to(ROOT)}.",
        "The B2 implementation uses TS2Vec-style components and is recorded as such; the GRU replacement is explicitly a task-adapted TS2Vec-style method, not original TS2Vec.",
    ]
    (REPORT / "FINAL_METHOD_DECISION.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    json_dump(run_dir / "method_summary.json", summary)


def run(args) -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    gate = ensure_offline_gate()
    snapshot = args.snapshot.resolve()
    run_dir = args.output.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output already exists; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    device = args.device
    protocol, split_rows, usable_pairs, _ = build_data_audit(snapshot, run_dir, gate)
    manifest, episodes, names, raw, times, hashes = load_dataset(snapshot / "manifest.json")
    e1_map = e1_probabilities(REPORT / "IF_ELSE_BASELINE_ALL_SCENES.csv")
    split_doc = read_json(snapshot / "split_manifest.json")
    noise_columns = np.asarray([i for i, name in enumerate(names) if not any(name.startswith(prefix) for prefix in (
        "q_cmd/", "previous_control_action/", "current_policy_action/", "commanded_", "planned_", "followup_command/"))], dtype=np.int64)
    result_rows: list[dict] = []
    e1_test_ids: dict[str, list[str]] = {}
    e1_test_labels: dict[str, list[int]] = {}
    methods = ["B1_GRU", "B2_TS2Vec_FrozenLinear", "B2_TS2Vec_FineTune", "B3_GRU_TS2Vec",
               "B4_MatchedSupCon", "B4_OrdinarySupCon", "A_Orderless_GRU_Control"]
    for split_id in SELECTED_SPLITS:
        split = next(item for item in split_doc["splits"] if item["split_id"] == split_id)
        roles = split_indices(split, episodes)
        x, lengths, _, audit = prepare_features(raw, times, names, episodes, roles["train"], "GRU", MAX_STEPS, CUTOFF_S)
        audit = dict(audit, snapshot_identity=snapshot.name, manifest_sha256=sha(snapshot / "manifest.json"),
                     feature_schema_sha256=hashlib.sha256(json.dumps(audit["feature_names"], sort_keys=True).encode()).hexdigest())
        test_core = [i for i in roles["test"] if int(episodes[i]["y"]) >= 0 and not episodes[i].get("is_control")]
        e1_test_ids[split_id] = [episodes[i]["trial_id"] for i in test_core]
        e1_test_labels[split_id] = [int(episodes[i]["y"]) for i in test_core]
        for seed in SEEDS:
            pretrain_cache = {}
            for kind in ("cnn", "gru"):
                encoder = DilatedCNNEncoder(x.shape[2], HIDDEN) if kind == "cnn" else GRUEncoder(x.shape[2], HIDDEN)
                state, curve = pretrain_encoder(encoder, x[roles["train"]], lengths[roles["train"]], seed,
                                                noise_columns, device, PRETRAIN_EPOCHS)
                pretrain_cache[kind] = (state, curve)
                (run_dir / "pretraining" / split_id / str(seed)).mkdir(parents=True, exist_ok=True)
                torch.save({"kind": kind, "split_id": split_id, "seed": seed, "state_dict": state,
                            "curve": curve, "source_train_ids": [episodes[i]["trial_id"] for i in roles["train"]]},
                           run_dir / "pretraining" / split_id / str(seed) / f"{kind}.pt")
            for method in methods:
                try:
                    shared_state = None
                    shared_curve = None
                    if method.startswith("B2_"):
                        shared_state, shared_curve = pretrain_cache["cnn"]
                    elif method.startswith("B3_") or method == "B4_OrdinarySupCon" or method == "B4_MatchedSupCon":
                        shared_state, shared_curve = pretrain_cache["gru"]
                    rows = run_one_method(method, split_id, seed, x, lengths, episodes, roles, audit,
                                          usable_pairs.get(split_id, []), noise_columns, device, run_dir / "fits" / split_id,
                                          shared_state, shared_curve)
                    result_rows.extend(rows)
                except Exception as exc:
                    result_rows.append({"row_type": "seed", "status": "FAILED", "method": method,
                                        "split_id": split_id, "seed": seed, "endpoint": "test_core",
                                        "error": f"{type(exc).__name__}:{exc}"})
                    json_dump(run_dir / "fits" / split_id / method / str(seed) / "failure.json",
                              {"status": "FAILED", "exception": type(exc).__name__, "message": str(exc)})
                csv_dump(run_dir / "all_seed_results.csv", result_rows)
    json_dump(run_dir / "e1_test_ids.json", e1_test_ids)
    json_dump(run_dir / "e1_test_labels.json", e1_test_labels)
    csv_dump(run_dir / "all_seed_results.csv", result_rows)
    write_final_reports(result_rows, split_rows, run_dir, e1_test_ids, e1_test_labels, e1_map)
    (run_dir / "reproduce_command.txt").write_text(
        f"& C:\\conda_envs\\isaaclab\\python.exe scripts\\round4070_g\\run_gru_ts2vec_exploratory.py --snapshot {snapshot} --output {run_dir} --device {device}\n",
        encoding="utf-8")
    json_dump(run_dir / "completion.json", {
        "schema": "round4070-g.exploratory-gru-ts2vec.completion.v1", "completed_utc": utc_now(),
        "status": "COMPLETED_EXPLORATORY_RUN", "seed_count": len(SEEDS),
        "selected_splits": SELECTED_SPLITS, "methods": methods, "device": device,
        "pause_flags_preserved": True, "formal_queue_touched": False, "result_rows": len(result_rows),
        "source_script_sha256": sha(Path(__file__)),
    })
    print(json.dumps({"status": "COMPLETED_EXPLORATORY_RUN", "run": str(run_dir), "rows": len(result_rows)}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
