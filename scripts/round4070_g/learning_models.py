"""Small, explicitly named existing model families for the G benchmark.

Transformer is a history-classifier adaptation, not a PROBE reproduction.
ResidualGRU/OldM2 inherit the CoPRE-style idea, not CoPRE's contact estimator.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


METHOD_GROUPS = {
    "standard": ["Logistic", "GradientBoosting", "GaussianAR1", "StatsMLP", "HistoryMLP", "GRU", "LSTM", "TCN", "Transformer", "ResidualGRU", "OldM2"],
    "contrastive": ["GRUSupCon", "GRUMatched", "MatchedNoHardNegative", "MatchedNoCrossPositive"],
    "temporal": ["GRUCommand", "GRUTimeProgress", "GRUParamMatched", "GRUShuffled", "Instantaneous", "GRUPrefix05", "GRUPrefix10", "GRUOrderless", "GRU10Hz", "GRU20Hz", "DurationOnlyLogistic"],
    "response": ["GRUResponseAux", "GRUPretrainFinetune"],
    "cross_hand": ["GRUArm", "GRUHandRemap"],
    "observation_dr": ["GRUObsDR"],
}
METHODS = sum(METHOD_GROUPS.values(), [])
ENGINEERING = ["AlwaysPassable", "AlwaysBlocked", "E0", "E1TimeCorrected", "E2TimeCorrected"]
CONTRASTIVE = set(METHOD_GROUPS["contrastive"])
MATCHED = {"GRUMatched", "MatchedNoHardNegative", "MatchedNoCrossPositive"}
RESPONSE = {"GRUResponseAux", "GRUPretrainFinetune"}


def mask_for(lengths, steps):
    return torch.arange(steps, device=lengths.device)[None, :] < lengths[:, None]


def masked_stats(x, lengths):
    mask = mask_for(lengths, x.shape[1]).unsqueeze(-1)
    mean = (x * mask).sum(1) / lengths[:, None]
    variance = ((x-mean[:, None]).square()*mask).sum(1)/lengths[:, None]
    minimum = x.masked_fill(~mask, torch.inf).min(1).values
    maximum = x.masked_fill(~mask, -torch.inf).max(1).values
    return torch.cat((mean, variance.clamp_min(0).sqrt(), minimum, maximum), 1)


class SmallModel(nn.Module):
    def __init__(self, method, inputs, max_steps=128, hidden=64, arm=None, body=None, response_indices=None, parameter_target=None):
        super().__init__()
        self.method, self.arm, self.body = method, arm or [], body or []
        self.response_indices = response_indices or []
        self.hidden = hidden
        self.encoder = None
        if method == "GRUParamMatched" and parameter_target:
            # Fixed architecture-based budget matching; no label/test search.
            hidden = min(range(8, 129), key=lambda h: abs((3*h*(inputs+h+2)+h+1)-parameter_target))
            self.hidden = hidden
        if method == "StatsMLP":
            self.encoder = nn.Sequential(nn.Linear(inputs*4, hidden), nn.ReLU())
        elif method == "HistoryMLP":
            self.encoder = nn.Sequential(nn.Linear(inputs, 24), nn.ReLU())
            self.history = nn.Sequential(nn.Linear(max_steps*24, hidden), nn.ReLU())
        elif method in ("Instantaneous", "GRUOrderless"):
            self.encoder = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU())
        elif method == "TCN":
            self.encoder = nn.Sequential(nn.Conv1d(inputs, 64, 3, padding=2), nn.ReLU(), nn.Conv1d(64, hidden, 3, dilation=2, padding=4), nn.ReLU())
        elif method == "Transformer":
            self.project = nn.Linear(inputs, hidden)
            layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=128, dropout=.1, batch_first=True, norm_first=False)
            self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
            pos = torch.arange(max_steps).float()[:, None]
            div = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000.)/hidden))
            pe = torch.zeros(max_steps, hidden)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos*div), torch.cos(pos*div)
            self.register_buffer("position", pe)
        elif method == "OldM2":
            if not self.arm or not self.body:
                raise ValueError("OldM2 requires actual arm and body channel indices")
            self.encoder = nn.GRU(len(self.arm), hidden, batch_first=True)
            self.condition = nn.Sequential(nn.Linear(len(self.body), 32), nn.ReLU(), nn.Linear(32, 2*hidden))
        else:
            recurrent = nn.LSTM if method == "LSTM" else nn.GRU
            self.encoder = recurrent(inputs, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.projector = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 32)) if method in CONTRASTIVE else None
        self.response_head = nn.Linear(hidden, len(self.response_indices)) if method in RESPONSE else None

    def forward(self, x, lengths, return_aux=False):
        method = self.method
        mask = mask_for(lengths, x.shape[1])
        sequence = None
        if method == "StatsMLP":
            representation = self.encoder(masked_stats(x, lengths))
        elif method == "HistoryMLP":
            z = self.encoder(x) * mask.unsqueeze(-1)
            representation = self.history(z.flatten(1))
        elif method == "Instantaneous":
            representation = self.encoder(x[torch.arange(len(x), device=x.device), lengths-1])
        elif method == "GRUOrderless":
            representation = self.encoder((x*mask.unsqueeze(-1)).sum(1)/lengths[:, None])
        elif method == "TCN":
            # Causal left-context padding is implemented by cropping the suffix.
            first = self.encoder[1](self.encoder[0](x.transpose(1, 2)))[:, :, :x.shape[1]]
            encoded = self.encoder[3](self.encoder[2](first))[:, :, :x.shape[1]].transpose(1, 2)
            representation = (encoded*mask.unsqueeze(-1)).sum(1)/lengths[:, None]
        elif method == "Transformer":
            z = self.project(x) + self.position[:x.shape[1]]
            # All tokens are already before the same fixed legal decision cutoff.
            encoded = self.encoder(z, src_key_padding_mask=~mask)
            representation = (encoded*mask.unsqueeze(-1)).sum(1)/lengths[:, None]
        else:
            values = x[:, :, self.arm] if method == "OldM2" else x
            packed = pack_padded_sequence(values, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
            output, state = self.encoder(packed)
            representation = (state[0] if method == "LSTM" else state)[-1]
            if method == "OldM2":
                body = (x[:, :, self.body]*mask.unsqueeze(-1)).sum(1)/lengths[:, None]
                scale, bias = self.condition(body).chunk(2, 1)
                representation = representation * (1+.1*torch.tanh(scale)) + .1*bias
            if return_aux and self.response_head is not None:
                sequence, _ = pad_packed_sequence(output, batch_first=True, total_length=x.shape[1])
        logits = self.head(representation).squeeze(-1)
        if not return_aux:
            return logits
        projected = nn.functional.normalize(self.projector(representation), dim=1) if self.projector is not None else None
        response = self.response_head(sequence) if sequence is not None else None
        return logits, projected, response


def supcon_loss(z, labels, temperature=.1):
    """Single-view supervised contrastive loss; self pairs excluded, no fake pairs."""
    n = len(z)
    if n < 3:
        return z.sum()*0, 0
    identity = torch.eye(n, dtype=torch.bool, device=z.device)
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    logits = (z @ z.T)/temperature
    log_probs = logits - torch.logsumexp(logits.masked_fill(identity, -torch.inf), dim=1, keepdim=True)
    count = positives.sum(1)
    valid = count > 0
    if not valid.any():
        return z.sum()*0, 0
    loss = -((log_probs.masked_fill(~positives, 0)).sum(1)/count.clamp_min(1))[valid].mean()
    return loss, int(valid.sum())


def matched_pair_loss(z, local_pairs, method):
    """Standard cosine contrastive objective on independently verified pairs."""
    terms, positive_n, negative_n = [], 0, 0
    for a, b, kind in local_pairs:
        similarity = (z[a]*z[b]).sum()
        if kind == "cross_configuration_positive" and method != "MatchedNoCrossPositive":
            terms.append(1-similarity)
            positive_n += 1
        elif kind == "hard_negative" and method != "MatchedNoHardNegative":
            terms.append(nn.functional.relu(similarity-.2))
            negative_n += 1
    return (torch.stack(terms).mean() if terms else z.sum()*0), positive_n, negative_n


class VectorStumpBoost:
    """Fixed 80-stage Bernoulli gradient boosting with depth-one regressors."""
    def fit(self, x, y):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=float)
        self.bias = float(np.log((y.sum()+.5)/(len(y)-y.sum()+.5)))
        candidates = [(j, float(t)) for j in range(x.shape[1]) for t in np.unique(np.quantile(x[:, j], [.1,.2,.3,.4,.5,.6,.7,.8,.9]))]
        self.candidates = candidates
        mask = np.stack([x[:, j] <= t for j, t in candidates]).astype(np.float64)
        count = mask.sum(1)
        valid = (count >= 2) & (count <= len(y)-2)
        self.trees, self.curve = [], []
        f = np.full(len(y), self.bias)
        for step in range(80):
            p = 1/(1+np.exp(-np.clip(f, -35, 35)))
            residual = y-p
            left_sum = mask @ residual
            right_sum = residual.sum()-left_sum
            score = np.where(valid, left_sum**2/np.maximum(count, 1)+right_sum**2/np.maximum(len(y)-count, 1), -np.inf)
            if not np.isfinite(score).any():
                break
            k = int(score.argmax())
            a, b = left_sum[k]/count[k], right_sum[k]/(len(y)-count[k])
            j, t = candidates[k]
            self.trees.append((j, t, float(a), float(b)))
            f += .2*np.where(mask[k], a, b)
            self.curve.append(dict(epoch=step, train_loss=float(np.mean(np.logaddexp(0, f)-y*f))))
        return self

    def predict(self, x):
        f = np.full(len(x), self.bias)
        for j, t, a, b in self.trees:
            f += .2*np.where(x[:, j] <= t, a, b)
        return 1/(1+np.exp(-np.clip(f, -35, 35)))

    def export(self):
        return dict(bias=self.bias, trees=self.trees, learning_rate=.2, algorithm="Bernoulli gradient boosting, depth-one, 80 fixed stages")


class GaussianAR1:
    """Small class-conditional diagonal Gaussian AR(1), shared random projection.

    This is a probabilistic temporal baseline, not a calibrated generative truth.
    Its 8-D projection is deterministic and independent of data and class labels.
    """
    def fit(self, x, lengths, y):
        rng = np.random.default_rng(407007)
        self.projection = rng.normal(size=(x.shape[2], min(8, x.shape[2])))/math.sqrt(x.shape[2])
        sequences = [v[:n] @ self.projection for v, n in zip(x, lengths)]
        self.models = []
        for cls in (0, 1):
            seq = [z for z, lab in zip(sequences, y) if lab == cls]
            previous = np.concatenate([z[:-1] for z in seq])
            target = np.concatenate([z[1:] for z in seq])
            pm, tm = previous.mean(0), target.mean(0)
            slope = ((previous-pm)*(target-tm)).sum(0)/(((previous-pm)**2).sum(0)+1.)
            intercept = tm-slope*pm
            variance = np.maximum(((target-previous*slope-intercept)**2).mean(0), .01)
            initial = np.stack([z[0] for z in seq])
            self.models.append((slope, intercept, variance, initial.mean(0), np.maximum(initial.var(0), .01)))
        return self

    def predict(self, x, lengths):
        out = []
        for v, n in zip(x, lengths):
            z = v[:n] @ self.projection
            scores = []
            for slope, intercept, variance, mean0, variance0 in self.models:
                transition = -.5*(np.log(2*np.pi*variance)+(z[1:]-z[:-1]*slope-intercept)**2/variance)
                initial = -.5*(np.log(2*np.pi*variance0)+(z[0]-mean0)**2/variance0)
                scores.append(float((transition.sum()+initial.sum())/(len(z)*z.shape[1])))
            out.append(1/(1+math.exp(float(np.clip(scores[0]-scores[1], -35, 35)))))
        return np.asarray(out)

    def export(self):
        return dict(projection=self.projection.tolist(), class_models=[[v.tolist() for v in m] for m in self.models],
                    likelihood="length-normalized diagonal AR1; equal class prior; calibrate only on calibration split")
