"""Causal, reproducible sensor sensitivity transformations, not new episodes.

Ranges are modest simulation sensitivity assumptions, NOT hardware calibration.
Commands/actions remain exact. FK is an independent robot-state proxy here; this
is not a complete physically coupled encoder-to-FK sensor simulator.
"""
from __future__ import annotations
import hashlib
import numpy as np

CONFIG = dict(q_noise_std_rad=.001, qdot_noise_std_rad_s=.01,
              effort_noise_fraction=.02, effort_bias_fraction=.01,
              imu_rate_noise_std_rad_s=.005, imu_orientation_noise_std_rad=.001,
              timestamp_jitter_bound_s=.002, observation_delay_max_s=.02,
              delay_distribution="per-episode 0 or20ms, equal probability; actual saved samples only",
              drop_probability=.01, minimum_timestamp_increment_s=1e-6,
              scope="SIMULATION_SENSITIVITY_NOT_HARDWARE_CALIBRATED")


def fit_scale(raw, names, train_indices, trial_ids):
    if not train_indices:
        raise ValueError("Observation scales require nonempty training IDs")
    columns = [i for i, name in enumerate(names) if name.startswith("effort/")]
    values = np.concatenate([raw[i][:, columns] for i in train_indices])
    scale = np.maximum(np.maximum(values.std(0), np.mean(np.abs(values), axis=0)), 1e-3)
    return dict(effort_columns=columns, effort_scale=scale.tolist(),
                scale_definition="max(train std,train mean absolute,0.001 N.m) per effort channel",
                scale_train_ids=[trial_ids[i] for i in train_indices], config=dict(CONFIG))


def episode_seed(master_seed, trial_id):
    salt = int(hashlib.sha256(str(trial_id).encode()).hexdigest()[:12], 16)
    return (int(master_seed)+salt) % (2**63-1)


def transform(x, times, names, fitted, seed, cutoff_s=2.):
    x, times = np.asarray(x), np.asarray(times)
    # Select by TRUE acquisition cutoff before jitter; never bring a future row
    # into a prefix by perturbing its timestamp backwards.
    eligible = np.flatnonzero(times <= cutoff_s+1e-9)
    if len(eligible) < 2:
        raise ValueError("Too few true pre-cutoff observations for sensor DR")
    x, t = x[eligible].copy(), times[eligible].copy()
    rng = np.random.default_rng(seed)
    requested_delay = float(rng.choice([0., CONFIG["observation_delay_max_s"]]))
    sources = np.searchsorted(t, t-requested_delay+1e-9, side="right")-1
    dropped = rng.random(len(t)) < CONFIG["drop_probability"]
    receipt, selected, last_source = [], [], -1
    for i, source in enumerate(sources):
        if source < 0 or source <= last_source or dropped[i]:
            continue
        if t[i]-t[source] > CONFIG["observation_delay_max_s"]+1e-7:
            # A 10 Hz log cannot simulate a <=20 ms delay by inventing samples.
            continue
        receipt.append(i)
        selected.append(int(source))
        last_source = int(source)
    if len(selected) < 2:
        # Do not invent observations or silently disable the requested transform.
        raise ValueError("OBSERVATION_DR_INSUFFICIENT_CAUSAL_ROWS_AFTER_DELAY_DROP")
    receipt, selected = np.asarray(receipt), np.asarray(selected)
    output = x[selected].astype(np.float32)
    receipt_time = t[receipt]
    jitter = rng.uniform(-CONFIG["timestamp_jitter_bound_s"], CONFIG["timestamp_jitter_bound_s"], len(receipt))
    reported = np.maximum(0., receipt_time+jitter)
    for j in range(1, len(reported)):
        reported[j] = max(reported[j], reported[j-1]+CONFIG["minimum_timestamp_increment_s"])
    # Sensor values are perturbed after the delayed source has been selected.
    for j, name in enumerate(names):
        std = (CONFIG["q_noise_std_rad"] if name.startswith("q/") else
               CONFIG["qdot_noise_std_rad_s"] if name.startswith("qdot/") else
               CONFIG["imu_orientation_noise_std_rad"] if name.startswith("imu_rpy/") else
               CONFIG["imu_rate_noise_std_rad_s"] if name.startswith(("imu_angular_velocity_b/", "gyro_")) else 0.)
        if std:
            output[:, j] += rng.normal(0., std, len(output))
    effort = fitted["effort_columns"]
    scale = np.asarray(fitted["effort_scale"])
    if effort:
        bias = rng.normal(0., CONFIG["effort_bias_fraction"]*scale)
        noise = rng.normal(size=(len(output), len(effort)))*CONFIG["effort_noise_fraction"]*scale
        output[:, effort] += bias+noise
    else:
        bias = np.empty(0)
    # Received commands are known at the receipt instant; do not time-delay the
    # robot's own command/action plan along with delayed sensory feedback.
    exact_prefixes = ("q_cmd/", "previous_control_action/", "current_policy_action/", "commanded_", "planned_followup_", "followup_", "probe_side/", "command_progress/")
    for j, name in enumerate(names):
        if name.startswith(exact_prefixes):
            output[:, j] = x[receipt, j]
        if name in ("control_age_s", "command_walk_mode"):
            output[:, j] = x[receipt, j]
        if name in ("elapsed_s", "elapsed/s"):
            output[:, j] = reported
        elif name in ("dt_s", "dt/s"):
            output[:, j] = np.diff(reported, prepend=reported[0])
    audit = dict(seed=int(seed), original_rows=len(times), cutoff_eligible_rows=len(eligible), received_rows=len(output),
                 source_indices=eligible[selected].tolist(), receipt_indices=eligible[receipt].tolist(),
                 source_acquisition_times=t[selected].tolist(), receipt_true_times=receipt_time.tolist(),
                 reported_times=reported.tolist(), effort_bias_nm=bias.tolist(), config=dict(CONFIG),
                 requested_episode_delay_s=requested_delay,
                 minimum_actual_delay_s=float(np.min(receipt_time-t[selected])),
                 max_actual_delay_s=float(np.max(receipt_time-t[selected])),
                 delay_note="past source only, actual age capped at20ms; sparse logs can make this diagnostic unavailable",
                 invariant="no future source; no interpolation; no new labels, IDs or independent episodes")
    assert np.all(t[selected] <= receipt_time) and np.all(np.diff(reported) > 0)
    return output, reported, audit


def transform_indices(raw, times, names, fitted, indices, trial_ids, master_seed, cutoff_s=2.):
    transformed, stamps, audits = list(raw), list(times), []
    for i in indices:
        transformed[i], stamps[i], audit = transform(raw[i], times[i], names, fitted, episode_seed(master_seed, trial_ids[i]), cutoff_s)
        audits.append(dict(trial_id=trial_ids[i], **audit))
    return transformed, stamps, dict(master_seed=int(master_seed), fitted_scale=fitted, episodes=audits)
