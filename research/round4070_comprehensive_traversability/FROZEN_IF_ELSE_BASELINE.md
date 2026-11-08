# Frozen Round 4070-G If–Else Baseline

## Freeze record

- Baseline identifier: `E1_FIXED_G_V1_REAL_DT`
- Freeze/replay time (UTC): `2026-11-08T03:24:45.869948+00:00`
- Current Git HEAD at replay: `8540f44c718fb9c88940e43e220af0ca05d1d210`
- Branch: `codex/round4070-g-comprehensive-traversability`
- Worktree was dirty at replay: `True`
- Status: `IN_PROGRESS_NOT_COMPLETE`
- This artifact is a deployment-rule baseline, not a completed benchmark or a claim of method advantage.

## Scope and inputs

The replay covers **21** legal current-primary G1 histories: **18** admitted histories from the primary manifest plus **3** histories retained in the pending collision-audit manifest. Superseded pilot histories are excluded. The pending B10/D01 rows remain `LABEL_UNKNOWN` and are never promoted by this evaluator. Two EMPTY controls are used only as same-hand response references.

The evaluator consumes the frozen 252-channel sensor schema, actual observation timestamps, and the legal physical probe/probe-hold window. Time is made relative to the first recorded sample. The cutoff is at most 2.0 seconds; only observed samples are used, with no interpolation and no terminal extrapolation.

Named hand channels are resolved directly:

- left: `left_hand_delta_initial_b/x`, `left_hand_velocity_initial_b/x`
- right: `right_hand_delta_initial_b/x`, `right_hand_velocity_initial_b/x`

The engineering coordinate is the initial robot-body forward x axis. No scene ID, physical material, topology, contact truth, label, follow-up result, or future observation enters the decision.

## Frozen thresholds and rule order

The exact threshold source is `configs/proprioceptive_thresholds.json`:

```json
{
  "blocked_effort_deviation_min": 2.2690875035237013,
  "blocked_progress_max_m": 0.009077692776918412,
  "blocked_progress_ratio_max": 0.6,
  "blocked_stall_duration_s": 0.15,
  "low_hand_speed_mps": 0.004,
  "minimum_samples": 4,
  "responsive_progress_m": 0.0076811246573925025,
  "responsive_progress_ratio_min": 0.75
}
```

The if–else order is fixed:

1. Fewer than `minimum_samples` observations -> `UNCERTAIN`.
2. Progress and same-hand reference ratio satisfy the responsive thresholds -> `RESPONSIVE`.
3. Otherwise, low progress/ratio together with effort deviation and real-time low-speed duration satisfy the blocked thresholds -> `BLOCKED`.
4. Otherwise -> `UNCERTAIN`.

The stall term is `sum(diff(actual_time) * low_speed_at_left_endpoint)`. The corrected unit is seconds; the former fixed-sample-period assumption is not used.

## Native outputs and upper-layer mapping

The raw E1 output is exactly one of `RESPONSIVE`, `BLOCKED`, or `UNCERTAIN`.

| Raw E1 output | Upper action | Offline operational encoding |
| --- | --- | --- |
| `RESPONSIVE` | `CAUTIOUS_ADVANCE` | `PASSABLE` only for metric compatibility |
| `BLOCKED` | `RETRACT` | `BLOCKED` |
| `UNCERTAIN` | `RETRACT` | `UNCERTAIN` |

`RESPONSIVE` is not a ground-truth `PASSABLE` label. The `PASSABLE` spelling exists only as an explicit numeric/metric compatibility encoding in the existing learning adapter.

Same-hand EMPTY references are frozen from the following legal controls:

- left: `G_EMPTY_left_empty_control_00_nominal_s74800_rng1_act2__attempt02`; progress `0.0727402865887` m
- right: `G_EMPTY_right_empty_control_00_nominal_s74800_rng1_act2__attempt01`; progress `0.0872250050306` m

## Replay and compatibility audit

- Run directory: `outputs/round4070_comprehensive_traversability/if_else_baseline_61154c50277590796dc1`
- Current-primary histories replayed: `21`
- Raw decision counts: `{"BLOCKED": 4, "RESPONSIVE": 15, "UNCERTAIN": 2}`
- Known labels across all scopes: `14` (including `2` EMPTY controls)
- Non-control core metric rows: `12`
- Core decision coverage: `0.8333333333333334`
- Existing `engineering_rules.py` compatibility: `PASS`
- Numeric feature compatibility: `PASS_FLOAT_PRECISION_TOLERANCE` (maximum absolute delta `3.7478e-06`, tolerance `1e-05`)
- Contract/regression test suite: `PASS`

The historical E1 implementation, the G compatibility adapter, the semantic observation module, and the threshold file were hashed at freeze time:

```text
canonical_e1_fixed: 7c264b68f74df161e3d317ed858acba0f91daef888da9c25f7600414d62fefb2
contract_test_source: 7bb8b6ac343f42520f8bf00b68b197024f27f51ed4ed172ef75868fce64b0488
g_compatibility_adapter: be495dffe9b9e30aa44d417a1505bb2ecf39d753d35aa26f73fdd6921750f0f7
historical_e1_reference: af997e57b06895beffb71cc348ab1a384f59fd393d552017c47a97b8d60a61d8
proprioceptive_thresholds: 7970fff8fafbf17a947585c4dfeee235a6807be88d048d3df17c329c5dd3ddf7
secondary_rule_freeze: f217241714ddb1a05150f8d834cf2d9757e80b7a786c1e5943ec8e573e81edc8
semantic_observation_reference: 32585387e214483d710e6c1915ad037ee79381598420b684f095066418632c5a
```

The replay command is:

```powershell
& C:\conda_envs\isaaclab\python.exe scripts/round4070_g/run_if_else_baseline.py
```

The runner executes `python -m unittest scripts/round4070_g/test_if_else_baseline.py -q` before publishing this report; the recorded result is `PASS`.

No G1 process was started by this replay, and the global G1 pause files remain outside the evaluator's write set.
