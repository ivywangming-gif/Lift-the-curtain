# Round 4070-D Run Instructions

All runs are headless, one Isaac environment, one process at a time. Existing C/B outputs are read-only references and are never overwritten.

## 1. Generate and validate the frozen manifest

```powershell
& 'C:\conda_envs\isaaclab\python.exe' scripts/round4070_fixed10cm_judgment/generate_manifest.py
& 'C:\conda_envs\isaaclab\python.exe' scripts/round4070_fixed10cm_judgment/validate_contract.py
```

The generated manifest contains 250 registered trials: 200 core rows, 20 empty/rigid controls and 30 matched no-wall controls. The launch budget is 260.

## 2. Run physics-only gates first

```powershell
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -File scripts/round4070_fixed10cm_judgment/run_physics_matrix.ps1 -Steps 720 -OverallTimeoutSeconds 900
```

The matrix is serial. A failed gate stops that condition branch; it does not become a G1 negative sample.

## 3. Run safety validation

Run the first registered left/right trials for `EMPTY_CONTROL`, `RIGID_CONTROL` and `FABRIC_NOMINAL` with `-ProbeValidationOnly`. They must all have a complete summary, `PROBE_10CM`, safe recovery and explicit target/actual/post-contact/stop fields before formal bilateral comparisons begin. The target is 0.10 m, but the protocol does not force actual displacement; a safe `PROBE_INCOMPLETE` is recorded and never becomes an automatic BLOCKED label.

```powershell
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -Command '$ids=@("<trial-id-1>","<trial-id-2>"); & "./scripts/round4070_fixed10cm_judgment/run_serial_trials.ps1" -TrialId $ids -ProbeValidationOnly -OverallTimeoutSeconds 1200'
& 'C:\conda_envs\isaaclab\python.exe' scripts/round4070_fixed10cm_judgment/validate_contract.py
```

## 4. Launch frozen trials

```powershell
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -Command '$ids=@("<trial-id>"); & "./scripts/round4070_fixed10cm_judgment/run_serial_trials.ps1" -TrialId $ids -OverallTimeoutSeconds 1200'
```

The wrapper refuses IDs outside the manifest, refuses non-`PROBE_10CM` modes, preserves non-empty incomplete directories and skips only a complete summary. Do not use parallel launches.

## 5. Evaluate offline

```powershell
& 'C:\conda_envs\isaaclab\python.exe' scripts/round4070_fixed10cm_judgment/validate_contract.py
& 'C:\conda_envs\isaaclab\python.exe' scripts/round4070_fixed10cm_judgment/evaluate_campaign.py
```

E0/E1/E2 consume only deployment observation histories. Evaluation truth is read after the decision and recorded in `LABEL_AUDIT.csv`. Runtime-invalid, protocol-mismatched, safety-aborted, `PROBE_INCOMPLETE` and physics-invalid rows remain `UNKNOWN`. Safety-validation-only outputs and all calibration-only outputs, including custom target profiles, are excluded from benchmark denominators.

## 6. Optional calibration-only full-arm target

`run_episode.py` accepts `--probe-target-upper-json` only together with `--probe-validation-only`. The JSON must contain a finite 14-element `upper_target`, an optional matching `probe_side`, and a descriptive `profile`. This override is for an isolated target-trajectory feasibility check; it does not amend `PREREGISTERED_MANIFEST.json` and must not be passed to formal benchmark trials.

The Round 4070-D recheck outputs are preserved under `outputs/round4070_fixed10cm_judgment/calibration_target_fk10cm/` and summarized in `PROBE_CALIBRATION_RECHECK.md`.
