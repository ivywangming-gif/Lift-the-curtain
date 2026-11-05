# Probe Calibration Recheck

Status: **BILATERAL_CALIBRATION_NOT_PASSED**

This is an isolated calibration-only recheck. It is outside the preregistered 250-trial manifest, does not alter the frozen `.85 rad` formal contract, and produces no passability label.

Completion criterion: actual signed hand FK displacement >= `0.095 m`, `TARGET_REACHED`, runtime-finalize valid, no safety stop, and successful retract/neutralize/stand recovery.

| Side | Target m | Actual m | Stop | Runtime | Safe recovery | Safety stop | Recovery gates | Profile |
|---|---:|---:|---|---|---|---|---|---|
| left | 0.1 | 0.09350532293319702 | PROBE_INCOMPLETE | True | True | None | `retract=True neutralize=True stand=True` | `FULL_ARM_FK_10CM_LEFT_CALIBRATION` |
| right | 0.1 | 0.10037761926651001 | TARGET_REACHED | True | True | None | `retract=True neutralize=True stand=True` | `FULL_ARM_FK_10CM_RIGHT_CALIBRATION` |

Interpretation: a right-side candidate reached the target in this run, while the left-side candidate remained incomplete. The bilateral calibration gate therefore remains unmet; neither calibration output is included in formal benchmark metrics.

Raw summaries are preserved under `outputs/round4070_fixed10cm_judgment/calibration_target_fk10cm/`.
