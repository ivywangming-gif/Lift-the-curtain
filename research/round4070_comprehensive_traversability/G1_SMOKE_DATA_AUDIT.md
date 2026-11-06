# G1 actual logging smoke audit

Completed probe-valid histories inspected: 10. No model fitting or simulation was started by this audit.

This report audits raw timing and legal robot channels only. Its displayed labels are original online outcomes, not independently approved primary labels. All V3 pilots are superseded and excluded from primary supervision, normalization, auxiliary response fitting and evaluation. See dataset_status.json and LABEL_VALIDITY_REPORT.md for formal admission.

- G_A01_left_coverage_00_nominal_s70000 / attempt01: label=LABEL_UNKNOWN, runtime=True, probe=True, physics=True; 102 real probe samples, median dt=0.020000 s (50.0 Hz), span=1.985 s; previous-action updates verified=101; causal cutoff=True, actual upper q_cmd=True; acceleration=SIGNAL_PRESENT_NOT_HARDWARE_CALIBRATED.
- G_A01_right_coverage_00_nominal_s70000 / attempt01: label=LABEL_UNKNOWN, runtime=True, probe=True, physics=True; 102 real probe samples, median dt=0.020000 s (50.0 Hz), span=1.985 s; previous-action updates verified=101; causal cutoff=True, actual upper q_cmd=True; acceleration=SIGNAL_PRESENT_NOT_HARDWARE_CALIBRATED.
- G_A02_left_coverage_00_nominal_s70100 / attempt01: label=LABEL_UNKNOWN, runtime=True, probe=True, physics=True; 102 real probe samples, median dt=0.020000 s (50.0 Hz), span=1.985 s; previous-action updates verified=101; causal cutoff=True, actual upper q_cmd=True; acceleration=SIGNAL_PRESENT_NOT_HARDWARE_CALIBRATED.
- G_A02_right_coverage_00_nominal_s70100 / attempt01: label=LABEL_UNKNOWN, runtime=True, probe=True, physics=True; 102 real probe samples, median dt=0.020000 s (50.0 Hz), span=1.985 s; previous-action updates verified=101; causal cutoff=True, actual upper q_cmd=True; acceleration=SIGNAL_PRESENT_NOT_HARDWARE_CALIBRATED.
- G_C01_left_coverage_00_nominal_s71900 / attempt01: label=PASSABLE_FOR_ACTION, runtime=True, probe=True, physics=True; 102 real probe samples, median dt=0.020000 s (50.0 Hz), span=1.985 s; previous-action updates verified=101; causal cutoff=True, actual upper q_cmd=True; acceleration=SIGNAL_PRESENT_NOT_HARDWARE_CALIBRATED.
- G_C01_right_coverage_00_nominal_s71900 / attempt01: label=PASSABLE_FOR_ACTION, runtime=True, probe=True, physics=True; 102 real probe samples, median dt=0.020000 s (50.0 Hz), span=1.985 s; previous-action updates verified=101; causal cutoff=True, actual upper q_cmd=True; acceleration=SIGNAL_PRESENT_NOT_HARDWARE_CALIBRATED.
- G_D01_left_coverage_00_nominal_s72700 / attempt01: label=BLOCKED, runtime=True, probe=True, physics=True; 102 real probe samples, median dt=0.020000 s (50.0 Hz), span=1.995 s; previous-action updates verified=101; causal cutoff=True, actual upper q_cmd=True; acceleration=SIGNAL_PRESENT_NOT_HARDWARE_CALIBRATED.
- G_D01_right_coverage_00_nominal_s72700 / attempt01: label=BLOCKED, runtime=True, probe=True, physics=True; 102 real probe samples, median dt=0.020000 s (50.0 Hz), span=1.995 s; previous-action updates verified=101; causal cutoff=True, actual upper q_cmd=True; acceleration=SIGNAL_PRESENT_NOT_HARDWARE_CALIBRATED.
- G_EMPTY_left_empty_control_00_nominal_s74800 / attempt01: label=PASSABLE_FOR_ACTION, runtime=True, probe=True, physics=True; 102 real probe samples, median dt=0.020000 s (50.0 Hz), span=1.995 s; previous-action updates verified=101; causal cutoff=True, actual upper q_cmd=True; acceleration=SIGNAL_PRESENT_NOT_HARDWARE_CALIBRATED.
- G_EMPTY_right_empty_control_00_nominal_s74800 / attempt01: label=PASSABLE_FOR_ACTION, runtime=True, probe=True, physics=True; 102 real probe samples, median dt=0.020000 s (50.0 Hz), span=1.995 s; previous-action updates verified=101; causal cutoff=True, actual upper q_cmd=True; acceleration=SIGNAL_PRESENT_NOT_HARDWARE_CALIBRATED.

The acceleration candidate is retained robot state, not hardware IMU calibration. Unless an explicit independent validation flag is present, it remains unavailable to every model with an explicit missingness mask. Primary UNKNOWN response admission additionally requires the frozen V4 RNG protocol and an independent VERIFIED whole-body collision audit. A valid robot probe alone does not establish that physical validity or whole-episode runtime validity.

## V5 actual EMPTY-left, 2026-11-06

The preceding ten-episode section is the superseded pilot snapshot. The first accepted V5 episode is `G_EMPTY_left_empty_control_00_nominal_s74800_rng1_act2__attempt02`, collected from 15:28 UTC. It is a control, not a core positive configuration. Formal import contains one control PASSABLE, zero core episodes and ten isolated pilots; all eight split endpoints remain NOT_EVALUABLE. No model was fitted by this audit.

Actual seed 74800 replay and readback passed. The sampled additive robot offsets were y=0.000911096268606405 m and yaw=-0.0006024924879857415 rad; actual root y was -0.003088903846219182 m. EMPTY has zero cloth nodes, so this validates the robot reset and empty node-array contract, not cloth randomization. Native binding verified 39 actual shapes: 27 exact and 12 conservative envelopes, against library_v3 and the actual per-trial inventory.

The exported probe contains 100 real rows by 252 legal features: 75 probe and 25 hold, median dt 0.020 s, total observed span 1.98 s. All 99 adjacent previous-action updates match, and upper commanded reference exactly equals the actually sent q_cmd subset. Effort is the applied-torque proxy; acceleration is unavailable for all 100 rows. No geometry or followup truth enters seq.

Body evidence consists of 13 immutable blocks: twelve 500-step blocks and a 68-step block, covering every actual step 1–6068. A separate actual reset snapshot is step zero. All block/index hashes, phase ledger and absolute start/end anchors passed the independent streaming validator. Recorded actual callback dt was 0.004999999888241291 s, the float32 representation of the nominal 0.005 s. Initial simulator clock was 0.009999999776482582 s; final clock was 30.349999321624637 s. The 30.34 s episode and reset warmup offset are explicitly distinct; no clock was interpolated or rewritten.

The independent collision audit evaluated 6,069 states including reset. It verified the full supported discrete-state scope; the action audit checked 3,416 entry-to-hold states without a tube violation and found maximum adjacent witness x displacement 0.0010021490176922443 m. Actual whole-body conservative minimum x at the final hold was 1.1532347088833943 m, exceeding the unchanged 0.950 m threshold. The final hold, release, neutralization and recovery each executed 300 actual steps. Runtime, probe, physics and safe recovery are all valid, and formal PASSABLE is supported for this EMPTY control only.

The compressed 200 Hz history totals 5,998,649 bytes; all then-present run files totaled 12,679,352 bytes. Peak measured working set was 2,875.848 MiB, private memory 7,280.113 MiB and GPU memory 2,663 MiB; minimum system-available RAM was 4,700.785 MiB. These are actual EMPTY measurements, not estimates for cloth scenes.

The original watchdog recorded `CLEANUP_TIMEOUT_AFTER_FINALIZE_30s`, final stage PASS, and total process wall time 177.344 s. This post-finalization shutdown failure remains preserved separately from the finalized simulation evidence. A subsequent authorized cleanup-only callback now sets the established IsaacLab stop flag after all recorders, summaries and finalization have completed, before the existing sim.stop/APP.close. Its success still requires the next actual run. The initial App-before-start admission failure remains attempt01 and is not deleted or converted into data. Repair evidence is archived under `protocol_repairs/v5_runtime_gate_flag_20261106` and `protocol_repairs/v5_post_finalize_cleanup_20261106`.

## V5 actual EMPTY-right and cleanup validation

`G_EMPTY_right_empty_control_00_nominal_s74800_rng1_act2__attempt01` started at 15:45:34 UTC and completed with guard return code 0, no timeout, and 127.7388867 s process wall time. Its real `post_finalize_cleanup.json` confirms all recorders closed and history finalized before the stop flag was set. The watchdog's underlying process exit-code field is null; the recorded guard return code is zero. These are separate fields, not an inferred process exit code. The cleanup-only patched source SHA is `8aa2ddeaea45f9feec5c4534295039f242613ace836461afa09e18e977190dd4`.

The right-hand probe also has 100 real 50 Hz rows by 252 features, with 75 probe/25 hold rows, 1.98 s span, 99 verified previous-action transitions and 100 unavailable-acceleration masks. Probe and followup hand are both right. Its actual RNG offsets, full root pose and all reset-array contents equal the independently recorded left control at the same seed; file-container hashes are not substituted for array equality.

Independent evidence verifies 5,698 poststep body states plus the actual reset state. The action tube has no violation, maximum adjacent witness x displacement is 0.0010674288835432755 m, and the final whole-body conservative minimum x is 1.2525206672782205 m against the unchanged 0.950 m threshold. Safe recovery is verified. Formal import now contains two PASSABLE controls, one per hand, zero core episodes and ten isolated pilots. No benchmark or left core-positive coverage is claimed from these controls.

Actual cleanup validation and source hashes are recorded in `protocol_repairs/v5_post_finalize_cleanup_20261106/actual_right_cleanup_validation.json`. All eight primary split endpoints remain NOT_EVALUABLE, and no model fitting was started.

## V5 actual C01 bilateral cloth, 2026-11-06 16:15 UTC snapshot

The preceding sections remain historical snapshots. Both `G_C01_left_coverage_00_nominal_s71900_rng1_act2__attempt01` and `G_C01_right_coverage_00_nominal_s71900_rng1_act2__attempt01` now have independent VERIFIED collision/action audits and formal PASSABLE_FOR_ACTION labels. Runtime, probe, physics and safe recovery are valid. They provide one core positive episode per hand, but share one physical configuration and one seed; they are not two independent positive configurations. Current formal data comprise these two core positives plus two EMPTY controls. Ten superseded pilots remain isolated. The actual-start inventory contains 15 launches and 14 finalized outcomes, including the preserved pre-App admission failure without metadata/outcome; metadata-import excluded=0 does not mean every launch succeeded.

Seed 71900 replay and readback passed for both cloth episodes. The actual additive robot offsets are y=0.0006948812198332803 m and yaw=0.0003754698614306098 rad. Each has 825 cloth nodes: 800 free and 25 fixed. Free-node injected velocities span -0.0009995464934036136 to 0.0009983050404116511 m/s, while every fixed-node injected velocity is exactly zero. Both hands have identical robot draws, free/fixed index arrays, applied node velocities and readback node velocities, checked by array contents rather than container hashes. The shared applied velocity-array SHA is `075622888d9a9b0527a2a11192f166c733b4ceb21de66a7069d3316ffdd9ef6b`. This validates actual cloth randomization and the matched-seed implementation; it does not establish seed diversity.

Both exported probes contain 102 actual observations by 252 legal features: 76 physical_probe and 26 physical_probe_hold rows over 1.99 s. Median dt is 0.020 s; actual phase-boundary intervals include 0.010 s, so the record is nominal 50 Hz with additional real boundary samples, not a uniform reconstructed grid. Each has 101 verified adjacent previous-action transitions, the actual commanded upper reference matches sent q_cmd, and all 102 acceleration observations remain explicitly unavailable. Effort remains the applied-torque proxy. No cloth geometry, wall/native truth, future followup, action label or world pose enters seq.

| Actual evidence | Left C01 | Right C01 |
| --- | ---: | ---: |
| Actual poststep states at 200 Hz | 9,110 | 6,661 |
| States independently audited including reset | 9,111 | 6,662 |
| Immutable history chunks | 19 | 14 |
| Nominal episode duration, s | 45.550 | 33.305 |
| Final actual simulator clock, s | 45.559998981654644 | 33.31499925535172 |
| Original narrow-phase triangle tests | 248,149 | 99,739 |
| Closed intersections / strict 0, 1, 5 mm counts | 0 / 0, 0, 0 | 0 / 0, 0, 0 |
| Guarded entry-to-hold states | 6,448 | 3,999 |
| Maximum adjacent witness x displacement, m | 0.0009767831891749346 | 0.0007491093388105607 |
| Final-hold actual whole-body conservative min x, m | 1.051496012033043 | 1.1934766399635564 |
| Original required clearance threshold, m | 0.950 | 0.950 |
| Final-hold actual physics step | 8,210 | 5,761 |
| Compressed history-chunk bytes | 164,630,838 | 117,976,385 |
| Legal exported NPZ bytes | 130,108 | 130,410 |
| Then-present run-directory bytes | 182,574,399 | 132,178,778 |
| Recorded watchdog wall time, s | 242.0945506 | 182.4958036 |
| Peak measured process working set, MiB | 2,903.129 | 2,898.328 |
| Peak measured process private memory, MiB | 8,127.066 | 8,121.195 |
| Peak recorded GPU memory, MB | 3,514 | 3,513 |
| Minimum system-available RAM, MiB | 4,524.488 | 4,844.836 |

The initial actual clock is 0.009999999776482582 s in both runs; the full history retains the actual float32-derived 0.004999999888241291 s callback interval and independent absolute step anchors. Resource telemetry's last sampled step/time counters are coarser (left 9,100 / 45.505 s, right 6,640 / 33.205 s), so they are not substituted for full history totals. Run-directory byte totals are a measured snapshot and may grow when additional derived audits are archived; immutable chunk byte totals are separately stated. Both watchdogs record no timeout and an underlying process exit-code field of null, not a claimed simulator exit code of zero.

Every recorded state and the measured reset state were checked using the bound native library: all 39 actual shapes across 33 collision-bearing links, with poses retained for all 39 links. The geometry comprises 27 exact shapes and 12 conservative native-vertex envelopes. The latter certify separation and conservative clearance lower bounds only; overlap would remain NOT_VERIFIED. Both C01 scenes independently record VERIFIED_NO_WALL. Their original audit SHAs are left `ecb8a5d308581beb24998b35885cb0fcc6db6c410b2a97b960c503862b10f31e` and right `83d29eccc4e2cf513a460d5673656ad0c1464d434f6dc1cd0c0fa9abcea0f475`. The source-bound copies in the dataset preserve those specific audits if a later implementation comparison creates a new derived audit.

The [bilateral probe plot](PRIMARY_BILATERAL_PROBE_AUDIT.png) and its [source/hash record](PRIMARY_BILATERAL_PROBE_AUDIT.json) use only these two actual legal NPZ histories and official active-arm joint-name mappings. They are descriptive robot-response diagnostics, not model accuracy, a left/right equivalence proof, or a comparison across physical conditions.

These results establish the frozen V5 action's recorded 200 Hz discrete-state validity for two executions of one configuration. They do not establish continuous-time nonpenetration, hardware safety, success across seeds/poses/materials, independent four-seed confirmation, or a balanced classifier benchmark. No reliable core negative exists at this snapshot. All eight split endpoints remain NOT_EVALUABLE. The frozen learning run `learning_campaign_12ac229296d8078b9d9a` therefore records 1,520 main tasks and 320 partial-label tasks as BLOCKED with zero fits; the lack of class/configuration support is retained rather than manufacturing labels or weakening admission.
