# Lift-the-curtain Engineering Baseline Report

**Date:** October 5, 2026  
**Repository:** `C:\Robotics\Lift-the-curtain`  
**Overall status:** M0 PASS; M1 BLOCKED; M2 and M3 NOT RUN

## 1. Machine and Environment

The experiments used one headless Isaac Sim process and one simulation
environment. No GUI, reinforcement learning, teacher-student training, tactile
sensing, or dexterous hand was used.

| Item | Observed value |
|---|---|
| OS | Windows 11 Enterprise 25H2, build 26200 |
| CPU | 13th Gen Intel Core i9-13900HX, 24 cores / 32 logical CPUs |
| RAM | 32,452 MB reported by Isaac Sim; 3,398 MB free at one M1 startup |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| GPU memory | 7,948 MB reported by Isaac Sim |
| Python | 3.11.15 |
| PyTorch | 2.7.0+cu128 |
| CUDA runtime | 12.8 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | 0.54.4 |
| Conda environment | `C:\conda_envs\isaaclab` |

The required random-agent smoke test was run with `--num_envs 1 --headless`.
The first invocation selected the unrelated Anaconda interpreter because
`CONDA_PREFIX` was not set; the successful equivalent explicitly set the
requested environment before invoking `isaaclab.bat`.

## 2. Source Revisions

| Repository | Revision | Use |
|---|---|---|
| `falcon_g1_reference` | `e393d2c10286b1802d94fc061eaee40abaf0eff2` | Read-only local FALCON-port reference |
| `FALCON-upstream` | `a967a6d8494f57777cf8d266a644ac8e45833301` | Official FALCON contract, ONNX, URDF, MIT-licensed source |
| `MoDeSuite-reference` | `470681a7ff117e487060ad52bc4f199e3fa34474` | Read-only curtain reference |
| `IsaacLab` | `b4c321024792976150ca55fddb26fa34480d974e` | Isaac Lab runtime reference |

Target repository commits created during this work:

- `3dfd4c3` `feat: add FALCON adapter and baseline primitives`
- `363fd99` `test: add headless FALCON conversion and M0 harness`
- `c64a6ed` `fix: use NumPy FALCON graph evaluator in Isaac Sim`
- `18852a3` `fix: map FALCON angular velocity field`
- `d9e6149` `test: record M0 FALCON rollout artifacts`
- `8f6ead1` `feat: add Isaac Sim 5.1 curtain sanity harness`
- `007cbb5` `test: record M1 curtain compatibility results`

## 3. Reused and Newly Written Code

The adapter reuses the previous FALCON-port contract rather than copying the
old repository wholesale:

- The 29-DoF joint order, default pose, gains, limits, action scale, and
  observation layout were taken from the FALCON reference contract.
- The bounded SE(2) feedback logic follows the reference
  `se2_path_controller.py` and `push_path_feedback.py` behavior.
- The official FALCON ONNX model is referenced at
  `C:\Robotics\FALCON-upstream\sim2real\models\falcon\g1_29dof.onnx`.
- The rubber-hand G1 URDF was converted locally with Isaac Lab's converter;
  the external URDF and checkpoint were not copied into this repository.

New code is limited to the local adapter, NumPy ONNX graph evaluator,
SE(2)-feedback wrapper, headless M0 harness, current-API curtain builder, M1
probe harness, configuration, and artifact/report plumbing. The local asset
conversion is reproducible from `scripts/convert_falcon_urdf.py` and is ignored
by Git under `assets/generated/`.

## 4. M0 Results: FALCON, G1, and SE(2) Control

The Isaac Lab random-agent smoke test loaded G1, initialized PhysX and CUDA,
and ran headlessly with one environment without GPU OOM or a visible physics
failure. The final M0 harness then ran four 2-second cases, 400 physics steps
per case, using 5 ms physics and 20 ms control periods.

| Case | Result | Evidence |
|---|---|---|
| Stand | PASS | 400/400 steps, no fall; XY displacement X = -0.039 m |
| Forward | PASS | 400/400 steps, no fall; X displacement = +0.326 m at 0.20 m/s reference |
| Yaw left | PASS | 400/400 steps, no fall; yaw change = +0.328 rad |
| Yaw right | PASS | 400/400 steps, no fall; yaw change = -0.251 rad |

The adapter exposes base pose, body-frame velocity, joint position, joint
velocity, and `applied_torque` as the available effort signal. Telemetry records
commanded and measured `vx`, `vy`, and `wz`, base XY, yaw, fall status, and
effort. The exact results are in
`reports/artifacts/m0/falcon_adapter_run/summary.json` and
`reports/artifacts/m0/falcon_adapter_run/telemetry.csv`.

Two narrow integration failures were encountered and preserved in the M0
artifacts: importing `onnx.reference` caused a Windows access violation inside
the Isaac process, so the adapter uses a small NumPy evaluator; a field-name
typo for body angular velocity was corrected to `base_ang_vel_b`. No software
environment changes were made.

## 5. M1 Results: Curtain Physics

### External MoDeSuite asset

The external file was inspected and loaded directly from
`C:\Robotics\MoDeSuite-reference\asset\objects\curtain.usd`; it was not
copied into the public repository. The checkout has no explicit `LICENSE` file,
so redistribution permission was treated as unclear.

The asset loads and remains finite for 300 headless steps. Its cloth mesh has
2,601 points and legacy particle-cloth schemas:
`PhysxParticleClothAPI`, `PhysxParticleAPI`,
`PhysxAutoParticleClothAPI`, and `PhysicsMassAPI`. It does not expose the
current surface-deformable schema used by the fallback. The kinematic probe
produced **0.0 m** maximum mesh displacement, so deformation was not
validated. PhysX also reported that several dynamic triangle-mesh collisions
were falling back to convex-hull approximations.

### Current Isaac Sim 5.1 fallback

The fallback creates a 1.4 m by 1.8 m triangular sheet with 16 columns and 20
rows, applies the current surface-deformable API and material helper, and
creates a static top support plus auto attachment. The API and material load
successfully, with 320 deformable nodes.

The probe run is not physically stable: final maximum displacement is
1.333852 m and the top edge moves grossly instead of remaining attached. The
large extent ratio in `fallback/summary.json` is consistent with this support
failure, not realistic curtain deformation. After the small number of narrow
compatibility fixes needed to reach this test, M1 was stopped as required.

M1 configuration, summaries, telemetry, and console logs are under
`reports/artifacts/m1/`. The aggregate decision is in
`reports/artifacts/m1/summary.json`. No rendered frames were produced because
the test was intentionally headless and did not add a camera.

## 6. M2 and M3 Gate

M2 was **not started** because M1 did not establish a stable, physically
meaningful curtain. Consequently there are no defensible probe thresholds,
rigid-versus-deformable classifier results, sweep results, or effort-response
measurements.

M3 was **not started** for the same reason. No RGB-D camera, color/depth
barrier detector, passable/non-passable state machine rollout, rigid detour, or
end-to-end goal run is claimed.

## 7. Reproduction Commands

Run commands from the indicated working directory and keep the environment
fixed at `C:\conda_envs\isaaclab`.

### Environment smoke test

```powershell
Set-Location C:\Robotics\IsaacLab
$env:CONDA_PREFIX='C:\conda_envs\isaaclab'
& C:\Robotics\IsaacLab\isaaclab.bat -p scripts\environments\random_agent.py --task Isaac-Velocity-Flat-G1-v0 --num_envs 1 --headless
```

### Convert the local rubber-hand asset

```powershell
Set-Location C:\Robotics\Lift-the-curtain
$env:CONDA_PREFIX='C:\conda_envs\isaaclab'
& C:\Robotics\IsaacLab\isaaclab.bat -p C:\Robotics\Lift-the-curtain\scripts\convert_falcon_urdf.py --headless
```

### M0

```powershell
Set-Location C:\Robotics\Lift-the-curtain
$env:CONDA_PREFIX='C:\conda_envs\isaaclab'
& C:\Robotics\IsaacLab\isaaclab.bat -p C:\Robotics\Lift-the-curtain\scripts\test_falcon.py --headless
```

### M1 external reference

```powershell
Set-Location C:\Robotics\Lift-the-curtain
$env:CONDA_PREFIX='C:\conda_envs\isaaclab'
& C:\Robotics\IsaacLab\isaaclab.bat -p C:\Robotics\Lift-the-curtain\scripts\test_curtain.py --headless --mode external --steps 300
```

### M1 current-API fallback

```powershell
Set-Location C:\Robotics\Lift-the-curtain
$env:CONDA_PREFIX='C:\conda_envs\isaaclab'
& C:\Robotics\IsaacLab\isaaclab.bat -p C:\Robotics\Lift-the-curtain\scripts\test_curtain.py --headless --mode fallback --steps 300
```

The M1 script enables the Isaac Sim 5.1 deformable beta setting for the
current-API experiment. It does not install or alter any package.

## 8. Resource Observations

All runs used one environment and no GUI. The M0 harness measured a peak
PyTorch allocation of approximately 0.217 MB and PyTorch reserved memory of
2.0 MB; M1 measured approximately 0.010 MB allocated and 2.0 MB reserved.
These are allocator-level measurements, not total Isaac Sim process VRAM.
Isaac Sim reported 7,948 MB GPU memory and 32,452 MB system memory. No OOM
occurred.

## 9. Failure Modes and Required Assumptions

- The launcher requires an explicit `CONDA_PREFIX` on this machine to select
  the requested Python environment.
- The old ONNX reference evaluator is unsafe to import in this Isaac Sim
  process on Windows; the local NumPy graph evaluator is used instead.
- The MoDeSuite USD is a legacy particle-cloth asset with unclear
  redistribution status and no observed deformation in this isolated test.
- The current surface-deformable fallback has not achieved a stable fixed top
  edge with the attempted auto attachment.
- The probe is kinematic and moves once along the thinnest world axis. Node
  overlap is only a coarse penetration proxy, not a contact-force measurement.
- M0's `applied_torque` is the available effort signal; no tactile signal is
  present.
- Physical parameters are deliberately initial engineering values, not fitted
  material identification.

## 10. What Worked and What Failed Systematically

The local FALCON rubber-hand articulation conversion, exact 29-DoF contract,
official policy evaluation, joint actuation, and bounded SE(2) feedback all
worked well enough for the M0 acceptance cases. The one-environment headless
configuration also stayed within the available GPU budget.

The systematic failure is the deformable-object scene: the legacy external
asset does not respond in the isolated 5.1 probe, and the current replacement
does not maintain its support attachment. This prevents scientifically useful
probe, sweep, and walk-through conclusions.

## 11. Capability Table

| Capability | PASS/FAIL | Evidence |
|---|---|---|
| FALCON stand | PASS | M0 summary, 400/400 steps, no fall |
| FALCON forward | PASS | M0 forward case, +0.326 m X displacement |
| FALCON yaw | PASS | M0 left/right cases, +0.328 / -0.251 rad |
| Curtain stable | FAIL | External finite-only; current fallback support instability |
| Curtain deforms realistically | FAIL | External 0.0 m displacement; fallback gross support motion |
| Rigid vs deformable probe | NOT RUN | M2 gated on M1 |
| Curtain sweep | NOT RUN | M2 gated on M1 |
| Hold + walk through | NOT RUN | M2/M3 gated on M1 |
| Rigid detour | NOT RUN | M3 gated on M1 |
| End-to-end baseline | NOT RUN | M3 gated on M1 |

## 12. Most Important Conclusion

This task does **not** yet work reliably with a simple engineered solution end
to end. There is a clear failure mode, but it is currently the construction
and validation of a stable, supported deformable curtain in Isaac Sim 5.1,
not evidence that learning is required. The correct next engineering decision
is to resolve or replace the curtain asset/support model before adding
perception, probe heuristics, or learning. Reinforcement learning should not be
started from the present evidence.
