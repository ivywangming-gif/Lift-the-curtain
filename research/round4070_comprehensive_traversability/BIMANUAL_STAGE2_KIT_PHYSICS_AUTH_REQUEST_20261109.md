# BIMANUAL_V1 Stage 2A Kit/Physics-Only Authorization Request

Request date: 2026-11-10 (America/Phoenix)
Status: **PENDING PROJECT OWNER AND INDEPENDENT REVIEWER SIGNATURES; NOT AUTHORIZED**
Execution: **NOT STARTED; RUNNER, NATIVE WORKER, AND DOOR CONTRACT IMPLEMENTED; OFFLINE TESTS ONLY**

## Decision

No Isaac Sim/Kit process, native USD authoring, physics step, physics gate, G1 model, training job, or formal collection was started. Both pause files remain present and unchanged. This document requests a narrowly scoped test-stage authorization; it is not that authorization.

The current `PAUSE.flag` blocks the existing campaign's physics mode, and the Stage 2 builder explicitly leaves Kit/USD authoring paused. An isolated Stage 2A runner while retaining the flags therefore requires a recorded, independently reviewed, append-only exception to the global hold. It must not delete, rename, mask, or move either flag, or call the campaign queue.

## Stage 2A Scope Requested

Authorize one headless Kit/PhysX process at a time, with one candidate scene per process and no scene batching. Assess the 8 GiB RTX 4070 Laptop first with one ordinary curtain. If startup, telemetry, or physics initialization fails, preserve the failure evidence and stop.

The proposed matrix is 8 proxy scenes x 3 test-only initial-condition seeds = 24 isolated runs. Each scene USD is authored once, saved, reopened, and hash-pinned; each isolated run checks schemas/native runtime bindings, initializes PhysX at 200 Hz, samples scene/contact/release/reset/collision telemetry at 50 Hz, records solver/resource telemetry, and closes Kit. These are not G1 seeds or independent physical configurations.

| Scene | Coverage | Current state | Material interpretation |
|---|---|---|---|
| `BI_A01_BASIC_CLOTH` | Ordinary top-fixed cloth | BUILDER_READY; no USD | Uncalibrated proxy |
| `BI_C_THREE_MISSING_TOP_LEFT` | Three-corner topology | BUILDER_READY; no USD | Uncalibrated proxy |
| `BI_C_FOUR_CORNERS` | Four-corner topology | BUILDER_READY; no USD | Uncalibrated proxy |
| `BI_B_WALL_04CM` | Hidden wall, 4 cm gap | BUILDER_READY; no USD | Uncalibrated cloth proxy |
| `BI_G_DOOR_ORDINARY` | Hinged door composite | BUILDER_READY; no USD | Door mass/resistance uncalibrated |
| `BI_E_OPEN_MESH_PROXY` | Open-mesh proxy | BUILDER_READY; no USD | Not real mesh certification |
| `BI_E_PVC_INDEPENDENT_PROXY` | Independent PVC-strip proxy | BUILDER_READY; no USD | Not real PVC certification |
| `BI_E_PVC_OVERLAP_PROXY` | Overlapping PVC-strip proxy | BUILDER_READY; no USD | Not real PVC certification |

Reserved test-only seeds: `4070701`, `4070702`, `4070703`. The worker verifies reproducible, distinct probe/free-node initial-condition perturbations and records Python, NumPy, and Torch seeds. It does **not** configure or verify a PhysX-internal RNG seed. The installed Isaac Lab `SimulationContext` source contains no seed setter; the installed PhysX extension is a native binary, so the worker cannot claim internal seed control. If the independent reviewer approves the operational screen with this limitation, all three runs are retained as initial-condition repetitions, not independent PhysX-seeded validation. Any scene remains `UNRESOLVED_SEED_OR_GATE`; no scene can become `PHYSICS_VALID` until a supported and independently verified PhysX seed mechanism is added and separately hash-reviewed.

`BI_E_REAL_CLOTH`, `BI_E_REAL_MESH`, and `BI_E_REAL_PVC` are excluded because their source assets are missing. Proxy results cannot promote those rows or certify real materials.

## Test Contract

- Preserve the existing physics-gate values: `dt=0.005 s`, 24 s, probe force cap 3 N, finite states, neighbor stretch <= 4, fixed-node/support drift <= 0.01 m where applicable, no nodes below ground, zero sampled interior penetrations, demonstrated contact, clear release, and late speed < 0.25 m/s. Record that sampled collision checks do not establish continuous-time collision freedom.
- Bind the final plan, runner, worker, source, and door-contract hashes before execution. Do not tune thresholds after results. Door-specific acceptance limits are preregistered in `BIMANUAL_STAGE2_DOOR_ACCEPTANCE_CONTRACT_20261109.json`; its drive/mass remain uncalibrated even if proxy mechanics pass.
- Assign `USD_AUTHORED` only after save/reopen and schema/binding checks. Assign `PHYSICS_VALID` only if all three independent seeded runs pass applicable preregistered checks. Actual physical violations are `PHYSICS_INVALID`; startup/import/timeout/instrumentation errors remain unresolved runtime failures.
- Record GPU telemetry with `nvidia-smi` and process working/private memory via process APIs. A prior wrapper could not read Windows' `Global` PerformanceCounter registry key. Capture actual solver type, iterations, dt, GPU/contact settings, and runtime versions from the active stage.
- Use only the new append-only root `outputs/round4070_comprehensive_traversability/physics_diagnostics/bimanual_v1_kit_physics_stage2_20261109/`. Never overwrite USD or run evidence. Do not edit the matrix until each status has supporting evidence.
- Requested resource ceilings: one Kit process, one scene, one GPU; 600 seconds per run, 14,400 seconds total, 10 GiB diagnostic output, and stop before GPU memory use reaches 7168 MiB. No automatic retries. Stop on the first source/schema error or first preregistered physical failure for a scene; any repair/retry needs a new reviewed source hash and approval.

## Exclusions

Stage 2A must not load G1 USD/hand colliders, instantiate FALCON, create a controller/action, execute STOP/retract, invoke `run_round4070_g.ps1`, use the V5 queue, create an episode/label, consume V5 G1 launch budget, or modify the manifest. It cannot certify V6, native hand collision, G1 contact, or G1 readiness.

Stage 2B (G1 USD, exact bound `PxConvexMesh` inspection, and V6 simulated STOP/retract replay) is separate and needs its own explicit approval. Formal G1 collection remains NO-GO regardless of Stage 2A results.

## Approval Required

Before any Kit process starts, the project owner and an independent safety reviewer who did not implement this stage must approve the exact scope and hashes in `BIMANUAL_STAGE2_KIT_PHYSICS_APPROVAL_TEMPLATE_20261109.json`. The reviewer must explicitly authorize a test-only Kit launch while both pause flags remain present, approve a scoped addendum to the global hold (not a bypass), review final runner/worker/source/door-contract hashes, approve resource/output limits and stop conditions, and explicitly acknowledge that the three current runs do not verify PhysX RNG independence and cannot promote a scene to `PHYSICS_VALID`. Without both approvals, no Kit operation is permitted.

The authorization, if eventually granted, covers only the 24 planned test-only runs, headless Kit, one scene/process at a time, one GPU, and new diagnostic outputs. Scope/hash/threshold changes invalidate approval. The final offline supervisor, native USD/Physics-only worker, and scene-bound door acceptance contract are implemented. The Stage 2A worker will preserve all three repetitions if PhysX RNG remains unverified, but the scene status remains unresolved. Stage 2B and formal G1 collection remain separately gated.

## Offline Verification (2026-11-10)

- The supervisor checks the exact 24-run scope, plan/runner/worker/test/scene/door hashes, both pause-file hashes, the V5 manifest hash, output-path containment, owner/reviewer signed-evidence references, runtime/resource ceilings, and door acceptance contract before launching anything.
- The worker imports without Kit, refuses before `AppLauncher` absent authorization, authors one non-overwritable scene USD per candidate, reopens and hash-binds it, and distinguishes runtime validity from physics-contract validity.
- Offline tests cover protected-state immutability, owner/reviewer exact-bundle binding, non-door scenes under the shared door contract, SAT margin direction, worker refusal, seed-claim rejection, and scene-builder contracts. Final test totals and hashes are in `BIMANUAL_STAGE2_KIT_PHYSICS_CHECKPOINT_20261109.json`.
- This remains offline contract coverage only. It is not native USD authoring, a Physics Gate, a Kit startup test, a scene validation, a verified PhysX RNG seed, or G1 authorization.

## Current Evidence and Recovery

- Installed runtime snapshot: Isaac Sim 5.1.0.0, Isaac Lab 0.54.4, PhysX extension 107.3.26, PyTorch 2.7.0+cu128, CUDA 12.8, RTX 4070 Laptop GPU 8188 MiB.
- A prior native polygon probe stopped at `BEFORE_APP` after 180 seconds with no Kit/physics result. Its telemetry wrapper also logged access denied for the Windows `Global` PerformanceCounter registry key. Current inspection found no Isaac/Kit/Python process and no active campaign/workflow/resource lock.
- V6 remains `NO_GO_FOR_FORMAL_G1_COLLECTION`; live hand polygon planes and physical STOP/retract are not certified.
- Current 56-row state remains 53 `BUILDER_READY` with no USD and 3 `NOT_BUILT` due to absent real-material assets. This stage has produced no `USD_AUTHORED`, `PHYSICS_VALID`, or `PHYSICS_INVALID` row.

Resume from `BIMANUAL_STAGE2_KIT_PHYSICS_CHECKPOINT_20261109.json`. Do not regenerate the matrix or reset the manifest. Before any Kit run, obtain exact-hash owner and independent-reviewer signatures in the approval template, including explicit acknowledgement of the PhysX-seed limitation. If approved, execute only the 24-run test scope with both flags retained; expect unresolved seed status unless a separately reviewed PhysX-seed mechanism is established. No Stage 2A result authorizes Stage 2B or G1.
