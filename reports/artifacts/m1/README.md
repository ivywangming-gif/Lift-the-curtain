# M1 Curtain Physics Artifacts

The MoDeSuite curtain remains a read-only external reference. It is not copied
into this repository because the checkout has no explicit `LICENSE` file.

The external run records that Isaac Sim 5.1 can open the USD and step it for
300 headless physics steps, but the legacy particle-cloth mesh shows zero
measurable displacement under the kinematic probe. The local current-API
fallback records successful schema/material creation but unstable support
dynamics: the final top edge moves by approximately 1.33 m.

Configuration is in `configs/curtain.yaml`. The external and fallback resolved
configurations, console logs, telemetry, and summaries are kept in their
respective subdirectories. No camera or GUI was used, so no rendered frames
were produced.

These results stop the M1 phase. M2 and M3 are intentionally not started.
