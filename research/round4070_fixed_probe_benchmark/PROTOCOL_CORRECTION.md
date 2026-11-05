# Round 4070-E Protocol Correction

The nominal command is fixed at `0.10 m`; actual hand FK feedback is an observed outcome, not a `PROBE_VALID` gate.

## Four statuses

- `RUNTIME_VALID`: normal simulator execution, finite state, and complete required logs.
- `PROBE_VALID`: the legal one-shot command executed with a complete proprioceptive history; contact-limited motion and an early safe stop may still be valid outcomes.
- `LABEL_VALID`: independent physical/geometric evidence supports `PASSABLE`, `BLOCKED`, or an explicit `UNKNOWN`; classifier output and PointGoal failure are not truth.
- `TASK_SUCCESS`: secondary navigation/PointGoal outcome; it is never substituted for a traversability label.

A valid trial may have actual hand displacement below 0.10 m. It is retained with its outcome class and is not automatically relabeled `BLOCKED`.

Observed summaries: `104`; runtime-valid `104`; probe-valid `104`; label-valid `44`.
Observed actual displacement range: `-0.002920..0.089980 m`.
Registered rows are not counted as completed experiments; only recorded output directories are execution evidence.
