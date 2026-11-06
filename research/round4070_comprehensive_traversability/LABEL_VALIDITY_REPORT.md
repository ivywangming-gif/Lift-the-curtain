# Round4070-G frozen action label contract

Formal label: `PASSABLE_FOR_ACTION:G_HOLD_PROBE_POSE_FORWARD24_V1`. Probe hand and followup hand are separate fields. V1 holds the executed final probe pose with the same hand and requests frozen R1 bounded forward motion: at most 24 segments of 0.12 m toward (1.30, 0), cruise 0.25 m/s, max vx 0.30 m/s, 7 s segment timeout, followed by 1.5 s release, 1.5 s neutralization and 1.5 s stable recovery. The old classifier is audit-only and does not select which cases receive followup. Existing safety vetoes remain active.

PASSABLE_FOR_ACTION requires an actual recorded followup, robot whole-body candidate-envelope clearance, finite valid physics/runtime/probe and verified stable release/recovery. Merely opening cloth, safe retraction, detouring or reaching PointGoal is insufficient. BLOCKED requires an independently constructed fixed full-height/full-width near-field wall covering the frozen forward corridor; timeout, low progress and controller refusal alone remain LABEL_UNKNOWN. Partial coverage does not receive this proof. D07 outside the probe range is always excluded from near-field supervision and deferred to visual reobservation.

A phase-level safety stop can leave the probe history valid but the action outcome UNKNOWN. Online deformable failures invalidate physics and prevent all supervised/response admission. New labels are recomputed from evidence at import, not read from material names. Historical E/F clearance-based labels are not automatically equivalent to this intervention-independent G contract.

Current accepted core labels: `{}`; control episodes: 2. Left/right counts and unique positive physical groups are exported separately. All unexecuted coverage goals remain pending.
