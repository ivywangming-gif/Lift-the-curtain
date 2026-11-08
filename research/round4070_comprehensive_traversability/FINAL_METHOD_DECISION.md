# Final Method Decision

## Decision Status

EXPLORATORY ONLY. No formal G1 learning admission, physical collection, or deployment certification is granted.

Descriptive main candidate for the next offline iteration: B2_TS2Vec_FineTune.
Descriptive backup candidate: B2_TS2Vec_FrozenLinear.
These names are provisional rankings within the two cross-hand diagnostics, not a claim of independent physical-config generalization.

## Evidence

- Per-method summary: {"A_Orderless_GRU_Control": {"mean_balanced_accuracy": 0.6, "seed_rows": 10, "splits": ["split4_left_to_right", "split4_right_to_left"], "std_balanced_accuracy": 0.11055415967851333}, "B1_GRU": {"mean_balanced_accuracy": 0.525, "seed_rows": 10, "splits": ["split4_left_to_right", "split4_right_to_left"], "std_balanced_accuracy": 0.075}, "B2_TS2Vec_FineTune": {"mean_balanced_accuracy": 0.725, "seed_rows": 10, "splits": ["split4_left_to_right", "split4_right_to_left"], "std_balanced_accuracy": 0.20766559657295186}, "B2_TS2Vec_FrozenLinear": {"mean_balanced_accuracy": 0.5416666666666666, "seed_rows": 10, "splits": ["split4_left_to_right", "split4_right_to_left"], "std_balanced_accuracy": 0.08539125638299663}, "B3_GRU_TS2Vec": {"mean_balanced_accuracy": 0.5, "seed_rows": 10, "splits": ["split4_left_to_right", "split4_right_to_left"], "std_balanced_accuracy": 0.0}, "B4_MatchedSupCon": {"mean_balanced_accuracy": 0.55, "seed_rows": 5, "splits": ["split4_right_to_left"], "std_balanced_accuracy": 0.09999999999999999}, "B4_OrdinarySupCon": {"mean_balanced_accuracy": 0.5, "seed_rows": 10, "splits": ["split4_left_to_right", "split4_right_to_left"], "std_balanced_accuracy": 0.0}}.
- Frozen E1 per-split summary: {"split4_left_to_right": {"balanced_accuracy": 0.75, "blocked_recall": 0.5, "brier": 0.03125, "coverage": 1.0, "ece": 0.0625, "false_pass": 0.5, "false_reject": 0.0, "passable_recall": 1.0, "uncertain_rate": 0.0}, "split4_right_to_left": {"balanced_accuracy": 0.75, "blocked_recall": 0.5, "brier": 0.0625, "coverage": 1.0, "ece": 0.125, "false_pass": 0.5, "false_reject": 0.0, "passable_recall": 1.0, "uncertain_rate": 0.0}}.
- Candidate ranking excludes A_Orderless_GRU_Control because it is an order-removal ablation control, not a proposed method.
- E1_FIXED remains the sole engineering baseline; thresholds and decision logic were not changed.
- All models use the same episode cutoff, legal feature history, and split roles. Training seeds are not independent physical samples.

## Interpretation Rules Applied

- The original TS2Vec-style CNN fine-tune is retained as the descriptive main candidate only because it is the strongest complete two-split exploratory result; this is not a formal promotion.
- The GRU + TS2Vec-style pretraining result (B3) is not retained as an improvement over B1 in this snapshot; the matched SupCon result (B4) is incomplete on the left-to-right split and is not rank-eligible.
- No method is promoted based on a single hand or a single seed.
- B4 is considered useful only when a split has at least two real positive and two real hard-negative matched pairs. The left-to-right split does not meet that support and is recorded as data-insufficient.
- No independent physical-configuration split has both core classes in the test endpoint. Therefore no formal OOD ranking, calibration claim, or safety advantage is asserted.
- B10 and D01 remain LABEL_UNKNOWN and never enter supervision or self-supervised pretraining.

## Recovery Gate

PAUSE.flag and PAUSE_G1.flag remain untouched. Formal recovery still requires the project's independent safety review and approval; this report does not substitute for that approval. The existing project resume command must not be executed while the pause flags remain present.

## Reproduction

Run directory: outputs\round4070_comprehensive_traversability\exploratory_gru_ts2vec_20261109_5e3c3fba_v3.
The B2 implementation uses TS2Vec-style components and is recorded as such; the GRU replacement is explicitly a task-adapted TS2Vec-style method, not original TS2Vec.
