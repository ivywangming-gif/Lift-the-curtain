# Scientific decision at the actual checkpoint

Status: **IN_PROGRESS_NOT_COMPLETE**. No method winner or method contribution is selected.

已实际启动 physics 72 次、G1 15 次；三独立 seed 门禁通过配置 17 个（含控制/组合）。指定 48 个命名场景中有 21 个具备三 seed 门禁证据；相同物理配置的别名不算独立验证。

主实验导入 4 条有效探测历史，其中空场控制 2 条，核心标签 `{'PASSABLE_FOR_ACTION': 2}`。另有 10 次已启动旧 pilot，仅保留诊断，不进入主训练或统计。当前冻结主学习状态 `{'BLOCKED_WITH_REASON': 1520}`，实际主 EVALUATED 任务 0 个；独立单类诊断 EVALUATED 0 个。合成 smoke 不进入这些计数。

Do not assign LEARNING_EVIDENCE_INSUFFICIENT as a substitute for unrun learning: that user-defined status requires completed model training. Do not assign NO_METHOD_ADVANTAGE without the comparisons. METHOD_ADVANTAGE_SUPPORTED requires repeated independent evidence at matched false-pass risk and a valid calibration/test separation. FULL_BENCHMARK_COMPLETE and COMPLETE_WITH_SCENE_LIMITATIONS require all executable planned comparisons, not merely implementation or queue creation.

The decision remains open until independent physical labels, all standard baselines, candidate ablations, cross-hand/pose/constraint tests and the confirmation set are actually evaluated. Physical aliases, stochastic training seeds, augmented observation copies, synthetic smoke cases and empty controls do not create new independent positive obstacle families.
