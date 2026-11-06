# Round 4070-G 运行中研究

本目录保存预注册清单、可追溯阶段报告和实际数据汇总。完整研究尚未完成；具体运行命令见 `RUN_INSTRUCTIONS.md`，当前状态见 `PROGRESS_CHECKPOINT.json` 和 `SCENE_COVERAGE_MATRIX.csv`。`campaign_manifest_v3.json` 及相邻 SHA 文件是不可变的完整976条计划；fresh checkout 由 `physics_restore_manifest.py` 精确恢复，而不是重新采样。

覆盖表的 `CONTROL_EMPTY` 是派生显示名称：冻结 v3 中 EMPTY 的原始 family 因初始化函数而为 `E`，报告将其显示为独立空场控制。`is_control=True` 的空场不计入柔性正例或顶部机构 E 家族。此显示修正没有修改冻结物理配置、seed或标签。DR组合明确是 `DR_COMBINATION`，与隐藏墙 D 家族区分。

物理材料/约束的真实性边界见 `PHYSICAL_VALIDITY_REPORT.md` 与 `MATERIAL_FIDELITY_AND_5090_LIMITATIONS.md`。某项状态为 PENDING 或 BLOCKED 不是成功验证；process退出不等于科学任务完成。
