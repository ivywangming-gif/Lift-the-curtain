# Round 4070 最终研究报告

日期：2026-11-03。状态：`COMPLETE_EVIDENCE_FROZEN`。

## 结论摘要

本轮启动 38 条新增 G1 物理 trial，37 条完整有效，1 条 runtime-invalid；没有 OOM、CUDA、renderer 或重复求解器失败。历史 24 条原始运行另外完成只读重建，未被修改。有效新样本中 35 条为非空候选：14 条 whole-body clearance、19 条 S3、2 条 rigid S2；37/37 安全收尾且无 fall。

冻结 T1 匹配在确认集得到 4 对 nominal/bottom-right 同 side/seed 条件，其中 3 对早期本体特征匹配但 clearance 不同：

- `C01_NOMINAL_LEFT_S49` vs `C02_BOTTOM_RIGHT_LEFT_S49`: False vs True
- `C01_NOMINAL_RIGHT_S49` vs `C02_BOTTOM_RIGHT_RIGHT_S49`: False vs True
- `C03_NOMINAL_LEFT_S50` vs `C04_BOTTOM_RIGHT_LEFT_S50`: False vs True
- `C03_NOMINAL_RIGHT_S50` vs `C04_BOTTOM_RIGHT_RIGHT_S50`: True vs True

这支持 H3：局部响应不是 whole-body clearance 标签。H1（纯时间历史收益）为 `INCONCLUSIVE`，H2（空间主动探测）为 `INCONCLUSIVE`：F2/F3 physics gate 4/4 失败，主动 probe 试验数为 0。

最终选择 **C. Action-Conditioned Traversability**，但当前只支持继续做小型、可解释、deployment-only 的 shadow baseline；不支持 GRU/Transformer、PPO/WBC、VLM end-to-end 或主动信息增益算法开发。

## 基线结果

确认集为 16 条，训练集为 19 条。T1：A frozen response AUROC 0.500、false-pass 7/7；C residual logistic AUROC 0.714，但 0.5 阈值下 false-reject 9/9；D T1 logistic AUROC 0.460。T2 residual/logistic 均 AUROC 1.0，但 T2 包含 advance 后的动作结果，只能作为 exploratory，不能证明纯历史因果收益。E history MLP/GRU 未训练，原因是独立条件和动作分支不足。

## 交付位置

- 研究主报告：[research/round4070_problem_discovery/RESULTS.md](C:/Robotics/Lift-the-curtain/research/round4070_problem_discovery/RESULTS.md)
- Q1–Q10：[research/round4070_problem_discovery/SCIENTIFIC_PROBLEM.md](C:/Robotics/Lift-the-curtain/research/round4070_problem_discovery/SCIENTIFIC_PROBLEM.md)
- 实现审计：[research/round4070_problem_discovery/IMPLEMENTATION_AUDIT.md](C:/Robotics/Lift-the-curtain/research/round4070_problem_discovery/IMPLEMENTATION_AUDIT.md)
- 物理场景审计：[research/round4070_problem_discovery/PHYSICAL_SCENE_VALIDATION.md](C:/Robotics/Lift-the-curtain/research/round4070_problem_discovery/PHYSICAL_SCENE_VALIDATION.md)
- 完整性审计：[research/round4070_problem_discovery/TRIAL_INTEGRITY_AUDIT.json](C:/Robotics/Lift-the-curtain/research/round4070_problem_discovery/TRIAL_INTEGRITY_AUDIT.json)
- 基线结果：[research/round4070_problem_discovery/BASELINE_EVALUATION.json](C:/Robotics/Lift-the-curtain/research/round4070_problem_discovery/BASELINE_EVALUATION.json)
- 原始新 trial：[outputs/round4070_problem_discovery/trials](C:/Robotics/Lift-the-curtain/outputs/round4070_problem_discovery/trials)

## 复现命令

```powershell
& 'C:\conda_envs\isaaclab\python.exe' scripts/round4070_problem_discovery/analyze_campaign.py
& 'C:\conda_envs\isaaclab\python.exe' scripts/round4070_problem_discovery/evaluate_baselines.py
& 'C:\conda_envs\isaaclab\python.exe' scripts/round4070_problem_discovery/audit_trial_integrity.py
& 'C:\conda_envs\isaaclab\python.exe' -m pytest -q tests/round4070_problem_discovery/test_preregistration.py tests/round4070_problem_discovery/test_scene_specs.py
```

本轮没有自动 commit、push 或创建 PR。
