# Round 4070 进度

更新时间：2026-11-02（Asia/Hong_Kong）。

## 完成

- 审计 Round12/20/21/22/23/32 与最终 4070 campaign：历史 24 条逐条有效重建，原始 summary 的 16/17 柔性分母差异已标出。
- 固定 Git HEAD `1e18825a42bf5cc19a39bf63f76fa172869cc6cd`；保留已有 `online_occupancy.py` 用户修改；无历史 artifact 被覆盖。
- 生成新的 22 条 G1 初始预注册矩阵；保留确认/主动探测预算，硬上限 48。
- 新 runner 增加逐时刻 effort source，`unavailable` 时 fail-closed；新 runner 空场右手 trial `D00_EMPTY_RIGHT_S47` 完整 PASS，PointGoal 到达，来源 `applied_torque`。
- 物理 sanity：共面 split-free 及 ±4 mm、±12 mm 分层 no-probe 各完成一次，均 `FAIL_PHYSICS_SANITY`；F2 暂停，不运行对应 probe 或 G1。
- 场景/预注册单元测试 5/5 通过；Python compile 与 PowerShell parse 通过。

## 正在推进

- 预注册 baseline G1 队列：从空场左手开始串行执行；之后 rigid/nominal 左右与第二 seed。
- 后续只有当单片 F1 条件与新 runner 回归有效时，才推进密度/锚点探索；全部新 run 经 analyzer 入账。

## 主要假设与证据

- H2：同一个局部右手/左手早期 T1 反馈是否能匹配而后续 S4 不同。阈值已写入预注册；尚无本轮确认样本。
- H3：局部手部进展不保证全身 clearance。历史支持；本轮将重新获得来源可追溯的部署数据。
- F2/F3：多片独立 surface overlap 必须先 physics-valid。当前 split-free 至 ±12 mm layer spacing 仍显著振荡，属于场景模型失败，不是 G1 负样本。

## 运行账本

详见 `research/round4070_problem_discovery/TRIAL_MANIFEST.csv`、`outputs/round4070_problem_discovery/` 和 `research/round4070_problem_discovery/EVIDENCE_MATRIX.csv`。进度由 `scripts/round4070_problem_discovery/analyze_campaign.py` 重建；runtime-invalid、physical failure 与 not-assessable 分开。
