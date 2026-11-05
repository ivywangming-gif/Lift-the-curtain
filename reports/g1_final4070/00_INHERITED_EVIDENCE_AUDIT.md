# Final RTX 4070 Campaign：继承证据审计

## 审计范围

本轮只把历史结果作为先验，不把历史 trial 计入本轮新物理样本。审计对象包括：

- `reports/g1_research_feasibility/FINAL_RESEARCH_FEASIBILITY_VERDICT_ZH.md`
- `reports/g1_research_feasibility/02_MATCHED_G1_PHYSICAL_RESULTS.md`
- `reports/g1_research_feasibility/03_WHOLE_BODY_TRAVERSABILITY_EVIDENCE.md`
- `outputs/g1_research_feasibility/trial_index.json`
- Round 20、26、32 报告和 Round 25B--27 的 policy interface / runner。

当前 Git HEAD：`1e18825a42bf5cc19a39bf63f76fa172869cc6cd`，分支：`main`。工作树原有修改 `lift_curtain/perception/online_occupancy.py` 保持不动；本轮没有 commit 或 push。

## 历史结论与本轮结论的边界

| 来源 | 历史事实 | 本轮处理方式 |
|---|---|---|
| Round 12 nominal / rigid | 已有 Oracle map、PointGoal、rigid 安全拒绝和绕行链路 | 作为接口基线，不重新计数 |
| Round 20、26、32 | nominal 局部响应不稳定，bottom-right 曾出现全身清空，rigid 能安全拒绝 | 用于提出 P1--P3，不替代新的匹配矩阵 |
| `g1_research_feasibility` | 20 次启动、12 条正式有效非空 trial；S4 集中于 bottom-right | 作为先验，单独列为 inherited evidence |
| 本轮 24 条 trial | R1/R2、左右 probe、nominal/bottom-left/bottom-right/rigid 的新运行 | 唯一计入本轮物理账本 |

历史不能直接推出“学习必要”，也不能把旧的 `PASS` 字段解释为所有 gate 都通过。特别是：candidate clearance、PointGoal、终点 yaw、safe retract 和 runtime validity 必须分开。

## 本轮冻结项

- Frozen FALCON、原 Stand/Walk、SE(2) feedback、安全阈值和物理参数。
- headless、`num_envs=1`、单 Isaac/Kit 串行运行。
- PointGoal：XY 位置误差 `<= 0.08 m`、停稳、无跌倒和安全通过；终点 yaw 只作诊断。
- R1/R2 共享同一个 probe、同一动作集合和独立安全监督。
- 物体类别、真实接触、布料节点和最终通行结果只进入 offline evaluation。

## 本轮样本账本

- 预注册矩阵：24 条。
- 实际启动：24 条。
- 完整 artifact：24 条。
- `RUNTIME_INVALID`：0 条。
- 资源基线：空场 2、rigid 2、柔性 bottom-right 2。
- 匹配动作：6 条；匹配状态/左右：8 条；关键重复：4 条。

证据等级均为 `MATCHED_INITIAL_CONDITION`；没有把柔性 PhysX 内部状态恢复到 `EXACT_MATCHED_STATE`，因此不声称精确反事实。

## 关键继承假设

本轮验证的三个问题是：

1. 局部 `RESPONSIVE` 是否足以保证全身 candidate clearance？
2. 合法高层动作选择是否会改变实际任务结果？
3. 在决策截止点，部署可得历史是否已有可泛化的预测信息？

最终裁决见 `FINAL_4070_SCIENTIFIC_VERDICT_ZH.md`。原始 trial、summary、events、telemetry 和评价真值均保留在 `outputs/g1_final4070/trials/`。

