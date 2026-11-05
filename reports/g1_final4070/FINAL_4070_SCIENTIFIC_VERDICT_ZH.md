# Final RTX 4070 Scientific Verdict

## 首页裁决

### Q1：当前研究问题在机器人学上是否真实成立？

成立，但范围是“G1 当前动作能力下的 action-conditioned whole-body traversability”，不是一般意义上的帘幕分类。24 条新 trial 中，柔性候选均可产生 `RESPONSIVE`，但 whole-body clearance 只有 4 条；rigid 则稳定 `BLOCKED`、撤回、绕行。这排除了“局部接触等于全身通过”的简单解释。

### Q2：同一种物理条件下，不同合法动作或机器人状态能否产生不同的有效全身结果？

有条件支持。bottom-right 的匹配条件中，R1 `CAUTIOUS_ADVANCE` 出现 clearance，而 R2 `RETRACT` 放弃近路；nominal seed46 也出现同样差异。可是 R1/R2 不是两种独立物理技能，R2 没有继续执行，因此这还不是“动作改变物理可通行性”的完整因果证明。nominal seed45/46 的差异还显示机器人状态/动力学响应具有影响。

### Q3：可部署本体信息是否足以预测部分未来通行结果？

T0/T1 尚不足以支持稳定的预动作预测：positive/negative 的 hand progress、tracking、effort 和 base drift 高度重叠。T2 在执行短时继续动作后出现了新的 progress/track 信息，但正样本只有 4 条且机制混杂。因此结论是“短时闭环反馈有信息价值”，不是“已有历史已经足以泛化预测”。

### Q4：简单规则是否已经足够？

对 rigid 安全拒绝和绕行，简单规则足够且稳定。对候选近路利用，R2 的固定 stall 门控明显过于保守：10/10 非空 R2 都没有 clearance，错过了 R1 已观察到的 4 条 clearance。简单规则尚未同时实现安全性和近路利用，但也没有证据说明需要大型学习模型。

### Q5：当前问题是否存在明确的算法创新空间？

存在一个窄的、条件性的空间：deployment-only 的短时动作后果验证、风险校准和 whole-body clearance 证书。它必须在同一机制内有平衡正负样本、独立 held-out 条件，并实地产生不同动作，否则只是把工程 if–else 换成更复杂的网络。

### Q6：4070 究竟限制了哪些实验？

4070 限制吞吐和并行规模：本轮 simulated/wall 中位数约 0.237，柔性运行耗时更长，内存 headroom 约 4.8--5.5 GB。它没有阻止 24 条 headless 物理运行，也没有造成 OOM 或 CUDA failure。renderer/RGB-D 是独立的 Windows/Isaac 兼容问题，不能归因于显存不足。

### Q7：哪些问题不是 4070 的错，而是仿真接口或现有执行能力的问题？

- Frozen FALCON/上层动作接口：只有部分状态和锚定条件能清空身体包络。
- R2 的固定 stall 阈值：把柔性 probe 的正常低速阶段当成风险，导致系统性 conservative reject。
- deployment observability：固定 probe 截止点的信息高度重叠。
- Kit 安装目录 `user.config.json` 写权限警告和历史 renderer crash：属于平台/软件分支，不是本体通行性结论。

### Q8：应该继续投资 RTX 5090 与两台 G1，还是停止该课题？

不建议因为 4070 资源问题立即迁移，也不建议现在宣称 RA-L 已经成立。当前建议是 `CONTINUE_METHOD_DEVELOPMENT`：先完成一个明确的 deployment-only adaptive short-advance 对照；若它在 held-out 条件仍不能减少 false-reject/false-pass，再停止当前主题。5090 可作为扩展重复数和后续传感器实验平台，但不是解决当前核心问题的必要条件。

### Q9：如果继续，下一阶段唯一最值得开展的科学实验是什么？

在同一个候选机制内，预注册相同 probe、相同安全预算和相同 Frozen FALCON，比较：

1. 固定 R1 continue；
2. 一段 deployment-only 短时 advance 后再决定 continue/retract；
3. 立即 retract/detour。

必须同时采集成功和失败后果，并在未见机制上测试校准风险规则。目标是证明“新增短时进度反馈”而不是物体类别，能降低 false-reject 或 false-pass。

### Q10：如果停止，为什么更大算力不能解决目前缺少创新的问题？

本轮暂不停止，但止损逻辑明确：如果上述匹配实验显示简单规则已达到同等安全和效率，或部署观测在短时反馈后仍不可区分，那么更大 GPU 只会增加重复数量，不会创造缺失的观测、动作能力或方法创新。

## PASS / FAIL / BLOCKED / NOT RUN

### PASS

- 24/24 headless runtime artifact 有效。
- Empty、rigid、柔性资源基线可重复运行。
- Rigid 5/5 `BLOCKED`、安全撤回、地图更新和 PointGoal 绕行。
- Bottom-right 的 R1 在独立 seed/左右配对中出现真实 candidate clearance。
- PointGoal contract 与 yaw 解耦；终点 yaw 未作为 PointGoal gate。
- 188 次高层决策的特权信息泄漏审计通过。

### FAIL

- 不能把 `RESPONSIVE` 作为全身通行保证。
- R1 在 nominal seed45 未完成 clearance，且 PointGoal 失败。
- R2 在本轮所有非空柔性运行都未利用近路，存在明显 false-reject。
- 当前数据不足以证明学习模型的 held-out 泛化收益。

### BLOCKED

- RGB-D/renderer 真实部署分支仍受历史 Windows/native compatibility blocker 影响；本轮没有重启已知 crash。
- T0/T1 的通行性预测因正负支持和机制留出不足，不能做有意义的学习验证。

### NOT RUN

- 新的 GRU/MLP/world model/PPO 训练。
- 真实 G1。
- 依赖 RGB-D renderer 的完整视觉导航。
- 大规模并行 benchmark。

## 独立最终字段

```text
SCIENTIFIC_VERDICT = SCIENTIFIC_CONDITIONAL_GO
RESOURCE_VERDICT = 4070_SUFFICIENT_FOR_CURRENT_RESEARCH
EXECUTION_CAPABILITY_VERDICT = PARTIAL_FROZEN_FALCON_CAPABILITY_LIMIT_WITH_CONDITION_DEPENDENT_PASSAGE
RAL_RESEARCH_RECOMMENDATION = CONTINUE_METHOD_DEVELOPMENT
ENGINEERING_BASELINE_COMPLETE = FALSE
```

## 证据入口

- 继承审计：`reports/g1_final4070/00_INHERITED_EVIDENCE_AUDIT.md`
- 限制诊断：`reports/g1_final4070/01_LIMITATION_DIAGNOSIS.md`
- 匹配实验：`reports/g1_final4070/04_MATCHED_PHYSICAL_CAMPAIGN.md`
- 动作后果：`reports/g1_final4070/05_ACTION_CONSEQUENCE_COMPARISON.md`
- 部署观测：`reports/g1_final4070/06_DEPLOYMENT_OBSERVABILITY.md`
- 机器账本：`outputs/g1_final4070/trial_index.json`
- 配对结果：`outputs/g1_final4070/action_consequence_comparison.json`
- 特权信息审计：`outputs/g1_final4070/deployment_observability.json`
- 资源/来源：`outputs/g1_final4070/resource_profile_final.json`、`outputs/g1_final4070/provenance.json`

