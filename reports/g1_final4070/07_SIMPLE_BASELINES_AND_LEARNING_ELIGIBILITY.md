# 简单基线与学习资格

## 已实际比较的强基线

本轮真实物理比较了：

- R1：RESPONSIVE 后继续谨慎推进；
- R2：probe 后用固定 progress/stall/tracking 门控，不满足即撤回；
- rigid 的 BLOCKED 撤回、2→1 和 0-only detour。

它们共享底层 FALCON、地图、动作权限和安全监督，因此比较的是高层决策后果，而不是不同 WBC 技能。

## 为什么没有训练新模型

本轮非空柔性 clearance 只有 4 条：bottom-right 3 条、nominal 1 条。负样本主要来自 bottom-left 和 nominal，正负几乎按物理机制分组；没有一个独立 held-out 机制同时提供足够正负支持。T2 只有 9 条 R1 继续样本，R2 在 post-probe 就撤回，没有可比的后续序列。

在这种分布上训练 Logistic、Gradient Boosting、History MLP、GRU 或 world model，会把场景身份/动作规则当作结果，不能证明泛化。按照预注册停止规则，本轮学习资格为 `NOT_ELIGIBLE_FOR_GENERALIZATION`，没有用极小数据制造算法优势。

## 可报告的学习信号

T0/T1 部署特征重叠，说明仅增加模型容量不能解决截止点信息缺失。T2 的短时 progress 差异说明应优先收集“同一动作、同一机制、不同后果”的成对数据，再做小模型。

## 需要满足的下一轮数据条件

1. 至少两个物理机制各自同时包含有效和无效后果；
2. 同一机制内正负样本不能只由 object label 分开；
3. 独立 held-out condition 或 seed 不能参与阈值选择；
4. 每个样本保留从 probe 到 clearance/recovery/PointGoal 的完整过程；
5. 先比较校准风险规则、Logistic/树模型和短历史模型，再决定是否需要序列学习。

