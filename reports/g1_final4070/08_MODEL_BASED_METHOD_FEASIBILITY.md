# Model-based 方法可行性

## 本轮判断

本轮没有启动新的 world model、PPO 或大模型训练。原因不是工程懒惰，而是前置条件尚未同时满足：

- 已有不同动作后果：部分满足；
- 部署历史含有有效信息：T2 部分满足，T0/T1 未证明；
- 简单规则存在可重复错误：R2 的 false-reject 已观察到；
- 数据足以训练/验证：不满足；
- 训练后能否改变真实决策：尚未可测。

Round25/26 的 surrogate 模型也不能替代本轮真实物理证据。它们若选择 DETOUR，只能说明训练分布和风险目标下的保守偏置，不能证明物理近路不存在。

## 最小合理方法方向

若继续，建议只实现一个 deployment-only 的短时动作后果模型/风险规划器：

- 输入：T0/T1 历史、候选动作和 map/goal；
- 动作：`CAUTIOUS_ADVANCE`、`RETRACT`、必要时第二处合法 probe；
- 预测：短时 base progress、tracking deviation、stall/recovery 风险；
- 输出：继续、停止或撤回的校准风险，而不是物体类别；
- 安全监督：独立且不可被模型绕过。

直接枚举少量合法动作即可，不需要梯度规划或视频 world model。

## 进入训练前的硬 gate

- 同一物理机制内有平衡的正/负结果；
- 有独立 held-out condition；
- T2 的新增信息在 deployment-only 特征上可重复；
- 模型决策必须在真实 Isaac 中改变动作，而不只改变 offline score；
- 评价同时报告 false-pass、false-reject、safe recovery、路径代价和 PointGoal。

在这些条件满足前，继续扩大网络只会放大数据覆盖和执行接口的混淆。

