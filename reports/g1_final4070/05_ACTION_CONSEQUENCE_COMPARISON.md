# 合法动作后果比较

## R1/R2 的真实语义

源码 `lift_curtain/navigation/round25b_policy_interface.py` 显示：

- R1 在 `RESPONSIVE` 后直接请求 `CAUTIOUS_ADVANCE`。
- R2 还要求 `hand_forward_progress >= 0.008 m`、`stall <= 0.15 s`、`tracking <= 0.30 rad`。
- 当前 probe 的柔性样本 stall 约 0.73--0.77 s，因此 R2 在所有柔性响应 trial 中都请求 `RETRACT`。
- Rigid 已经是 `BLOCKED`，两者都撤回。

所以 R1/R2 是同一个物理技能后的不同高层门控，不是两个独立推帘技能。这个限制必须写入科研解释。

## 匹配配对

| 场景/侧/seed | R1 首动作 | R1 clearance | R2 首动作 | R2 clearance | 主要含义 |
|---|---|---:|---|---:|---|
| bottom-left / left / 45 | advance | 否 | retract | 否 | 动作不同，但两者都绕行 |
| bottom-left / right / 45 | advance | 否 | retract | 否 | 左右交换未改变结果 |
| bottom-left / right / 46 | advance | 否 | retract | 否 | 重复仍未出现 clearance |
| bottom-right / left / 45 | advance | 是 | retract | 否 | 同一条件下动作选择改变近路利用 |
| nominal / left / 45 | advance | 否 | retract | 否 | R1 继续并最终 PointGoal 失败，R2 安全绕行 |
| nominal / right / 45 | advance | 否 | retract | 否 | 同上 |
| nominal / right / 46 | advance | 是 | retract | 否 | 独立 seed 下 R1 利用近路，R2 放弃近路 |
| rigid / left / 45 | retract | 否 | retract | 否 | 两者一致安全拒绝 |

## 汇总

- 非空 R1：12 条，clearance 4 条，PointGoal 10 条。
- 非空 R2：10 条，clearance 0 条，PointGoal 10 条。
- Rigid：5/5 `BLOCKED`，无 false-pass。
- R2：所有柔性 `RESPONSIVE` 试验都保守撤回，安全但错过了 4 条 R1 已观察到的 clearance 机会。

这给出明确的 false-reject 证据，但不能把 R2 的撤回解释为“物体被证明不可通过”。代码和 summary 均将其标为 `UNVERIFIED_AVOIDED`，候选区域没有被永久写成 1。

## 因果边界

可以支持的结论：

1. 合法高层动作会改变任务结果：继续可能利用近路，撤回必然转向绕行。
2. 当前 R2 的 frozen stall 门限过于保守，未能利用已有的 post-probe 局部响应。
3. 同一物理条件下存在“安全到达但错过近路”和“尝试近路但可能失败/成功”的真实 trade-off。

不能支持的结论：

1. R2 失败证明某种物理材料不可通行；
2. R1/R2 已经构成跨技能物理泛化；
3. 仅凭这 24 条 trial 就能学习任意新锚定机制的策略。

## 研究含义

下一步应把动作后果预测定义为“短时合法推进是否值得继续”，而不是物体类别分类。需要在同一机制内取得足够的正/负后果，并让短时推进、停止和撤回共享相同安全预算。

