# RA-L 研究价值与原创性边界

## 机器人学问题是否真实

是。G1 在相同 0/1/2 候选语义下出现：

- 柔性条件都能产生局部 `RESPONSIVE`；
- whole-body clearance 只在部分锚定/状态/动作组合出现；
- rigid 能稳定拒绝并绕行；
- bottom-right 的 clearance 与 nominal/bottom-left 的失败不是单一 PointGoal yaw gate 造成。

这与普通静态几何导航不同，因为结果依赖机器人当前执行能力、接触后的短时运动和释放/恢复过程。

## 目前不能声称的贡献

本轮没有证明：

- 一个通用的跨帘幕物理泛化模型；
- 自动选侧已成功；
- 视觉导航已完成；
- learning 比强规则更好；
- “首次人形机器人穿帘子”。

相关交互导航、可通行障碍、触觉/接触导航和 MoDeSuite 工作已经覆盖“主动交互”“地图更新”“移动障碍”等概念本身。因此仅增加一个 probe 或把动作接入 RL 不足以构成贡献。

## 仍有条件价值的候选切口

**候选 1：embodiment-aware whole-body passage verification。** 不是分类 curtain/rigid，而是用部署可得的短时 progress、tracking 和恢复风险，判断当前 G1 动作是否值得继续，并把 candidate clearance 与 PointGoal 分开。

**候选 2：action-conditioned safe shortcut utilization。** 在同一未知候选语义下，让机器人在有限安全预算中比较继续、撤回和绕行，明确报告 missed-shortcut 与 false-pass，而不是把安全绕行的 100% 成功当作完整解决。

两者目前都只是有证据支持的研究方向，不是已完成的方法贡献。

## 主要 RA-L 风险

1. 正样本集中于 bottom-right，容易被审稿人判为场景特例。
2. R1/R2 不是独立物理技能，动作因果证据还不够强。
3. T0/T1 可观测性重叠，学习泛化尚未成立。
4. Frozen FALCON 的执行边界可能成为主要结果，而不是高层算法。
5. 当前为 Oracle map，不能包装成完整 RGB-D autonomy。

## 裁决

研究问题值得保留，但只能给 `SCIENTIFIC_CONDITIONAL_GO`。在投入 RA-L 主线前，唯一优先事项是完成同一动作/同一机制内的平衡短时推进对照，并证明 deployment-only 风险规则能在 held-out 条件减少 false-reject 或 false-pass。

