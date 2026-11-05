# 部署可观测性与特权信息审计

## 允许进入决策的字段

本轮 policy decisions 使用的字段属于部署可得范围：

- `q`、`qdot`、`q_cmd`；
- motor effort proxy；
- previous action；
- 左/右手 FK 位置与速度；
- base pose/velocity（当前明确标为 Isaac root pose localization assumption）；
- phase、map label、relative goal；
- 由上述字段计算的 tracking、progress、stall 和 history。

## 禁止字段

以下只写入 evaluation ground truth 或 external safety monitor，没有进入 classifier、memory、planner 或 locomotion controller：

- object kind；
- simulator contact flag/contact force truth；
- cloth node positions/displacement；
- true hand-obstacle gap；
- true passability；
- future episode outcome。

## 机器审计

- 本轮 policy decision 总数：188。
- `forbidden_fields_attempted`：空。
- 所有 decisions 的 `deployment_fields_used` 均没有越界字段。
- `evaluation_ground_truth.json` 明确标注 offline-only。
- Hand contact reporter 即使初始化，也只用于 evaluation；部署 classifier 不读取它。

这意味着本轮没有用 simulator 真值偷偷补全决策。但也意味着“接触力不可部署获得”的现实限制仍然存在：当前工程规则主要依赖 FK 进展、q tracking、速度和 effort proxy，不能把 effort proxy 称为六维力传感。

## T0/T1/T2 证据

在 nominal、bottom-left、bottom-right 的非空柔性样本中：

- T0：正 4、负 13；hand progress、base drift、tracking 和 effort 的范围高度重叠。
- T1：正 4、负 13；固定 probe 后仍高度重叠。
- T2：正 4、负 5；继续动作后的 hand/base progress 和 tracking 有明显差异，但这是动作已执行后的反馈，且正样本集中于 bottom-right 加一个 nominal seed。

因此：

- 固定 probe 截止点没有足够证据支持稳定的预动作分类器；
- 合法短时推进可产生新信息，值得研究 phase-aware risk monitor；
- 不能把 T2 的 post-action 分离能力误写成 T1 的普适可预测性。

## 真机准备状态

当前 phase-1 允许仿真 root pose 作为定位假设。正式真机声明前仍需替换为可获得的定位源，并确认 actuator effort、左右手 FK 和时序接口与仿真 schema 一致。这个缺口是 deployment readiness 限制，不是本轮 4070 资源限制。

