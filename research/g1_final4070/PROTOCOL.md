# Final RTX 4070 G1 Feasibility Protocol

本目录定义 Windows RTX 4070 上最后一轮 G1 交互通行可行性实验。历史 Round 12--32、Scout02 和 `g1_research_feasibility` 只作为继承证据，不计为本轮新物理样本。

## 冻结项

- Unitree G1、FALCON checkpoint、Stand/Walk 契约、SE(2) 闭环和正式安全阈值不变。
- Isaac Sim/Lab、CUDA、PyTorch、driver、物理后端和第三方资产不修改。
- Oracle 0/1/2 map、PointGoal `XY <= 0.08 m`、candidate clearance 和安全结果独立报告。
- 部署决策只允许使用 q、qdot、q_cmd、effort proxy、FK、base pose/velocity 定位假设、previous action、phase、map 和 goal。
- object identity、contact truth、cloth nodes、true passability 和未来 outcome 只允许进入 evaluation。

## 假设

- P1：局部 `RESPONSIVE` 不足以保证 whole-body candidate clearance。
- P2：在匹配物理条件下，机器人状态或合法高层动作会改变有效全身结果。
- P3：T0/T1/T2 的部署可得历史包含优于强简单规则的可利用信息。

P1、P2、P3 分别评价；P1 成立不自动推出 P2 或 P3。

## 证据等级

1. `EXACT_MATCHED_STATE`：仅在完整物理内部状态可恢复时使用；当前柔性场景默认不具备。
2. `MATCHED_INITIAL_CONDITION`：相同配置、seed、reset、目标、安全预算和独立 episode。
3. `RANDOMIZED_GROUP_COMPARISON`：预注册分组与随机动作分配，用平均结果比较。

## 第一批运行矩阵

资源基线先运行空场、rigid、柔性场景各两条。正式机制矩阵优先覆盖 nominal、bottom-right、bottom-left、rigid，比较 R1/R2、right/left probe，并以 seed 45 为主配对，关键条件用 seed 46 独立重复。

R1/R2 共享同一动作集合和安全监督。源码审计结论：R1 在 `RESPONSIVE` 后请求 `CAUTIOUS_ADVANCE`；R2 还要求 hand progress、stall duration 和 upper tracking 满足冻结阈值，否则 `RETRACT`。这是一种高层动作后果比较，不是跨技能泛化。

## 每条 trial 的独立 gate

- `CONTACT_ACQUISITION`
- `LOCAL_RESPONSE`
- `EXECUTION_PROGRESS`
- `SUSTAINED_PROGRESS`
- `WHOLE_BODY_CLEARANCE`
- `SAFE_RETRACT`
- `DETOUR_SUCCESS`
- `POINTGOAL_SUCCESS`
- `RUNTIME_VALIDITY`

S1--S4 只由独立 episode 产生；rigid 绕行不计 S4，candidate clearance 后 PointGoal 失败不计完整导航成功。

## 停止规则

- 新 G1 启动最多 48 条，目标 18--24 条有效匹配物理结果。
- 同一运行时失败最多两次有依据修正；没有新假设时不重复。
- 若出现 OOM、device lost、持续内存压力或明显吞吐崩溃，停止对应负载并转入资源/迁移诊断。
- 若已有数据足以否定某项假设，停止无信息的新 seed。
- 小型学习只在至少两个物理条件具有正负支持、可做独立 held-out condition 时运行。

