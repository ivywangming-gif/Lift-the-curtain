# Round 16 工程基线对比与特权信息审计

## 状态分层

| 层级 | 结论 | 不能推出的结论 |
|---|---|---|
| Round 12 nominal | PASS；curtain 和 rigid 正式 episode 均完成声明的地图闭环 | 不能推出对任意柔性约束的鲁棒通行 |
| Round 13 nominal frozen reproduction | PASS；冻结参数可复现 | 不能推出多场景成功率 |
| Round 13 扩展矩阵 | 有效暴露 probe-to-traverse gap | 不能把所有失败归因于物理模型 |
| Round 14 PointGoal/SE(2) | PASS；位置/停稳与旧 yaw gate 分离，世界坐标反馈已审计 | 不能把 yaw 失败当作地图或接触失败 |
| Round 15 monitor/spatial/hidden | 软件生命周期和空间记忆 contract 有进展；严格 hidden 和物理双站位未完成 | 不能宣称 B/C 提高物理 Goal success |
| Round 16 | 证据账本和研究边界完成 | 扩展工程 baseline 仍未 COMPLETE |

## 已被普通工程修复的问题

### 1. 终点 gate 混用

旧 rigid episode 的位置误差为 `0.051807 m`，但 yaw error `2.600305 rad`。Round14 引入 PointGoal 后，位置、停稳、有限性和安全 gate 可独立判断；历史 PoseGoal yaw 结果继续保留为审计数据，不再作为所有任务的统一 completion gate。

### 2. Monitor 跨阶段继承

旧 nominal replay 在 clearance 后触发 false stagnant。Round15 用 phase-aware lifecycle 在 robot-owned candidate clearance 处关闭 monitor；阈值未改变，合成 stall 仍可检测。这个修复解决的是状态机生命周期，不是柔性物理适应。

### 3. Telemetry 采样与持久化

Round15 将 monitor 对齐到已有 `.10 s` telemetry cadence，避免把 `.005 s` physics integration 误当作观测窗口；阶段 heartbeat 和最终 summary 可回溯。相关修复使后续失败可解释，但不改变物理结论。

### 4. 上肢 FK/IK 映射

Round15 修正 Isaac 到官方关节映射，并以 bounded DLS 做左手诊断；`.04 m` kinematic target error 为 `.001420 m`，nominal hand-level 运动和 curtain displacement 可观测。该修复排除了明显坐标映射错误，但没有解决持续全身穿越。

### 5. Stand/Walk 和 SE(2)

当前 FALCON adapter 有显式 `policy_walk_flag`，Stand 不接受非零行走 command；runner 使用 `.25 m/s` cruise、`.30 m/s` 上限。SE(2) 同时反馈世界 `x/y` 和 wrapped yaw，具备饱和和 command rate limit。静态测试 PASS，长时柔性接触仍是独立问题。

## 未被普通工程解决的问题

1. 相似的局部 `RESPONSIVE` 不能可靠预测全身 candidate clearance。
2. 持续接触下的 upper/lower-body 协调仍可能停滞或产生位置相关结果。
3. 默认空间 merge radius 与实际左右 probe 站位尺度不匹配。
4. hidden-rigid 的严格 matched deployment comparison 缺共同可达 pre-contact region。
5. 物理双站位策略 C 尚未运行，扩展场景也无冻结参数重复成功率。

## Privileged Information Leakage Audit

### Deployment observation：允许

当前 deployment observation contract 明确列出：

- `q`、`qdot`、`q_cmd`；
- 与真机对应的 `motor_effort`/执行器反馈 proxy；
- `previous_action`；
- 由自身关节状态和模型 FK 计算的手部位置/速度；
- base position/velocity、IMU orientation/angular velocity；
- 已知 0/1/2 map label；
- relative goal；
- 阶段一允许的 simulator pose localization assumption。

这些字段可以在真实机器人上获得，或已明确标注为定位假设。

### Evaluation ground truth：禁止进入部署决策

下列字段只可用于离线评价、物理 sanity 或外部 safety monitor：

- `object_kind`；
- simulator true contact flag/contact point；
- true gap、真实 stiffness/friction/fixed constraint；
- curtain runtime node position/velocity/displacement；
- true passability 和 outcome oracle。

### 代码路径核对

- `lift_curtain/control/deployment_observation.py` 将 deployment fields 与 evaluation fields 分成两个明确集合。
- `scripts/test_round15_oracle_map_episode.py` 中 `scene_kind` 用于 scene construction；在 responsive traverse 阶段可触发 rigid geometry 的外部安全终止。这是允许的 evaluation/safety path，不是部署分类器或导航动作分支。
- curtain displacement、stretch 和 `evaluation_contact_confirmed` 写入 summary 供事后评价；不进入 FALCON actor、probe classifier、map planner 或 branch selection。
- 控制器未读取 true curtain node、物体类别或 true passability。

### 剩余部署缺口

`base position` 当前来自 simulator pose 的阶段一定位假设；真实 G1 仍需等价的定位输入。若实际部署无法提供等价定位，正式 baseline 只能标为“仿真定位假设下 PASS”，不能宣称真机可部署。

## 工程基线判定

`Nominal baseline = PASS（限定场景）`

`Robust extension baseline = NOT COMPLETE`

`Privileged information isolation = PASS（含定位假设 caveat）`

`Learning readiness = NOT READY`

报告和 artifact 中保留 `COMPLETE_WITH_RESEARCH_BLOCKERS`，而不是把 nominal PASS 改写成整个项目完成。
