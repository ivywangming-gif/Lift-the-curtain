# Round 16 研究诊断

## 结论摘要

本轮最可靠的结论不是“所有柔性障碍都不可通行”，而是：在已冻结的 nominal 控制器和相同局部试探窗口下，多个场景给出几乎相同的 `RESPONSIVE` 本体输出，但后续全身 envelope 是否清空候选区域、是否能继续前进和是否到达 PointGoal 明显不同。因此，单次局部手部响应不能被解释为 `PASSABLE`。

Round 12 的 nominal curtain/rigid 两个正式 episode 仍然成立；Round 13--15 证明的是 nominal 之外的工程边界和研究问题，不应合并成一个笼统的 COMPLETE/INCOMPLETE 标签。

## 证据等级

### 名义工程基线：PASS（限定范围）

- Round 12 curtain：经过候选 2，完成受限交互、全身通过、地图和目标 gate。
- Round 12 rigid：`BLOCKED/UNCERTAIN -> 2->1 -> zero-only detour -> Goal`。
- Round 13 frozen nominal reproduction：`RESPONSIVE`，candidate clearance true，goal stable true，no fall。

这只证明固定 nominal 场景、既定近场几何和既定一次试探预算下的工程闭环可运行。

### 扩展工程失败或限制：FAIL/BLOCKED

- double-density、bottom-left、both-bottom 和 H2 在手部层面均为 `RESPONSIVE`，但 candidate clearance false 或最终 Goal gate false。
- Round 13 rigid 的位置误差为 `0.051807 m`，但旧 PoseGoal yaw error 为 `2.600305 rad`；这不是地图更新失败，而是旧终点 yaw 合约与 PointGoal 目标混用造成的工程接口问题。
- Round14 发现 monitor 在 robot-owned clearance 之后继承旧阶段状态；Round15 已用 phase-aware lifecycle 修复，阈值未放宽。
- 默认 spatial merge radius `.35 m` 大于左右站位间距 `.30 m`，默认配置会将两个位置合并；显式 `.20 m` 软件契约可保留两个站位，但未完成实际物理验证。

### 运行时/仿真失败：独立记录

过去的 telemetry 落盘、阶段 heartbeat 和 monitor 采样节奏问题已经被定位并修正。它们不能再被当作柔性物理不可通行的证据。RGB-D renderer 的硬件/运行环境 blocker 也不在本轮新启动；没有再次触发 native crash。

### 严格不可评估：NOT ASSESSABLE

- close-gap hidden-rigid 的离线布料物理对照已通过，但没有共同有效的 nominal G1 hidden deployment 预接触配置，不能报告严格本体分类结果。
- C 策略只有 spatial memory 软件 replay，尚无第二处物理试探、撤回和重新定位的 matched episode。
- 尚无冻结参数下的多次独立重复成功率。

## 最早有效扩展失败

在排除旧 telemetry、PointGoal/PoseGoal 混淆、monitor lifecycle 和静态 SE(2) 接口问题之后，最早的有效扩展失败是：

> probe 阶段本体判定为 `RESPONSIVE`，但后续机器人自己的 collision envelope 不能持续通过候选区域，或全身候选清空 gate 不能成立。

这比“布料发生了位移”更严格，也比“单个 hand contact 成功”更接近真正的通行性问题。证据路径：

- `reports/artifacts/round13/local_probe_representativeness.json`
- `reports/artifacts/round13/episodes/double_density_frozen/summary.json`
- `reports/artifacts/round13/episodes/bottom_left_anchor_safety_fix_revalidation/summary.json`
- `reports/artifacts/round13/episodes/bottom_right_anchor_safety_fix_revalidation/summary.json`
- `reports/artifacts/round13/episodes/both_bottom_corners_safety_fix_revalidation/summary.json`
- `reports/artifacts/round13/episodes/curtain_hidden_rigid_h2_revalidation/summary.json`

## 局部试探与全身结果的因果比较

三个底部约束场景的局部特征如下：

| 场景 | 决策 | 手部进展比 | 上肢 tracking RMS | effort deviation | base disturbance | 全身结果 |
|---|---:|---:|---:|---:|---:|---|
| bottom-left | RESPONSIVE | 0.976825 | 0.061774 rad | 13.1728 | 0.001541 m | clearance FAIL，Goal FAIL |
| bottom-right | RESPONSIVE | 0.973907 | 0.061737 rad | 13.1739 | 0.001549 m | clearance PASS，Goal stable PASS |
| both-bottom | RESPONSIVE | 0.974330 | 0.061740 rad | 13.1759 | 0.001531 m | clearance FAIL，Goal FAIL |

这组差异足以支持“局部信息不足”的研究假设，但不足以支持材料分类器或 hidden-wall 归因。bottom-right 的通过机会与 hand-level response 相近，提示动作位置、约束分布和身体 envelope 的耦合很重要。

Nominal 与 double-density 也显示相同方向：nominal ratio `0.976709`，double-density `0.955485`，两者都为 finite/no-fall，但后者 exit error `0.219409 m`、clearance false、goal false。该差异可能来自物理约束、持续接触控制或二者耦合，不能仅凭结果归因于物理模型。

## 失败分类

| 类别 | 当前判断 | 证据与边界 |
|---|---|---|
| `PHYSICAL_MODEL_FAILURE` | 非 nominal 的主要候选，但未完全证明 | bottom anchor、density 和 hidden physics 有差异；需要严格共同预接触及重复实验 |
| `LOCOMOTION_CONTROL_FAILURE` | 已有证据 | 旧 PoseGoal yaw、多个 traverse 停滞；FALCON 低速响应和 upper/lower 耦合仍限制长时接触 |
| `CONTACT_ACQUISITION_FAILURE` | 局部存在 | hidden G1 deployment 的 hand progress `0.004405 m`、ratio `0.31898`、effort deviation `32.3423`；严格 matched nominal 不可评估 |
| `PROPRIOCEPTIVE_AMBIGUITY` | 已有证据 | 三个底部场景同为 RESPONSIVE，却出现 pass/fail 分化 |
| `SPATIAL_PROBE_SELECTION_FAILURE` | 软件限制已发现，物理未证实 | `.30 m` 站位距离被默认 `.35 m` 合并；`.20 m` contract replay PASS |
| `LONG_HORIZON_CONTACT_ADAPTATION_FAILURE` | 已有证据 | 手部响应后持续前进/clearance gate 失败 |
| `SAFE_RECOVERY_FAILURE` | 不是最早 blocker | retract/detour 逻辑和 2->1 软件路径存在；hidden 最终稳定 gate 仍失败，需单独修复 |

## 三种策略的研究解释

### A：固定一次试探 + 固定穿越动作

对 nominal 有效，但对 distributed flexible constraints 不鲁棒。它把瞬时 `RESPONSIVE` 隐含地当作足够的通行预测，无法解释 bottom-left/bottom-right 的位置相关差异。

### B：一次试探 + phase-aware monitor + 安全撤回/绕行

Phase-aware replay 证明了 monitor 生命周期修复有效：nominal clearance 后不再误停，合成 stall 仍在约 `0.30 s` 检出。它提高了状态机安全性和可审计性，但目前没有物理证据证明最终 Goal success rate 已提高。

### C：最多两处空间化试探

软件 contract 已表达“两个空间上可区分的位置、预算耗尽后不再尝试”。物理实验尚未运行，所以不能宣称 C 提供了额外成功率。它的价值在于为“位置相关的可通行机会”提供可验证动作接口，而不是直接提供答案。

## 是否已经需要学习

没有。当前证据还混合了未完成的严格 matched comparison、长时 locomotion 接口限制和物理双站位缺失。更早引入学习会把接口误差、终点 gate、空间记忆和真实物理效应混在一起，无法识别增益来源。

当前最合理的下一研究问题是：在只使用部署可得本体历史、robot envelope、已知 0/1/2 地图和有限空间化试探的情况下，设计一个有明确安全预算的 `MOVABLE/RESPONSIVE` 到 `PASSABLE` 过渡协议，并用匹配重复实验验证其因果性。

## 十个问题逐项回答

1. **Nominal 工程 baseline 是否仍然成立？** 成立，但仅限 Round12 声明的 nominal curtain/rigid 场景、冻结控制器和已知地图闭环；扩展鲁棒性不成立。
2. **现有扩展场景最早的有效失败是什么？** hand-level `RESPONSIVE` 之后，whole-body candidate clearance 或持续前进失败；不是旧 telemetry 或单纯 yaw gate。
3. **哪些问题已经被普通工程修复？** Stand/Walk 合约、SE(2) 世界坐标反馈、PointGoal 与 PoseGoal 分离、monitor phase lifecycle、telemetry cadence、FK/IK 映射和 map 2->1 软件契约。
4. **是否存在可复现的局部试探信息不足？** 存在；bottom-left、bottom-right、both-bottom 的局部特征近似，但全身结果分别 FAIL、PASS、FAIL。
5. **两处试探是否提供额外有用信息？** 软件空间记忆 contract 提供了可区分站位和预算信息；真实物理双站位尚未运行，因此物理增益为 NOT ASSESSABLE。
6. **持续反馈是否改善 Goal success 和安全性？** 改善了阶段安全性和误停行为；replay 中 nominal clearance 后不再误停，合成 stall 仍可检出；尚无新的物理 Goal success 增益证明。
7. **哪些场景存在动作/位置相关的可通行机会？** bottom-right 单次扩展场景通过，而 bottom-left/both-bottom 在近似局部响应下失败；这支持位置相关机会，但不支持泛化率。
8. **哪些失败其实是物理模型或 WBC 接口问题？** 长时 traverse、旧 yaw 终点、hidden pre-contact 和 upper/lower 协调包含 WBC/接口因素；close-gap 物理差异已经有离线证据，但不能直接等同部署分类。
9. **目前最值得研究的一个方法问题是什么？** 如何把有限本体历史、robot envelope、progress monitor 和最多两处 spatial probe 组合成安全的 `RESPONSIVE -> PASSABLE/BLOCKED` hybrid contract。
10. **是否有足够证据开始学习高层策略？** 没有。严格 hidden matched、物理双站位和冻结参数重复集尚未完成，继续纯工程路线。
