# Round 12：Oracle 0/1/2 地图交互导航闭环工程报告

**日期：** 2026-10-10
**工作目录：** `C:\Robotics\Lift-the-curtain`
**运行约束：** headless、`num_envs=1`、单 Isaac/Kit 进程、冻结 FALCON、无 learning、无实际 RGB-D renderer、无真实机器人。
**正式结论：** `ENGINEERING_BASELINE_COMPLETE = TRUE`
**范围限制：** 冻结参数鲁棒性重复矩阵未完成；真机 readiness 仍为 `NOT_READY`。

## 1. 结论总览

| Gate | 状态 | 证据结论 |
|---|---|---|
| 初始候选路线选择 | `PASS` | 规划器选择经过候选 `2` 的路线，候选代价 `2.4 m`，零单元绕行代价 `4.1 m`。 |
| Curtain probe 与本体决策 | `PASS` | 本体历史输出 `RESPONSIVE`；仿真物理响应只作事后评价。 |
| Curtain 全身受控穿越与目标导航 | `PASS` | 身体包络清空候选区，释放后重锚定，最终 `GOAL_REACHED`。 |
| Rigid probe 与本体决策 | `PASS` | 本体历史输出 `BLOCKED`，执行撤手。 |
| Rigid `2 -> 1` 更新与绕行 | `PASS` | 在线地图实际更新，零单元 detour 到达目标。 |
| 特权信息隔离 | `PASS_WITH_DEPLOYMENT_ASSUMPTIONS` | 未发现特权字段进入部署决策路径；定位和 effort proxy 仍是假设。 |
| 冻结参数重复矩阵 | `BLOCKED/INCOMPLETE` | 首个扰动重复无 telemetry/summary，未计入成功率，也未伪造结果。 |
| 是否需要 learning | `NOT REQUIRED` | 当前证据支持继续纯工程路线。 |

正式双 episode 和地图交互契约均通过，因此按本轮任务书将工程 baseline 标记为 `TRUE`。这不表示已经完成真机部署或鲁棒性统计。

## 2. Round 11 失败审计

### 2.1 坐标和 gate 混用

Round 11 的 `responsive_traverse` 错把最终 goal `(1.35, 0)` 当作穿越阶段目标。穿越、候选区清空、释放恢复和最终目标导航没有独立状态。

在 Round 11 C2 中：

- traverse arrival：`(1.283064, -0.099392) m`；
- Stand 后：`(1.249986, -0.098965) m`；
- Stand 期间 base x 后退 `0.033081 m`；
- arrival 到 `(1.35, 0)` 的位置误差约 `0.119830 m`，大于中间 waypoint 的 `0.08 m`，所以 `motion_reached=false`；
- Stand 后独立目标误差 `0.140702 m`，超过已有 planner 最终位置 gate `0.14 m` 约 `0.000702 m`；
- 身体包络已经越过 curtain plane，未跌倒，也没有 external safety stop。

因此 `motion_reached=false` 不是唯一问题：它反映了错误复用中间 waypoint gate；另一个独立问题是 Stand 后的实际 base 漂移和最终位置 gate。不能通过把一个 tolerance 放宽来掩盖这两个问题。

### 2.2 Round 12 修复

状态明确拆分为：

`APPROACH_2 -> STAND_AND_PROBE -> CAUTIOUS_TRAVERSE -> CLEAR_CANDIDATE_REGION -> RELEASE_AND_RECOVER -> NAVIGATE_TO_GOAL -> GOAL_REACHED`

修复只做了两件工程性工作：

1. `CAUTIOUS_TRAVERSE` 使用独立的 `traverse_exit` 和机器人自身 collision envelope 清空 gate；
2. 最终 goal waypoint 使用已有 planner `0.14 m` 位置容差，普通中间 detour waypoint 仍使用 `0.08 m`，并保留 `0.15 rad` yaw gate、速度驻留和无跌倒条件。

没有修改 FALCON checkpoint、物理参数、probe 阈值、cruise speed、`vx_max=0.30 m/s` 或真实场景约束。

## 3. Case A：固定顶部 Curtain

### 3.1 初始路线与接近

Oracle map 语义为 `0=FREE`、`1=FIXED_OBSTACLE`、`2=INTERACTION_CANDIDATE_UNKNOWN_RESULT`。候选区域在 curtain 和 rigid 两个 case 中完全相同，初始地图不携带物体类别。

初始比较结果：

- candidate route：`2.4 m`；
- zero-only detour：`4.1 m`；
- 选择：`candidate_2`；
- 直线段包含 label `2`，因此不能直接当作 FREE 执行；
- pre-contact waypoint：约 `(0.45, 0) m`；
- 实际到达并稳定在 `(0.409215, -0.042965) m`，预接触区域和低速 Stand gate 通过。

### 3.2 本体试探

部署决策只使用 `deployment_observation` 历史：

- decision：`RESPONSIVE`；
- 样本数：`400`；
- hand forward progress：`0.013521 m`；
- no-contact baseline progress：`0.013809 m`；
- progress ratio：`0.9791`；
- upper tracking RMS：`0.06177 rad`；
- effort deviation：`13.1731`；
- base disturbance：`0.001524 m`。

仿真 runtime curtain displacement 为 `0.045629 m`，用于离线确认确实发生了物理布料响应，不进入 classifier、memory、planner 或动作选择。`RESPONSIVE` 只表示在这段受限试探中仍有运动响应，不等于 `PASSABLE`。

### 3.3 持续接触行走、清空和释放

Curtain 分支执行低速 SE(2) 闭环和 upper reference 更新。独立 traverse exit 目标是 `(1.30, 0) m`，不是 final goal。

结果：

- traverse arrival：`(1.258843, -0.102523) m`；
- 独立 traverse exit error：`0.126850 m`，该 waypoint 的 `motion_reached=false`；
- Stand 后仍有限且无跌倒；
- 最小机器人尾端 x：`0.978374 m`；
- 候选区域远端 x 约 `0.91 m`，加 `0.04 m` 安全余量后要求为 `0.95 m`；
- 所有机器人身体和非手部身体均通过 `x >= 0.95 m` 的 clearance gate；
- `opening_verified_from_robot_geometry=true`；
- curtain top drift 最大约 `8.5e-5 m`；
- 邻接 stretch ratio 最大约 `1.061`；
- 无 curtain external safety stop、无跌倒。

这说明机器人虽然没有把一个独立的理想 exit waypoint 以 `0.08 m` 局部条件精确停住，但已经满足更重要的候选区身体清空条件。随后执行 retract/neutralize/Stand，依据实际 base/yaw 重新建立常规目标 reference。

### 3.4 最终目标

- final base：`(1.244247, -0.011206) m`；
- position error：`0.106345 m <= 0.14 m`；
- final yaw error：`0.131116 rad <= 0.15 rad`；
- speed dwell：通过；
- finite/no-fall：通过；
- state：`GOAL_REACHED`；
- repeated probe：`false`。

因此 Case A 的正式状态为 `PASS`。报告保留了 traverse exit `motion_reached=false` 这个事实，没有将它改写为“每个 waypoint 都精确到达”。

## 4. Case B：固定 Rigid Panel

### 4.1 相同地图和相同执行器

Rigid 只改变场景构建。地图标签、pre-contact 区域、probe 轨迹、FALCON、SE(2) controller、Stand/Walk contract、本体阈值和状态机均相同。执行端没有 `if object_is_curtain: pass else: detour` 分支。

### 4.2 本体试探与撤手

部署历史输出：

- decision：`BLOCKED`；
- hand forward progress：`0.000453 m`；
- progress ratio：`0.03277`；
- stall duration：`2.0 s`；
- effort deviation：`26.7832`；
- base disturbance：`0.007692 m`；
- retract/neutralize/post-retract Stand：finite、stable、无跌倒；
- `probes_used=1`，`repeated_probe_allowed=false`。

仿真接触力最大约 `13.4685 N` 只出现在 `evaluation_ground_truth.json`，不是本体判别输入。

### 4.3 在线地图更新和 detour

初始候选区域有 `20` 个 label-2 单元。BLOCKED 后：

- `navigation_map` 实际更新 `20` 个单元 `2 -> 1`；
- `initial_oracle_map` 保持原始 `2`，用于重现；
- 更新地图计数：`0=871、1=29、2=0`；
- 新规划不穿过任何 `1`，实际 detour 路径只采样 label `0`；
- 不因换视角或新 component id 获得第二次 probe；
- final goal error：`0.051807 m`；
- state：`GOAL_REACHED`。

因此 Case B 的正式状态为 `PASS`。

## 5. 地图和状态机契约

### 5.1 初始和在线地图

- origin：`(-0.80, -1.50) m`；
- resolution：`0.10 m`；
- size：`30 x 30`；
- candidate curtain/rigid：相同区域、相同 label `2`；
- 结构固定障碍：label `1`；
- 2 未 probe 前不是 FREE；
- 机器人不能直接跨越 2。

### 5.2 交互后的语义

- `RESPONSIVE`：只允许进入受限 `CAUTIOUS_TRAVERSE`，不立即把 2 改成 0；
- 身体清空和安全条件通过后，允许释放并导航；
- `BLOCKED` 或 `UNCERTAIN`：停止、撤手、消耗一次 probe、把当前候选区域 `2 -> 1`、沿 0 重规划；
- 本 episode 最多一次主动 probe。

纯逻辑和跨模块回归共 `23 passed`，覆盖候选 route 比较、2 非 FREE、状态转移、2->1、零单元 detour、一次 probe memory 和运行时字段隔离。

## 6. Privileged Information Leakage Audit

机器可读审计：`reports/artifacts/round12/privileged_information_leakage_audit.json`。

### 部署决策实际允许读取

- `q`、`qdot`、`q_cmd`；
- 与真机对应的 actuator effort estimate/proxy；
- `previous_action`；
- 由自身关节状态和运动学得到的 hand FK position/velocity；
- IMU 派生姿态和角速度；
- 当前阶段允许的 Isaac root pose localization assumption；
- 已知 0/1/2 map、当前 pose、relative goal 和 episode interaction memory。

### 禁止进入决策路径

`object_kind`、simulator contact flag/point/force、true hand-obstacle gap、curtain node positions/displacement、真实 mass/stiffness/friction/fixed constraints、true passability、episode outcome oracle。

当前 runner 的 `scene_kind` 只用于场景构建、离线评价和外部安全监控；本体 decision 先于评价字段读取，分类输入来自 deployment history。当前审计没有发现特权信息泄漏，但真机部署仍缺少经验证的定位源、effort 标定和可部署安全监督，因此 real-robot readiness 为 `NOT_READY`。

## 7. 重复实验和资源 blocker

正式双 episode 通过后尝试了一个冻结参数扰动：curtain reset `y=+0.04 m`、yaw `+0.03 rad`。约 12 分钟只产生 `resolved_config.json`，没有 telemetry/summary；为避免单 Isaac 进程无限占用，已结束该进程，状态记为 `BLOCKED_RUNTIME_NO_ARTIFACT`。

因此：

- 没有把该次重复计入成功率；
- 没有据此推断控制失败或成功；
- 其余每种场景 2–3 次重复为 `NOT RUN`；
- 鲁棒性统计不是本轮正式双 episode 完成 gate，仍需后续有资源时单独完成。

## 8. 对任务问题的明确回答

1. **候选 2 路线是否被优先选择？** 是。候选代价 `2.4 m` 小于零绕行 `4.1 m`，但 2 没有被当作 FREE。
2. **Curtain 是否由本体规则允许继续？** 是，matched history 输出 `RESPONSIVE`；这不是读取 curtain 类别，也不等于 passability。
3. **Curtain 是否完成真实全身受控穿越？** 在仿真 formal episode 中是。物理评价确认布料非零响应，机器人自身 envelope 清空候选区，释放后到达目标；但独立 traverse-exit waypoint 没有以 `0.08 m` 局部 gate 到达，这一限制已如实保留。
4. **Rigid 是否安全阻挡并绕行？** 是。`BLOCKED -> retract -> 2->1 -> zero-only detour -> goal` 全链路通过。
5. **原始 2 是否真的更新为 1？** 是，20 个候选单元在 `navigation_map.json` 中变为 1；不是只更新 memory。
6. **是否重复 probe？** 否。两个 formal episode 都是一次 probe，`repeated_probe_allowed=false`。
7. **目标侧 gate 是否与穿越 gate 分离？** 是。`CLEAR_CANDIDATE_REGION`、`RELEASE_AND_RECOVER` 和 `GOAL_REACHED` 分开记录；Round 11 的混用已修复。
8. **是否使用了仿真物体类别或接触真值做动作决策？** 没有。相关字段只进入评价/安全 artifact。
9. **两个正式 episode 是否分别通过？** 是，curtain 和 rigid 的 summary 状态均为 `PASS_ROUND12_PHYSICAL_*_EPISODE`，且最终状态为 `GOAL_REACHED`。
10. **鲁棒性是否已经完成？** 没有。首个扰动重复无 artifact，后续重复未运行，不能宣称统计鲁棒。
11. **当前最早工程 blocker 是什么？** 不是地图语义或一次 probe memory；剩余最早 blocker 是在不同 reset 扰动下的 Isaac 运行可复现性，以及接触行走对 upper/lower 协调和定位/effort 假设的真机验证。
12. **是否有证据必须使用 learning？** 没有。当前结果可由状态机 gate、SE(2) 闭环、部署可观测本体历史和地图更新契约解释，继续保持纯工程路线。

## 9. 证据索引

- 进度：`reports/ROUND12_PROGRESS_ZH.md`
- manifest：`reports/artifacts/round12/run_manifest.json`
- 复现：`reports/ROUND12_REPRODUCE.ps1`
- curtain summary/events/telemetry：`reports/artifacts/round12/episodes/curtain_final_gate_fix/`
- rigid summary/events/telemetry：`reports/artifacts/round12/episodes/rigid_primary/`
- 特权信息审计：`reports/artifacts/round12/privileged_information_leakage_audit.json`
- 状态机/地图实现：`lift_curtain/navigation/round12_state_machine.py`、`lift_curtain/navigation/oracle_grid_map.py`
- 统一 runner：`scripts/test_round12_oracle_map_episode.py`

本报告没有把诊断性接触、评价侧真值或未完成重复实验包装成额外成功；正式完成结论仅基于两个真实 Isaac G1 formal episode、独立状态 gate 和在线地图文件证据。
