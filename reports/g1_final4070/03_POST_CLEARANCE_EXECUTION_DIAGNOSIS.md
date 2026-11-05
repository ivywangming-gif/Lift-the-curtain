# Candidate Clearance 与 PointGoal 执行诊断

## 契约

本轮正式任务是 PointGoal：

- XY error `<= 0.08 m`；
- 速度停稳并保持 dwell；
- 无跌倒、无不安全刚性碰撞；
- candidate clearance、safe retract、detour 和 PointGoal 独立记录；
- 终点 yaw 只作诊断，不进入 PointGoal gate。

## 代表性结果

| Trial | Candidate clearance | PointGoal | XY error | 解释 |
|---|---:|---:|---:|---|
| `B1_NOMINAL_R1_right_seed45` | 否 | 否 | 0.2140 m | 局部响应后持续推进未清空候选区域，PointGoal 原因是 XY_ERROR |
| `B2_NOMINAL_R1_right_seed46` | 是 | 是 | 0.0703 m | 同一名义动作在另一独立 seed 完成 clearance 和 PointGoal |
| `R2_BOTTOM_RIGHT_R1_right_seed45` | 是 | 是 | 0.0679 m | bottom-right 的条件化全身通行 |
| `R1_RIGID_R1_right_seed45` | 否 | 是 | 0.0743 m | rigid 安全拒绝后绕行；PoseGoal yaw 失败，但 PointGoal 通过 |

`R1_RIGID_R1_right_seed45` 的 final yaw error 约 2.145 rad，`pose_goal_pass=false`，但 `point_goal_pass=true`。这验证了 yaw 不能再混入 PointGoal 成功判定。

## 状态切换审计

当前 runner 已把 candidate clearance 后的 release/recovery 与 final goal 分开；`traverse_exit_is_final_goal=false`。刚体场景将候选区更新/绕行与最终目标停止分开，地图更新计数为 20 个 cell 的 2→1。

需要注意：历史 summary 中 `detour_waypoints_reached` 是路线完成字段，不能单独解释为“物理穿越”。本报告以 `round26_evidence_status`、`candidate_clearance_verified` 和 PointGoal evaluation 组合判定。

## 执行能力含义

nominal seed45 的失败不是 yaw gate，也不是仅仅 final stop 误差；其最终 XY 已偏离目标 0.214--0.265 m。相反，rigid 运行在拒绝后可以通过 0 区域到达 PointGoal。这说明后续导航能力存在，但候选区域内持续接触行走仍受 Frozen FALCON、身体/手部姿态和柔性约束共同限制。

## 结论

candidate clearance 不是 final goal，PointGoal 也不是物理可通行性的唯一指标。后续研究必须同时报告：

1. 是否清空候选区域；
2. 是否安全释放；
3. 是否走完绕行或通过路线；
4. 是否到达 PointGoal；
5. yaw 仅作诊断。

