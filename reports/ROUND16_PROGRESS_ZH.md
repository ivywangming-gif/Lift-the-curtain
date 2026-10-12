# Round 16 进度记录

日期：2026-10-11

工作目录：`C:\Robotics\Lift-the-curtain`

## 总状态

`COMPLETE_WITH_RESEARCH_BLOCKERS`

本轮完成了 Round 12--15 的证据审计、控制接口审计、特权信息隔离审计、匹配场景证据账本和研究方向评估。没有启动新的 Isaac 进程，没有改变 FALCON、物理参数、环境依赖或既有阈值。

`ENGINEERING_BASELINE_COMPLETE = FALSE`（扩展鲁棒性和严格匹配实验仍未完成）。Round 12 的 nominal 双 episode 仍保留为其声明范围内的历史 PASS，不被本轮扩展失败覆盖。

## 约束核对

| 项目 | 结果 |
|---|---|
| 单 Isaac 进程、headless、num_envs=1 | PASS；本轮新 Isaac 启动数为 0 |
| frozen FALCON | PASS |
| 不启动 RL、VLM、teacher-student、实际 RGB-D | PASS |
| 不修改 Isaac Sim/Lab、驱动、CUDA、PyTorch、外部仓库 | PASS |
| 不连接真实机器人 | PASS |
| 不 commit/push | PASS |
| 工作区写权限 | PASS；Round2 临时写入测试已完成并清理 |
| 软件契约测试 | PASS；`36 passed` |

## 本轮阶段记录

### 1. 证据审计：PASS

读取并交叉核对了：

- `reports/ROUND12_ORACLE_MAP_END_TO_END_REPORT_ZH.md`
- `reports/ROUND13_ROBUSTNESS_REPORT_ZH.md`
- `reports/ROUND13_SCENARIO_MATRIX_ZH.md`
- `reports/ROUND13_FAILURE_TAXONOMY_ZH.md`
- `reports/ROUND14_POINTGOAL_CONTROL_REPORT_ZH.md`
- `reports/ROUND14_SPATIAL_PROBING_REPORT_ZH.md`
- `reports/ROUND15_ENGINEERING_REPORT_ZH.md`
- `reports/ROUND15_SPATIAL_PROBING_REPORT_ZH.md`
- `reports/ROUND15_HIDDEN_RIGID_REPORT_ZH.md`
- Round12--15 的 manifest、summary、events、telemetry 和 replay artifact。

已将证据分为：名义基线、工程失败/限制、运行时或仿真失败、严格不可评估四类。

### 2. 控制和接口干扰审计：PASS

- FALCON 的 `.25 m/s` cruise、`.30 m/s` `vx` 上限及显式 Stand/Walk 合约仍一致。
- SE(2) runner 使用世界坐标 `x/y/yaw` 反馈、饱和及 rate limit。
- PointGoal 的位置/停稳 gate 与历史 PoseGoal yaw gate 已分离。
- Phase-aware progress monitor 的阈值未改变；旧 nominal false stop 在 clearance 后被关闭，合成 stall 仍可检测。
- 空间记忆的默认 merge radius 为 `.35 m`，而两个候选站位相距 `.30 m`；软件契约已证明默认配置会合并两个站位。显式 `.20 m` 配置可保留两次预算，但真实双站位物理 episode 尚未运行。

### 3. 匹配证据账本：PASS（离线生成）

生成了 7 行场景摘要，统一保存 hand progress、progress ratio、tracking residual、effort proxy、base disturbance、traverse/clearance/goal gates 和事后物理评价字段。

关键结果：

- nominal：`RESPONSIVE`，hand progress `0.013487 m`，ratio `0.976709`，base disturbance `0.001557 m`，candidate clearance 和 goal stable 均为 true。
- double-density：`RESPONSIVE`，ratio `0.955485`，base disturbance `0.001553 m`，clearance false，goal false。
- bottom-left：`RESPONSIVE`，ratio `0.976825`，base disturbance `0.001541 m`，clearance false，goal false。
- bottom-right：`RESPONSIVE`，ratio `0.973907`，base disturbance `0.001549 m`，clearance true，goal stable true。
- both-bottom：`RESPONSIVE`，ratio `0.974330`，base disturbance `0.001531 m`，clearance false，goal false。
- hidden-rigid H2：`RESPONSIVE`，ratio `0.978150`，clearance false，goal false；严格 nominal 对照尚未在同一 G1 近场条件下完成。
- rigid：`BLOCKED`，ratio `0.032770`，base disturbance `0.007692 m`，候选区域未清空；Round14 PointGoal 的位置 gate 可通过，但旧 PoseGoal 的 yaw gate 失败。

### 4. 策略 A/B/C：部分完成

| 策略 | 状态 | 结论 |
|---|---|---|
| A：一次试探 + 固定穿越 | FAIL（扩展鲁棒性） | nominal 可复现；多个柔性约束下同为 RESPONSIVE 但无法清空候选区域 |
| B：一次试探 + phase-aware monitor + retract/detour | PASS（软件生命周期）/物理收益未证实 | monitor replay 消除旧误停并保留 stall 检测；没有新的物理 Goal success 证据 |
| C：最多两处空间化试探 | NOT ASSESSABLE（物理） | 软件 memory contract PASS；默认 merge 配置不允许第二站位，物理双站位未运行 |

## 生成的 Round16 产物

- `reports/artifacts/round16/evidence_ledger.json`
- `reports/artifacts/round16/engineering_confounds.json`
- `reports/artifacts/round16/matched_experiment_summary.json`
- `reports/artifacts/round16/run_manifest.json`
- `scripts/analyze_round16_evidence.py`

## 当前 blocker

1. 局部 hand-level 本体响应对多个柔性约束条件几乎相同，但全身 candidate-clearance 结果不同；这是真实研究问题的候选证据，不应通过放宽阈值消除。
2. hidden-rigid 的严格 matched G1 deployment comparison 尚未建立，不能把离线布料/墙面响应直接当成本体分类证据。
3. C 策略只有软件空间记忆 replay，没有实际第二站位的物理 episode。
4. 多场景尚无冻结参数下的独立重复成功率，不能宣称扩展鲁棒性。

## 下一步门槛

在任何 learning 决策前，必须先完成：

- 同一初始状态、同一 probe、同一时间窗的匹配重复实验；
- hidden-rigid 的共同预接触配置及无墙/有墙对照；
- 两站位真实物理试探、撤回、重新定位和安全预算验证；
- 把全身 progress、robot envelope 和安全恢复纳入统一的部署可得历史。

本轮没有证据支持启动学习。
