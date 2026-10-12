# Round 10 FALCON 速度控制与接触行走工程报告

**日期：** 2026-10-09
**阶段：** FALCON 速度工作区间、Stand/Walk 切换与接触行走修复
**总体状态：** `COMPLETE_WITH_SCOPE_LIMITS`
**工程 baseline：** `ENGINEERING_BASELINE_COMPLETE = FALSE`
**Learning：** `NOT REQUIRED BY CURRENT EVIDENCE`

## 1. 执行摘要

Round 9 的最早控制 blocker 已定位并修复：walkthrough 的 `vx` 被外层 tracker 限制为 `0.08 m/s`，同时旧 adapter 无法显式表达“零速度 Walk”和“Stand”的区别。该组合让低速步态响应差、停止语义不正确，并造成 `71.9424%` 的外层命令饱和。

Round 10 完成了以下有实际机器人运动证据的诊断：

- 10 组无障碍速度工作区间测试，确认 `0.25 m/s` 是当前 checkpoint 的保守有效工作点，外层前向上限取 `0.30 m/s`。
- 显式 `Walk -> Decelerate -> Stand` 测试，确认零速度 Walk 不能替代 Stand；显式 Stand 后 neutral/probe-hold 分别在 `0.525/0.515 s` 内满足停稳驻留条件。
- formal 2 m matched E0/E1：无帘子与有帘子均完成全身越过评价平面。帘子使实际速度显著下降，但未阻止闭环收敛，布料 top drift 和 stretch 有界。
- 实体 formal rigid panel：G1 实际走到预接触区域，完成一次受限试探，本体规则输出 `BLOCKED`，随后撤手、写入 episode memory，并仅沿 label-0 路径绕行到目标。
- privileged information leakage audit 未发现仿真物体类别、contact flag、布料节点或 true passability 进入工程决策路径。

仍未执行一个从 Oracle Map 远端起点开始、走到候选 2、真实试探 formal curtain、受控穿越并到达目标的统一 curtain episode。因此本轮不能宣称双分支完整工程 baseline 已完成。真实部署还缺可用定位源、真实 G1 effort proxy 标定和真机安全监控验证。

## 2. FALCON 命令契约审计

### 2.1 训练期小速度过滤

官方 FALCON 源码在训练命令重采样阶段执行 `norm(command_xy) > 0.2`；不满足时将平面命令置零。该逻辑位于训练环境 `_resample_commands()`，不是 ONNX 推理图中的 deadband，本地 adapter 也没有额外添加 deadband。

当前模型：

- FALCON 提交：`a967a6d8494f57777cf8d266a644ac8e45833301`
- ONNX SHA256：`8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0`
- 输入/输出：`[1,575] -> [1,29]`
- `575 = 115 x 5`，与当前单帧 observation contract 和 5 帧 history 精确一致。

ONNX 本身没有嵌入训练配置元数据，因此“该 checkpoint 对应此训练配置”是由版本化模型路径、固定 hash、官方 README 和精确 observation contract 联合支持，而不是由 ONNX 内部元数据单独证明。

### 2.2 Stand/Walk 语义

官方字段名为 `command_stand`，但数值语义是：

- `0`：stance/Stand；
- `1`：Walk。

零速度与 Stand 是两个不同状态。Round 10 在 `VelocityCommand` 中加入显式 `walk_mode`：

- `walk_mode=True` 可表达零速度 Walk；
- `walk_mode=False` 表达 Stand；
- Stand 配合非零速度会被拒绝；
- `walk_mode=None` 保留旧调用兼容行为。

actor observation 解码确认实际送入策略的 `command_lin_vel` 和 `command_stand` 与请求一致。

### 2.3 Round 9 饱和来源

Round 9 的 `71.9424% = 4000/5560`。4000 行恰好对应 20 秒 walkthrough 在 `dt=0.005 s` 下的全部 physics samples，全部被 walkthrough `PushPathTracker(vx_max=0.08)` 截到上限。它不是 ONNX action clip、电机 effort 限制或 adapter 缩放错误。

## 3. 无障碍速度工作区间

| Upper posture | 请求 vx (m/s) | 稳态实测 vx (m/s) | 响应比 | 6 s world dx (m) |
|---|---:|---:|---:|---:|
| neutral | 0.00 | 0.0041 | n/a | -0.0016 |
| neutral | 0.08 | 0.0435 | 0.544 | 0.2179 |
| neutral | 0.15 | 0.1275 | 0.850 | 0.7014 |
| neutral | 0.25 | 0.2264 | 0.906 | 1.2742 |
| neutral | 0.35 | 0.3314 | 0.947 | 1.8840 |
| probe_hold | 0.00 | 0.0030 | n/a | -0.0047 |
| probe_hold | 0.08 | 0.0554 | 0.692 | 0.2821 |
| probe_hold | 0.15 | 0.1407 | 0.938 | 0.7422 |
| probe_hold | 0.25 | 0.2460 | 0.984 | 1.3233 |
| probe_hold | 0.35 | 0.3476 | 0.993 | 1.8976 |

结论：

- `0.08 m/s` 并非完全不走，左右脚有约 `22-24` 次接触切换，但响应比低、分布宽，不适合作为精确 approach 或 stop 工作点。
- `0.15 m/s` 已明显响应，证明推理端不存在硬 `0.2 m/s` deadband；但它低于训练采样保留边界，不能由一次平地测试推断普遍鲁棒性。
- `0.25/0.35 m/s` 的响应比高且无跌倒。本轮选择较保守的 `0.25 m/s` cruise 和 `0.30 m/s` 外层限幅。
- probe-hold 没有降低前向响应，反而略高；因此 Round 9 不能归因于“伸手姿态使 FALCON 无法前进”。
- probe-hold 的常速度开环横向漂移更大，例如 `0.25 m/s` 时 neutral/probe-hold 的 `dy` 约为 `-0.273/-0.496 m`，说明 upper posture 会影响耦合响应，必须保留世界坐标 SE(2) feedback。

## 4. SE(2) 与 Stand/Walk 切换

第一次尝试保留零速度 Walk 2 秒：

- neutral 最终平面速度 `0.11691 m/s`；
- probe-hold 最终平面速度 `0.15816 m/s`；
- 两组路径误差有界，但都未达到 `<=0.06 m/s` 持续 `0.30 s` 的停止 gate。

这次结果为 `FAIL_DIAGNOSTIC`，明确证明零速度 Walk 仍维持步态。

第二次只增加显式 Stand，不改变阈值、SE(2) 参数或 history：

| 姿态 | Stand 后达到驻留 gate | 最终速度 (m/s) | Stand 漂移 (m) | max cross-track (m) | max yaw error (rad) |
|---|---:|---:|---:|---:|---:|
| neutral | 0.525 s | 0.00348 | 0.00900 | 0.07585 | 0.09160 |
| probe_hold | 0.515 s | 0.00282 | 0.01945 | 0.09378 | 0.05022 |

两组 actor flag 均正确，policy history 未在模式切换时重置，finite/no-fall。正确停止契约是：

`WALK -> feedforward DECELERATE -> explicit STAND -> measured-speed dwell confirmation`

## 5. Oracle Map 预接触与停稳

采用相同 `0.25 m/s` feedforward、`0.30 m/s` clamp、SE(2) feedback 和显式 Stand 后：

- 无物体地图执行器到达预接触实际位置 `(-0.09275,-0.02492) m`；
- 实际速度 `0.00199 m/s`；
- Stand 漂移 `0.03051 m`；
- 一次同目标、基于实测位姿重建的 receding-horizon 修正后，最终目标误差从 `0.16591 m` 收敛到 `0.06522 m`；
- 全部 detour telemetry 保持 label `0`，未重复 probe，finite/no-fall。

首次短转弯超调被保留为 `FAIL_DIAGNOSTIC_MAP_LOCOMOTION`，没有通过放宽容差制造 PASS。

## 6. Formal 2 m E0/E1 匹配对照

### 6.1 E0：无帘子

状态：`PASS_DIAGNOSTIC`

- 本体规则：`RESPONSIVE`；
- target error：`0.07982 m`；
- base progress：`0.82927 m`；
- 全部机器人 collision-envelope 后缘越过评价平面；
- final speed：`0.02463 m/s`，停止驻留 PASS；
- command saturation ratio：`0.00754`；
- finite/no-fall。

### 6.2 E1：fixed-top formal 2 m curtain

状态：`PASS_DIAGNOSTIC_NEAR_FIELD`

- hand-curtain contact gate：`PASS`；
- deployment-only 本体规则：`RESPONSIVE`；
- probe/sweep 布料响应：`0.01142/0.01040 m`；
- target error：`0.07980 m`；
- base progress：`0.80382 m`；
- 全部机器人 collision-envelope 后缘越过评价平面；
- command/measured vx：`0.23302/0.12983 m/s`；
- final speed：`0.00285 m/s`，停止驻留 PASS；
- command saturation ratio：`0.14918`；
- max upper tracking RMS：`0.09942 rad`；
- 无 safety stop、无跌倒。

布料离线安全核验：

- max neighbor stretch：`1.05002`；
- final neighbor stretch：`1.04419`；
- max top drift：`8.01e-05 m`；
- max node displacement：`0.95502 m`；
- control-rate sampled max node speed：`1.01766 m/s`；
- 节点均 finite。

E1 相对 E0 的实际平均 `vx` 下降 `0.07637 m/s`，运动阶段完成时间增加 `2.555 s`。离线分析中的 `0.00030 m` 是只比较 `walkthrough` 运动行的 motion-only progress；`summary.json` 的 `0.82927/0.80382 m` 则把后续 `walkthrough_stop` 也合并进去，不能直接相减。两种指标都保留，定义已写入 `matched_effective_speed_analysis.json`。这说明帘子接触负载真实存在，同时速度工作区间修正和 SE(2) 闭环使系统能够继续推进。

该结论只覆盖 reset-only 近场受控穿越，不覆盖从远端 Oracle Map approach 开始的统一 curtain episode。

## 7. 实体刚板试探、撤手与绕行

状态：`PASS_DIAGNOSTIC_PHYSICAL_RIGID_EPISODE`

正式 `2.0 m x 1.2 m` 静态刚板场景中：

- 预接触位置 `(-0.05578,-0.05927) m`，速度 `0.01242 m/s`，显式 Stand PASS；
- hand progress `0.001219 m`，相对 no-contact baseline 的 ratio `0.08827`；
- stall `2.0 s`，effort deviation `27.863`，base disturbance `0.00624 m`；
- deployment-only 本体规则输出 `BLOCKED`；
- evaluation-only ContactSensor 确认 active probe/hold 的最大右手接触合力 `10.969 N`，且该字段没有进入 classifier、memory、planner 或 locomotion controller；
- 撤手、回中后速度 `0.00259 m/s`，Stand 漂移 `0.00073 m`；
- memory 为 `DEFERRED_THIS_EPISODE`，剩余 probe budget 为 `0`，不同 component id 也不能再次 probe；
- 4834 个 detour physics samples 全为 label `0`；
- 最终目标误差 `0.05898 m`，finite/no-fall。

离线几何评价中，base center 到刚板矩形的最小距离为 `0.37825 m`。两次确定性运行的 decision、probe features、precontact、final pose、goal error 和 detour path 一致。

## 8. Privileged Information Leakage Audit

状态：`PASS_WITH_DEPLOYMENT_ASSUMPTIONS`，未发现工程决策路径读取仿真特权信息。

### 8.1 控制器运行时读取字段

| 类别 | 实际字段 | 真机可获得性 |
|---|---|---|
| 关节 | `q`, `qdot` | G1 编码器/状态估计可提供 |
| 控制历史 | `q_cmd`, `previous_action` | 控制器内部状态 |
| 力矩代理 | `motor_effort` | 需要真实 G1 执行器估计标定，不是腕部六维力 |
| 手部运动 | `hand_fk_position_w`, `hand_fk_velocity_w` | 由关节状态、运动学与定位计算 |
| 基座 | `base_position_w`, `base_velocity_w` | 当前使用 simulator pose；真机定位尚未集成 |
| IMU | orientation/projected gravity、angular velocity | 真机 IMU 与状态估计可提供 |
| 导航 | 0/1/2 map、relative goal、interaction memory | 任务输入与内部状态 |
| FALCON | previous actions、IMU、commands、joint state、upper reference | 均来自上述部署字段或内部命令 |

### 8.2 仅用于评价或外部安全监控

- `object_kind`；
- simulator contact flag/contact point/contact force；
- true hand-to-obstacle gap；
- curtain runtime nodes/deformation；
- true stiffness/mass/friction/fixed constraints；
- true passability/episode outcome oracle；
- top drift、neighbor stretch、全身是否越过评价平面；
- 刚板真实身份与精确几何。

这些字段用于实验有效性核验、布料安全评价或外部终止，没有进入 probe classifier、动作分支、interaction memory 或路径规划。场景命令行选择 curtain/rigid 是实验搭建，不是运行时物体分类。

### 8.3 部署缺口

- simulator root pose 仍是第一阶段定位假设；
- `motor_effort` 阈值尚未在真实 G1 上标定；
- simulator contact/node safety monitor 不能直接移植到真机；
- 本项目阶段没有连接或测试真实机器人。

因此当前系统不能标记为真实机器人部署就绪。

## 9. Gate 汇总

| Gate | 状态 | 结论 |
|---|---|---|
| FALCON 训练/部署 contract 审计 | `PASS_AUDIT` | 找到训练期 0.2 过滤与官方 mode 语义 |
| 速度工作区间 | `PASS_DIAGNOSTIC` | 选定 0.25 cruise / 0.30 clamp |
| 零速度 Walk 作为停止 | `FAIL_DIAGNOSTIC` | 不能停止，不再使用 |
| 显式 Stand 停稳 | `PASS_DIAGNOSTIC` | 两种 upper posture 均通过 |
| SE(2) 直行/yaw 保持 | `PASS_DIAGNOSTIC` | 路径误差有界 |
| Oracle Map 预接触/停稳 | `PASS_DIAGNOSTIC` | 实际到位并停稳 |
| Formal E0 no-curtain | `PASS_DIAGNOSTIC` | 全身越过评价平面 |
| Formal E1 curtain | `PASS_DIAGNOSTIC_NEAR_FIELD` | 近场受控接触穿越通过 |
| 实体 rigid probe/retract/detour | `PASS_DIAGNOSTIC_PHYSICAL_RIGID_EPISODE` | 实体接触、本体阻挡、撤手和绕行通过 |
| Privileged leakage audit | `PASS_WITH_DEPLOYMENT_ASSUMPTIONS` | 无检测到的决策泄漏 |
| Map-integrated curtain episode | `NOT RUN` | 尚未把远端 approach 与 formal curtain 合并 |
| 真实机器人部署 | `NOT RUN` | 定位、effort 标定和安全监控未验证 |
| 双分支完整 baseline | `BLOCKED_BY_MISSING_INTEGRATION_EVIDENCE` | `ENGINEERING_BASELINE_COMPLETE = FALSE` |

## 10. 任务问题逐项回答

1. **当前 checkpoint 的训练数据是否存在小速度命令过滤？**  存在。训练命令重采样对平面命令使用 `norm(command_xy) > 0.2`，否则置零；这不是推理 deadband。当前 checkpoint 与该配置的对应关系由版本、模型路径、hash 和 observation contract 支持，但 ONNX 没有内嵌训练元数据。

2. **Walk/Stand 的指令含义和实际实现是否一致？**  Round 10 后一致。官方 `command_stand=0` 为 Stand，`1` 为 Walk。adapter 已能显式表达零速度 Walk 和 Stand，并拒绝 Stand+非零速度。

3. **Round 9 的 71.9% 饱和来自哪里？**  来自 walkthrough 外层 `PushPathTracker(vx_max=0.08)`。全部 4000 个 walkthrough physics samples 被该 clamp 截断，不是策略 action 或执行器饱和。

4. **0.08 与 0.25/0.35 的实际速度响应有何差别？**  `0.08` 的实测响应比只有 `0.544/0.692`，虽然会迈步但速度分布宽；`0.25` 为 `0.906/0.984`，`0.35` 为 `0.947/0.993`，明显更稳定。为安全余量选择 `0.25`，没有使用更激进的 `0.35`。

5. **Upper posture 是否明显影响步行响应？**  不会使前向响应失效，probe-hold 的前向响应甚至略高；但它加大开环横向漂移并改变耦合动态，所以必须使用 SE(2) feedback，不能把 neutral 结果直接外推。

6. **SE(2) feedback 是否能稳定直走与保持 yaw？**  能。neutral/probe-hold 的最大 cross-track 为 `0.07585/0.09378 m`，最大 yaw error 为 `0.09160/0.05022 rad`，finite/no-fall，显著优于常速度开环漂移。

7. **G1 能否在有效步行速度下稳定进入预接触站位并停稳？**  能。无物体地图与实体刚板 episode 均实际进入几何预接触区，速度分别为 `0.00199/0.01242 m/s`，显式 Stand 和速度驻留通过。

8. **速度区间修正以后能否改善 curtain 接触穿越？**  能。Round 9 在 `0.08` clamp 下失败；Round 10 formal E1 在真实接触负载下完成 `0.80382 m` base progress、全身越过平面并停稳，布料 attachment/stretch 有界。但结论限于近场诊断。

9. **Rigid 分支能否正确停止、撤手和绕行？**  能。实体刚板接触经 evaluation-only sensor 证实，本体规则独立输出 `BLOCKED`；机器人完成撤手、memory 锁定、label-0 detour 并以 `0.05898 m` 误差到达目标。

10. **当前最早的工程 blocker 是否已经改变？**  已改变。低速工作区间、mode 语义、预接触停稳和 FALCON 接触推进能力不再是最早 blocker。当前最早项目级 blocker 是缺少一个统一的 Oracle Map curtain episode；面向真机则更早缺少部署定位和 effort proxy 标定。

11. **是否有足够证据开始考虑 learning？**  没有。已有失败可由命令工作区间、模式契约和执行器 reference 重建等工程问题解释并修复；当前剩余的是集成和部署验证缺口，没有出现必须由 learning 才能解决的系统性失败。

## 11. 复现与证据

当前 Codex 进程中 `IsaacLab\isaaclab.bat -p` 因没有继承 `CONDA_PREFIX` 不可用；本轮未修改环境，使用已有解释器 `C:\conda_envs\isaaclab\python.exe` 直接启动。

关键命令：

```powershell
C:\conda_envs\isaaclab\python.exe scripts\test_round10_falcon_speed_workspace.py --headless --run-root reports\artifacts\round10\speed_workspace
C:\conda_envs\isaaclab\python.exe scripts\test_round10_se2_walk_stand.py --headless --run-root reports\artifacts\round10\se2_walk_stand_explicit_stand
C:\conda_envs\isaaclab\python.exe scripts\test_round7_oracle_map_falcon_navigation.py --headless --physical-rigid --run-root reports\artifacts\round10\oracle_map_physical_rigid_contact_reporter --reset-root-x -0.60 --position-tolerance 0.08 --cruise-speed 0.25 --vx-max 0.30 --stand-duration 1.5 --stop-dwell-duration 0.30 --precontact-offset 0.45 --precontact-region-x-min -0.13 --precontact-region-x-max -0.0231029034 --precontact-region-y-half-width 0.082285403 --precontact-region-speed-max 0.06
C:\conda_envs\isaaclab\python.exe scripts\analyze_round10_effective_speed_matched.py
C:\conda_envs\isaaclab\python.exe scripts\analyze_round10_physical_rigid_episode.py
C:\conda_envs\isaaclab\python.exe scripts\audit_round10_privileged_information.py
C:\conda_envs\isaaclab\python.exe -m pytest -q -p no:cacheprovider tests\test_round10_falcon_command_modes.py tests\test_round2_perception_navigation.py tests\test_round7_runtime_isolation.py
```

最终回归：`19 passed in 0.93 s`；相关模块和脚本 `py_compile` 全部通过。

主要 evidence：

- `reports/artifacts/round10/falcon_command_contract_audit.json`
- `reports/artifacts/round10/speed_workspace/summary.json`
- `reports/artifacts/round10/se2_walk_stand_explicit_stand/summary.json`
- `reports/artifacts/round10/formal_matched_effective_speed_analysis/matched_effective_speed_analysis.json`
- `reports/artifacts/round10/oracle_map_physical_rigid_analysis/physical_rigid_episode_analysis.json`
- `reports/artifacts/round10/privileged_information_leakage_audit.json`
- `reports/artifacts/round10/run_manifest.json`

## 12. 最终结论

Round 10 证明：FALCON 在正确的命令工作区间、显式 Stand/Walk 契约和世界坐标 SE(2) feedback 下，能够完成 formal 2 m 帘子的近场受控接触穿越；同一工程栈也能在实体刚板上完成保守本体判断、撤手、一次试探记忆和 label-0 绕行。

这消除了 Round 9 的最早控制 blocker，但没有完成一个地图集成 curtain episode，也没有验证真实 G1 部署输入。因此：

`ENGINEERING_BASELINE_COMPLETE = FALSE`

`LEARNING_REQUIRED = FALSE`
