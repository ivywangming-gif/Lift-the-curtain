# 限制因素诊断

## 总结

本轮没有把任何一个问题简单归因于 RTX 4070。24/24 条 headless G1 运行到达 `RUNTIME_PASS_FINALIZE_REACHED`，没有 OOM、CUDA allocation failure、device lost 或热降频证据。当前最明确的限制是：Frozen FALCON 和现有高层动作只支持条件化、局部的全身通行；而不是稳定的通用通行能力。

## H1：GPU / CPU / Memory

结论：`NOT_CONFIRMED`。

- GPU：RTX 4070 Laptop，VRAM 8188 MB，driver 610.60。
- 24 条运行的 GPU peak：2588--3441 MB。
- 进程 private peak：7241.8--8142.3 MB。
- 系统 RAM 可用最小值：4800.8--5541.6 MB。
- physics throughput：43.05--67.74 steps/s。
- simulated/wall：0.215--0.339。
- 没有资源错误导致的 invalid trial。

4070 足以完成当前小规模串行 headless 研究；它限制的是吞吐和并行规模，不是本轮物理结论的有效性。

## H2：Windows / Isaac / Sensor

结论：`CONFIRMED_FOR_INSTALLATION_CONFIG_AND_RENDERER_BRANCH_ONLY`。

所有运行的 stderr 都有 Isaac Kit 无法保存安装目录 `user.config.json` 的警告。该警告没有阻止 scene、physics、telemetry 或 FINALIZE。历史 RGB-D/renderer native crash 仍是 Windows/renderer 分支限制，本轮没有重新触发已知 crash path，也没有把它写成 G1 科研失败。

因此：

- headless physics：本轮可用；
- renderer/RGB-D：仍未作为本轮正式部署输入；
- 更换 5090 不能自动解决 sensor API 或观测契约问题。

## H3：Execution / FALCON capability

结论：`PARTIAL_CAPABILITY_LIMIT_CONFIRMED`。

证据：

- 非空 R1：12 条，其中 4 条完成 candidate clearance；
- nominal seed45 R1：`RESPONSIVE`，未清空候选区域，最终 XY error 0.2140/0.2653 m；
- nominal seed46 R1：完成 clearance，PointGoal error 0.0703 m；
- bottom-right：R1 在独立 seed 和左/右 probe 下出现 clearance；
- rigid：5/5 `BLOCKED`、安全撤回并绕行；
- 所有运行无跌倒，说明不是普遍站立崩溃。

这证明当前动作接口有条件地能完成目标行为，但不能证明所有视觉相似帘幕都可通过，也不能把某次 R1 失败直接当作“物理绝对不可通行”。

## H4：Observability

结论：`PARTIAL_OBSERVABILITY_LIMIT`。

T0/T1 的 deployment-only 特征在 clearance 与 non-clearance 样本间高度重叠；固定 probe 后仅凭 q/qdot/q_cmd、effort proxy、FK 和 base 反馈，不能作出已有证据支持的通用预判。T2 在执行一段合法 `CAUTIOUS_ADVANCE` 后，base/hand progress 和 tracking 出现差异，但正样本只有 4 条，且集中在 bottom-right 加一个 nominal seed，仍被物理条件混杂。

因此本轮只证明“后续短时反馈可能增加信息”，没有证明已有本体历史足以支持泛化学习器。

## H5：Scientific / Algorithmic

结论：`CONDITIONAL_RESEARCH_SPACE`。

P1 有证据：局部 `RESPONSIVE` 不等于 whole-body clearance。

P2 有部分证据：在匹配初始条件下，R1 请求继续、R2 请求撤回，会改变是否利用候选近路；但两者不是两个独立物理技能，R2 本质是更保守的高层门控。

P3 尚未成立：没有足够的 held-out 机制和正负支持，不能声称学习模型已找到可泛化的隐藏物理信息。

## 最早有效失败

不是启动失败，也不是 PointGoal yaw。最早的科学失败 gate 是：`RESPONSIVE` 之后，机器人未必完成 candidate clearance。对 nominal seed45，probe 已有局部响应，但 R1 的持续推进没有在有限预算内清空身体包络；随后 PointGoal 也失败。这个失败必须与刚体安全拒绝和后续目标误差分开。

