# RTX 5090 迁移或止损计划

## 当前资源裁决

`RESOURCE_VERDICT = 4070_SUFFICIENT_FOR_CURRENT_RESEARCH`。

本轮没有证据要求立即迁移：24/24 headless trial 有效，GPU peak 仅约 2.6--3.4 GB，失败没有显示显存耗尽。4070 的实际瓶颈是单条 rollout 的低于实时吞吐和有限并行能力，不是当前科学结论无法取得。

## 若继续使用 5090

5090 的合理用途是：

- 增加同一机制内的独立 seed 和正/负动作配对；
- 扩大 held-out condition；
- 在不改物理参数的前提下提高串行/小批量吞吐；
- 在得到新的传感器/renderer 运行许可后独立验证观测链路。

不应做的事：用 5090 直接改变 FALCON、物理参数、PointGoal gate 或把渲染真值接入部署决策。

## 可迁移包

当前 `outputs/g1_final4070/` 已保留：

- 原始 24 条 trial 的 summary/events/telemetry/deployment observation；
- `run_manifest.json`、`physical_matrix.json`；
- `provenance.json`、Git HEAD、代码/配置 hash；
- local rubber-hand USD 的 hash；
- 外部 FALCON ONNX 的只读路径和 hash，未复制第三方资产；
- 可恢复 runner：`scripts/g1_final4070/run_queue.ps1`、`run_physical_trial.ps1`、`run_guarded_with_resources.ps1`。

5090 需要重新验证：Isaac/driver 组合、环境安装、单条 smoke run 和资源 profile。无需重做历史 headless 结论；只有当物理时间步、solver、资产或动作契约改变时，才需重新建立 matched evidence。

## 止损条件

若后续在满足 deployment-only、同一机制正负平衡和 held-out 条件后，简单风险规则已经达到同等安全通行与成本，或模型无法产生真实动作差异，则应停止当前 RA-L 主线。更大算力不会凭空创造缺失的观测信息或算法创新。

