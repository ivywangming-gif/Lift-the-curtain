# V5 首次启动准入修复

2026-11-06 15:06:45 UTC，EMPTY-left 的 attempt01 在 App 启动前因 `Runtime plan differs from frozen campaign` 退出。冻结计划中的 `physics_valid=False` 表示尚待准入，队列在三个真实门禁通过后置为 True；原严格逐字段比较误把证据状态晋升当作实验参数变化。没有机器人轨迹、模型历史或完成结果，独立碰撞复核为 NOT_VERIFIED，不能给通行标签。这一次实际启动保留并计入 1000 次总预算。

修复仅允许有独立门禁证据的 False→True 晋升；场景、动作、机器人条件、seed 和其余冻结字段仍必须一致。wrapper 和 importer 同步修正。34 项 CPU 合同通过，实际失败 runtime plan 通过三 seed 证据复核；缺门禁和修改 robot/seed/action 的反例被拒绝。匹配对证据同时改为引用各 episode 真实动作与标签合同，拒绝 V1/V5 混配。

修改前的两份源码由精确逆补丁重建，其 SHA 与冻结 V5 source_sha256 完全一致；重建方法如实记录。新旧源码、diff、失败日志 SHA、实际 runtime plan 和修复证据位于 `outputs/round4070_comprehensive_traversability/protocol_repairs/v5_runtime_gate_flag_20261106/`。失败 raw、manifest、动作 sidecar 均未改动；冻结 E 控制补丁 SHA 仍为 `5bbb678b007cc7f53600394a9e761c8c46565ab1645e0a979aecc527a59ffe87`。

另在队列启动 guard 前归档全部关键源码的原始 bytes，并由 campaign_launch 绑定 archive path、SHA 与 size。两项归档合同验证 CRLF、去重及后续源码修改隔离。失败发生在 App 之前也能保留当次实现。

2026-11-06 15:26 UTC，独立目录中的实际 wrapper dry-run 返回 0，耗时 1.513 秒，stderr 为空；该检查没有启动仿真、不计 G1。15:28:06 UTC，以同一冻结试次参数授权唯一 attempt02，原失败目录中的 recapture_authorization.json 绑定修复证据及新源码。此时累计实际 G1 启动 12 次（10 个已废弃旧协议 pilot、V5 的失败 attempt01 和正在验证的 attempt02）；976 个主计划全执行后仍有 13 次额外启动余量。重采不以标签好坏为依据。

本记录只证明实现修复与受控重试依据，真实运行和独立碰撞审计的最终状态以 PROGRESS_CHECKPOINT.json、TRIAL_PROGRESS_LEDGER.csv 和原始审计为准。正式 G 学习性能仍无证据。

15:31 UTC，attempt02 实际 rollout 完成并通过独立审计：6,068 个真实 200 Hz 状态加初始状态，13 个分块，所有源 SHA 核对，动作边界无违例。最终 hold 的全身最小 x 保守下界为 1.15323471 m，大于 0.95 m 阈值，安全恢复有效。合法模型观察 100×252、50 Hz、1.98 s；空场控制不计核心柔性正例。审计 SHA 为 `40d6c194d3c1567c2282ca38c46927eef45668f2fca66ce4f472d068b7e4246a`。完成结果落盘后 Kit 清理超过 30 s，由原 guard 仅终止当次 PID；此清理状态与物理 rollout 分开记录。
