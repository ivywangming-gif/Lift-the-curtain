# Round 4070-G 执行与恢复

工作目录为 `C:\Robotics\Lift-the-curtain`，Python 为现有 `C:\conda_envs\isaaclab\python.exe`。不升级 Isaac、PyTorch 或驱动。当前任务使用单个 headless Isaac 环境与冻结 FALCON。所有新原始数据仅写入 `outputs/round4070_comprehensive_traversability/`。

从冻结队列继续完整流程：

```powershell
Set-Location -LiteralPath 'C:\Robotics\Lift-the-curtain'
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_round4070_g.ps1 -Mode all -RetryRuntimeFailures
```

`all` 每 12 个实际启动的 physics/G1 作业返回一次，自动导入真实记录并生成覆盖及完整性检查点，然后继续剩余任务。physics gate 不计入 1000 条 G1 总启动预算；失败 G1 与重试计入。先覆盖各独立配置的双手，再扩充重复、确认和随机化。一个场景未通过门禁不会取消其他场景。相同全局实现异常连续出现时停止并留下修复原因。

当当前可执行采集队列耗尽，主入口将实际数据元数据复制到不可变 `dataset_snapshots/<SHA256>/`，冻结全部学习任务，在确认没有仿真进程及采集锁后创建有期限的训练窗口。仅运行满足标签、划分及方法前提的任务；其余保留 `BLOCKED_WITH_REASON`。训练中每个 epoch 和每次拟合均有恢复状态。Isaac 与正式训练共用排他资源锁。数据、代码或协议改变会生成新的学习 run，不改写此前冻结结果。

即使程序返回 0，也不代表 `FULL_BENCHMARK_COMPLETE`。最终必须检查物理有效性、左右手核心标签、独立划分、实际训练/评价任务、确认集及材料可信度。没有可评估标签的空表不是实验结果。

其他可执行入口：

```powershell
# 只更新当前场景状态；不启动仿真。
.\run_round4070_g.ps1 -Mode status

# 导入目前真实数据，冻结不可变学习快照并生成审计；不训练。
.\run_round4070_g.ps1 -Mode analyze

# 单次只启动最多 12 个实际作业，之后审计并对当前可评估划分训练。
.\run_round4070_g.ps1 -Mode all -MaxJobs 12 -RetryRuntimeFailures

# 单场景三 seed 物理门禁；不启动 G1。
.\run_round4070_g.ps1 -Mode physics -Scene A01 -RetryRuntimeFailures

# 只运行已通过门禁的 G1 剩余队列。
.\run_round4070_g.ps1 -Mode g1 -RetryRuntimeFailures

# 数据已就绪且所有采集进程已停止时，恢复对应冻结学习 run。
.\run_round4070_g.ps1 -Mode train
```

`physics`/`g1` 是低层调试入口；退出后可运行 `-Mode analyze`。正式完整运行应使用 `-Mode all`。`-MaxFits N` 可限制本次学习拟合数量；不改变总冻结任务。正常执行无需修改任何 JSON。首次检出自动从研究目录 `CURRENT_CAMPAIGN.json` 指向的 manifest v5 与 SHA256 恢复配置、原生证据和 976 个 `_rng1_act2` 主计划，不重新采样。旧 v3 pilot 原始证据保留且计入 1000 总预算；它们不进入主数据。V5 冻结时记录 10 次旧启动，完整主计划之外预留 14 次额外尝试；这不是后续恢复时重新获得的重试额度。当前硬上限按冻结历史、研究目录持久账本 `G1_LAUNCH_LEDGER.json` 和全部本地 `campaign_launch.json` 路径的并集计算，同一路径必须保持原 SHA256。修订原因及准入条件见 `PROTOCOL_AMENDMENT_V5.md`，历史 V4 文件仍保留。

采集器在持有仿真资源锁时再次检查预算，先保存实际启动 JSON 和源码，再将该尝试追加到账本并落盘，最后才启动 guard/App。账本追加失败时不得启动 guard。App 前失败、guard 启动失败，以及已经持久记录但随后崩溃的启动意图都保守消耗一次预算；重复读取同一路径不重复计数。已有账本条目只能追加，不能删除、改写 SHA 或按成功标签挑选。当前账本必须随正常 Git 提交/交接保存；一个未收到最新账本的旧 checkout 无法知道其他工作区后来发生的启动，不能作为重新运行的授权。

新 checkout 即使没有原始轨迹，账本已知尝试仍消耗预算。若已登记的 `campaign_launch.json` 缺失，该完整 trial plan 返回 `KNOWN_ATTEMPT_RAW_MISSING_RESTORE_OR_REVIEW_REQUIRED` 并被暂缓；不能用同一路径或下一 attempt 编号重跑来替代恢复。若启动 JSON 还在而轨迹、outcome 或独立审计缺失，科学准入仍不成立，重采仍需既有明确修复授权。不得删除或重建账本来恢复额度。恢复器会原字节恢复研究目录中的 quarantine record/sha；已有不同字节直接拒绝。恢复元数据不生成历史 raw，也不把缺失 raw 视为可重试。

单计划 quarantine 是排程隔离，不是新标签或物理有效性豁免。`g1_quarantine.quarantine_state` 在旧记录自动重审之前读取与 SHA 绑定的授权记录：`VALIDATED_WITHHOLD` 只跳过记录中的精确 trial plan，`NOT_QUARANTINED` 仍须走完整普通门禁，`GLOBAL_PAUSE_REQUIRED` 则暂停等待重新分类。当前 D01-left 与 D01-right 各自有独立授权，均保留冻结 V5 的包络/墙重叠 `NOT_VERIFIED`；不是从左手结论推断右手结果，也不是隔离所有墙场景。其他 seed、动作或场景不会因这一记录自动获得隔离或准入。原 audit、RNG、身份、raw/source SHA 或 attempt 集合变化均不能自动解除隔离；需要新的明确审查。诊断性的 1/5 mm 支持计算不能越过冻结 V5 的包络重叠规则。未知实现错误、哈希/身份/RNG/完整历史问题仍须全局暂停，不能归入这种局部隔离。

隔离记录位于 `research/round4070_comprehensive_traversability/quarantine/` 的 Git 恢复副本及 `outputs/round4070_comprehensive_traversability/quarantine/` 的执行副本；正文和 `.sha256` 必须原字节一致。quarantine 不改变计划总数、角色、预算或标签，不自动重试，不进入监督或响应训练。若没有剩余可执行任务但仍有隔离、缺失 raw 或未验证计划，状态仍是有原因的未完成，不能把队列暂时排尽解释为完整 benchmark 已完成。

恢复前检查 `workflow_lock.json`、`campaign_lock.json`、`resource_lock.json`。活跃 PID 存在时不要启动第二份同类队列。若崩溃留下锁，必须确认锁中 PID 已退出、相应仿真及其 guard 已退出后，将该锁归档；不得直接按文件年龄删除，不得结束其他任务进程。`PAUSE.flag` 在作业边界暂停全部采集；`PAUSE_G1.flag` 只暂停 G1 准入并继续 physics gate，且禁止正式训练。不得为绕过审计缺口直接删除 G1 准入标志。训练窗口过期自动停止启动新的训练工作；它不是后台自动续期承诺。

主要证据：

- `campaign_events.jsonl`、`workflow_events.jsonl`：实际启动、返回、暂停及错误。
- 研究目录 `G1_LAUNCH_LEDGER.json`：持久的已耗 G1 尝试、原始启动 SHA 和总量完整性；缺 raw 不退款。
- `quarantine/<exact_trial_id>.json` 与 `.sha256`：明确授权的单计划排程隔离及其原审计/原始证据绑定，不是正式标签。
- `physics/<scene>/seed*_attempt*/`：各次原始门禁、几何轨迹、24 秒门禁证据、监控及 summary。
- `trials/<trial>/attempt*/`：唯一实际 G1 尝试、源码归档、50 Hz 合法观察、200 Hz 逐步几何分块及其索引、环境审计、heartbeat/events/telemetry、独立 outcome。
- `dataset/`：持续更新的实际导出；既有 NPZ 复读比较后复用，不重采样。
- `dataset_snapshots/`、`learning_campaign_*/`：冻结输入、划分、源码哈希、任务状态、模型 checkpoint、校准与评价。
- 研究目录 `PROGRESS_CHECKPOINT.json`、`TRIAL_PROGRESS_LEDGER.csv`、`SCENE_COVERAGE_MATRIX.csv`：可读阶段状态。原始数据和模型不纳入 Git。

本体输入仅截取真实 probe/probe_hold 的至多 2 秒记录，严格早于后续行动。UNKNOWN 可按训练组规则参与响应学习，不能作为第三监督类别。旧 F 标签没有自动转成 G 的固定动作标签；空场控制 PASSABLE 不计核心正例。第二层障碍必须新视觉/深度观测重新建立候选，地图接口测试不是完整导航实验证明。

资源实测补记（2026-11-06）：10 次旧 G1 pilot 的 guard 总 wall time 为 97.55–312.88 s，均值 197.10 s，包含当时 30 s 的退出清理等待。直接将这个旧均值乘以 976 约为 53.4 小时；这只是历史吞吐量的量级换算，既不是新版采集完成时间承诺，也不包含全部物理门禁、200 Hz 审计开销、失败修复或训练。新结构、更长动作和数据记录方式可能改变成本。不得因此把未执行队列当成完成结果。

输出盘在本次检查约有 251 GiB 可用。新版资源门禁在可用空间低于 20 GiB 时等待并记录原因，不自动删除任何原始试验。200 Hz 几何记录已在 V5 左手空场实际试次中验证 6,068 步、13 块及初始状态的完整性，模型合法历史为 100×252 的真实 50 Hz 样本；这不代替布帘/墙体或完整 benchmark 验证。任务可以在明确的作业边界恢复，聊天结束本身不代表代理会继续审查运行结果。

V5 实际布帘开销更新（2026-11-06 16:15 UTC）：C01 左手 guard wall time 242.095 s，随后独立 CPU 全步审计约 262 s；9,110 个状态的压缩 200 Hz history 为 164,630,838 bytes。右手对应实际数据如下，均属单个配置的初测，不能据此保证全队列时间或空间。审计优化必须保持全部真实样本、原公差、计数及证据一致，不能靠降采样减轻开销。


| 手侧 | 已审计实际步数 | 含初态状态数 | 最终全身 x 下界 (m) | 压缩 history (MiB) | guard wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| left | 9110 | 9111 | 1.051496 | 157.00 | 242.095 |
| right | 6661 | 6662 | 1.193477 | 112.51 | 182.496 |
