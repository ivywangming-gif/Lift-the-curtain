# Round 4070-G Git交付记录

启动HEAD=`1d5136d85066c922b29e4901b18edd7fc9e977f7`，实际fetch后与远端F研究分支一致。G独立分支为`codex/round4070-g-comprehensive-traversability`，远端`https://github.com/ivywangming-gif/Lift-the-curtain.git`。

本轮只提交` scripts/round4070_g/ `和` research/round4070_comprehensive_traversability/ `内有范围源码、配置、测试、小型结果与报告；如入口位于仓库根目录，将单独列出并显式暂存。原始trial、checkpoint与大型数组保留独立outputs。禁止force push和无筛选git add。

已有dirty worktree及历史证据保全见`ENVIRONMENT_AUDIT.md`与`PRESERVATION_AUDIT.json`。2026-11-06 15:18 UTC 再次实际核验 2012 个历史文件、729999234 bytes 全部一致，`online_occupancy.py` 的原有 104 行修改及 SHA256 保持不变。

已实际完成的 G 提交：`c929722fa93c74cf1159b27dd86a99f19d94d3ad`，题目 `research: audit Round 4070-G provenance literature and map contract`。该提交已成功 push 到独立 G 分支并建立 upstream，仅含 7 个审计、文献和地图合同文件。

第二阶段已实际提交并成功 push：`e9bdca5eb360a4eccede86318c6d58c7d87430c0`，题目 `research: implement resumable G physics data and learning campaign`。包含可执行场景/门禁/采集/学习/恢复源码、冻结 manifest v3、实际双手空场 smoke 及明确未完成的研究报告。没有正式 G 模型拟合结果；1280 个学习任务的阻塞状态是实际 preflight 结果，不是性能分数。

检出完整性审查发现 Git `core.autocrlf=true` 会把 manifest 的提交 blob 与冻结 Windows 文件换行区分；已增加该目录的精确范围 `.gitattributes`，对冻结 manifest 与 SHA 文件禁用文本换行转换，以后续提交保留原始冻结字节。冻结物理内容、seed、动作合同及运行中的 v3 SHA 不改变。

后续已实际提交并成功 push：

- `82991b61d538a44ccd804003fc6db5e5dae2a97e`：冻结低标签预算与对应评价协议；同 25%/50% 标签子集用于三个预注册方法和五训练 seed。
- `9904c75d74f5d3c7327fbe9d162a842f8fdd5e15`：保留 manifest 原始字节、恢复快照测试和实际 pilot 历史图。暂存 blob SHA 已与冻结文件精确核对；其 CRLF 按文件属性被保留，不是修改预注册物理参数。

上述两提交已随 `git push` 从远端 `e9bdca5` 推进到 `9904c75`，没有 force push。之后的 V4 随机初态/实际碰撞体准入修订、confirmation 隔离与原 pilot 排除仍处于实现检查阶段；实际科研完成状态始终为 `IN_PROGRESS_NOT_COMPLETE`。V4 已冻结并实际通过 fresh restore/幂等恢复，尚无正式 G1 主样本或模型拟合，不能把 staged 源码称为完成实验。


V5 阶段交付准备（2026-11-06 16:05 UTC，尚未提交本批更改）：冻结并激活 V5 manifest/action/native evidence；真实逐 seed reset、200 Hz 全步碰撞、静态墙读回与分离证书、50 Hz 合法输入、旧 pilot 隔离、confirmation 及独立单类 OOD 状态均已实现。双手 EMPTY 共两条真实控制已通过独立审计，C01 左右手首轮均已通过独立审计并导入（核心 P2、控制 P2）；无正式 G 模型拟合结果。全部源码/小型证据待本阶段实际验证后显式范围提交。大型 raw/chunks/checkpoints 留在 outputs。当前源码及文件状态不能替代已执行的 benchmark。
