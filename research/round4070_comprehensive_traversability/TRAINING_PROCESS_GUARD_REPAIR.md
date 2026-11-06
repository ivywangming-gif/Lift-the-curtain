# Training process guard repair

记录时间：2026-11-07 HKT。状态：**进程误报已修复并经过实际训练验证；新训练仍在运行，研究未完成。** 本次修复涉及资源调度识别，不改变 G1 物理、标签、模型结构或训练超参数。

旧规则对每个进程的全部命令行参数取 basename，只要出现 `g1_trial.py` 等名称就判为仿真。后台 Git 扫描使用 `:(literal)scripts/round4070_g/g1_trial.py` 等文件参数，因此触发误报。前两次训练尝试被拒绝，均未产生 fit；第三次在旧 run `85a8eff5a17f1f36e678` 完成 11 项评价后暂停。11 项具体为五个确定性工程基线、五个 Logistic seed 和一个 GradientBoosting seed，不能称为 11 个神经网络训练。

第三次暂停的直接证据是一致的 PID：旧队列记录 `SIMULATION_PROCESS_ACTIVE:29468`；100 ms 主机监视日志在 `2026-11-06 17:58:17 UTC` 实际记录 PID 29468 为 `git.exe`，其命中参数是 Git literal pathspec。[原始进程监视日志](C:/Robotics/Lift-the-curtain/outputs/round4070_comprehensive_traversability/handovers/v5_activation_20261106/training_guard_process_matches.jsonl)，[旧队列归档](C:/Robotics/Lift-the-curtain/outputs/round4070_comprehensive_traversability/handovers/v5_activation_20261106/training_guard_fix_before_20261106T180141Z/task_queue.json)

修复前已归档旧协议对应的全部九个学习源码、workflow 源码、identity test、冻结 protocol、task queue 和误报日志。九个源码 SHA 与旧协议逐项一致；修复后再次核验旧 protocol、queue 和日志原文件 SHA 未变，11 项结果原样保留。旧 protocol SHA-256 为 `1a1ee37da4c3284f02d0a1381821744b7dcba0457842900c51202133c34902d4`。[归档清单与全部 SHA](C:/Robotics/Lift-the-curtain/outputs/round4070_comprehensive_traversability/handovers/v5_activation_20261106/training_guard_fix_before_20261106T180141Z/archive_manifest.json)，[修复后不变性核验](C:/Robotics/Lift-the-curtain/outputs/round4070_comprehensive_traversability/handovers/v5_activation_20261106/training_guard_fix_before_20261106T180141Z/verification_after_fix.json)

新增公共 [simulation_process.py](C:/Robotics/Lift-the-curtain/scripts/round4070_g/simulation_process.py)，由 trainer 的 `window_status` 与 workflow 的 `active_simulations` 一致调用。它先识别实际 executable，再解析 Python 的真正 script/module 入口；支持 `-u`、`-X utf8`、`-W`、组合选项和 `--`。Git、rg、`python -m py_compile`、读取脚本的后续文件参数不再被当作运行中的仿真。真实 Python 仿真入口及 native Kit/Isaac 进程仍被识别。资源锁、明确允许且未过期的训练窗口、`isaac_stopped_confirmed` 和边界检查保持不变。此 matcher 不根据任意 `-c`/stdin/shell 文本猜测动态程序，不能替代资源锁。

新模块纳入 trainer 和 workflow 共同校验的十文件代码注册表，完整 SHA 存在上述核验 JSON 和新冻结协议中。新模块 SHA-256 为 `a149af84fe485b08276e745a6c51d70074102a8694c16dd8dc12f9daa7747370`；新 trainer 为 `40f640e33af1dac17026bc9afdd9c768cc1095dc03616eb67da39fb53902fa6a`。新代码不能静默 resume 旧九文件协议，也没有将旧 11 项复制成新 run 的结果。

针对真实 Git 参数、Python 启动形式、两个调用方一致性、窗口拒绝条件及不可变协议身份的六项 CPU 测试通过（1.183 s）。随后完整 CPU 合同套件 **157 项通过，28.637 s**，`CUDA_VISIBLE_DEVICES=-1`；这些是实现测试，不是科研 trial 或模型 fit。[完整套件记录](C:/Robotics/Lift-the-curtain/research/round4070_comprehensive_traversability/PROCESS_GUARD_CHECKPOINT_TESTS.json)

新 run `3e48cd09d5df3559fdb6` 于 `2026-11-06 18:09:51 UTC` 实际启动，已越过此前触发误报的后台 Git 扫描并继续训练。新 [frozen_protocol.json](C:/Robotics/Lift-the-curtain/outputs/round4070_comprehensive_traversability/learning_campaign_3e48cd09d5df3559fdb6/frozen_protocol.json) SHA-256 为 `f9d1757d028c114eb4e149c71adcc57094cb397ea0b5ad723698fa04fe11328c`，十个源文件 SHA 均已匹配。运行期间另有 checkpoint I/O 失败正在独立排查，不归因于进程 guard；已执行、失败、待执行任务须以实际队列分别报告。[启动记录](C:/Robotics/Lift-the-curtain/outputs/round4070_comprehensive_traversability/handovers/v5_activation_20261106/learning_after_guard_fix_launch.json)

新协议使用独立冻结的 **8 条真实 admitted episode：核心 P=2、B=4、EMPTY controls=2**；六条核心数据仅来自两个物理配置。旧 run 使用六条总 episode，因此两版 run 同时具有数据快照差异，不能合并指标或把两版性能差归因于 guard。当前执行仅是小样本、具有支持的任务训练，不代表全部场景、五 seed 全方法、独立确认或 OOD benchmark 已完成，不能据此宣称方法优势或完整研究完成。
