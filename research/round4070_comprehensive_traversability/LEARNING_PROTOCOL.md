# G 学习执行协议与恢复入口

当前代码实现 31 个学习方法/消融及 5 个工程/退化基线，默认学习 seed 为 40700–40704。八个具体划分的全标签预算共 1280 个计划任务（每划分 31×5 个学习 fit 和 5 个确定性规则）；另加三方法×两种低标签预算×五 seed×八划分的 240 个任务，共 1520 个。实现完成与真实训练完成分开记录。`LEARNING_RUN_STATUS.csv` 的每个任务状态及理由来自冻结队列；没有实际 fit 的结果表只有列名，不填造分数。

## 已实施的方法

- M0：always PASSABLE/BLOCKED、E0、真实时间修正 E1/E2、C=1 凸 Logistic、80-stage Bernoulli 决策桩 boosting、统计量 MLP、8维固定随机投影的类条件 Gaussian AR(1)。工程参考只来自对应 split 的同手训练空场，缺参考的行为单列审计；不从测试控制拟合。
- M1：History MLP、GRU、LSTM、TCN、单层轻量 Transformer、CoPRE-style 残差 GRU、旧 M2 FiLM。Transformer 不是 PROBE 完整复现，正常响应 Ridge 也不是 CoPRE 完整接触估计器。
- M2：普通 SupCon、离线物理匹配 cosine contrastive、去困难负对照、去跨配置正对照。三个匹配变体要求同一训练数据确有两类有证据配对，否则明确受阻。模型输入不含配对元数据。
- M3：命令进程、时间间隔加进程、按结构参数数匹配的 GRU、固定 seed 乱序、去累计信息的末观测、固定 .5/1s 前缀、orderless、真实 10/20Hz 降频、仅采样点数/观测时长的 DurationOnlyLogistic。
- M4：GRU 联合下一观测 q/qdot 辅助预测；先响应预训练再监督微调。只用当前训练分组的 UNKNOWN，不使用校准/测试历史。
- M5：身体通道消融、按真实关节名称重排 active/opposite 手且保留侧别，配合 split 的双向跨手/姿态测试。该重排不冒称经过认证的带符号运动学镜像，不镜像标签。
- M6：原始 forced=.5、固定拒判=.2/.8、独立校准温度、校准集选择的风险/覆盖阈值，以及有限样本 label-conditional conformal 风格集合。conformal 分数使用冻结模型的 raw 概率，不能使用已由同一 calibration 标签拟合温度后的分数。不支持双类校准时，相关方法状态为不可用，不能反用测试集。
- 观测随机化：GRUObsDR 与普通 GRU 同结构、同 clean 训练归一化，只扰动训练观察；全部学习模型另执行固定 seed 的 clean 校准→noisy test 敏感性诊断。它不改变标签、split 或独立 episode 数。

神经模型固定 hidden=64、80 epoch、AdamW lr=.001、weight decay=.01、batch32；SupCon/真实配对使用训练标签全批以保留真实配对。分阶段预训练额外40 epoch，须披露其额外预算。梯度裁剪5；辅助与对比损失权重均.1。选择固定最后 epoch，不看测试挑 checkpoint。凸 Logistic 为 `mean BCE + ||w||²/(2*n_train)`，bias不正则，float64 LBFGS最多200次，记录真实梯度而不预设收敛。确定性的 boosting/AR重复拟合五次也不作为五份独立随机证据。

## 标签效率协议

GRU、GRUResponseAux 与 GRUPretrainFinetune 预先固定 25%、50%、100% 训练标签预算。每个 split 内按原训练标签分层，使用固定 SHA256 排序后逐类 `floor(fraction*n_class)` 取嵌套子集；三个方法及全部五个模型 seed 共用完全相同的标签 IDs。计数包括训练 controls，另记录核心 P/B 数；预算后核心双类不足也明确阻塞，不能仅靠空场正例支撑标签效率结论。预算比例、hash 盐及选择规则不读取校准/测试或模型预测。

未选中的已知训练标签只在当前 fit 的视图中改为 UNKNOWN；原始 manifest、全部训练历史、训练归一化来源、校准集和测试集都保持不变。辅助方法可使用这些真实训练历史进行响应学习，纯监督 GRU 不获得被隐藏的标签。`frozen_protocol.json` 保存原训练标签审计、选中/隐藏 IDs 和实际数量，每个 fit 另写 `label_budget_audit.json`。100% 复用原有任务而不重复训练；25/50% 使用独立 display method 和 `label_fraction`，不会与全预算结果混合求均值。`LABEL_BUDGET_FROZEN_SUBSETS.csv` 与 `LABEL_EFFICIENCY_RESULTS.csv` 分别记录冻结子集和真实完成的模型结果；统计配对仅在相同标签预算内比较。

## 输入与因果边界

正式监督只接受 `G_HOLD_PROBE_POSE_FORWARD24_V1`；旧 F 保持原有标签契约，不偷偷混入 G 分类。旧数据若用作单独诊断，必须 `--legacy`，其范围标为 `LEGACY_F_CONTRACT_DIAGNOSTIC_NOT_G_BENCHMARK`。当前管线不会自动把旧 F 预训练到 G，也不会凭映射补造缺失传感器。

每条最多2s、最多128个真实采样点。低频数据不插值；超过128点时按固定物理时间网格取 past 索引并去重。所有基线用同样截止；前缀是单独消融。10/20Hz消融复用同episode和截止、只选已有观测。安全提前停止保留真实长度，不补齐反馈；padding/长度可能成为合法但依赖安全策略的捷径，因此另有点数/时长诊断。

普通 GRU 与 shuffled 的通道相同，都移除显式 elapsed/dt/command_progress；q_cmd 仍可能携带动作阶段，所以 shuffled 不能单独证明顺序无信息。严格末观测进一步移除手部累计进展、stall 和累计命令信息；它依然接收该时刻合法记录的当前/上一控制动作，不冒称只有无记忆传感器读数。

归一化仅训练组真实有效行（可含 UNKNOWN），不对padding或测试历史拟合。残差正常模型只训练空场，0.4s状态排除与暖机零残差；残差分支仍间接使用全身状态。损失中的下一观测仅来自训练历史，不在推理时读取未来反馈。

## 观测随机化合同

`observation_randomization.py` 实施 q 噪声 .001rad、qdot .01rad/s、effort 训练尺度的 2% 逐点噪声及 1% 每条偏置、IMU 角速度 .005rad/s 与姿态 .001rad、时间戳 ±.002s 抖动、1% 丢样。effort 尺度为每通道训练组 std、训练 mean absolute 与 .001N·m 的最大值；不使用校准/测试幅度。范围是尚未经硬件标定的仿真敏感性假设，不能视作真实传感器误差标定。

在实际 50Hz 离散历史上，每条等概率选择 0 或 20ms 延迟端点，以已有 past 样本模拟；不插值实现虚构的更细延迟。源观测必须早于收到时刻，实际年龄不超过 20ms，收到时间戳保持严格递增。稀疏记录无法满足延迟时明确不可用，不造观察。机器人已知的命令、动作及动作进程使用收到时刻的记录，传感器才延迟。FK 作为独立状态代理被延迟，未因编码器噪声重新计算，因此不是完整耦合硬件模型。

所有模型使用同一 heldout 噪声 seed=40706000；GRUObsDR 的训练噪声按模型 seed 加固定偏移确定，逐 episode seed 由 trial ID 稳定派生。保存训练尺度来源、原始/源/收到索引、真实与报告时间、每条 effort 偏置和 seed。前缀先按真实收到时间裁剪，再应用报告时间截止，禁止时间抖动把未来反馈移入前缀。noisy test 只替换冻结模型的测试预测，calibration 概率与 clean 评价完全相同；无 noisy test 调参。`OBSERVATION_RANDOMIZATION_RESULTS.csv` 和 `OBSERVATION_DR_TRAINING_RESULTS.csv` 仅由真实完成的 fit 生成，未训练时只有表头。

## 冻结、运行、恢复

先由数据导出器保存一个独立数据快照；`manifest.json`、`split_manifest.json`、`matched_pairs.json` 必须共同固定。`freeze` 保存输入/代码hash、所有split ID、超参、五个seed以及逐方法数据缺口。不能在训练后把同一路径的manifest换成新采集数据。新数据批次应另建snapshot与run目录。

```powershell
& C:/conda_envs/isaaclab/python.exe scripts/round4070_g/train_benchmark.py freeze --manifest outputs/round4070_comprehensive_traversability/dataset_snapshot_001/manifest.json --splits outputs/round4070_comprehensive_traversability/dataset_snapshot_001/split_manifest.json --pairs outputs/round4070_comprehensive_traversability/dataset_snapshot_001/matched_pairs.json --output outputs/round4070_comprehensive_traversability/learning_001

& C:/conda_envs/isaaclab/python.exe scripts/round4070_g/train_benchmark.py run --output outputs/round4070_comprehensive_traversability/learning_001 --training-window outputs/round4070_comprehensive_traversability/training_window.json --device cuda

& C:/conda_envs/isaaclab/python.exe scripts/round4070_g/learning_report.py --run outputs/round4070_comprehensive_traversability/learning_001
```

以上snapshot_001是未来恢复入口示例，不是声称它已存在或已执行。当前真实预检快照为 `learning_preflight_000`；其零G样本、全部任务数据不足，不能运行成完整benchmark。代码审计后有更新时须新freeze，不改写冻结hash。

训练窗口由协调者在确认Isaac停止后写入：`allow_training=true`、带UTC时区的`expires_utc`、`isaac_stopped_confirmed=true`。每fit/epoch检查窗口，整个训练run持有原子 `resource_lock.json`；物理试验使用相同锁。超时不启动下一epoch，保存partial checkpoint退出。锁包含pid/token，只释放本进程拥有的锁；崩溃后残留锁需先核对PID退出，不能猜测删除。

再次运行同一`run`命令即恢复：完成fit不覆盖，partial保存模型、optimizer、RNG、epoch及累计时间。`--max-fits N`允许有界批次。FAILED任务再次调用可重试，BLOCKED任务必须由新增独立数据快照解决。进程退出后不会自动继续。

checkpoint、curve、校准参数、真实预测、指标和completion分别存在每个fit目录。推理计时仅网络/统计模型前向，不包含采集、预处理、残差计算或I/O；不能称端到端实时保证。报告按完整五seed的共同episode做物理配置聚类分析，初始化不当独立物理样本。

## 当前实现验证

十一项小合成契约检查已实际通过：因果截止、训练归一化隔离/严格瞬时通道、实际采样降频、分组/配对真值、防 UNKNOWN 进入监督/拒判与校准指标、关节重排、各神经模型前反向、资源锁互斥、观测 DR 的训练尺度隔离/因果索引/单调时钟/确定性复现、conformal 分数不复用温度拟合标签，以及标签预算嵌套/原标签与holdout不变/结果分离。全部 31 个学习方法的两 epoch 合成 fit（分阶段方法额外一轮预训练）、checkpoint 读取与数值回放、固定噪声测试推理、校准，以及 GRU partial 恢复一致性也全部通过；最新完整拟合证据位于 `outputs/round4070_comprehensive_traversability/implementation_smoke/92d44ab68cbe47588fe2ab3150b8e723/implementation_test_results.json`。完整拟合 smoke 使用 CPU 两线程，Isaac 已退出且持共享资源锁；结束后关闭训练窗口。随后修正 conformal 的 raw 分数语义并新增标签预算调度，实际通过全部十一项 CPU 合同，未重新运行拟合 smoke。这些均为实现测试，不是物理 G 数据，不进入任何科研分数。

## 仍未评价的范围

低标签预算入口已经实现，当前标签效率结果尚未实际训练；不能宣称提高标签效率。即使后续 full-label 辅助模型更好，也只能解释为同一监督量下的泛化变化。匹配正对照目前仅使用真实跨配置 P-P；并未声称同时使用跨配置 B-B 或存在真实跨材料配对。split7 按姿态/动作条件划分，含速度变体，评价按 robot_variant 分层，不能将总体差异归因于纯姿态。所有正式学习结果仍须等待新 G 标签及可评价划分，代码和合成验证不替代这些证据。
