# Round 4070-F 数据审计

审计日期：2026-11-06（Asia/Hong_Kong）。脚本实际运行：`scripts/round4070_f/build_dataset.py`。旧日志只读；独立产物写入 `outputs/round4070_learning_traversability/dataset/`。本报告的计数是 Phase A 的 E 数据快照，新增 F 物理结果另列。

## 可以用于哪些学习目标

| 数据范围 | episode 数 | 正例/负例 | 可以做什么 |
|---|---:|---:|---|
| E 全部完整合法探测 | 104 | 不要求标签 | 仅在训练物理配置内做短期响应预测/表示预训练；不能使用 holdout 的 UNKNOWN 历史 |
| E 核心独立真值 | 28 | PASSABLE=2，BLOCKED=26 | 直接通行性监督与探索性评价 |
| E 控制 | 16 | CONTROL_PASSABLE=14，CONTROL_BLOCKED=2 | 空场响应建模、独立报告的控制辅助监督；不冒充柔性正例 |
| E UNKNOWN | 60 | 无有效监督真值 | 训练配置内响应学习；不作第三类、不作负例、不给监督指标贡献分母 |
| E 原始注册但未运行 | 8 | 无数据 | 不进入任何训练或评价分母 |

104/104 runtime-valid、probe-valid，44/104 label-valid（含控制），与旧报告一致。总共 2240 个落盘 probe 点，每条 20–22 点、观察跨度 1.90–1.99 s。共有 13 个 canonical physical groups；不能把 2240 个时间点当作 2240 个独立样本。

## 发现的时间错误

旧 E `run_episode.py` 的 physics DT=0.005 s、EVALUATION_DECIMATION=20，CSV 常规采样间隔实际为 **0.1 s**。逐条验证所有 104 条 median dt=0.1 s；阶段边界允许 0.005–0.1 s 不等间隔，时间严格递增。旧 `evaluate_benchmark.py` 却对落盘 CSV 使用 `sample_period_s=0.005` 计算 stall，近似低估 20 倍。

本轮原样复现旧 E0/E1/E2；另在同一窗口、同一冻结阈值下把 stall 改成真实时间左积分 `sum(diff(time) * low_speed[:-1])`，不对首尾外推、不调参。结果：

| 方法 | false-pass / 26 | 正确 BLOCKED / 26 | 正确 PASSABLE / 2 | UNCERTAIN / 28 | coverage | balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| E0 原样 | 26 | 0 | 2 | 0 | 1.0000 | 0.5000 |
| E1 原样 | 4 | 0 | 2 | 22 | 0.2143 | 0.5000 |
| E2 原样 | 1 | 0 | 2 | 25 | 0.1071 | 0.5000 |
| E1 时间修正 | 4 | 8 | 2 | 14 | 0.5000 | 0.6538 |
| E2 时间修正 | 1 | 4 | 2 | 21 | 0.2500 | 0.5769 |

旧 E1/E2 缺少 BLOCKED 判断部分来自时间单位错误，不能把修正这一错误的收益算作神经网络贡献。旧在线程序曾维护 200 Hz 的内存历史，但没有完整落盘；本轮历史模型只能使用实际保存的约 10 Hz 观测，不能宣称使用了全部 200 Hz 信号。`timing_audit.csv` 保存每条的时间与 stall 差异。

## 统一字段与来源

`manifest.json` 包含 feature_names、feature_blocks、feature_units_by_block、joint_order、arm_indices、body_indices；每个 `episodes/<trial_id>.npz` 包含 `seq[T,166]`、`time`、`phase`、raw q/qdot/q_cmd/effort、动作来源和缺失字段。元数据/标签与 seq 分离。学习代码只加载 `seq`，不能扫描 NPZ 其余 audit-only 数组。

| 字段 | 来源与单位 | 审计结论 |
|---|---|---|
| q / qdot | post-step joint_pos / joint_vel；rad / rad/s | 104 条逐行 29 维、有限，使用 official joint order |
| q_cmd | 已传给 `set_joint_position_target` 的 last_target，逆映射 official order；rad | 是实际下发并保持的目标；同一行反馈发生在它下发后的 physics step |
| effort | `after.joint_effort`；来源全部 applied_torque；N.m | 合法执行器 applied-torque 仿真 proxy；没有零填充/缺失，不等于真实电流/真机扭矩传感器验证 |
| policy_action_logged | 原 CSV `deployment_previous_action_json`；归一化 action | adapter 在 compute_target 内已把 previous_action 改成当次 action；该日志名有误导性，导出更名，不能宣称精确的上一控制周期 action |
| previous_logged_action | 对已记录 action 作一行因果滞后，并保存 previous_logged_action_time | 首行缺失置 NaN；是上一日志点的 action，约 0.1 s，不能冒充 0.02 s 控制周期的上一 action；不进入 seq |
| 双手 FK | robot articulation 的 hand body pose，经本体 RPY 与 base pose 转成 body frame；m | 原字段虽叫 FK，实际来自仿真机器人刚体运动学状态；是机器人自身状态 proxy，未经独立真机 FK 误差标定 |
| 手部进展 | 手世界位姿减去 probe 第一时刻手位姿，再转初始 body frame；m | 仅用当前与首帧，可因果计算；绝对世界位置不进入 seq |
| IMU 姿态 | telemetry roll/pitch/yaw，同一 phase+time 精确 join；rad | 机器人 orientation proxy，全部有限；没有补造 IMU 观测 |
| IMU 角速度 | telemetry measured_wz；rad/s | 仅 z 保存；x/y 缺失，raw 以 NaN 表示，不进入 seq |
| IMU 加速度 | 未保存 | raw NaN；不能宣称有完整六轴 IMU |
| 身体反馈 | base_lin_vel_b，m/s；q 中腿/腰状态 | 与 q 同时间；排除绝对 base xyz、goal、map |
| 动作/阶段/时间 | probe_side 与 causal elapsed 进入 seq；phase 单独保存 | 只含 physical_probe/physical_probe_hold，排除之后 retract/advance/navigation |

手部世界位姿、base 世界位置仅作为明确命名的 `*_audit_only` 数组留存，部署输入中不含它们。`deployment_observation.csv` 本身还包含 map/target 等列；采用白名单而非把全部 CSV 转成模型特征。

关节映射追踪 `falcon_contract.py` 的 29 个关节名称与 ISAACLAB_TO_OFFICIAL/逆映射，验证双向置换恒等。0–14 为腿/腰，15–21 左臂，22–28 右臂；不是凭索引猜测。旧工程 feature 名 `upper_tracking_rms` 实际对全部 29 个关节取 RMS，本轮复现其真实实现，未偷换成仅手臂 RMS。

## 标签与缺失机制

两条正式 PASSABLE 来自 OFFLINE_WHOLE_BODY_CLEARANCE，26 条 BLOCKED 均来自 OFFLINE_RIGID_CONSTRAINT，60 条 UNKNOWN 均为 INSUFFICIENT_INDEPENDENT_EVIDENCE。标签依赖独立 clearance/contact/geometry 与 recovery 证据；这些字段只在 manifest label_evidence 中保存，不能输入模型。没有将 PointGoal 失败、低实际位移、未达到 0.100 m 或局部 RESPONSIVE 直接转成监督标签。

UNKNOWN 不是随机缺失：它们主要属于没有被独立验证整体通行性的柔性条件，而负标签集中于后方刚墙；学习器可能学到受限任务分布而非普适通行性。两条正例都来自 right / frozen_confirmation，各物理配置仅一个；侧别和阶段与正例高度混杂。部署条件不能由此声称覆盖左手或新姿态。

## 历史轮次读取核对

已读取 Round 4070 主报告、problem discovery RESULTS、B/C/D FINAL_REPORT 与真实原始 summary/CSV；完整路径、逐文件 SHA-256、phase 行数、字段维度记录在 `historical_inventory.json`。以下是**实际存在 deployment_observation.csv 的数量**：

| 来源 | 原始历史数 | 报告语义核对 | 本轮用途 |
|---|---:|---|---|
| g1_final4070 历史原始 | 24 | 与 Round 4070 报告另计的 24 条历史一致 | 审计，异协议不混入 E |
| Round 4070 problem discovery | 37 | 38 启动、37 完整、1 execution-invalid；14 clearance 不是把其余自动变负例 | 审计；T2 已含 advance，禁进入本轮 probe 输入 |
| Round 4070-B failure discovery | 38 | 42 启动、38 valid；隐藏刚板与左右手反例，4 invalid 无完整 history | 审计；非统一 10 cm 协议，不借旧正例补 E 正例数 |
| Round 4070-C structural | 27 | 30 启动、27 valid；不同深度/拓扑，1 个非控制 clearance | 审计；深度差异未混入训练 |
| Round 4070-D fixed10cm | 10 | 8 个原 manifest/验证输出 + 2 个独立全臂校准复核；正式核心有效标签为 0 | 审计；不把校准/协议中断转成 BLOCKED |

目录中 summary.json 数还包括 physics gates 等，不能直接当 G1 episode 数。历史读取未增加任何新的监督标签。总审计了 136 条旧原始历史 + E 104 条；主学习数据仍明确是 E 104 条。

## 可复现与数据完整性

运行 `C:/conda_envs/isaaclab/python.exe scripts/round4070_f/build_dataset.py` 即重建全部 Phase A 独立处理产物。104 条的 source_sha256 同时覆盖 deployment CSV、telemetry CSV 和 summary；原历史路径不写入。断言检查：精确时间 join 唯一、时间递增、合法阶段、29 维与有限值、applied_torque 来源、关节双向置换、104 条及元数据分离。运行没有启动 Isaac、没有改变环境或 FALCON。

Phase A 的正式探索训练产物使用 `outputs/round4070_learning_traversability/phase_a_valid/`。更早 `phase_a/` 是代码审查前的执行记录，因前缀时长和残差 warmup 实现问题作废，不能混入最终模型表；修正不依据测试效果挑选模型。

## 首批11条F与Phase C冻结学习快照

首批实际新增 **11 条 runtime/probe-valid** G1：PASSABLE=5、BLOCKED=3、UNKNOWN=3。另有 2 次沙箱启动失败没有完整 summary/观测；由物理报告列为 execution-invalid，不进入数据集、不作负例。11 条 F 每条 22 个合法 probe 点，共 242 点。

独立 `dataset_phase_c/manifest.json` 冻结学习快照为 **115 条 / 2482 个真实采样点**：正式 PASSABLE=7、BLOCKED=29、UNKNOWN=63；控制仍为 CONTROL_PASSABLE=14、CONTROL_BLOCKED=2。监督指标核心分母为 36，含控制的有效标签数为 52，控制不冒充正式柔性正例。原 E 104 条文件和 manifest 未改。

新 F 从每条 trial_metadata 的显式 equivalent_e_condition_id、curtain_length 读取配置；缺字段直接报错，不根据名称猜组。同 2.0 m 配置与 E 合并；两个 1.95 m 配置各自建立完整配置组，但保留原结构 family。全部 F 标签用原 E independent_label 从原 summary/协议/physics gate 独立重算，与 physics 标签文件逐条一致。新 PASS_F_PHYSICS 名称在旧 E evaluator 中不存在，导出器先读取真实 no_probe/probe summary 均为 PASS_PHYSICS_SANITY、保存 SHA-256，再规范映射为 PASS；不修改旧 evaluator 或标签标准。

逐条 F 配置、种子、侧别、姿态与标签见 `F_EPISODE_LABELS.csv`；所有 E+F 配置计数见 `PHASE_C_CONFIGURATION_COUNTS.csv`。`PHASE_C_BASELINE_RESULTS.csv` 保持 36 个核心标签分母，原 `BASELINE_RESULTS.csv` 仍只对应 E 的 28 条，不能混淆。

最终 F 冻结评估由 `evaluate_frozen.py` 实际运行，未拟合任何权重、scaler、响应模型或阈值。先回放 30 个 Phase-A checkpoint 的 420 个 E holdout 概率，最大误差 `2.3283064365386963e-10`（阈值 1e-6），再对 11 条 F 产生 165 行方法×seed预测。UNKNOWN 有预测但不进监督分母。详见 `F_FROZEN_EVALUATION.csv` 与 `F_FROZEN_EVALUATION_REPORT.md`。

## 最终左手覆盖补充：总数据117，训练快照仍115

为实际检查左手正标签缺口，另完成两个已验证1.95 m配置的左手G1：`F_LEFT_COMPLETION_nominal_left_S420401` 与 `F_LEFT_COMPLETION_bottom_right_anchor_left_S420402`。两条均runtime/probe-valid、各22个合法采样点，独立标签复核均为 **UNKNOWN**，没有为补齐左手正类而改标签。

最终 `dataset_final/manifest.json` 为 **117条 / 2526个采样点**：正式PASSABLE=7、BLOCKED=29、UNKNOWN=65，控制P14/B2。F最终13条=P5/B3/U5；另外2次早期startup-invalid仍在物理审计中独立记录。核心有效标签仍36、含控制有效标签仍52；完整正物理配置仍4个，7个正式正例仍全部右手。

这两个新UNKNOWN只保存为可供未来严格分组响应学习的数据。没有新增监督标签，因此没有重复Phase C训练；`dataset_phase_c`115条、`phase_c`模型、scaler与响应模型均保持原快照。`dataset_left_completion`106条仅由E104+新左2构成，专门用于独立冻结预测。它不是新的监督训练集合。

原11条标签逐字节归档为 `physics/first11_frozen.json`，SHA256=`6609dcfa98c838ba49f5189c987b08005ddcfc1e81a944090b3dfb26e47de1f8`；旁置SHA清单记录331个原始训练数据、checkpoint/config和首批F报告文件。已核对原11行在最终13行中原顺序保留、全部331个保护hash不变，原115 manifest与训练配置内hash一致。

新左手冻结推理以Phase-A模型实际产生30行预测（2 episode×5方法×3seed）。监督评价分母为0，所有accuracy/false-pass率/recall/coverage/Brier均不报告；导出的相应率字段为空。E420概率回放仍通过。结果和可复现命令见 `LEFT_COMPLETION_REPORT.md` 与 `LEFT_COMPLETION_F_FROZEN_EVALUATION.csv`，没有覆盖原F8标签评价报告。
