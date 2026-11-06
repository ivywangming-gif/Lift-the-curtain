# Round 4070-F：文献、问题边界与方法贡献审查

检索与核验日期：2026-11-06（Asia/Hong_Kong）。本文件是针对指定问题的定向原始来源核验，不是覆盖全部机器人文献的系统综述。论文事实、对本轮的推论、尚待实验支持的主张分开列出；不预设 M2 有效，不宣称首次。

## 1. 核验范围与来源等级

已读取 Blind Dexterity、CoPRE、PROBE 的 arXiv 正文方法与实验段；TOIF 的作者托管八页全文；触觉 HMM/LSTM 工作的修订正文；Grounding DINO 的 ECCV 论文及 traversable-obstacle navigation 的最新版 arXiv 正文。Prágr 工作核对了出版社元数据、摘要和作者托管稿的可检索正文（完整 PDF 因浏览工具大小限制未全文展开），因此不声称逐式复现其 GP 或复核全部实验。检索引擎、聚合网站只用于找原文，以下结论以作者论文、项目页和出版社为依据。

同时读取本仓库 E 轮最终报告、Round 4070 问题分析及旧文献报告。旧报告中的“尚未复核原文”不能替代本次核验；旧轮 T2 包含 cautious advance，不能用来支持本轮单次 probe 截止前的时间学习收益。

## 2. 指定工作的事实对照

| 工作与年份 | 实际平台、输入及研究目标 | 与本轮的关系和边界 |
|---|---|---|
| **Blind Dexterity: Whole-Body Humanoid Manipulation via Pure Proprioception**，Bhatt 等，2026 年预印本 | Unitree G1；关节状态、previous joint target、部分任务的 IMU；PPO 学习足球停球、滑板登乘、行李箱交互等，另训练任务状态估计器 | 已直接覆盖 G1、纯本体、短历史和主动接触；主要学习交互控制和对象状态，不是冻结控制器下的一次探测通行标签。[正文](https://arxiv.org/html/2608.29487v1) |
| **CoPRE: Improving Sensitivity in Proprioceptive Contact Detection for Low-Cost Robot Arms**，Zhu 等，2026 年预印本 | ARX L5 与 G1 右臂；从无接触状态和命令学习正常 torque estimate，残差检测弱接触；ARX 展示避碰放置与插书重试 | “命令条件正常响应—残差”已有直接先例；检测接触不等价于判断 G1 全身通道可利用性。[正文](https://arxiv.org/html/2609.27381v1) |
| **PROBE: Proprioceptive Obstacle Detection and Estimation while Navigating in Clutter**，Ramesh 等，2025 年预印本 | Unitree Go1 四足；关节位置/速度/力矩、位姿、命令历史，Transformer 重建矩形障碍存在、尺寸、SE(2) 位姿及固定/可移动状态 | 已利用本体历史处理遮挡与嵌套交互；输出目标与本轮不同。本轮不能把其隐藏几何监督照搬为通行标签，也不需要其大型重建网络。[正文](https://arxiv.org/html/2505.11848v1) |
| **Thinking on Its Feet: Terrain-Aware Hierarchical Control with Proprioceptive Feedback for Humanoid Navigation**，Wang 等，2026；作者列为 Humanoids 2026 | G1；轻量地形异常检测，经时间聚合触发语义重规划及调速；真实地形包括不平、台阶和滑面 | 已覆盖人形本体反馈修正导航决策；不是手臂一次有限探测后的柔性障碍整体通行概率。[作者全文](https://rongxuan-zhou.github.io/Research/TOIF_Humanoids2026.pdf)、[项目页](https://yuhaiw.github.io/TOIF/) |
| **Autonomous exploration with online learning of traversable yet visually rigid obstacles**，Prágr、Bayer、Faigl；2022 年在线，2023 年期刊卷 47，161–180 | 多足机器人，外感知和前部 haptic bumper；GP 在线预测穿越阻力，结合信息增益探索遮挡后区域 | 是问题层面最直接的先例：视觉像障碍、交互确认后利用。不能声称首次探索可穿越遮挡；区别应落在双足全身约束、部署本体输入和固定动作预算的证据问题。[出版社](https://link.springer.com/article/10.1007/s10514-022-10075-4)、[作者稿](https://comrob.fel.cvut.cz/papers/auro22escape.pdf) |
| **Inferring Object Properties with a Tactile-Sensing Array Given Varying Joint Stiffness and Velocity**，Bhattacharjee、Rehg、Kemp；2014 初稿、2017 修订、2018 年 IJHR 15(1) | Cody 轮式移动人形的前臂触觉皮肤；HMM/LSTM/1-NN 用力、接触面积和接触运动时序预测 hard/soft 与 moved/unmoved；留出刚度/速度组合 | 时序和跨机器人动作条件识别早有研究；“用 GRU 替代 HMM/LSTM”不构成机制贡献。本轮不得复制其材料属性监督。[修订正文](https://arxiv.org/html/1409.4972v2)、[期刊元数据](https://experts.illinois.edu/en/publications/inferring-object-properties-with-a-tactile-sensing-array-given-va/) |
| **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection**，Liu 等；2023 初稿、ECCV 2024 | 图像与语言驱动的开放类别检测 | 输出对象框/语义，不是固定 G1 的物理通行保证；适合作为候选区域生成接口。[ECCV 正文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06319.pdf) |
| **Interactive Navigation in Environments with Traversable Obstacles Using Large Language and Vision-Language Models**，Zhang 等；2023 初稿、ICRA 2024 | Unitree A1 四足；GPT-3.5、Grounding DINO、LiDAR 构建 action-aware costmap，展示穿帘与草地导航 | 已有 Grounding DINO→通行属性地图→导航实例；其属性由语言指令赋予，不是本轮探测反馈验证。不能把“帘幕”检测结果直接写成 map=0。[最新版正文](https://arxiv.org/html/2310.08873v3)、[发表元数据](https://arxiv.org/abs/2310.08873) |

## 3. 需要进入实现和消融的具体差别

### 3.1 Blind Dexterity：历史信息存在，不等于当前任务可辨识

该文对象交互用 5 帧、0.1 s 历史；previous action 与编码器可形成跟踪误差。辅助估计器为独立监督 MLP，不把其损失反传给 PPO；作者也没有主张未见对象几何的类别级泛化。其 estimator feedback 并无一致平均增益。这支持先比较原始历史及简单模型，不能据此推断本轮完整 M2 必要。[方法 IV-A/IV-C 与讨论](https://arxiv.org/html/2608.29487v1)

对本轮的推论：历史是否包含足够信息，要由**相同探测窗口、独立物理条件**的通行标签检验。成功完成停球/搬箱不构成柔性隔挡通行性已解决的证据；反之，任务不同也不足以构成新方法贡献。

### 3.2 CoPRE：不能把普通残差头称为原论文复现

CoPRE Eq.4 从截至 `t-H` 的状态历史及截至 `t` 的已知命令预测正常响应，排除预测区间内的实测状态；用当前 torque residual 和按无接触噪声加权的 Jacobian 聚合检测。训练、验证、无接触阈值校准分离，q/qdot 是辅助预测，检测使用末步 torque estimate。[方法 IV-A–C](https://arxiv.org/html/2609.27381v1)

本轮实现应称 **CoPRE-style response residual baseline**，不能报告“超过 CoPRE”。实际 Ridge 目标为 q 与日志中的 effort，输入为至少 0.4 s 前的 q/qdot 和当前已执行 q_cmd；排除前 0.4 s 拟合样本，并将该暖机区间残差显式置零。因此已有 state exclusion 和正常 effort 预测，不能误写为这两项完全缺失；但并未复现原论文网络、无接触噪声加权 Jacobian 聚合及完整接触检测校准协议。effort 的来源及与物理关节力矩的对应仍须按数据审计说明，不能以字段名视为理想 torque 真值。正常响应只用训练组内空场控制拟合；缺失 torque 不得用零值伪装为真实测量，暖机残差置零也不能被解释为实测无接触。

### 3.3 PROBE：隐藏几何可重建的结果不可跨协议移植

PROBE 的 ORM 面向导航过程中的持续 whole-body 交互，学习显式场景重建，数据由多种导航策略收集，并讨论策略导致的接触覆盖偏差。[方法 IV-C–D](https://arxiv.org/html/2505.11848v1)

对本轮的推论：一次最长 10 cm 手部试探可能没有触及第二层；此时不能以网络容量替代观测。当前本体历史能否区分“局部可推动但身体无法利用”必须独立验证，不能把模型预测后衬类别当作任务达成。

### 3.4 TOIF：小特征模型是实质基线，接触真值不能借入

全文采用 15 帧投票、leaky integrator 和异常分数调速；真实模型包含 AdaBoost、XGBoost、LOF。模拟紧凑特征中含 foot forces/air times；真实训练另用已标注 G1 数据。其 80/20 时间划分不等于本轮要求的完整物理配置留出。[方法 III-E/F 与实验 IV/V](https://rongxuan-zhou.github.io/Research/TOIF_Humanoids2026.pdf)

对本轮的推论：加入全身状态可能减少动作补偿混淆，也可能引入冗余、初始姿态捷径或噪声。必须比较 arm-only、body-only 诊断、联合模型及无条件 M2。不能把 TOIF 模拟接触传感器值当作本轮合法部署输入；不要因全身状态“看起来更完整”便预定有效。

### 3.5 触觉时序与 Prágr：区分三种监督语义

触觉研究的目标是对象属性；Prágr 的目标是穿越阻力及探索；本轮目标是在固定机器人能力与安全动作下能否利用候选通道。三者相关，但监督不可互换。被动柔性、低阻力或接触存在，都不能逻辑推出全身 PASSABLE；一次执行未完成也不能自动推出物理 BLOCKED。[触觉任务定义与实验](https://arxiv.org/html/1409.4972v2)、[Prágr 任务定义](https://link.springer.com/article/10.1007/s10514-022-10075-4)

## 4. 六个新颖性问题的回答

1. **是否已有 G1/人形同题研究？** 已有 G1 纯本体交互、G1 接触检测及 G1 本体反馈导航；上述检索中没有核实到同时满足“冻结控制器、单次有限手部探测、柔性/后衬条件、整体通行标签、独立物理配置留出”的完整同题工作。这是有边界的检索结论，不是不存在同题工作的证明，更不是首次声明。
2. **谁检测接触，谁预测通行？** CoPRE 直接检测接触；Blind Dexterity 学控制/估计对象状态；PROBE 重建障碍；TOIF 判地形执行异常并重规划；Prágr 预测穿越所需阻力并探索；Zhang 等根据语言与视觉生成通行地图。本轮若成功，需直接支持冻结能力下的通行决策，而非复述这些中间目标。
3. **M2 是否解决普通 GRU/残差基线不能解决的问题？** 文献不能回答本仓库实验问题。必须同划分、同输入截止、同训练预算比较普通 GRU、残差 GRU、完整 M2，并报告真实计数和种子差异。无显著且可重复的额外收益，就不支持 M2 必要性。
4. **状态条件化是否确实改善跨姿态、左右手和约束泛化？** 目前只是待检验机制。若测试集只有单一姿态/尺寸，不能据不同 seed 声称跨姿态泛化；若两个正例分别被分到不同集合，方差和类覆盖必须直接披露。
5. **仅换网络是否足够？** 不足。GRU、残差、双编码器及三态输出本身都是常见工具；把它们拼接并在极小样本上胜出，不能自动成为 RA-L/T-RO 方法贡献。
6. **最有价值的潜在机制？** 是把机器人自运动导致的响应变化与接触证据分开，并识别当前一次动作是否已经提供足够的**任务相关证据**。需用匹配干预证明这一点，而不是用隐藏材料或拓扑分类间接宣称成功。

## 5. 最多一个主方向和一个备选的研究假设

以下最终排序依据 `phase_a_valid`、F 冻结模型评价及合并数据 Phase C 的实际输出；第 8 节保留 Phase A 结果，第 9–10 节给出补充证据和优化敏感性审计。它是后续研究的复杂度选择，不是从同批测试中选出模型后再将其成绩当作独立确认。

**主方向：普通小 GRU 的可审计通行判断最小基准。** 以合法本体输入和固定探测窗口研究标签、观测边界与独立物理条件评估。Phase C 中普通 GRU 三个 seed 均误放行 3/29；残差 GRU 只在一个 seed 少错一条，完整 M2 反而均值更差；F 的八条新有效标签连原始统计量 Logistic 也全部判对。因此保留普通 GRU 作为探索基准，不将其称为已证实最优模型，更不宣称其时间顺序机制或新方法贡献已成立。

最低可证伪实验仍为：同材料、匹配命令、相近初始状态的独立正负条件；同一物理条件左右手及姿态变化；训练外的约束/尺寸/阻力条件；无校正、无身体条件、无时间顺序、参数量相近 GRU 的配对消融。无接触验证损失不能替代通行性能。完整 M2 保留为已测试的机制假设，当前不列为额外主要方法方向；只有这些干预中出现可重复收益，才能重开其必要性论证。

**备选：简化的命令响应残差 + 小 GRU。** 只在训练组内拟合正常响应，将合法残差与手臂历史用于通行判断；它在 Phase A 有探索性收益，但 Phase C 相对普通 GRU 的优势缩小且未形成稳定机制证据。该方向是对 CoPRE 等已有残差思想的任务适配，须由独立匹配干预验证，不能用命名包装成新残差方法。只有获得判别能力后才考虑前缀拒判/提前停止；无判别能力时加拒判只是减少覆盖，不是解决任务。

不同时扩张主动多次探测、材料分类、world model 或低层控制学习；它们会改变本轮要回答的问题。

## 6. 形成论文方案的证据门

本次文献核验本身不能授予 `METHOD_SUPPORTED`。需要跨独立物理配置的可通行正例、足够且语义一致的阻挡负例、完整运行有效性、真实合法输入来源和有意义的测试正负覆盖。反复确认同一正例不是独立物理配置泛化。

误放行降低必须与 BLOCKED 正确计数、PASSABLE recall、false-reject、UNCERTAIN rate、coverage 同时报告。若仅靠拒判或 always-BLOCKED 得到好看的总体 accuracy，没有方法贡献证据。UNKNOWN 真值的缺失可能受失败难度影响，不能视为随机缺失；监督集上的概率校准结论不自动适用于全部候选区域。

若数据不足，准确结论是 `DATA_INSUFFICIENT`；若匹配物理条件证明合法历史不可区分，才有依据讨论 `OBSERVATION_INSUFFICIENT`。不能以学习失败替代信息论不可能性证明。现有文献使问题有研究动机，但 RA-L/T-RO 方法方案是否成立，必须由本轮及后续独立实验决定。

## 7. 检索审计记录

定向检索包含精确标题及组合：`Blind Dexterity robot`、`CoPRE proprioceptive residual`、`PROBE proprioceptive obstacle`、`Thinking on Its Feet`、`Prágr visually rigid traversable obstacles`、`tactile HMM LSTM object properties`、`Grounding DINO traversable obstacles navigation`、`G1 curtain proprioceptive`、`humanoid traversability judgment probing`。已追踪可获得的作者原文，未以综述或搜索摘要代替可读正文。

年份特别校正：Prágr 是 2022 online / 2023 volume；触觉 HMM/LSTM 工作是 2014 arXiv 初稿 / 2017 修订 / 2018 期刊卷，不能仅从 DOI 的 `1750024` 猜年份；Grounding DINO 是 ECCV 2024 而非搜索课件误写的 ICLR 2024；TOIF 项目页的 Paper 链接未到正文，但已从合作者主页找到上文明确链接的完整 PDF。所有数值比较应留在各自任务语境，不与本仓库评估指标直接横向排行。

## 8. Phase A 实际训练后的方法判据与独立统计审阅

数据依据：`outputs/round4070_learning_traversability/phase_a_valid/` 的实际训练输出，以及本目录 `LEARNING_MODEL_COMPARISON.csv`、`ABLATION_RESULTS.csv`、`MODEL_MEAN_STD.csv`、`BASELINE_RESULTS.csv`。下表每个 seed 合并两个互斥配置测试折，均为同一批 **28 个非控制有标签 episode（PASSABLE=2、BLOCKED=26）**；种子依次为 4070、4071、4072。全为 forced 阈值 0.5，coverage=1，false-reject=0，两个正例均被放行。

| 方法 | 三个 seed 的 false-pass 计数（每次分母 26） | 均值 ± seed 标准差 | 正确 BLOCKED 计数 | Brier 均值 |
|---|---|---|---|---|
| 普通 GRU | 3、1、3 | 2.333 ± 1.155 | 23、25、23 | 0.05785 |
| TCN | 3、4、4 | 3.667 ± 0.577 | 23、22、22 | 0.12675 |
| ResidualGRU | 0、0、2 | 0.667 ± 1.155 | 26、26、24 | 0.02641 |
| M2NoBody | 0、0、2 | 0.667 ± 1.155 | 26、26、24 | 0.02641 |
| 完整 M2 | 0、1、0 | 0.333 ± 0.577 | 26、25、26 | 0.02365 |

**可支持的有限结论。** 这些模型在当前探索测试中确实识别了大量 BLOCKED：forced 模式没有 UNCERTAIN，因此收益不能解释为只靠拒判。原 E0/E1/E2 的 false-pass 分别为 26/4/1，而 coverage 为 1/.214/.107；修正时间累计后的 E1/E2 coverage 为 .50/.25，正确 BLOCKED 为 8/4。应同时展示修正工程基线，不能只用旧零 BLOCKED-recall 的结果突出模型。

**不能据此宣称普通 GRU/TCN 全面优于 E2。** 它们的 forced false-pass 均值高于 E2 的 1 次，但能给出更多正确确定判断，属于不同风险与覆盖权衡。ResidualGRU 的 selective 模式在本测试观察到 false-pass=0、coverage 均值 .845；完整 M2 为 0、.857。该结果值得继续验证，但选择性阈值的零观察误放行不等于部署风险为零，更不等于固定风险下已有统计保证。

**完整 M2 的独立收益没有得到可重复证明。** 相对于残差 GRU，其 false-pass 总数只从三个 seed 合计 2 次降为 1 次，而且 seed 4071 由正确变为 1 次误放行，seed 4072 才从 2 次改善到 0。不能把均值差 0.333/26 或 Brier 差约 .00276 当作稳定机制优势。M2NoBody 与 ResidualGRU 当前实现及预测相同，属于同一个消融参照的两个名称，不是两份独立重复证据。无残差 M2 的 false-pass 均值为 9/26，支持继续检查正常响应校正的作用，但不证明校正已剥离真实因果干扰。

**NoBody 的准确含义是移除显式身体 FiLM 分支。** `normal_response` 仍用全部 29 关节的 q/qdot 和已执行 q_cmd 拟合正常响应；因此分类 GRU 即使仅接收手臂原始通道和手臂残差，残差也间接包含腿/腰等身体信息。它不是完全去除身体信息的消融，不能证明所有机器人状态条件化都没有作用。ResidualGRU 与原始 GRUArm 的比较也同时改变了响应校正和间接身体信息，不能把差距纯归因于手臂残差。当前数据不足以支持必要性结论，不根据本批测试结果继续追加模型调优。

**时间顺序收益未证实。** 普通 GRU shuffled 的 false-pass 均值为 1.667/26，低于有序 GRU 的 2.333/26；均值压缩的 M2Orderless 较完整 M2 差，也可能来自丢失幅值分布、时间点信息及改变聚合机制。二者不能合并为“严格时间顺序必不可少”。命令进程还可提供阶段线索。当前只能说完整历史/残差含有可利用关联，不能说顺序机制已被识别。

`SingleFrame` 的准确含义是**最后一行摘要（含累计进展）**：该行保留双手的六维 `hand_delta_initial_b`，需要当前与探测首帧计算，因而不是严格瞬时、无历史基线。其 false-pass 均值 3.333/26 不能用于证明历史相对纯瞬时信息的收益。后续语义审计增加的 `Instantaneous` 删除这六维累计进展、仅取最后观测，已单独实际训练，Phase A forced false-pass 均值为 3.333/26、seed 标准差 .577，Brier 为 .07581；结果保存在 `STRICT_INSTANT_*`，不回写原 17 方法。它是事后语义审计新增的探索性消融，仍不能与上述 shuffled 结果合并成时间顺序必要性证据。

**身体状态必要性未证实。** 原始 GRU 全身输入优于 arm-only 的平均误放行（2.333 对 9），这是值得分析的观察；但移除显式 FiLM 分支的残差 GRU 已接近完整 M2，且两个正例均为右手、同阶段、同初始姿态。训练中身体状态可能携带姿态/执行阶段关联。没有足够跨姿态正负反事实，不宜把该差距称为跨姿态泛化机制；也不从上述含间接身体输入的 NoBody 消融推出身体信息不必要。

**样本与不确定性边界。** 三个初始化重复利用同样 28 条测试数据，不能计为 84 条独立试验；两个正例不能计为 6 个。这里的标准差只表示初始化波动，不是物理配置泛化的置信区间。fold 0 没有非控制 BLOCKED 训练样本，依靠刚板控制负例；所有非控制负例来自隐藏墙隙的三个配置，不能外推到未见固定拓扑、尺寸、重量和材料。Brier 是这批被选择有标签样本上的平方概率误差；尚无足够正例证明概率校准，UNKNOWN 的非随机缺失也限制外推。

上述仅是 Phase A 的探索性证据，不能据其均值预定最终方向。F 补充及 Phase C 实际结果改变了结构排序；最终主方向为普通 GRU 最小基准，备选为残差 GRU，详见下节。整体仍为 `DATA_INSUFFICIENT`，不授予 `METHOD_SUPPORTED`，也不把小数据学习结果提升为已成立的 RA-L/T-RO 新方法贡献。

## 9. F 冻结评价与 Phase C：最终方法判断

**两种评价回答不同问题。** `F_FROZEN_EVALUATION.csv` 使用原 Phase A checkpoint 和原阈值，未用 F 训练或校准；首批冻结评价快照的 11 条完整 episode 中 P5/B3/U3，只有八条有标签数据进入监督指标。原 Logistic、GRU、TCN、ResidualGRU、M2 在三个 seed 下均为 TP5、TN3、FP0、FR0，forced 与 .2/.8 selective 都没有拒判。这里的 Logistic 是原 60-step AdamW 版本，不是第 10 节的凸审计模型。该结果证明这批运行上存在可利用关联，却无法区分统计特征、时间顺序、残差或显式身体分支的必要性；三次初始化不使八条物理数据变成 24 条。

最终 F 为13条有效探测P5/B3/U5，追加两条左手均UNKNOWN，独立冻结预测的监督分母为0。统一数据共117条，但没有再次重训；以下Phase C仍对应原115条快照，不能把追加未知样本当作泛化成功。

Phase C 则在合并的 115 条 E+F episode 上重新训练，使用预定的配置分组与固定 60 epoch；正式测试合计 **36 条（P7/B29）**。两个配置折 train/validation/test 的总行数为 72/13/20 和 78/8/16，train 总行数包含用于归一化等处理的 UNKNOWN；真正监督训练分别为 19 条（含控制 P17/B2）和 28 条（含控制 P18/B10）。Phase C 包含原 E 及同批 F，因此既不是独立于 Phase A 的第二批 36 条新物理试验，也不能把重训练结果直接解释为冻结模型的外推结果。新增 UNKNOWN 还会改变训练组归一化，不能把阶段间差异完全归因为新增有标签监督。

以下来自 `PHASE_C_LEARNING_MODEL_COMPARISON.csv`、`PHASE_C_ABLATION_RESULTS.csv` 和 `PHASE_C_MODEL_MEAN_STD.csv`，seed 顺序仍为 4070、4071、4072。forced 均 coverage=1、TP7、FR0。

| 方法 | 三个 seed 的 false-pass（每次分母 29） | 均值 ± seed 标准差 | Brier 均值 |
|---|---|---|---|
| 普通 GRU | 3、3、3 | 3.000 ± 0.000 | .07579 |
| ResidualGRU / M2NoBody（同一实现） | 2、3、3 | 2.667 ± .577 | .06400 |
| 完整 M2 | 3、6、3 | 4.000 ± 1.732 | .07964 |
| GRUShuffled | 3、3、3 | 3.000 ± 0.000 | .06800 |
| Instantaneous（删累计进展的末观测） | 4、3、3 | 3.333 ± .577 | .08338 |

完整 M2 相对残差 GRU 的均值方向从 Phase A 的 .333 对 .667 反转为 Phase C 的 4.000 对 2.667；相对普通 GRU 也没有优势。这说明结构排序对有限训练数据敏感，不证明某模块在所有分布上有害。残差 GRU 相对普通 GRU 的净收益仅为三个 seed 合计少一次误放行，且其 selective coverage 均值 .8704，低于普通 GRU 的 .9815；必须连同 false-pass、正确 BLOCKED 和拒判一起比较，不能仅按选择性 FP 排行。GRUShuffled 与普通 GRU 的 forced 计数相同，严格 Instantaneous 仅小幅较差，时间顺序必要性依旧未证实。

最终保留**普通 GRU 主方向、残差 GRU 备选**，按最小结构进行后续研究；这不等于证实普通 GRU 显著优于残差 GRU。完整 M2 的结论为 `NO_ADVANTAGE_OVER_BASELINES`：当前未证明独立、可重复增益，不是永远无优势的数学结论。整体为 `DATA_INSUFFICIENT`。合并七个正例仍全部为右手，只覆盖 nominal/bottom-right 两个结构 family、2.0/1.95 m 四个尺寸×锚定配置；每个新长度只有一个正例。新增姿态范围小，没有质量、摩擦、阻力 OOD 证据。F 的三个负例即使零观察误放行，其 95% Wilson 上界仍约 .5615；不能认证部署安全，也不足以支持 RA-L/T-RO 新机制。

## 10. 标准凸 Logistic：事后优化与正则化敏感性审计

原 Logistic 使用固定 60-step AdamW，在 Phase A 的三个 seed 上 FP 为 11、4、6。这种变化不能直接归因为统计表示不足。为检查有限优化与正则化的敏感性，新增独立的 `converged_logistic.py`，冻结原 Phase A manifest、两个物理配置折、165 个合法通道（去 elapsed）、48 点 past-hold、训练组归一化与裁剪，以及 mean/population-std/min/max 的 660 维统计特征。只有可靠训练标签进入损失，两个折分别为 17、25 条；其他 UNKNOWN 只按原协议参与训练组预处理。

预先固定目标为 `mean BCE + sum(weight²)/(2*n_train)`，即 L2 Logistic `C=1`、intercept 不正则；使用 float64 LBFGS、strong-Wolfe、最多 200 迭代、梯度容差 1e-8，不依据测试结果选 C、容差或 checkpoint。三个 seed、两个折全部实际拟合，**6/6 满足梯度容差**，36–46 次迭代，最大梯度无穷范数 9.39e-9；checkpoint、256 行 closure/最终参数曲线和逐折预测均已保存。closure 记录可能是线搜索试点，不伪称每行都是接受的优化步。

`CONVERGED_LOGISTIC_RESULTS.csv` 的每个 seed 仍汇总同一批 P2/B26 共 28 条正式 Phase A 测试，三 seed 分类计数一致：forced 和 .2/.8 selective 均 FP12、TN14、TP2、FR0、UNCERTAIN0，coverage=1、balanced accuracy=.76923、Brier 均值 .42787635；旧式 validation-zero-FP 为 FP7、TN14、TP2、FR0、UNCERTAIN5，coverage=23/28=.82143。后者只是验证负例确定的经验阈值，不能保证测试零误放行。

这项审计消除了该固定凸目标下的明显初始化差异，但没有改善正式测试。**它同时改变了优化器、正则化形式及数值精度，并不等同于把原 AdamW 目标充分训练到收敛**，所以不能把变化纯归因于优化器，更不能把原模型较好成绩自动称为表示优势。`C=1` 仅是一个预先固定的标准正则化强度；没有 C 扫描、没有嵌套验证，不代表最优 Logistic，也不能证明所有统计特征线性模型必然失败。该审计仅针对 Phase A，未在 Phase C 或 F 冻结评估中运行；不得把它的 FP12/26 与 Phase C 的分母 29 混比或替换原 17 方法结果。它是透明保留的事后优化/正则化敏感性复核，不是新的确认性胜负试验。
