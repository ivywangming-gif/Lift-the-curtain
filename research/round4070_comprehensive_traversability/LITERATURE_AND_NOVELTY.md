# Round 4070-G 文献与创新性审计

核验日期：2026-11-06。本文区分原始论文事实、本轮拟实现的适配、真实实验结果。实验状态以自动生成的结果表与执行账本为准；实现代码或冻结清单不表示实验完成。

## 原始来源与任务边界

| 工作、年份 | 已有任务、模型和训练思想 | G轮改变什么；不能据此声称什么 |
|---|---|---|
| **Blind Dexterity: Whole-Body Humanoid Manipulation via Pure Proprioception**，Bhatt等，2026预印本 | G1纯本体全身交互，强化学习控制及独立任务状态估计；关节反馈短历史用于接触后对象信息推断。[原文](https://arxiv.org/html/2608.29487v1) | 本轮冻结控制器、固定探测与后续动作，预测通道利用结果；不能宣称首次G1纯本体交互。 |
| **CoPRE: Improving Sensitivity in Proprioceptive Contact Detection for Low-Cost Robot Arms**，Zhu等，2026预印本 | ARX与G1接触检测；命令条件正常力矩预测，排除近期状态，再以残差和噪声加权Jacobian构造接触分数。[原文](https://arxiv.org/html/2609.27381v1) | G轮残差GRU将简化响应残差用于通行标签，未完整复制原接触检测器；不能把接触检出等同于全身可通，也不能称超过CoPRE。 |
| **PROBE: Proprioceptive Obstacle Detection and Estimation while Navigating in Clutter**，Ramesh等，2025预印本 | Go1本体历史Transformer，重建遮挡矩形障碍的存在、几何与位姿；持续导航接触下构建数据。[原文](https://arxiv.org/html/2505.11848v1) | 本轮轻Transformer仅借用历史编码思想，输出动作条件通行概率，不使用隐藏几何监督；不是PROBE完整复现。 |
| **Thinking on Its Feet: Terrain-Aware Hierarchical Control with Proprioceptive Feedback for Humanoid Navigation**，Wang等，2026，作者列为Humanoids 2026 | G1本体异常检测、时间证据积累、语义重规划与速度调节，面向不平/台阶/滑面。[作者项目](https://yuhaiw.github.io/TOIF/)、[作者全文链接](https://rongxuan-zhou.github.io/Research/TOIF_Humanoids2026.pdf) | 本轮面向一次手部探测后的隔挡通行判断。地形异常重规划已有直接先例；模拟foot-contact特征不能借入本轮部署白名单。 |
| **TAPNAV: Humanoid Navigation through Tactile Active Perception**，Liu等，2026-10预印本 | G1在已知静态地图上无视觉导航，使用定制TouchTronix触觉末端、里程计、IMU；依据预期信息增益选择左右手探测，校正定位信念并规划。[原文方法III–IV](https://arxiv.org/html/2610.10748v1) | 本轮不增加触觉皮肤、不学习主动探测策略，问题是固定动作的障碍通行性。TAPNAV并非纯关节反馈的柔性障碍分类，也不能忽略其“人形触觉主动导航”直接相关性。 |
| **Interactive Navigation in Environments with Traversable Obstacles Using Large Language and Vision-Language Models**，Zhang等，ICRA 2024 | Unitree A1；GPT-3.5、Grounding DINO、LiDAR与action-aware costmap，验证帘幕和草地导航。[原文](https://arxiv.org/html/2310.08873v3) | 视觉语义候选地图与穿帘导航已有先例；本轮使用实际探测反馈核验候选区域，不从“curtain”名称直接赋通行真值。 |
| **Autonomous exploration with online learning of traversable yet visually rigid obstacles**，Prágr、Bayer、Faigl，2023卷（2022在线） | 多足机器人以前部触觉探测与外感知学习穿越阻力，GP在线预测并按信息增益探索。[出版社](https://link.springer.com/article/10.1007/s10514-022-10075-4)、[作者稿](https://comrob.fel.cvut.cz/papers/auro22escape.pdf) | 视觉障碍经交互后可利用不是新问题。本轮研究双足固定能力、仅本体输入、约束与动作条件泛化，不宣称首创交互通行性。 |
| **Supervised Contrastive Learning**，Khosla等，NeurIPS 2020 | 利用类别标签构造多个正对与负对，在嵌入空间做监督对比学习。[论文](https://papers.nips.cc/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf) | G轮物理匹配只改变训练对的选择与任务适配；SupCon损失和对比学习不是原创。必须用真实同材料异结果、跨配置同结果配对，并与普通SupCon比较。 |
| **Selective Classification for Deep Neural Networks**，Geifman、El-Yaniv，NeurIPS 2017 | 已训练网络的置信分数配合选择函数，在风险与覆盖率间取舍；使用独立数据估计风险界。[论文](https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks.pdf) | G轮关注有方向的误放行风险，还须报告真正通行机会与正确拒绝。普通选择性错误率不等同于false-pass；有限相关仿真样本、OOD与非随机缺失标签不支持真机安全保证。 |
| **SelectiveNet: A Deep Neural Network with an Integrated Reject Option**，Geifman、El-Yaniv，ICML 2019 | 分类器与选择头联合优化，显式约束覆盖率。[原始会议页](https://proceedings.mlr.press/v97/geifman19a.html) | 是可选已有基线；本轮阈值拒判或conformal适配不能冒称实现了SelectiveNet，未运行方法须标明未运行。 |
| **Inferring Object Properties with a Tactile-Sensing Array Given Varying Joint Stiffness and Velocity**，Bhattacharjee等，2018期刊（2014初稿/2017修订） | Cody触觉阵列，以HMM/LSTM等分析动作条件下的接触时序、对象属性。[修订原文](https://arxiv.org/html/1409.4972v2) | 跨速度/刚度时序学习并非新思想；本轮不采用材料分类目标。换成GRU或加入时间戳本身不足以形成方法贡献。 |

核验范围：本轮打开上述可用原始页面并核对任务/方法；TAPNAV额外核对已知地图、定制触觉传感器、左右手动作与定位目标。TOIF作者PDF本轮抓取失败，采用作者项目页并对照F轮已保存的全文阅读审计；Prágr作者稿超浏览工具大小限制，因此出版社摘要与元数据的核验不冒称逐式复现全文。未以搜索聚合站的描述代替可获取原始来源。

## 对G轮设计的可检验要求

以下是本轮研究设计推论，不是文献已经证明的G1结果。

1. **标签必须是动作条件结果。** 同一帘幕的局部响应和后续身体清障是不同量；采样、模型选择与标签生成必须分开。固定forward动作、安全监管和恢复门槛后，控制失败仍可能只产生LABEL_UNKNOWN。
2. **匹配比网络名称更关键。** 对比学习若只有隐藏墙负例与普通帘正例，可能强化场景捷径。离线配对必须留下物理差异、动作差异、真实标签与训练归属证据；测试配置不可参与配对、归一化或自监督。
3. **时间机制要有同信息消融。** 真实50Hz与其10/20Hz降采样可比较，但插值不增加物理信息。shuffled、严格瞬时、命令进程、真实dt及参数匹配编码器应使用相同截止点。累计位移不是严格瞬时观测。
4. **响应学习不自动带来标签效率。** 训练组UNKNOWN有可用历史，但自监督损失下降只证明响应拟合；必须在独立通行标签、固定监督预算下比较纯监督、联合与分阶段训练。
5. **左右手不能靠镜像标签补齐。** 真实轴向/关节映射、身体坐标和动作契约都需检查；可对合法观测做重映射，但不推断另一只手的通行结果。缺一侧正类时应报告不可评价。
6. **风险结果必须带分母。** 固定阈值、校准阈值、双阈值与可能的conformal集合分别报告TP/TN/FP/FN/拒判。风险—覆盖曲线不等于任何OOD风险保证；所有拒判也不是成功。

## 创新性与结论更新规则

已有任务包括本体接触推断、人形交互与主动触觉导航、可穿越视觉障碍探索。已有结构包括GRU/LSTM/Transformer、响应残差及状态条件化；已有训练思想包括辅助预测、监督对比、预训练及选择性分类。G轮可能有实证价值的变化是：严格固定动作语义、真实多拓扑和阻力场景、匹配物理对照、真实采样率消融、独立配置与侧别评价，以及针对错误放行的风险覆盖率比较。

这些变化是否足够形成方法贡献，必须由实际数据回答。必须同时满足多个独立正负物理配置、左右手双类支持、无泄漏校准、跨条件可重复收益及错误机制分析。训练seed只度量优化波动，不能充当独立物理样本。若差异只有一两个episode，不宣称稳定优势；若某物理模型只有代理结构数值稳定，不能报告真实材料泛化。

当前F轮证据仅支持复杂M2无已证实优势、整体数据不足；它不能预先决定G轮方法胜负。G轮未执行完毕的候选保持PLANNED或BLOCKED_WITH_REASON，不能用F成绩填充G表。最终方法状态由实际G结果更新到`SCIENTIFIC_METHOD_DECISION.md`。
