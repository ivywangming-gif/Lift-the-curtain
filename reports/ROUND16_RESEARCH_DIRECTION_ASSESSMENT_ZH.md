# Round 16 研究方向评估

## 研究问题定位

当前最值得保留的问题是：

> 对一个仅标为 2、结果未知的候选障碍，机器人如何用有限、可撤回、部署可得的本体历史和少量空间化试探，区分“局部可响应”与“全身可通行”，并在证据不足时安全绕行？

这里的关键不是把 curtain/rigid 做成语义分类，而是处理三种不同层次：

1. hand-level `RESPONSIVE`；
2. whole-body progress/clearance；
3. 受安全预算限制的 `PASSABLE` 或 `BLOCKED/DEFERRED`。

Round 12 已给出一个窄范围的工程闭环；Round 13--15 暴露了局部 probe 信息不足、动作位置依赖和长时接触适应的边界。

## 与近邻工作的关系

### Interactive Navigation with Traversable Obstacles（ICRA 2024）

该方向把可交互障碍纳入导航，并通过交互轨迹提高通过率；其核心是学习交互/导航策略。我们的差异不在于重新证明“障碍可以被交互”，而在于研究已知 0/1/2 map 下、不预知候选物理类别、仅用本体历史和有限 probe 的可审计 hybrid contract。当前不应把我们的结果表述为对该类方法的替代或更高成功率。

### HAVEN、VLM-GroNav

这类工作使用视觉语言或视觉语言动作模型处理开放世界语义、空间先验和操作规划。我们的当前设置主动去掉 VLM 和 RGB-D renderer，把问题缩小到物理可通行性边界和部署输入隔离；未来若加入视觉，应作为额外观测来源，而不是用语义标签替代本体证据。

### DCT、Blind Dexterity、DoorMan、SPARTA、TANGO

- DCT 的长时控制重点是 diffusion actor-critic 与 tree search；我们的首要缺口还在 deterministic contract 和 matched causal evidence，尚未到应引入长时学习搜索的阶段。
- Blind Dexterity 强调触觉/本体反馈下的盲操作；它说明没有视觉真值时反馈设计的重要性，但我们的任务是人形全身穿越与导航，不是灵巧抓取，且本轮明确不加入触觉。
- DoorMan 关注门操作的视觉到动作闭环；门的铰链、把手和可见几何与当前未知柔性约束不同，不能直接移植其任务成功定义。
- SPARTA 关注从示范中学习空间约束；我们的当前证据恰好显示空间位置和约束分布影响通行，但需要先用确定性空间 probe 建立数据和安全接口。
- TANGO 关注几何 traversability 与拓扑路径；我们的关键扩展是 candidate 2 在试探前不能当作 free，必须把交互结果和地图更新绑定。

这些工作共同说明该问题与交互导航、长时控制、空间约束和反馈操作相邻，但没有一项可以替代当前所需的严格 matched experiment。

## 最多两个候选贡献

### 候选一：部署可得的 Probe-to-Passability Hybrid Contract

**核心贡献。** 定义并验证一个不读取物体类别和仿真接触真值的状态协议：

`2 -> STAND_AND_PROBE -> RESPONSIVE/BLOCKED/UNCERTAIN -> whole-body progress monitor -> PASSABLE 或 2->1`。

该协议将 hand progress、tracking residual、effort proxy、base progress、robot-owned envelope 和 bounded recovery 统一起来，并明确 `RESPONSIVE != PASSABLE`。

**为什么值得做。** Round13 的三种 bottom anchor 已给出同样局部响应、不同全身结果的可复现线索；这比单纯材料分类更贴近真机安全决策。

**真正困难。** 需要严格匹配的重复样本、共同 pre-contact region、明确的 safety budget 和未参与阈值设定的验证集。若没有这些，容易把 WBC 停滞、几何 gate 或 physics artifact 误判为交互算法增益。

**最小技术方案。** 保持确定性规则；增加阶段化 progress monitor、连续窗口证据、robot envelope 清空、失败后地图 2->1 和一次 bounded retract。若后续仍存在稳定的 ambiguous cases，再考虑 model-based belief 或学习 residual，而不是直接 PPO。

**必须的对照。** nominal、double-density、bottom-left、bottom-right、both-bottom、matched rigid、close-gap no-wall/with-wall；同一 reset/probe/time window；A/B/C 三策略；报告 false-pass、false-block、uncertain、safe recovery 和 PointGoal。

**资源可行性。** G1、4070 Laptop 和有限 A100/A800 足以运行 headless、num_envs=1 的确定性 matched sweep；不需要 5090，也不需要训练大模型。主要成本是仿真时长和高质量 telemetry。

**T-RO 风险。** 若只有单次 nominal PASS 和若干未匹配失败，贡献会被评为工程集成而非研究；若把仿真 root pose 当真实定位而不做定位假设说明，也会削弱可部署性；若没有真实机器人或至少严格 sim ablation，结论外推风险高。

### 候选二：Spatially Active, Budgeted Probe Selection

**核心贡献。** 在同一 2 区域内选择最多两个几何验证的 probe site，用 persistent world/local map memory 合并重复入口；根据第一次的本体历史和身体 envelope 选择第二个位置，成功或失败后更新地图并停止无限试探。

**为什么值得做。** bottom-right 与 bottom-left/both-bottom 的差异表明位置可能改变身体通过机会；Round15 软件 contract 已说明现有默认 merge radius 会错误地合并 `.30 m` 的左右站位。

**真正困难。** 第二站位必须在真实可达工作空间内，且要证明它提供的是新信息而不是把失败换一个位置重复；试探次数有限，安全撤回和 map update 需要同一状态机；物理双站位结果目前完全缺失。

**最小技术方案。** 先不学习：使用候选 footprint 与手部 FK 的交集生成站位，显式配置 `.20 m` spatial merge radius、最多两次 probe；第二站位只在第一处 `BLOCKED/UNCERTAIN` 且仍有安全余量时启用；任何失败都写入 episode memory 并将当前候选区域标为 1。

**必须的对照。** 同一场景的固定站位 A、左右双站位 C、无第二站位 B；相同安全预算和时间窗；记录第二次是否改变 whole-body clearance、目标成功和重复 probe 数。

**资源可行性。** 软件和几何部分在当前仓库可完成；物理验证受单进程 Isaac 时间和 G1 upper/lower 协调限制，仍可用 4070 做少量重复，不需要学习训练。

**T-RO 风险。** 如果 C 只是软件 contract，没有物理成功率或信息增益统计，不能作为完整贡献；如果第二站位依赖场景真值或对象类别选择，特权信息审计会失败；如果没有 hidden/no-wall matched pair，位置选择结论可能只是控制器偶然性。

## 研究路线建议

1. 先完成候选一的严格 matched repeated experiment，并冻结所有阈值。
2. 只有在候选一的 failure taxonomy 稳定后，运行候选二的真实双站位物理实验。
3. 如果确定性策略仍在 matched unseen constraints 上出现稳定、可量化的 ambiguity，再考虑学习一个高层 probe-site/residual policy；训练输入必须仍然是部署可得历史，不得使用 object label、cloth nodes 或 outcome oracle。
4. 在当前证据状态下，不启动 PPO，不把学习作为解释工程失败的捷径。

## 学习门槛结论

当前 `evidence_sufficient_to_start_learning = FALSE`。

原因不是学习永远不需要，而是：

- 严格 hidden matched comparison 未完成；
- physical two-site 未运行；
- 扩展场景无冻结参数重复成功率；
- 长时 traverse failure 仍可能包含 locomotion/WBC 接口因素；
- 没有足够的因果标签来区分“物理不可通行”和“动作/位置选得不好”。

在这些门槛通过前，纯工程路线更能提高科学可解释性。
