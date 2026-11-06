# G 风险、覆盖率与校准

范围：`G_ACTION_CONTRACT_BENCHMARK`。真实已评价 fit：0 / 预定任务 1840。状态 `{'BLOCKED_WITH_REASON': 1520}`。

forced=0.5、固定拒判=.2/.8；温度拟合和风险阈值只用独立 calibration 组。阈值采用预先固定的 .025 网格，目标误放行率为 0/.01/.025/.05/.1/.2，同时约束校准 false-reject≤.1，再最大化确定覆盖率。该约束是校准样本上的经验约束，不是部署安全认证。缺任一校准类别则不拟合温度/经验阈值。

label-conditional split conformal 的 alpha=.1，分数使用冻结模型的raw概率，不能复用同calibration标签拟合temperature后的分数。有限样本分位数超出可用秩时使用 q=1 的保守集合，双类/空集合都输出 UNCERTAIN。物理配置采集存在选择偏差，交换性未验证，不宣称真实世界覆盖保证。

FP分母为全部可靠BLOCKED，FR为全部可靠PASSABLE；拒判不算TN/TP。selective risk只在已决策集合计错误，必须同时报告coverage与PASSABLE recall。UNKNOWN永远不进监督分母。工程规则的0/.5/1是决定编码，不作为概率报告Brier或校准。

`RISK_COVERAGE_POINTS.csv`、`CALIBRATION_BINS.csv`、各fit的calibration.json均来自实际执行。有实际有效bin时生成 `CALIBRATION_RELIABILITY.png`，按固定候选方法展示 clean test 的 raw/temperature 可靠性曲线；汇总权重是预测数，不把重复seed当独立物理样本、不画未经支持的误差带。完整方法及noisy诊断都保留在CSV。只有表头表示尚无实际可用评价，不是零风险。

`PAIRED_CONFIGURATION_ANALYSIS.csv`在forced和各预注册calibration风险模式下分别评价FPR/FR/coverage/PASSABLE recall差：先同episode平均五seed，再抽物理配置组；不足3个相应类别/总体配置不报告CI。CI为描述性、未多重比较校正，不能据此宣称显著新方法。相同calibration风险目标不保证相同heldout风险，coverage差必须与FPR联合解释。`false_pass_episode_binomial_wilson_*`仅为假设episode IID的二项描述区间，不是物理配置cluster区间或跨配置保证。

confirmation是第四个独立角色，完全排除train/calibration/main test。每个预注册模型和预算用同一已冻结模型及main calibration参数独立评价；不重新拟温度/阈值、不选赢家。`CONFIRMATION_*`表与主表分开，状态列明缺样本/缺类别/未运行。确认集只做clean诊断，未声称已做其观测DR。
