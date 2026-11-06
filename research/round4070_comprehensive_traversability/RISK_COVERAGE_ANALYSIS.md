# G 风险、覆盖率与校准

范围：`G_ACTION_CONTRACT_BENCHMARK`。真实已评价 fit：0 / 预定任务 1280。状态 `{'BLOCKED_WITH_REASON': 1280}`。

forced=0.5、固定拒判=.2/.8；温度拟合和风险阈值只用独立 calibration 组。阈值采用预先固定的 .025 网格，目标误放行率为 0/.01/.025/.05/.1/.2，同时约束校准 false-reject≤.1，再最大化确定覆盖率。该约束是校准样本上的经验约束，不是部署安全认证。缺任一校准类别则不拟合温度/经验阈值。

label-conditional split conformal 的 alpha=.1；有限样本分位数超出可用秩时使用 q=1 的保守集合，双类/空集合都输出 UNCERTAIN。物理配置采集存在选择偏差，交换性未验证，不宣称真实世界覆盖保证。

FP分母为全部可靠BLOCKED，FR为全部可靠PASSABLE；拒判不算TN/TP。selective risk只在已决策集合计错误，必须同时报告coverage与PASSABLE recall。UNKNOWN永远不进监督分母。工程规则的0/.5/1是决定编码，不作为概率报告Brier或校准。

`RISK_COVERAGE_POINTS.csv`、`CALIBRATION_BINS.csv`、各fit的calibration.json均来自实际执行。只有表头表示尚无实际可用评价，不是零风险。`PAIRED_CONFIGURATION_ANALYSIS.csv`先在同episode上平均五个seed的配对误差，再按物理配置聚类bootstrap；不足3个负配置不报告CI，seed不是独立物理重复。
