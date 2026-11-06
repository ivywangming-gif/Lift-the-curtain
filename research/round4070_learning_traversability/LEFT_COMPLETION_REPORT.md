# 左手正标签覆盖的最终补充

实际追加两条G1：1.95 m nominal左手seed420401、1.95 m bottom-right左手seed420402。场景在新增前已通过物理门，模型和参数冻结。两条runtime/probe-valid，**独立真值均为UNKNOWN**。没有获得左手PASSABLE真值，不能把UNKNOWN当BLOCKED。

最终F13条=P5/B3/U5；E+F总117条=P7/B29/U65/控制16，核心有效监督标签仍36。此前Phase C实际训练使用的115条快照保留，未因为没有新监督标签的两条补充而再次训练。新数据仅留作未来严格分组响应学习候选。

## 冻结推理完成，监督分母为0

Phase-A的Logistic、GRU、TCN、ResidualGRU和M2各3个seed生成30行新预测。nominal使用family_fold_1、bottom-right使用family_fold_0，训练时未见对应family；没有拟合scaler、响应模型、分类器或阈值。E30 checkpoint/420概率回放最大误差仍为2.33e-10。

两条缺少真值，**不报告accuracy、false-pass率、recall、coverage或Brier**。输出CSV的监督n=0，所有这些率字段为空。以下仅是对两条episode各3个初始化的预测描述，不是6个独立物理样本或正确性结果：

| 模型 | 6次概率范围 | forced PASSABLE/BLOCKED | selective PASSABLE/BLOCKED/UNCERTAIN |
|---|---:|---:|---:|
| Logistic | 0.2599–0.9171 | 3/3 | 3/0/3 |
| GRU | 0.0025–0.7118 | 2/4 | 0/4/2 |
| TCN | 0.0001–0.9987 | 3/3 | 3/3/0 |
| ResidualGRU | 0.0074–0.9574 | 2/4 | 2/3/1 |
| M2 | 0.0036–0.7121 | 2/4 | 0/3/3 |

不同预测只能说明模型输出存在差异，不能判断哪种模型正确。UNKNOWN是缺失监督真值，UNCERTAIN是模型拒判，两者没有被混用。

## 数据保全和复现

原11条标签逐字节复制到 `physics/first11_frozen.json`，附SHA清单。原115条训练manifest、Phase-A/Phase-C checkpoint/config及首批F报告共331个保护文件hash全部不变；原11行仍以原顺序构成最终13行标签文件的前缀。

```powershell
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/build_dataset.py --supplementary-labels outputs/round4070_learning_traversability/physics/supplementary_labels.json --output-dir outputs/round4070_learning_traversability/dataset_final --report-prefix PHASE_C_FINAL_
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/build_dataset.py --supplementary-labels outputs/round4070_learning_traversability/physics/supplementary_labels.json --output-dir outputs/round4070_learning_traversability/dataset_left_completion --report-prefix LEFT_COMPLETION_ --trial-ids F_LEFT_COMPLETION_nominal_left_S420401 F_LEFT_COMPLETION_bottom_right_anchor_left_S420402
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/evaluate_frozen.py --f-manifest outputs/round4070_learning_traversability/dataset_left_completion/manifest.json --output outputs/round4070_learning_traversability/frozen_left_completion --report-prefix LEFT_COMPLETION_
```

原115条数据、原F8标签冻结评价文件均未覆盖。最终配置和逐条标签见 `PHASE_C_FINAL_CONFIGURATION_COUNTS.csv`、`PHASE_C_FINAL_F_EPISODE_LABELS.csv`；左2预测见 `outputs/round4070_learning_traversability/frozen_left_completion/LEFT_COMPLETION_F_FROZEN_PREDICTIONS.csv`。
