# F 新数据的冻结模型评价

状态：真实运行，exploratory。模型为 `phase_a_valid/` 保存的 Phase-A checkpoint，没有 F 训练、再校准、阈值选择或模型选择。先验证30个checkpoint对420个E holdout概率的回放误差≤1e-6，实际最大误差为2.3283064365386963e-10。

## 数据与映射

11个完整新G1 episode：P5/B3/U3，只有8个有效标签进入下表。3个UNKNOWN保存预测但不进入任何监督指标。另2次启动失败没有完整history，由物理报告独立记录。11条F原始数据与E合并得到115条，但下表**仅评价F新增8条有效标签**。

nominal和gap05使用family_fold_1，bottom-right使用family_fold_0；对应完整family从未进入该fold监督、响应模型或标准化拟合。新1.95 m配置沿用对应family的holdout模型。原始E manifest SHA与训练freeze相符。

## 实际结果

以下学习模型每个独立初始化seed=4070/4071/4072均得到相同计数；不是把3次seed评价当成24个独立物理样本。forced阈值0.5与selective阈值0.2/0.8的计数在该小集合上相同。

| 方法 | TP/5 | TN/3 | False-pass/3 | False-reject/5 | UNCERTAIN/8 | coverage | balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic | 5 | 3 | 0 | 0 | 0 | 1.000 | 1.000 |
| GRU | 5 | 3 | 0 | 0 | 0 | 1.000 | 1.000 |
| TCN | 5 | 3 | 0 | 0 | 0 | 1.000 | 1.000 |
| ResidualGRU | 5 | 3 | 0 | 0 | 0 | 1.000 | 1.000 |
| M2 | 5 | 3 | 0 | 0 | 0 | 1.000 | 1.000 |
| always PASSABLE / E0（含时间修正） | 5 | 0 | 3 | 0 | 0 | 1.000 | 0.500 |
| always BLOCKED | 0 | 3 | 0 | 5 | 0 | 1.000 | 0.500 |
| E1 / E2（原样及时间修正） | 5 | 0 | 0 | 0 | 3 | 0.625 | 0.500 |

逐seed Brier范围：Logistic 0.001943–0.002391；GRU 2.230e-6–3.687e-6；TCN 4.663e-10–9.866e-9；ResidualGRU 1.412e-6–1.944e-6；M2 1.888e-6–2.330e-6。小Brier不是可靠校准证明：该集合小、条件被筛选，而且模型均非常自信。

`validation_zero_fp` 使用各Phase-A fold原先保存的验证阈值，完整数值保留CSV；该名称表示验证集经验0误放行，不是部署风险保证。UNKNOWN预测不判作对错。

## 科学解释与限制

在这8条有效新运行上，学习方法确定识别了3个BLOCKED，而E1/E2拒判了这3条；学习收益没有靠增加UNCERTAIN产生。但简单统计量Logistic与所有神经网络同样全对，**该补充集合不证明时间顺序、残差或机器人状态条件化的必要性**。M2没有独立可测的计数改进。

新增正例覆盖nominal/bottom-right的2.0/1.95 m四个尺寸×锚定配置，合并正式正例7个，仍只有两个结构family，且全部right。新长度各仅1个正例；左手仍没有正例真值。新姿态幅度小、没有质量/摩擦/阻力OOD验证，不能宣称未见结构或广泛材料泛化。

只有3个新负例，即使observed false-pass=0，95% Wilson上界约0.5615；不满足部署风险认证。Fscreening、尺寸与姿态阶段分开保存，冻结推理不是额外的未知物理family盲测。

## 复现

```powershell
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/build_dataset.py --supplementary-labels outputs/round4070_learning_traversability/physics/supplementary_labels.json
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/evaluate_frozen.py
```

主要证据：`F_FROZEN_EVALUATION.csv`、`F_EPISODE_LABELS.csv`、`PHASE_C_CONFIGURATION_COUNTS.csv`；逐预测、checkpoint SHA、E回放验证保存在 `outputs/round4070_learning_traversability/frozen_evaluation/`。原E基线文件未覆盖，合并36标签的规则另存 `PHASE_C_BASELINE_RESULTS.csv`。
