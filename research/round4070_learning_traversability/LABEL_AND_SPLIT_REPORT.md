# Round 4070-F 标签与划分契约

Phase A 样本：E 的 104 条 runtime/probe-valid。LABEL_VALID=44 包含16控制；非控制正式监督仅 PASSABLE=2、BLOCKED=26，UNKNOWN=60。UNKNOWN 的 `y=-1` 是缺失监督掩码，不是第三类。部署输出 UNCERTAIN 由概率/拒判规则产生，不监督预测 UNKNOWN。

## 冻结物理配置分组

canonical `group_id` 对同物理条件的所有种子、左右手、阶段、机器人初始姿态保持不变。所有 EMPTY_CONTROL 与三种 WALL_GAP_*_MATCHED_NO_WALL 合并为同一个 EMPTY_CONTROL 组，禁止把不同命名的空场当独立物理配置。其余按完整条件标识分组；物理设置、种子和动作参数完整放在 `physics_configuration_audit_only`，永不作为 seq 输入。

| group | n | PASSABLE | BLOCKED | UNKNOWN | 控制 P/B |
|---|---:|---:|---:|---:|---:|
| EMPTY_CONTROL | 14 | 0 | 0 | 0 | 14/0 |
| RIGID_CONTROL | 2 | 0 | 0 | 0 | 0/2 |
| FABRIC_NOMINAL | 12 | 1 | 0 | 11 | 0/0 |
| FABRIC_BOTTOM_RIGHT_ANCHOR | 10 | 1 | 0 | 9 | 0/0 |
| FABRIC_BOTTOM_LEFT_ANCHOR | 10 | 0 | 0 | 10 | 0/0 |
| FABRIC_BOTH_BOTTOM_CORNERS | 8 | 0 | 0 | 8 | 0/0 |
| FABRIC_HEAVY_CONSTRAINED | 10 | 0 | 0 | 10 | 0/0 |
| FABRIC_HEAVY_FREE | 4 | 0 | 0 | 4 | 0/0 |
| FABRIC_FOUR_CORNERS_ONLY | 4 | 0 | 0 | 4 | 0/0 |
| FABRIC_FULL_PERIMETER_FIXED | 4 | 0 | 0 | 4 | 0/0 |
| FABRIC_WALL_GAP_02 | 8 | 0 | 8 | 0 | 0/0 |
| FABRIC_WALL_GAP_05 | 10 | 0 | 10 | 0 | 0/0 |
| FABRIC_WALL_GAP_08 | 8 | 0 | 8 | 0 | 0/0 |

机器可读实际计数：`outputs/round4070_learning_traversability/dataset/scene_counts.csv`。不以同轨迹窗口随机划分，不按 seed 随机拆开同一物理配置。归一化、残差响应模型、监督训练和自监督训练仅使用该 fold 的训练组；预留验证组可用于声明的经验阈值校准，测试组不用于拟合或阈值选择。测试及验证组中的 UNKNOWN 也不进入训练。

## 小样本划分的实际限制

两个正式正例分别为 `E_PHASE_3_FABRIC_NOMINAL_R_05_S410029`、`E_PHASE_3_FABRIC_BOTTOM_RIGHT_ANCHOR_R_05_S410091`；均为右手、旧轮的 frozen_confirmation 阶段。直接沿用旧 discovery/confirmation 作训练测试会导致 discovery **没有正式正例**。本轮可进行固定超参数的两向物理配置 holdout 探索，将两个正例配置分开，但每 fold 的 train/test 正例都只有 1 个；没有足够独立正配置同时支持 train/validation/test。不能用该测试结果反复挑网络、阈值或训练轮数。

控制辅助监督要单列控制计数并只在训练侧使用，不得把控制的 14 个正标签宣称为柔性独立正配置。空场正常响应学习仅在训练控制上拟合。holdout controls 的时序不可用作监督或自监督；使用专用训练空场参考应在训练说明中注明。

左手正式正例=0，右手正式正例=2。左→右若只用正式标签是单类训练；右→左测试无正类，PASSABLE recall 和 balanced accuracy 必须留空/NA，不能把它们写成 0 或 1。仅在侧别切分而不按配置切分的结果，只能称 side-transfer diagnostic，不能声称未见物理配置泛化。

## 标签证据强度

同材质 fabric 允许 PASSABLE 与 BLOCKED；现有 BLOCKED 全是隐藏刚墙配置，不能证明已经覆盖同一材质各种锚定限制的两类标签。60 个 UNKNOWN 代表缺少独立真值，实际结局的低位移/PointGoal 失败不能补负标签。筛选条件后新增独立确认必须冻结条件与种子，不把 screening 当独立确认。

所有方法必须在同一个合法 probe 截止点比较，先报告 E0/E1/E2 原样复现及真实时间修正版。旧报告 E1/E2 低 stall 时长存在明确单位错误；新的学习比较须包含时间修正版，并披露正负样本的选择偏差。

## 评价约定

False-pass 分母为有效 BLOCKED 数，False-reject 分母为有效 PASSABLE 数；UNCERTAIN 单独统计，不计成 BLOCKED。PASSABLE/BLOCKED recall 的分母包括本类被拒判的样本；balanced accuracy 为两类上述 recall 均值，任何一类缺失则 NA。同时给出所有三种预测的真实计数、decision coverage 与 Brier，不得通过只计算已决策样本 accuracy 隐藏拒判。

仅有 26 个负例时，即使 observed false-pass=0，95% Wilson 上界约 0.1287；当前不能认证 5% 或更低部署风险。2 个正例的 recall 置信区间更宽。所谓固定风险 coverage 必须区分训练/验证经验约束与独立统计保证；若没有足够校准数据，应明确不能认证。

## 首批11条F后的Phase C配置与标签覆盖

F 实际新增 P5/B3/U3，合并 E+F 为 115 条：核心 P7/B29/U63，控制16。总 canonical groups 从13增至15；正式正例覆盖4个完整尺寸×锚定配置：

| 正例完整配置 | E PASSABLE | F PASSABLE | 合计 |
|---|---:|---:|---:|
| nominal，2.0 m | 1 | 1 | 2 |
| bottom-right，2.0 m | 1 | 2 | 3 |
| nominal，1.95 m | 0 | 1 | 1 |
| bottom-right，1.95 m | 0 | 1 | 1 |

这4个配置仅来自 **2个锚定结构家族**，尺寸只变化0.05 m；7个正式正例全部右手。没有独立左手正例，没有未见锚定家族的正例，没有不同宽度/阻力下充分的正例。姿态补充 +0.006 m / +0.008 rad 的成功是新姿态/新seed复核，不单独计为新物理障碍配置。

F 冻结评估按原结构 family 映射到 Phase-A 从未用于 fitting 的对应 fold（nominal 与 gap05 → family_fold_1；bottom-right → family_fold_0）。包含同 family 的新1.95 m尺寸也使用同一未见family fold；不能从两个模型中按 F 效果挑选。scaler、正常响应权重、分类权重、拒判阈值全部保留 Phase-A 数值。E manifest SHA 与 frozen_config 完全一致。

F 条件是依据历史物理可行性优先选择的探索补充；screening 与其后的尺寸/姿态扩展不能统称全新盲测条件。冻结推理提供独立新运行证据，但新seed、多随机初始化、同一材料尺寸微调，都不能冒充更多独立结构家族或更多独立测试样本。后续 Phase C 重训练须重新按结构 family 分组，并禁止读取 holdout family 的 UNKNOWN 原时序作自监督。

F 有效评价只有8条（P5/B3）；5种学习方法、3个seed均8/8正确且coverage1，Logistic同样全对，不能据此声称M2或时序结构必要。0/3 observed false-pass 的95% Wilson上界约0.5615，仍不能认证低风险部署。合并36标签的时间修正规则也需与对应 grouped 模型在完全相同测试分母比较，不能把F8结果直接对比E28或合并36结果。

## 最终117条与左手补充的划分边界

首批11条后的学习快照仍固定为115条；随后两个1.95 m左手补充均为UNKNOWN。最终总数据117条=P7/B29/U65/控制16，监督核心分母36与全部已训练模型保持不变。新左2分别并入既有 `FABRIC_NOMINAL|length=1.95`、`FABRIC_BOTTOM_RIGHT_ANCHOR|length=1.95` 完整配置；没有将新seed或左侧重命名为新物理组。正式左手正例在实际尝试后仍为0。

这两条从未进入Phase-A或Phase-C监督、标准化、空场响应或自监督拟合；仅以冻结Phase-A对应family holdout模型生成预测。它们缺少监督真值，评价分母为0，不能计算accuracy/误放行率或将预测BLOCKED当成可靠负标签；即使模型输出PASSABLE，也不能反过来补正标签。`LEFT_COMPLETION_F_FROZEN_EVALUATION.csv`明确保留n=0和空白率字段。

原first11标签、115条manifest、所有原checkpoint/config及首批F报告有SHA归档保护。后续复现导出必须使用新output-dir及report-prefix，不能把完整13条覆盖回原Phase C训练输入。最终117条供数据保存与未来分组响应学习；当前结论和重训结果仍基于冻结115条快照。
