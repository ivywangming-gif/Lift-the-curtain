# Round 4070-G：E1_FIXED If–Else 基线回放报告

回放时间（UTC）：`2026-11-08T03:24:45.869948+00:00`
基线版本：`E1_FIXED_G_V1_REAL_DT`
状态：`IN_PROGRESS_NOT_COMPLETE`

## 范围

本次只读回放覆盖当前主协议的 **21** 条合法 G1 历史：主 dataset manifest 中已接纳的 **18** 条，加上仍处于碰撞/action-tube 审计挂起状态的 **3** 条。10 条 superseded pilot 不进入基线。B10 左侧与 D01 左右侧继续保持 `LABEL_UNKNOWN`，没有被 E1 输出改标。

两条 EMPTY 控制（左右手各一条）只用于冻结同手参考响应，单独审计，不进入核心真假标签分母。所有决策使用同一套 252 通道、同一观察窗口、真实记录时间；stall 时长采用 `diff(actual_time)` 积分，不使用固定采样周期替代。

## 核心结果

当前数据中的已知标签共 **14** 条（原始计数 PASSABLE=10、BLOCKED=4），其中包含左右两条 EMPTY 控制。排除控制后，非控制核心有效标签为 **12** 条，其中 PASSABLE **8**、BLOCKED **4**；另有核心 `LABEL_UNKNOWN` **7** 条。控制不进入真假判断分母。

| 指标 | 数值 |
| --- | ---: |
| E1 原生 `RESPONSIVE` | 15 |
| E1 原生 `BLOCKED` | 4 |
| E1 原生 `UNCERTAIN` | 2 |
| 核心决策覆盖率（排除 UNKNOWN/控制） | 83.33% |
| PASSABLE recall | 100.00% |
| BLOCKED recall | 50.00% |
| balanced accuracy | 75.00% |
| false-pass | 0.00% |
| false-reject | 0.00% |
| UNCERTAIN | 2 |

## 按手侧

| 分组 | 总行数 | 核心有效 | UNKNOWN | UNCERTAIN | 覆盖率 | PASSABLE recall | BLOCKED recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| side left | 11 | 4 | 6 | 1 | 75.00% | 100.00% | 50.00% |
| side right | 10 | 8 | 1 | 1 | 87.50% | 100.00% | 50.00% |

## 按场景族

| 分组 | 总行数 | 核心有效 | UNKNOWN | UNCERTAIN | 覆盖率 | PASSABLE recall | BLOCKED recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| family A | 8 | 4 | 4 | 0 | 100.00% | 100.00% | NA |
| family B | 1 | 0 | 1 | 0 | NA | NA | NA |
| family C | 2 | 2 | 0 | 0 | 100.00% | 100.00% | NA |
| family D | 6 | 4 | 2 | 2 | 50.00% | NA | 50.00% |
| family E | 2 | 0 | 0 | 0 | NA | NA | NA |
| family F | 2 | 2 | 0 | 0 | 100.00% | 100.00% | NA |

同一物理配置 D03 的不同 seed/手侧记录保留为独立历史；其 E1 输出存在 `BLOCKED` 与 `UNCERTAIN` 的混合，未进行阈值调参或事后修正。D01 的两条 `BLOCKED` 输出仍只是对 UNKNOWN 历史的规则响应，不能当作有效 BLOCKED 真值。

## 按物理配置聚合

下表保留每个 episode 的独立记录，同时给出同一 `physical_config_id` 内的有效行数、UNKNOWN 数量与 UNCERTAIN 数量；它不是把训练 seed 当成独立物理配置的显著性检验。

| 分组 | 总行数 | 核心有效 | UNKNOWN | UNCERTAIN | 覆盖率 | PASSABLE recall | BLOCKED recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| config 032a27e19daa922f | 2 | 1 | 1 | 0 | 100.00% | 100.00% | NA |
| config 0f3241eff410af15 | 2 | 2 | 0 | 0 | 100.00% | 100.00% | NA |
| config 29bc5415aada6264 | 2 | 1 | 1 | 0 | 100.00% | 100.00% | NA |
| config 411fe3bc3a3eb695 | 2 | 1 | 1 | 0 | 100.00% | 100.00% | NA |
| config 450526391af8cd60 | 2 | 0 | 0 | 0 | NA | NA | NA |
| config 47ef0b84e422c3d9 | 2 | 2 | 0 | 0 | 100.00% | 100.00% | NA |
| config 63b80486f73ac370 | 2 | 0 | 2 | 0 | NA | NA | NA |
| config 7e95a1922fbb7042 | 2 | 1 | 1 | 0 | 100.00% | 100.00% | NA |
| config b524c675bce62908 | 4 | 4 | 0 | 2 | 50.00% | NA | 50.00% |
| config fc4d2265ce27db5d | 1 | 0 | 1 | 0 | NA | NA | NA |

## OOD 与兼容性

当前输入没有可用的 `test`、`confirmation` 或 `heldout_posture` 标签历史，因此 OOD 结果为不可评估，不报告泛化结论。
与现有 `scripts/round4070_g/engineering_rules.py` 的 E1 决策回归：**PASS**；其 `PASSABLE` 仅是数值指标兼容编码，不是 E1 原生语义。
数字特征逐项兼容性：**PASS_FLOAT_PRECISION_TOLERANCE**（最大绝对差 `3.7478e-06`，容差 `1e-05`；差异来自旧适配器的 `float32` 与冻结回放的 `float64` 计算精度，不改变任何 E1 决策）。
契约与回归测试：**PASS**。

## 研究状态

本次产物只冻结并回放官方工程 if–else 基线；没有启动新的 G1，没有删除或修改 `PAUSE_G1.flag`/`PAUSE.flag`，也没有宣称完整 benchmark、学习方法优势或研究完成。后续学习方法仍须在同一冻结基线上、同一历史和标签合同下进行。

产物目录：`outputs/round4070_comprehensive_traversability/if_else_baseline_61154c50277590796dc1`
