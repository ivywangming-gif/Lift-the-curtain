# Round 4070-D：固定 10 cm G1 本体通行性判断报告

## 执行状态

- 研究状态：`INCOMPLETE_PROTOCOL_FEASIBILITY`；预注册 trial：250；核心目标：200 条有效重复。
- 覆盖状态：`INSUFFICIENT_POSITIVE_COVERAGE`；没有用空场或刚板控制替代柔性障碍正负样本。
- 实际输出目录：8；runtime-valid：8（核心 2）；安全验证输出：8。
- 核心协议匹配行：2；核心有效物理标签：0（PASSABLE=0，BLOCKED=0）。
- 核心 `PROBE_INCOMPLETE`：2；由于固定动作的实际 FK 未达到目标，按预注册规则保持 `UNKNOWN`，不自动判为 BLOCKED。
- 固定协议：`PROBE_10CM`，目标 0.10 m，固定世界方向，单次连续轨迹，零重试；实际 FK、post-contact 位移和 stop reason 分开保存。
- 安全验证结论：空场、刚板、正常布帘左右两侧均能安全撤回，但有符号实际 FK 位移约 -0.014 m 至 0.088 m（刚板受阻时可为负），未形成可用于 PASSABLE/BLOCKED 的完整 10 cm benchmark 行。
- 独立几何复核：对当前对称肩 pitch/肘关节参考及固定 G1 URDF 做 FK 扫描，在允许的 step 区间 [0, 1.20] rad 内最大前向投影约 0.0889 m；要实现 0.10 m 必须重新预注册探测姿态/轨迹，不能靠追加力矩或重试补足。
- 独立全臂 FK 校准复核（不属于正式 manifest）：left=0.093505 m/PROBE_INCOMPLETE/safe=True; right=0.100378 m/TARGET_REACHED/safe=True；右侧达到目标、左侧未达到，双侧校准门槛仍未通过，正式发射保持 withheld。
- 因此本轮没有启动其余核心正式重复；这是探测协议可执行性门禁触发的停止，不是因为发现少量误判而提前停止。

## 实际运行分组

| Condition | Registered | Output | Runtime-valid | Protocol-match | Safety-only | Incomplete | Labels P/B/U | Actual FK m (min..max) | L/R runtime-valid |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| FABRIC_NOMINAL | 20 | 2 | 2 | 2 | 2 | 2 | 0/0/20 | 0.076..0.088 | L:1, R:1 |
| FABRIC_BOTTOM_RIGHT_ANCHOR | 20 | 0 | 0 | 0 | 0 | 0 | 0/0/20 | - | L:0, R:0 |
| FABRIC_BOTTOM_LEFT_ANCHOR | 20 | 0 | 0 | 0 | 0 | 0 | 0/0/20 | - | L:0, R:0 |
| FABRIC_FOUR_CORNERS_ONLY | 20 | 0 | 0 | 0 | 0 | 0 | 0/0/20 | - | L:0, R:0 |
| FABRIC_FULL_PERIMETER_FIXED | 20 | 0 | 0 | 0 | 0 | 0 | 0/0/20 | - | L:0, R:0 |
| FABRIC_WALL_GAP_02 | 20 | 0 | 0 | 0 | 0 | 0 | 0/0/20 | - | L:0, R:0 |
| FABRIC_WALL_GAP_05 | 20 | 0 | 0 | 0 | 0 | 0 | 0/0/20 | - | L:0, R:0 |
| FABRIC_WALL_GAP_08 | 20 | 0 | 0 | 0 | 0 | 0 | 0/0/20 | - | L:0, R:0 |
| FABRIC_HEAVY_FREE | 20 | 0 | 0 | 0 | 0 | 0 | 0/0/20 | - | L:0, R:0 |
| FABRIC_HEAVY_CONSTRAINED | 20 | 0 | 0 | 0 | 0 | 0 | 0/0/20 | - | L:0, R:0 |
| EMPTY_CONTROL | 10 | 4 | 4 | 2 | 4 | 4 | 0/0/0 | 0.080..0.087 | L:3, R:1 |
| RIGID_CONTROL | 10 | 2 | 2 | 2 | 2 | 2 | 0/0/0 | -0.014..0.012 | L:1, R:1 |
| WALL_GAP_02_MATCHED_NO_WALL | 10 | 0 | 0 | 0 | 0 | 0 | 0/0/10 | - | L:0, R:0 |
| WALL_GAP_05_MATCHED_NO_WALL | 10 | 0 | 0 | 0 | 0 | 0 | 0/0/10 | - | L:0, R:0 |
| WALL_GAP_08_MATCHED_NO_WALL | 10 | 0 | 0 | 0 | 0 | 0 | 0/0/10 | - | L:0, R:0 |

标签顺序为 `PASSABLE/BLOCKED/UNKNOWN`；`Output` 包括安全校准/验证输出，不把它们当作正式 benchmark。

## 方法对比

- E0：eligible=0, false-pass=0/0, false-reject=0/0, coverage=None, balanced_accuracy=None
- E1：eligible=0, false-pass=0/0, false-reject=0/0, coverage=None, balanced_accuracy=None
- E2：eligible=0, false-pass=0/0, false-reject=0/0, coverage=None, balanced_accuracy=None

当前没有核心有效正负标签，故不报告准确率、false-pass、false-reject 或 recall；安全验证行上的 E0/E1/E2 仅作特征可计算性记录，不进入分母。

## Q1-Q10

- **Q1**：`NOT_ESTABLISHED`。没有足够的核心 PASSABLE/BLOCKED 标签，不能声称 E1 可靠或失败。
- **Q2**：`NOT_ESTABLISHED`。E0 与 E1 在少量安全验证历史上可计算，但没有公平核心标签分母可比较。
- **Q3/Q4**：`NOT_ESTABLISHED`。E2 尚未在有效正负样本上证明能同时降低两类错误。
- **Q5**：本轮没有证据支持按材料分别调参；但也没有足够跨条件数据证明统一规则可泛化。
- **Q6**：`NOT_ESTABLISHED`。没有完成 discovery/confirmation 的核心 seed 覆盖。
- **Q7**：当前最直接的限制属于 Type E/协议执行边界：固定关节参考在安全运行下没有产生目标 0.10 m 的实际 FK 行程；不能把该缺口归因于分类器特征。
- **Q8**：没有依据判断无需学习模型；在探测本身未形成有效标签前，学习模型结论也不成立。
- **Q9**：没有合法匹配的 false-pass/false-reject 案例；`FALSE_PASS_CASES.md` 与 `FALSE_REJECT_CASES.md` 保持空案例审计。
- **Q10**：当前证据不足以直接进入本体历史学习。下一步应先完成可重复的 0.10 m 探测标定，或正式接受 `PROBE_INCOMPLETE -> UNKNOWN` 并重新定义可评价问题。

PVC 和多片中开帘没有独立 physics gate，因此不进入 benchmark。
