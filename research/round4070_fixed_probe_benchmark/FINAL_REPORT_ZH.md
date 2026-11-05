# Round 4070-E：固定命令本体通行性判断基准报告

## 结论

- 注册计划 `112` 条；实际有 summary 的输出 `104` 条。注册数量不等于实验完成数量。
- `RUNTIME_VALID=104`，`PROBE_VALID=104`，`LABEL_VALID=44`。
- 观测标签分布：PASSABLE=2，BLOCKED=26，UNKNOWN=60，CONTROL_PASSABLE=14，CONTROL_BLOCKED=2。
- 核心正式可评估行 `28`；`PASSABLE=2`，`BLOCKED=26`。
- 0.10 m 是固定名义命令，不是真实位移硬门槛；PointGoal 失败不自动产生 BLOCKED。

## 实际位移

- 观测到的实际手端点位移范围：`-0.002920..0.089980 m`。
- 接触受限、自由空间跟踪不足和安全停止是不同结果类别；只有独立物理/几何证据充分时才产生有效 BLOCKED。

## 阶段完成度

- `PHASE_0`: registered=6, observed=6, runtime=6, probe=6, label_valid=4。
- `PHASE_1`: registered=52, observed=44, runtime=44, probe=44, label_valid=12。
- `PHASE_2`: registered=32, observed=32, runtime=32, probe=32, label_valid=12。
- `PHASE_3`: registered=10, observed=10, runtime=10, probe=10, label_valid=4。
- `MATCHED_CONTROL`: registered=12, observed=12, runtime=12, probe=12, label_valid=12。
- Phase 1 formal core rows: registered=44, observed=44。
- Phase 1 未执行的 8 条非核心控制行未计入正式失败；Phase 0 控制与 12 条 matched controls 已实际完成。
- Phase 1+2 累计达到每侧至少 4 次的 P0 条件：FABRIC_BOTH_BOTTOM_CORNERS, FABRIC_BOTTOM_LEFT_ANCHOR, FABRIC_BOTTOM_RIGHT_ANCHOR, FABRIC_HEAVY_CONSTRAINED, FABRIC_NOMINAL, FABRIC_WALL_GAP_02, FABRIC_WALL_GAP_05, FABRIC_WALL_GAP_08。
- 尚未达到每侧 4 次累计重复的 P0 条件：FABRIC_FOUR_CORNERS_ONLY, FABRIC_FULL_PERIMETER_FIXED, FABRIC_HEAVY_FREE。
- Phase 3 独立确认条件：FABRIC_BOTTOM_LEFT_ANCHOR, FABRIC_BOTTOM_RIGHT_ANCHOR, FABRIC_HEAVY_CONSTRAINED, FABRIC_NOMINAL, FABRIC_WALL_GAP_05。

## 科学问题逐项回答

1. **E1 能否判断通行性？** 当前有效标签集为 PASSABLE=2、BLOCKED=26；另有观测 UNKNOWN=60。在这组有效标签上可以做有限判断，但必须把 UNKNOWN/覆盖率一起报告。
2. **E0/E1/E2 谁最好？** 不存在严格支配者：E0 覆盖=1.0000、误放行=26/26；E1 覆盖=0.2143、误放行=4/26；E2 覆盖=0.1071、误放行=1/26。E0 提供最高覆盖但放行全部有效 BLOCKED；E2 的误放行最低但覆盖最低；E1 是中间权衡。
3. **E2 是否解决了 E1 不能解决的真实场景？** 可审计的 E2 rescue trial 数为 `0`；当前没有可证实的 rescue 案例。
4. **E2 仍然失败的匹配条件？** `26` 条有效标签上 E2 未输出正确确定标签；其中输出 UNCERTAIN=25、PASSABLE=1、BLOCKED=0。这里的‘错误’包含未给出正确确定标签，不等同于全部是误放行；`E_PHASE_1_FABRIC_WALL_GAP_02_L_01_S408248`, `E_PHASE_1_FABRIC_WALL_GAP_02_R_01_S408249`, `E_PHASE_1_FABRIC_WALL_GAP_02_L_02_S408255`, `E_PHASE_1_FABRIC_WALL_GAP_02_R_02_S408256`, `E_PHASE_1_FABRIC_WALL_GAP_05_L_01_S408279`, `E_PHASE_1_FABRIC_WALL_GAP_05_R_01_S408280`, `E_PHASE_1_FABRIC_WALL_GAP_05_L_02_S408286`, `E_PHASE_1_FABRIC_WALL_GAP_05_R_02_S408287`
5. **失败来自信息未利用还是探测不足？** 柔性 UNKNOWN 行不能作为算法错误；它们表示单次局部探测没有独立获得整体 clearance 证据。隐藏刚性墙的有效 BLOCKED 则有独立几何/接触证据，属于可评价负例。
6. **结构条件结果：** `FABRIC_BOTTOM_RIGHT_ANCHOR`: PASSABLE=1, BLOCKED=0；`FABRIC_NOMINAL`: PASSABLE=1, BLOCKED=0；`FABRIC_WALL_GAP_02`: PASSABLE=0, BLOCKED=8；`FABRIC_WALL_GAP_05`: PASSABLE=0, BLOCKED=10；`FABRIC_WALL_GAP_08`: PASSABLE=0, BLOCKED=8
7. **PVC/中开帘：** P1 physics-only 记录数 `8`，状态 `FAIL_PHYSICS_SANITY`；real PVC modeled=False，不能把 surrogate 结果称为真实 PVC。
8. **是否足够支持学习式判断器？** `是，已有至少一组有效正负标签，但仅适合作为小规模工程基准`；不能把 UNKNOWN 或控制样本当作柔性正例。
9. **能否严格认定 E1 不足？** `可以在当前有效集合上报告其误放行/误拒绝证据，但不能外推到未覆盖物理条件`。
10. **下一步算法需求：** 先扩充独立 whole-body clearance 正例、约束结构负例和 matched near-confounder；保持同一单次探测截止点，加入不确定性输出与侧别/姿态鲁棒性验证，再决定是否需要学习模型。

## E0 / E1 / E2

- E0: eligible=28, false-pass=26, false-reject=0, coverage=1.0, balanced_accuracy=0.5.
- E1: eligible=28, false-pass=4, false-reject=0, coverage=0.21428571428571427, balanced_accuracy=0.5.
- E2: eligible=28, false-pass=1, false-reject=0, coverage=0.10714285714285714, balanced_accuracy=0.5.

## 证据边界

- P0 单片场景的 physics gate 与 G1 运行分别审计。
- P1 多片/条带尝试只作为 physics-only 证据；失败不能阻断 P0。
- `LABEL_VALID` 只在独立 clearance/contact/geometry 证据充分时为真；否则保留 UNKNOWN。
- 历史 Round 4070 正例不替代 E 内部试验；只有 E summary 中的机器人包络 clearance 才进入 E 正例标签。
- matched positive/negative CSV 按 probe side 与 repeat_index 配对；控制 seed 独立，因此它是 matched-condition control，不是同 seed 复现。
- 详细执行、标签和左右手比较见同目录 CSV 报告。
