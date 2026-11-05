# Round 4070：G1 主动本体试探研究

状态：`SUPERSEDED_BY_FINAL_REPORT`。最终计数、Q1–Q10 判定和基线结果见 `reports/ROUND4070_FINAL_REPORT_ZH.md` 与 `research/round4070_problem_discovery/RESULTS.md`。

## 当前最重要结果

继承 Round 4070 的 24 条 policy-in-the-loop 运行已逐条重建，均有完整有效 summary。记录显示柔性候选 whole-body clearance 4 条、rigid 5 条全部局部 `BLOCKED` 并安全绕行，空场控制 2 条通过。逐条场景数为 nominal 6、bottom-left 6、bottom-right 5，即柔性 17 条；历史某摘要的“4/16”分母不一致，本报告不沿用该分母。

旧部署记录没有保存每个样本的 FALCON effort source。adapter 可能选 `applied_torque`、`computed_torque`、`joint_effort_target`，或在全部 unavailable 时以零填充。Round 4070 新 runner 已增加来源字段，并在 probe effort unavailable 时拒绝作本体分类。新空场 `D00_EMPTY_RIGHT_S47` 已在真实 Isaac 中通过，PointGoal 完成，实际来源 `applied_torque`。

F2 split-free 探索发现当前 PhysX surface overlap 初始化不可靠。共面双片无 probe 就出现最高 11.19 m/s 与 0.46 m 位移；±4 mm 分层仍为 10.33 m/s、0.38 m；±12 mm 分层仍为 7.39 m/s、0.23 m。三者都不是可用负样本，且未运行 probe/G1。顶边 attachment 集合准确、无 NaN、stretch 与 ground gate 通过，但动力学不稳定。

## 方法与信息隔离

runner 输入白名单为 q/qdot/q_cmd、带 source 的 effort proxy、previous action、双手自身 FK、base/IMU 状态、Oracle 0/1/2 map、relative goal 和时间/phase。Curtain nodes、sim contact truth、物体类别、材料/锚点、真 clearance/outcome 仅在 evaluation/external safety 文件中。root pose 是允许的定位假设，不宣称已完成 RGB-D 定位。

详细逐 trial 重建、场景物理 gate、左右手结果、T0/T1/T2 特征、匹配反例、R1/R2 比较和科学假设判定由下列文件生成：

- `research/round4070_problem_discovery/RESULTS.md`
- `research/round4070_problem_discovery/EVIDENCE_MATRIX.csv`
- `research/round4070_problem_discovery/BASELINE_COMPARISON.csv`
- `research/round4070_problem_discovery/TRIAL_MANIFEST.csv`

最终结论只在独立 trial 分组支持充分后更新。本轮不训练新 WBC、RL、VLM 或大型时序模型。
