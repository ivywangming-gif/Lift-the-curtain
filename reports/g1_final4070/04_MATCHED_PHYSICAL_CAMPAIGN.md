# 匹配物理实验结果

## 实验设计

预注册矩阵保持物理参数、地图、目标、reset、probe、FALCON 和安全监督不变，比较 R1/R2、左右 probe 和独立 seed。所有运行均为 `MATCHED_INITIAL_CONDITION`，不是 exact PhysX state replay。

R1/R2 的差异来自高层决策：

- R1：`RESPONSIVE -> CAUTIOUS_ADVANCE`；
- R2：只有 progress/stall/tracking 满足冻结阈值才继续，否则 `RETRACT`。

它们共享底层 probe 和动作集合，不是两个不同的物理交互技能。

## 结果账本

| 场景 | 有效 trial | Responsive | Blocked | Clearance | PointGoal | 路线结论 |
|---|---:|---:|---:|---:|---:|---|
| Empty control | 2 | 2 | 0 | 2 | 2 | 控制参考 |
| Nominal top-only | 6 | 6 | 0 | 1 | 4 | 局部响应不保证 clearance |
| Bottom-right anchor | 5 | 5 | 0 | 3 | 5 | 存在条件化可通行机会 |
| Bottom-left anchor | 6 | 6 | 0 | 0 | 6 | 安全到达多来自保守绕行 |
| Matched rigid | 5 | 0 | 5 | 0 | 5 | 2→1、撤回、绕行稳定 |

全体新 trial：24/24 runtime valid，0 条 invalid。

## S1--S4 账本

- Control reference：2 条。
- S2（局部效果未建立，rigid）：5 条。
- S3（局部响应但没有全身 clearance）：13 条。
- S4（局部响应并完成 whole-body clearance）：4 条。
- SAFE_RETRACT：24 条。

rigid 的 PointGoal 绕行不计为 S4；candidate clearance 失败但最终绕行到目标也不计为物理通过。

## 接触与评价信息隔离

部署分类只消费 deployment observation。rigid 的 offline evaluation 记录了约 14.245 N 的接触力峰值；curtain 试探的力字段为 0，但 runtime curtain displacement 非零。这个 0 不能解释为“没有接触”，因此帘幕响应仅作为 offline 形变证据，不能进入决策器。

所有 curtain top drift 都保持很小，rigid 没有 cloth displacement；这些字段只用于事后安全/物理评价。

## 主要发现

1. `RESPONSIVE` 在 nominal、bottom-left、bottom-right 柔性条件均出现，但 clearance 只有部分场景出现。
2. bottom-right 的 R1 结果在 seed45/46 和左右配对中重复出现，说明不是单个偶然标签。
3. nominal seed45 与 seed46 的差异说明执行/物理响应仍有随机或状态敏感性，不能把单次成功推广为通用能力。
4. rigid 结果最稳定，说明安全拒绝和地图绕行基线有效。

