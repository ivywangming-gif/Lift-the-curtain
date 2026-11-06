# D02 interaction evidence: read-only diagnosis

本报告于 2026-11-07 HKT 根据已保存的真实 physics-only 输出进行事后诊断。没有启动 Isaac/GPU、追加 trial、修改场景/门禁/阈值、覆盖 raw，或将失败改成 PASS。诊断不产生 G1 标签和模型结果。

**结论：D02 40702/40703 失败的直接原因是全局位移没有超过 20 mm；记录更符合探针受到阻力、近处布料被背墙/接触壳限制的情形，不能据此断言没有原生接触。40701 的通过依赖远处边缘向后折弯，而非探针附近出现 20 mm 推进。** 接触壳/背墙夹持是有依据的解释，仍需独立仪器诊断确认；现有证据不足以识别真实接触对象及冲量。

## 原始依据与比较范围

结果表：[D02_INTERACTION_EVIDENCE_DIAGNOSIS.csv](D02_INTERACTION_EVIDENCE_DIAGNOSIS.csv)。机器可复核记录：[D02_INTERACTION_EVIDENCE_DIAGNOSIS.json](D02_INTERACTION_EVIDENCE_DIAGNOSIS.json)，包含 7 个实际 run、38 个 raw/config 文件路径与 SHA256、逐 run 原执行源码哈希、局部节点 ID、定位方法和本机 API 文件哈希。

以下 7 个 run 保存的四项源码 SHA 完全一致：`physics_gate.py`、`physics_scene_builder.py`、`physics_scenes.py` 和 guarded runner。gate SHA 为 `355feca00f394755ef7f98877d8d037c17b084a642a4b3e06f986263dfbb50e0`。这是已记录源码的匹配，不是完整依赖环境同一性的证明。

原始根目录为 `outputs/round4070_comprehensive_traversability/physics/`。A01 seed40701 使用 `attempt02`，其余表内使用 `attempt01`。旧 A01 seed40701 attempt01 没有相同的源码记录，且是旧装置失败，不合并作同版对照。EMPTY 40701/40702 没有源码哈希记录，故选有相同四项哈希的 EMPTY 40703。

| 场景/seed | 原门禁 | 接近采样数 | 接近期间最大全局位移 mm | 探针最大 x m | 近处增量位移 mm |
|---|---|---:|---:|---:|---:|
| D02/40701 | PASS | 215 | 33.190 | 0.770203 | 1.713 |
| D02/40702 | FAIL: contact_demonstrated | 211 | 2.792 | 0.770165 | 0.267 |
| D02/40703 | FAIL: contact_demonstrated | 211 | 2.812 | 0.770203 | 0.351 |
| A01/40701 attempt02 | PASS | 175 | 216.494 | 0.931939 | 175.162 |
| A01/40702 | PASS | 171 | 205.841 | 0.931966 | 174.993 |
| A01/40703 | PASS | 172 | 217.345 | 0.932404 | 175.631 |
| EMPTY/40703 | PASS（无接触要求） | 0 | N/A | 0.950000 | N/A |

“接近”严格沿用 `probe_touch_triangles > 0`，仅代表布料三角形与半宽 0.085 m 的探针检测盒重叠，不能称为测得接触。“近处增量位移”是事后定位指标：在保存的 6 s 状态，选取距探针初始 Y/Z 各小于 0.1 m 的 12 个布料节点，计算 8–12 s 保存状态相对 6 s 的最大欧氏位移。它不参与原门禁，也不拟作为从本次结果选出的新阈值。

## 失败与通过的具体机制

当前源码 `scripts/round4070_g/physics_gate.py:110` 定义施加的 PD 力 `80*(target-p)-8*vprobe`，并限幅 3 N；记录的 `probe_force_n` 是该指令力大小，**不是力传感器或原生 contact report**。第 120 行计算所有布料节点相对初始化状态的最大位移；第 131 行计算扩大的接近盒；第 142 行要求在 6–14 s 内满足“接近、全局位移 >20 mm、指令力 >0.05 N”，或探针接近背墙。

- 三个 D02 run 都完整运行 24 s。40702/40703 的其他门禁全部通过，唯一失败是 `contact_demonstrated`。两者接近窗口均为 8.26–12.46 s，指令力达到 3 N，目标位置与实际探针位置最大差约 180 mm。探针停在 x≈0.7702 m；同版 EMPTY 到达 x≈0.9500 m，A01 到达 x≈0.932 m。没有“探针根本没有前进”的迹象。
- 40702 最大全局位移 2.792 mm 早在约 4.02 s 已出现，主要是悬挂/落地后的远端变化。接近期间局部相对 6 s 的新增响应最大仅 0.267 mm；40703 为 0.351 mm。因此这两条既有记录无法满足 20 mm 判据。
- 40701 在 9.96–12.50 s 有 128 个 50 Hz 采样满足原判据。10 s 最大位移节点 697 位于约 `(0.8282, 0.4963, 0.3278)` m，距探针约 0.803 m，x 方向位移约 **−21.83 mm**。11 s 最大位移节点 749 位于约 `(0.8204, 0.5910, 0.1990)` m，距探针约 0.962 m，x 方向位移约 **−29.60 mm**。这是远侧边缘向机器人方向、远离背墙的折弯。其探针附近的增量响应仍仅 1.713 mm。

这说明原全局指标会把远处折弯与接近盒/推力指令在同一时刻出现作为通过证据。它没有证明该大位移发生在接触区域，也未单独证明由探针引起。40701 仍保留原 PASS，但不能将它表述为“三种子相同的局部接触证据，仅两个随机漏检”。

## 几何对照及可判定边界

以下是依据记录的 probe pose、0.12 m box 定义、真实布料节点/网格拓扑以及原 wall metadata 进行的离线几何计算。原记录的墙前面为 x=0.870 m、初始布料面为 x=0.850 m。**这些旧 physics-only 输出没有后来新增的墙实体原生 readback，故此处墙边界不是补做的原生形状认证。**

| D02 seed | 全程保存状态中 cube 前缘至墙前面最小 x 间隙 mm | 10 s 布料至 cube 最小 L∞ 扩张间隙 mm | 10 s 局部布料至墙前面间隙 mm |
|---|---:|---:|---:|
| 40701 | 39.695 | 18.960 | 19.731–19.978 |
| 40702 | 39.828 | 19.855 | 19.821–19.996 |
| 40703 | 39.716 | 19.731 | 19.706–19.998 |

10 s 三者布料三角形与半宽 0.06 m 的实体 cube 都没有严格内部相交。L∞ 间隙通过同时扩大 cube 三个局部半轴，查找三角形首次重叠求得；它不是欧氏最短距离，也不是原生接触壳厚度。节点/pose 快照为 10 Hz，原门禁 telemetry 为 50 Hz，不能据此声明 200 Hz 或连续时间无接触/无穿透。

几何结果反对“探针实体直接撞在背墙上”的解释；同时显示前后约 20 mm 的几何空隙仍存在。若自动接触/静止偏置占据这些间隙，局部向 +X 运动会被明显限制，这与微小局部响应、探针停滞及 3 N 指令相容。然而原 `mesh_collision_offsets` 的 contact/rest offset 均为 `null`，不可把约 20 mm 的可见间隙直接认作已测得的 offset。

也不能宣称当前判据在 D02 数学上绝不可能满足：全局欧氏位移可通过远处侧向/向后变形超过 20 mm，40701 就提供了实际例子。准确结论是**判据与局部受限接触不等价，且存在远处形变导致通过的依赖**。

## 原生 API 核实

本机版本路径以 `C:/conda_envs/isaaclab/Lib/site-packages/isaacsim/` 为根；文件完整路径与 SHA 见 JSON。

| 本机文件/行 | 已核实能力或限制 |
|---|---|
| `exts/isaacsim.core.api/isaacsim/core/api/sensors/rigid_contact_view.py:249` | `get_net_contact_forces(..., dt=1.0)`；实际步长应明确传 `.005`。默认 `dt=1` 返回底层冲量，不能直接当 N。 |
| 同文件 `:293` | `get_contact_force_matrix` 提供按过滤对象的 rigid contact 汇总；不能据接口存在推断会包含 deformable 接触。 |
| 同文件 `:101` 构造参数及 `:123–124` 上下文设置 | `disable_stablization=True` 是默认值，构造时会关闭稳定化。独立诊断必须明确 `disable_stablization=False`，避免把测量封装误作完全不改物理设置。 |
| `extscache/omni.physx-107.3.26+107.3.3.wx64.r.cp311.u353/omni/physx/bindings/_physx.pyi:1365,1466` | 原生逐步 contact report/订阅接口存在，缓冲仅对当前一步有效；并不证明 deformable contact 支持。 |
| `extscache/omni.physics.tensors-107.3.26+107.3.3.wx64.r.cp311.u353/omni/physics/tensors/impl/api.py:5094–5490` | 当前 `DeformableBodyView` 提供节点位置/速度、拓扑、变换等，没有 contact-force 或 native contact/rest-offset getter。 |
| `extscache/omni.usd.schema.physx-107.3.26+107.3.3.wx64.r.cp311.u353/plugins/PhysxSchema/resources/generatedSchema.usda:516,530` | 未显式设置的 contact/rest offset 由仿真决定；rigid rest 默认零，不能套用于布料。 |

匹配版本的 [NVIDIA Omni Physics 107.3 limitations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/current_limitations.html) 明确列出 deformable body contact reports 不受支持。因此，报告缺失或 tensor 输出零不能作为无布料接触证据。未来若尝试 rigid probe 的 contact view，必须先用独立的已知接触对照验证其是否覆盖布料；本次没有运行该验证。

[107.3 Beta Deformable 文档](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/deformables_beta/deformable_beta.html) 说明 surface deformable 使用 simulation mesh 碰撞，并提供显式 contact/rest-offset 示例；示例数值不是当前 raw 的运行时默认值。

## 独立预注册诊断建议（未实施）

1. 新建隔离的 physics-only 诊断协议，冻结原三个 seed、同一运动/力限幅/时间表、原 PASS/FAIL 门槛。比较 D02、A01、EMPTY，另列“相同背墙、无布料”机制对照；新增对照不得混入原矩阵或改写历史门禁。
2. 每个真实 200 Hz step 保存 probe pose/线速度、实际已施加世界系力向量、质量/重力/阻尼 readback、布料节点，以及接触 API 可用性和 native offset 的可验证来源；无法获得的量明确缺失。独立验证记录器不会额外推进仿真。
3. 预先固定局部区域和时间窗，分别报告局部向前响应、全局位移、远处折弯、接近时间、推进滞后、释放和尾段稳定性。不要用本次看到的 0.267/0.351/1.713 mm 反向选择“能使 D02 通过”的新位移阈值。
4. 若布料 contact report 不可用，可诊断积分残差 `m·Δv − ΣF_known·Δt`；必须核清施力时间对齐及全部已知外力/阻尼。它表示总未解释冲量，不自动给出布料/墙接触对象，也不是直接接触传感器。先在 EMPTY 与 rigid-wall-only 对照核验其解释范围。
5. 原 40702/40703 保持 FAIL。只有独立诊断完成后才能讨论是否需要另版本装置/证据协议；任何修订都要重新冻结并单列新试验，不能以本报告把旧失败追认成 PASS。
