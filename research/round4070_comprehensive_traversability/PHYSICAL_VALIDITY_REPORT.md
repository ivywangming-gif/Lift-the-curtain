# Round G 物理有效性审计（运行中）

本报告不把配置生成等同于实际验证；实际完成状态由 `outputs/round4070_comprehensive_traversability/campaign_progress.json` 与每条原始 `summary.json` 更新。当前完整研究尚未完成。

2026-11-06 11:28 UTC阶段事实：EMPTY三seed完整门禁通过且左右手G1 smoke各一条50Hz有效控制历史；A01在保留原装置失败attempt之后，新仪器版本三seed40701/40702/40703均24s全部门禁通过。其他物理配置仍待持续队列实际验证。这些结果不代表已有柔性通行正例。

## 冻结范围与旧成功配置

48 个指定场景名称具有 43 个不同的完整物理配置。A01/B01/C02/D05/E01/F01 是同一完整配置的语义索引，共享真实重复试验，并不构成 6 个独立物理条件。另有 12 个 Latin hypercube 组合及独立 EMPTY 控制配置。每个实际配置须获得 3 个不同 physics seed 的有效门禁，然后才能进入 G1；proxy 材料通过仿真门禁后允许 G1，但不声称真实材料已标定。

已核对 E/F 原始 summary，旧成功 nominal 的实际值是宽 1.2 m、长 2.0/1.95 m、25×33 网格、底边 0.02 m、密度 250 kg/m³、厚 0.002 m、stretch 500、shear 250、bend 0.02、damping 0.05，平移 x=0.85 m、yaw=π/2。G 的相应配置复用这些实际参数与原支持杆/附件构造，而非类默认的 1.4 m 宽与 20 行。

机器人朝 +X，因此世界 +Y 为机器人左侧。旧源码 `bottom_right` 实际映射世界 +Y；G B02 明确是机器人左下固定，B03 是机器人右下固定。这是坐标命名澄清，不是镜像标签。

## 真实几何与约束

`physics_scene_builder.py` 在内存中复用冻结 E 构造函数，原文件不改。新增布褶改变初始三维节点几何；网孔移除明确的内部网格单元并保留连续边界；PVC 是独立的多片 surface-deformable 几何代理；下缘配重是连接到底边节点的真实动态刚体，不能伪称仅提高密度。三角固定、四角固定、整边固定、单侧边缘分别记录实际低层 attachment 节点集合与世界坐标。

E02 是整根支撑沿世界 Y 的 prismatic joint；E03 是多个独立支撑刚体和关节，E04 另固定机器人左端吊点，E05 为两片独立滑动顶部。约束漂移在真实动态支撑局部坐标下计算，同时检查轨道 ±0.30 m、横向漂移与旋转；不豁免全部世界 Y 位移。它们是否实际成功必须由独立运行结果决定。

## 24 秒门禁

每个 seed：0–4 s 静置，4 s 对非约束节点注入 ±0.002 m/s 有界扰动；6–10 s 力控探针接近并推进，11–14 s 退回，14–24 s 释放与持续稳定。物理 200 Hz，几何与数值门禁 50 Hz，节点快照 10 Hz。探针最大力 3 N，修订仪器姿态力矩上限 0.03 Nm。该 physics-only 探针不是 G1 的 10 cm 动作。

检查 finite、节点速度≤5 m/s、实际网格边伸长比≤4、固定/相对支撑漂移≤10 mm、节点不低于地面 −5 mm、静置和末 4 s 速度<0.25 m/s、接触交互证据、退回清空、采样三角形与探针内部/墙内部相交，以及轨道约束。探针内部为 120 mm 立方体内缩 10 mm，墙内缩 5 mm。此为离散采样的几何非穿透证据，不是连续时间证明。

## 首次真实仪器诊断与保留

A01 seed40701 attempt01 实际跑完 24 s；finite、速度、伸长、锚定、地面、非穿透和持续稳定均满足，但原 70 mm 几何接触查询未命中，且自由旋转探针在末段离开目标。原始结果完整保留，未改写为 PASS，也不将其解释为布帘材料失效。

源码定位发现当前 IsaacLab `WrenchComposer` 的 permanent composer 缓存 link quaternion，直到 reset 才失效；在不 reset 的旋转探针上反复用 `is_global=True` 会使用过时姿态转换。修订仪器显式按当前姿态将世界力转入 body frame，并施加有界姿态恢复/阻尼。原实现、实际 trace 和修复理由保存在该 attempt 的 `apparatus_diagnosis.json`。

v3 在新运行前冻结了另一项审计修订：接触交互证据定义为 85 mm 近邻 box 内出现真实三角形，同时具有 >20 mm 布节点形变及 >0.05 N 有界施力响应，或真实 3D 墙近邻。原 trace 在 80 mm 查询已有三角形，而 70 mm 无。该规则是 **geometric interaction evidence**，不是 simulator contact-force truth；USD collision offset 原值另行记录，未确切解释的默认/自动 offset 不冒称已标定。严格内部穿透公差不变，新门禁与原门禁分版本报告。

部分墙的探针相交采用完整 15 轴 OBB/AABB SAT，包括 Y/Z 重叠与探针旋转，不能只凭 X 间距判穿透。

A01新运行的实际USD `contact_offset_attr` 与 `rest_offset_attr` 均返回 `None`，只有 `physics:collisionEnabled=True`。因此未读取到能确证85mm规则的显式solver offset，继续仅称几何交互代理证据。

## 运行与材料边界

当前材料均未校准到真实布、蚊帐或 PVC。短期/24 s 仿真稳定只支持该代理配置进入受保护的仿真 G1 试验。三 seed 门禁及 G1 仍可能暴露新的物理异常；此类试验独立标记无效，不作为监督标签。

每条运行由原有 Windows headless guard 监控 owned PID、阶段 heartbeat、显存/RAM/CPU/温度。空闲资源不足则 WAIT；不清理其他用户进程。成功 FINALIZE 后 Kit 清理超过 30 s 可由 guard 仅终止 owned PID，和未完成物理 rollout 的超时分开记录。
