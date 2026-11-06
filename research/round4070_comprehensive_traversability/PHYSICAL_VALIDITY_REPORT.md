# Round G 物理有效性审计（运行中）

本报告不把配置生成等同于实际验证；实际完成状态由 `outputs/round4070_comprehensive_traversability/campaign_progress.json` 与每条原始 `summary.json` 更新。当前完整研究尚未完成。

## 当前已核实状态（2026-11-06 15:57 UTC）

V5 已冻结并激活，manifest SHA256 为 `c311a6aa876bf5bae011148c3df5b6e87d986f16e6197bdb52bb67cfa828b200`。主数据已正式导入 **2 条 EMPTY 控制 PASSABLE（左右各1）**；原 10 条 V3 pilots 继续排除，不能混作主监督数据。累计13次实际G1启动，包括原10pilots、1次App前失败及2次有效控制；未启动条目不计预算。

EMPTY-left 首次启动因冻结计划 `physics_valid=False` 与合法运行门禁 `True` 的接口比较冲突，在 App 之前退出，无物理样本。原始失败与恢复后的精确旧源码字节保留；修复只允许经过真实三seed门禁的该证据字段转换，seed/机器人/动作等冻结条件不变。唯一受控 attempt02 完成，并通过全部独立审计：6068 个实际 200Hz 步及 reset/RNG 的 step0，共6069个几何状态；39个原生碰撞形状绑定、有限性、探测和安全恢复均有效。最终 hold 的实际碰撞后缘下界为1.15323471m，超过固定0.95m阈值；3416个动作约束状态无违例，最大逐步x位移1.00215mm。离线复算与已保存审计JSON完全一致。这仍是空场控制验证，不是障碍物通行性能。

EMPTY-right也完成5698个真实200Hz步及step0，共5699个几何状态，actual clearance、动作约束和安全恢复均经独立审核。新墙证据模块接入后的旧左右EMPTY全量只读复核仍为VERIFIED，原始文件未改写。RIGHT同时实际验证G1的post-finalize清理修复：127.739s自然退出，无cleanup timeout。

实际 native 几何提取已完成：27个 exact 形状与12个仅能证明分离的 native-vertex outer envelope；39个形状包括27 convex meshes、8 spheres、4 native zero-margin core cylinders。4个真实同tick link-pose样本及独立已知compound验证坐标与缩放。当前使用 `g1_native_collision_library_v3.json` 和 `g1_native_geometry_validation_v2.json`，详见 `BODY_COLLISION_AUDIT_PROTOCOL_V2.md`。12个外包络交叠只能NOT_VERIFIED，不伪称exact，也不能据此给无效穿透标签。

56个不同物理配置中已有17个通过三seed门禁（含B05）。A03/40703、B03/40702、C04/40701、E03/40701、E04/40703各只在late_stability未通过；D02/40702及40703只在contact_demonstrated未通过，其余checks全部通过。剩余覆盖依持久队列继续。CSV分别记录未执行、seed不足、门禁失败、装置/执行异常及各attempt证据路径，不把接触代理证据不足写成材料不稳定。

B05后续门禁实际验证了仅在FINALIZE后禁用交互式STOP渲染回调的清理修复：首两次为80.52s及70.963s正常退出，无30s清理超时，物理时序和门槛未变。每次启动现在先保存实际脚本、关键依赖及guard原始字节与SHA，因此pre-App失败也具可复现实证。

墙场景的实际Cube变换/尺寸/CollisionAPI/static属性、原生overlap精确碰撞路径，以及六轴原生进入面position/normal/distance对解析Box边界的闭环已实现并通过CPU反例测试。真实墙场景上的首次API执行仍待完成；缺失/禁用/动态/动画/变换或原生面不符都会在first-step前拒绝，不将请求参数metadata冒称实际墙证据。离线audit独立复查wall原始JSON、归档源码SHA与实际native二进制身份，只有通过才可审计body-wall和V5 BLOCKED证书。没有墙的EMPTY证据不受此额外证据要求影响。

## 历史检查点（13:24 UTC；不代表当前状态）

当时全seed聚合事实：EMPTY、A01、A02、B02、C01、D01、E02、F02 已各有三seed的24s有效门禁；A03完成1个seed。D02仅seed40701通过，另外两个seed的 `contact_demonstrated=false`（其余checks通过），因此尚不符合三seed门禁；不得仅看首seed概括全场景通过，也不把交互证据不足扩大成真实材料失效。原A01装置失败attempt完整保留。既有10条G1均为V3 pilot：缺少实际seed随机注入和完整机器人非穿透证据，不进入主学习，也不把其online正/负标签计作正式真值。V4主试验当时为0条。

V4新增实际seeded微扰与独立50Hz机器人/布节点记录；G1 admission暂因真实native collider几何审核装置未验证而暂停，环境physics-only队列持续运行。见 `BODY_COLLISION_AUDIT_PROTOCOL_V1.md`。原USD27个mesh碰撞位于父Xform；直接cooking API不支持该构造，故不能用子visual mesh或旧fallback bbox替代。property-query确实返回39个collider条目，native shape统计也非空，不能将API导出失败称为资产缺失。独立OmniPVD实际mesh提取仍在诊断。

有限宽墙覆盖固定局部corridor只证明局部直线路径阻挡；当前V4闭环动作允许横向/转向，不能据此直接证明完整动作BLOCKED。旧raw BLOCKED只保留诊断；formal标签由独立数据审核决定。

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
