# BIMANUAL_V1 数据与场景审计

状态：STAGE_1_ONLY。未启动 Isaac，未修改 manifest、V5/V6、E1_FIXED 或两个暂停标志。

## 旧状态

- 旧 A-F 命名场景：48；独立物理配置：43。
- 旧场景 physics gate：{'PHYSICS_VALID': 32, 'PHYSICS_GATE_FAILED': 16}。
- 当前合法 G1 历史：21（已接纳 18 + collision pending 3）。
- 已知标签（含 EMPTY 控制）：14。
- 排除 EMPTY 控制的核心标签：12，PASSABLE 8，BLOCKED 4。
- 合法 LABEL_UNKNOWN：7。
- 当前 split manifest 的独立物理配置：8。
- 合法双手同步 episode：0。

当前活动描述文件：research\round4070_comprehensive_traversability\CURRENT_CAMPAIGN.json；manifest：campaign_manifest_v5.json；descriptor 与冻结 manifest SHA 匹配：True。输出目录镜像不存在不影响研究目录 descriptor；审计没有伪造缺失文件。
审计输入文件已登记 8/8 个，并保存路径与 SHA256 到 BIMANUAL_PROGRESS_CHECKPOINT.json。

## 新矩阵与资产

- 总行数：104（旧参考 48 + 新候选 56）。
- 新候选物理配置 ID 数：50。
- 资产状态：{'RUNTIME_CONFIG_AND_BUILDER_SOURCE_NO_STATIC_SCENE_ASSET': 48, 'STATIC_GEOMETRY_CANDIDATE_NOT_RUNTIME_BUILT': 49, 'MISSING_EXTERNAL_MATERIAL_ASSET': 3, 'STATIC_DOOR_CANDIDATE_REQUIRES_DOOR_BUILDER': 4}。
- 双手 physics 状态：{'LEGACY_SINGLE_HAND_ONLY': 48, 'NOT_RUN_PAUSED': 56}。
- 按项目规定枚举的总门禁状态：{'PHYSICS_VALID': 32, 'PHYSICS_INVALID': 16, 'EXECUTION_BLOCKED': 56}；新候选统一为 EXECUTION_BLOCKED，不能视为已建模或已通过 physics gate。

新增 JSON/OBJ 是可检查的静态候选几何和配置描述，不是 native PhysX scene asset，也不构成 contact、release、recovery 或 penetration 证据。

## 关键缺口

1. 4 cm、6 cm 隐藏墙和底角/三角固定等新条件没有旧物理证据。
2. PVC 和网状场景仍是 articulated/surface proxy；不能宣称真实材料验证。
3. 门场景需要独立 hinged-door builder、铰链动力学和 curtain/composite 双标签记录，当前不存在。
4. 旧 gate 是单手证据，不能转化为双手同步 gate。
5. B10/V6 native collider、STOP/retract physics-only 验证和独立安全审批仍未通过。
6. 预算下限超过当前冻结 V5 剩余额度，需要正式预算 addendum 和批准。

正式 benchmark 只有在所有必需场景完成 3-seed physics gate、双手 STOP/retract 验证、B10/V6 安全审批和预算审批后才可启动。
