# Round 4070-F 物理配置与随机化契约

本文件区分已执行、待门禁的候选和未来方案；登记不等于完成。实际完成状态只以 `PHYSICS_VALIDITY_REPORT.md` 及新输出中的 summary 为准。

## 固定量与有效性

- 保持 E 的 Isaac/Isaac Lab 环境、G1 USD、冻结 FALCON ONNX、200 Hz physics、50 Hz control、单环境 headless；不调用 RTX render。
- 单次名义 0.10 m 探测，joint step 0.85 rad。实际 FK 行程不足 0.10 m 不是全局终止条件；安全监控仍生效。
- 名义长度 2.00 m、宽 1.20 m、厚 0.002 m 的已有 E physics gates 只用于相同物理配置。
- 新长度 1.95 m 必须分别在 nominal、bottom-right 配置通过无探测和探测 gate 才可进入 G1；不把失败场景写成 BLOCKED。
- gate 保留 top/anchor drift 10 mm、最大节点速率 5 m/s、位移 1.5 m、邻接伸长比 4、零显著地面穿透、probe 大于 matched gravity-only response。顶部滑轨仅允许世界 Y 方向移动；横向 X/Z 漂移仍有 10 mm 限值。

## 本轮有限采集设计

1. Screening：nominal 与 bottom-right、左右手，独立 seed 420101–420104。初始 lateral/yaw 均 -0.004 m/rad，复用历史 clearance 成功附近的姿态。
2. 新尺寸探索/匹配对照（旧CLI名 `confirm`）：长度 1.95 m 的 nominal 与 bottom-right 右手，seed 420201–420202；另有相同初姿和固定探测的 gap05 左右手对照，seed 420203–420204。新尺寸不是同条件独立确认。
3. 姿态扩展在任何 F 仿真前另行冻结：只有筛选出现可信 PASSABLE，才在 y=+0.006 m、yaw=+0.008 rad 下执行 nominal/right、bottom-right/right、gap05/right（seed 420301–420303）。此扩展不由模型测试性能决定。
4. 共 11 条 G1 候选是上限而非必须完成数；新尺寸 physics 不通过便跳过其 G1，未获得正例则不启动姿态扩展。不得用额外大量负例填充预算。
5. P1 六个不同构造先做无探测，seed 420071；只有初始化 gate 通过才做探测。只有两者通过才做 seed 420072 的有限重复。初始化失败则后续记录 withheld，不能当作已运行。P0 新尺寸遵循同样顺序。P1 没有 G1，不产生通行标签。

同一 physics config 的不同 seed 是重复，不自动代表 domain randomization。未显式施加参数噪声时，换 seed 不能证明物理随机性。本轮最多两个很近的初始姿态，不能据此宣称宽范围姿态 OOD。

姿态扩展同时使用新 seed，因此只是近邻姿态批次，不是同 seed 的姿态因果干预；初姿与种子/求解随机性无法被本设计分离。nominal 的 UNKNOWN 与 PASSABLE 分别表示证据不足和已验证通过，不能表述为已确认通行标签从阻挡翻转到通过。

## 独立物理配置的分组

分组键包含材质模型、宽/长/厚、density、stretch/shear/bend/damping、拓扑及具体锚点、障碍间距；机器人姿态可作为更细分组。相同配置全轨迹及所有窗口必须属于同一 split；左右手训练/留一侧是独立实验。新 trial 仅作为 supplementary exploratory 外部批次，不根据其结果回调已冻结模型超参。

场景元数据只存在 manifest 和离线评估中，模型白名单不得读取它们。physics-only 节点、接触真值、未来 clearance 只能验证场景/标签。

## 真实未覆盖的随机化与结构

宽度、面密度、厚度、阻尼、弯曲/拉伸、锚定位置、姿态范围、执行器噪声、观测延迟的联合随机化尚未执行。建议未来每一项先单变量 physics gate，再做 2–3 个离散水平的分组评估，不在当前小标签集上并行扩大全部维度。

已有 P0 覆盖顶部整边、顶部加单底角/双底角、真正四角、全周边、较重自由/受限、2/5/8 cm 后板。真正三角固定（其中一个顶部角也自由）没有在 F 新建或验证；顶部整边加一个底角不等于三角固定。P1 单侧受限是双片中一片额外底角约束，也不能重命名为真正三角拓扑。

P1 褶皱幅度 15 mm、波数 2，顶部保持平面；网状候选删除棋盘式内部面片，但为未校准 surface approximation。PVC 候选是三片独立 surface deformable sheet，不是 articulated surrogate，更不是真实 PVC 材料标定。滑轨采用动态 support 与 Y prismatic joint，未标定真实轨道摩擦；不声称真实材料泛化。

滑轨构造移动的是整根顶部 support，不是可独立滑动/聚拢的窗帘环；它验证整体顶部平移代理。原沿X手形probe产生的rail Y运动仅微米量级，不证明滑动开帘能力；另行登记的一次 mobility 检查直接给support施加 world-Y 0.5 N、0.5 s 的小力，属于physics-only自由度/限位检查，不是G1手部动作或通行验证。

## 两条左手标签补充（原11条完成后单独预注册）

原11条有效G1已冻结，正例仍全部来自右手。基于该物理标签覆盖缺口，预先登记且最多仅运行两条1.95m左手：`F_LEFT_COMPLETION_nominal_left_S420401`、`F_LEFT_COMPLETION_bottom_right_anchor_left_S420402`；phase=`F_LEFT_COMPLETION`，y/yaw=-0.004，固定0.85rad/名义0.10m探测。两场景1.95m均已通过无探测/探测物理门，原2.00m同动作左手也已安全完成。此补充只换已验证侧别，控制、动作、安全阈值不变。

选择依据是左手正例空白，不是模型误判或预测概率；无论结果如何仅两条，不追正例。先保持所有已训练模型冻结评价，原11行及其SHA保存于 `physics/supplementary_labels_before_left_completion.json` 和 `left_completion_preregistered.json`。新增行只追加，原Phase C评估保留。有效G1预注册上限从11增为13，加2次沙箱启动失败总attempt最多15，低于用户64上限。新增两条在所有P1及mobility物理分支完成后串行执行。
