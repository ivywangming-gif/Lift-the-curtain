# Round 16 匹配实验与策略比较

## 实验纪律

本文件只整理已有的、满足或接近匹配条件的 Round 13--15 artifact；本轮没有为了填表重新启动 Isaac。所有 deployment 判断都只允许使用 `q/qdot/q_cmd`、effort proxy、previous action、由自身 FK 得到的手部运动、base/IMU 历史、已知地图和相对目标。布料节点、真实 contact、物体类别、真实 stiffness/friction、hidden wall 几何和 true passability 只用于离线评价。

## 最小场景矩阵

| 场景 | 物理/控制状态 | 局部试探 | 全身 clearance | PointGoal/最终状态 | 证据等级 |
|---|---|---|---|---|---|
| Nominal top-fixed curtain | 正常固定顶边 | RESPONSIVE；hand `0.013487 m` | PASS | goal stable PASS；误差 `0.093299 m` | PASS（名义） |
| Double-density curtain | 密度改变 | RESPONSIVE；ratio `0.955485` | FAIL | goal false；误差 `0.310924 m` | FAIL（扩展） |
| Bottom-left anchored | 左下约束 | RESPONSIVE；ratio `0.976825` | FAIL | XY 误差 `0.061813 m`，yaw 误差 `2.236896 rad` | FAIL（扩展） |
| Bottom-right anchored | 右下约束 | RESPONSIVE；ratio `0.973907` | PASS | goal stable PASS；XY `0.065269 m` | PASS（单次扩展） |
| Both-bottom corners | 双下角约束 | RESPONSIVE；ratio `0.974330` | FAIL | XY `0.064459 m`，yaw `2.306422 rad` | FAIL（扩展） |
| Close-gap hidden rigid curtain | 帘后约 `0.045 m` 刚性面 | H2 deployment `RESPONSIVE`，但严格 matched 不成立 | FAIL | goal false；误差 `0.059129 m` | NOT ASSESSABLE（严格） |
| Matched full-height rigid panel | 固定刚板 | BLOCKED；ratio `0.032770` | 不进入 traverse | PointGoal 位置可满足，旧 PoseGoal yaw `2.600305 rad` 失败 | PASS（阻挡分支）/旧 gate FAIL |

## 本体特征比较

| 场景 | hand progress (m) | progress ratio | upper RMS (rad) | effort deviation | base disturbance (m) | 判定 |
|---|---:|---:|---:|---:|---:|---|
| nominal | 0.013487 | 0.976709 | 0.061780 | 13.1708 | 0.001557 | RESPONSIVE |
| double-density | 0.013194 | 0.955485 | 0.061646 | 13.1827 | 0.001553 | RESPONSIVE |
| bottom-left | 0.013489 | 0.976825 | 0.061774 | 13.1728 | 0.001541 | RESPONSIVE |
| bottom-right | 0.013448 | 0.973907 | 0.061737 | 13.1739 | 0.001549 | RESPONSIVE |
| both-bottom | 0.013454 | 0.974330 | 0.061740 | 13.1759 | 0.001531 | RESPONSIVE |
| H2 | 0.013507 | 0.978150 | 0.061784 | 13.1717 | 0.001536 | RESPONSIVE |
| rigid | 0.000453 | 0.032770 | 0.085429 | 26.7832 | 0.007692 | BLOCKED |

局部手部指标对 nominal、bottom-left、bottom-right、both-bottom 和 H2 的区分度很弱；全身结果的分化主要在持续前进、candidate-clearance 和恢复/目标阶段出现。该结果支持“短时本体信息不足”，但不支持把所有差异直接命名为材料类别差异。

## 机器人身体 envelope 关系

Round13 的 `local_probe_representativeness.json` 给出：

- bottom-left：物理 probe displacement `0.207044 m`，ratio `0.976825`，clearance false，goal false；
- bottom-right：物理 probe displacement `0.209780 m`，ratio `0.973907`，clearance true，goal true；
- both-bottom：物理 probe displacement `0.205879 m`，ratio `0.974330`，clearance false，goal false。

三者局部响应同属 `RESPONSIVE`，但只有 bottom-right 完成了当前 robot-owned envelope 的候选清空。这是本轮最强的“动作位置/约束分布相关机会”证据。

需要保持的解释边界：

1. `RESPONSIVE` 只表示当前受限动作期间的响应支持继续小幅运动，不等于 `PASSABLE`。
2. candidate-clearance 是由机器人当前 pose/FK、已知候选区域和安全余量计算的部署 gate，不读取 curtain nodes。
3. 物理节点位移、top drift、stretch 和 hidden wall crossing 只用于离线验证。

## Hidden-rigid matched 状态

离线 physics audit 的 close-gap 几何为：curtain back face `0.851 m`，wall front face `0.896 m`，face gap `0.045 m`。no-probe displacement `0.002407 m`；probe displacement `0.445682 m`；top drift `0.000106 m`；stretch `1.04578`；没有节点越过 wall front face。这证明了仿真场景具备可分析的物理差异，但没有 G1 deployment 本体对照。

Round15 hidden G1 deployment 的本体输出为 hand progress `0.004405 m`、ratio `0.31898`、upper RMS `0.10035`、effort deviation `32.3423`、base disturbance `0.01251 m`，最终 stable gate false。严格 nominal matched run 在同一 hidden region 未到达共同预接触区，因此该结果标记为 `NOT ASSESSABLE`，不能作为隐藏刚体分类器性能。

## 策略 A/B/C

### A：固定一次试探 + 固定穿越

状态：`FAIL_EXTENSION_ROBUSTNESS`。

Nominal 复现了既有工程闭环；扩展场景中固定穿越动作无法适应 bottom-left、both-bottom、double-density 等持续接触约束。失败发生在 hand-level probe 之后，因此不能只改 probe 阈值解决。

### B：一次试探 + Phase-aware Progress Monitor

Round15 replay：

- 窗口 `.8 s`；command threshold `.08 m/s`；minimum progress `.015 m`；recovery `.025 m`；连续 3 个 stagnant windows；
- nominal 在 `16.805 s` candidate clearance 后 monitor 关闭，不再继承旧 false stop；
- 合成 stalled 输入约 `0.30 s` 检出；
- 阈值未改变；
- 这是软件生命周期 PASS，不是新的 physical Goal success 证明。

### C：最多两处空间化试探

软件 contract：

- 左/右站位相距 `.30 m`；
- 默认 merge radius `.35 m`，第二位置会被视作同一入口，故默认不允许；
- 显式 merge radius `.20 m`、episode budget `2` 时两个记录可并存，第三次被拒绝；
- map update check PASS，候选区域可从 `2` 变为 `1`，initial snapshot 保持不变；
- physical two-site episode：`NOT_RUN`。

因此 C 目前只能标为“空间记忆接口已具备，物理收益未评估”。

## 需要的最小后续 matched 实验

1. 固定同一 reset、脚部配置、policy history、probe command、时间窗和 pre-contact pose。
2. 先做 no-obstacle 与无 hidden wall 的 curtain 对照，再加入 hidden wall；若共同预接触区不存在，直接 `NOT ASSESSABLE`。
3. 将连续前进期间的 hand progress、robot-owned envelope back/front、base progress、upper residual、effort proxy、stall history 和 safety reason 同步记录。
4. 在相同安全预算下执行两处空间化试探，第二次只能从不同 spatial memory 位置进入；禁止通过改变 scene label 选择控制分支。
5. 用未参与阈值设定的重复样本报告 false-pass、false-block、uncertain、撤回安全性和 Goal success。

在这些证据完成前，不把单次 bottom-right PASS 外推为泛化能力，也不把失败外推为“柔性障碍原则上不可通行”。
