# Native convex support and reconstructed-plane limits

核查日期：2026-11-07。此说明澄清几何证据的含义，不修改冻结协议、原始数据、几何库或标签。当前 12 个受影响凸体继续使用 `NATIVE_VERTEX_CONSERVATIVE_ENVELOPE`；在 V5 下，外包络重叠仍为 `NOT_VERIFIED`。

目前发现的是 **OVD 导出的三角面经重新计算法向后，与同一导出的完整原生顶点集合不自洽**。这不是已经测得 PhysX 原生 `PxHullPolygon.mPlane` 自相矛盾。OVD 导出中的原生顶点来源、重建三角面平面、原生存储平面，以及接触算法的数值行为，必须分别描述。

本机冻结库记录的绑定为 `omni.physx-107.3.26+107.3.3.wx64.r.cp311.u353/omni/physx/bindings/_physx.cp311-win_amd64.pyd`，SHA-256 为 `7fd515caa211084c1b4b2d57fbaba3fda232c280981ceab122143e4220b0265e`。本次查阅 NVIDIA 官方 `107.3-omni-and-physx-5.6.1` 发布分支；官方仓库将 107.3、PhysX 5.6.1 与 Isaac Sim 5.1.0 对应。此版本对应关系支持源码解释，但不证明本机二进制与该公开分支逐字节对应。[官方版本对应表](https://github.com/NVIDIA-Omniverse/PhysX)

**OVD 提供了什么。** `streamConvexMesh` 第 532–535 行读取原生 polygon/index buffer、`getVertices()` 和顶点数；第 550–556 行复制全部原生 cooked 顶点；第 559–573 行按原生 polygon 的索引将面转换成三角扇；第 576–577 行写出顶点和三角索引。该函数没有写出 `PxHullPolygon.mPlane`，也没有保留原 polygon 的分组字段。因此从导出三角形叉乘获得的平面不能称为直接读取的原生平面。[官方 writer，525–591 行](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/physx/source/physx/src/omnipvd/OmniPvdPxSampler.cpp#L525-L591)

原生 `getPolygonData` 的确能够返回真实存储平面：其实现第 43–49 行将 `HullPolygonData.mPlane`、顶点数及索引起点复制到 API 结构。这些平面没有通过上述 OVD mesh writer 保存；当前 artifact 不能补推它们的数值。[官方 `getPolygonData`，38–50 行](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/physx/source/geomutils/src/convex/GuConvexMesh.cpp#L38-L50)

**支持函数证明到哪里。** 对完整顶点集 V 及已经核验的仿射变换 T，在精确算术下，`max(v in V) dot(d, T(v))` 等于 `conv(T(V))` 的支持函数。官方 CPU 实现的穷举分支确实搜索全部顶点；其另一分支采用邻接图爬山，还依赖有效的凸体拓扑。EPA/GJK 的支持点经 `vertex2Shape` 变换返回。[CPU 顶点搜索，377–407 行](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/physx/source/geomutils/src/gjk/GuVecConvexHull.h#L377-L407)，[支持点变换，465–508 行](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/physx/source/geomutils/src/gjk/GuVecConvexHull.h#L465-L508)

GPU 数据布局把同一 native hull 的全部顶点复制到 GPU，并另外复制真实 polygon planes 和邻接数据。已核查的 CUDA 指针数组支持函数遍历 `nbVertices`，对点积取最大值。这支持“全原生顶点凸包表示理想支持几何”的判断；没有证明所有有限精度碰撞结果、接触流形或求解器响应均与独立 float64 凸包完全一致。[GPU 布局，482–523 行](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/physx/source/gpunarrowphase/src/PxgGeometryManager.cpp#L482-L523)，[CUDA 支持函数，171–193 行](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/physx/source/gpunarrowphase/src/CUDA/gjk.cuh#L171-L193)

**接触流形仍使用原生面。** CPU box–convex 路径读取 `polygon.mPlane`，并用 reference/incident polygon 的平面与索引顶点生成接触；GPU 路径同样读取、变换原生 polygon plane，再进行 `polyClip`。因此支持函数层面的等价不能消除尚未读取原生平面时的表示核验缺口。[CPU 原生平面读取，79–85 行](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/physx/source/geomutils/src/pcm/GuPCMContactGenBoxConvex.cpp#L79-L85)，[CPU 接触生成，331–355 行](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/physx/source/geomutils/src/pcm/GuPCMContactGenBoxConvex.cpp#L331-L355)，[GPU 平面裁剪，911–956 行](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/physx/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu#L911-L956)

**part28 的实际数值。** 当前 [native library v3](C:/Robotics/Lift-the-curtain/research/round4070_comprehensive_traversability/g1_native_collision_library_v3.json) 的 SHA-256 为 `03dfddb9dd026ef461130372fc1d59bc499deb7ada4314bdc2b793f8d0067fa8`。`parts[28]` 对应 `left_rubber_hand`、body index 37，记录了 `eSIMULATION_SHAPE`，mesh scale 为 `[1,1,1]`，raw shape-local pose 为单位变换。以下均来自其 `native_geometry_source`：

| 量 | 实际值与含义 |
|---|---|
| USD 导出点记录 / 精确去重后顶点 | 192 / 34；192 包含三角化后的重复记录，不能称为 192 个独立原生顶点 |
| 导出三角面数 | 64；不是已恢复的原 polygon 数 |
| 最大顶点越过重建面平面距离 | `0.0005733595558896958 m`，即 `0.5733595558896958 mm` |
| 最小导出三角形面积 | `4.774576289843889e-05 m²` |
| 最大越界所对应三角形面积 | `0.00016154590997023412 m²` |
| 全顶点外包络的点包含最大数值残差 | `2.7755575615628914e-17 m`；仅是顶点包含检查的残差 |

本机计算路径为 [physics_pvd_extract.py](C:/Robotics/Lift-the-curtain/scripts/round4070_g/physics_pvd_extract.py:225)：第 225–234 行从导出三角形叉乘、归一化、定向，再计算全部顶点到该平面的最大有符号距离；第 237–251 行建立完整顶点凸包并保存该诊断。因此 `native_face_inconsistency` 和 `exact_rejection` 是保留的历史字段名，其中的 “native” 指三角面来源，不能解释为已经读取并否定了原生 `mPlane`。该 0.57336 mm 也不是已测定的“重建面相对真实原生平面的误差”；真实原生平面数值尚缺，现有量是重建平面的顶点一致性偏差。

当前 v3 库绑定 [v7 / parser-v5 artifact](C:/Robotics/Lift-the-curtain/outputs/round4070_comprehensive_traversability/physics_diagnostics/collider_native_linkframe_v7_attempt01/native_pvd_usd_00/native_geometry_parser_v5.json)，其记录 SHA-256 为 `38c889e24da2257bf5739353b83530dd4094588228e1d98aa552a2c12d95135a`。39 个实际 shape 包含 27 个普通 convex mesh，其中 15 个通过导出面重建的一致性检查、12 个保留外包络模式；另外是 8 个 sphere 与 4 个 zero-margin native convex-core cylinder。此处没有把 15 个重建检查通过的 shape 描述成已经逐面测得原生 `mPlane`。

**缩放、cooking 与 margin 的界限。** OVD writer 记录 `g.scale.scale`，未明确写出 `PxMeshScale.rotation`。本机全部 27 个 ordinary convex mesh 的 scale 都是 `[1,1,1]`，因此这批数据中 scale-axis rotation 不影响缩放；此结论不适用于任意非均匀缩放。shape-local transform 与 body frame 仍须使用已验证的 raw native pose 证据。[writer，584–590 行](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/physx/source/physx/src/omnipvd/OmniPvdPxSampler.cpp#L584-L590)

官方 cooking 文档说明 `planeTolerance` 可参与点舍弃、面合并与平面变化，公开默认值为 `0.0007f`；但本次没有读取实际 cooking 参数，也没有证明该机制解释了 part28 的偏差，不能把默认值直接换算成这里的误差。原生 cooked 顶点已经是 cooking 的产物，不能再凭猜测添加一次“cooking 膨胀”。[官方 cooking 参数，207–227 行](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/physx/include/cooking/PxCooking.h#L207-L227)

所核查的 GPU 普通 convex 分支将 `expandedMargin` 置零，另行计算 PCM margin；接触距离随后单独组合。故不能将 PCM/GJK 数值阈值一概视为实体的 Minkowski 膨胀，也不能由此宣称运行时 contact/rest offset 已被读回或为零。native convex-core cylinder 是另一种几何类别。[普通 convex 分支，2482–2491 行](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/physx/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu#L2482-L2491)，[接触距离组合，2579 行](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/physx/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu#L2579)

下一步若需进一步确证原生面表示，应以同一 native mesh handle 导出真实 `getPolygonData(i).mPlane`、原 polygon 分组/索引、完整顶点及实际 scale/localPose，并逐项比较顶点半空间包含性、面顶点共面性和支持值；不能只重建三角法向后再次称为原生面。当前仅做源码和既有 artifact 的只读核查，没有执行该采集，也没有升级任何 shape 的证据模式。

V5 的 [冻结动作 sidecar 第 74 行](C:/Robotics/Lift-the-curtain/research/round4070_comprehensive_traversability/g1_action_contract_v2.json:74) 明确规定外包络分离可用、重叠为 `NOT_VERIFIED`。即使另外建立某个深度阈值下的充分分离证明，也不能静默改写这个冻结规则。本文不改变 D01 或其他轨迹的有效性、P/B/U 标签、0/1/5 mm 敏感性结果或训练准入。
