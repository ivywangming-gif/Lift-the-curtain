# RTX 4070 资源剖面

## 环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| VRAM | 8188 MB |
| Driver | 610.60 |
| Isaac Lab | 0.54.4 |
| PyTorch | 2.7.0+cu128 |
| CUDA runtime | 12.8 |
| CPU | 32 logical processors |
| System RAM | 32452.7 MB |
| 模式 | headless，`num_envs=1` |

## 24 条运行的统计

| 指标 | 最小 | 中位数 | 最大 |
|---|---:|---:|---:|
| Isaac 初始化（s） | 17.687 | 20.766 | 76.688 |
| rollout wall time（s） | 96.016 | 160.274 | 317.688 |
| simulated time（s） | 30.005 | 37.255 | 75.405 |
| physics steps/s | 43.054 | 47.440 | 67.741 |
| simulated/wall | 0.215 | 0.237 | 0.339 |
| GPU peak（MB） | 2588 | 3430 | 3441 |
| private process peak（MB） | 7241.8 | 8099.1 | 8142.3 |
| available RAM minimum（MB） | 4800.8 | 5057.0 | 5541.6 |
| GPU utilization peak | 41% | 43% | 48% |
| GPU temperature peak | 56 C | 61 C | 62 C |

## 资源基线结果

- Empty G1：2/2 有效，PointGoal 2/2。
- Rigid：2 条 resource baseline 有效，未出现 runtime invalid。
- Flexible bottom-right：2 条 resource baseline 有效，均达到 candidate clearance 和 PointGoal。
- 全部 24 条：没有 OOM、device lost、CUDA allocation failure 或持续进程崩溃。

## 已知非致命软件警告

Isaac Kit 尝试写安装目录下的 `user.config.json` 失败。该路径不在本轮授权修改范围内；由于每条运行仍生成完整 artifact，这被记录为安装/权限卫生问题，不作为 GPU 资源故障。

## 结论

`RESOURCE_VERDICT = 4070_SUFFICIENT_FOR_CURRENT_RESEARCH`。4070 不适合大规模并行或长时间 renderer 录像，但足以完成本轮 24 条小规模 headless 物理证据。5090 的合理价值是提高重复数、并行规模或后续传感器实验吞吐，而不是修复当前最早的执行/观测问题。

