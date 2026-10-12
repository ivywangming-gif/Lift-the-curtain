# Lift-the-curtain

这是一个面向 G1 橡胶手与固定顶部柔性障碍的工程 baseline 原型。仓库包含冻结 FALCON adapter、SE(2) 闭环、部署/评价观测隔离、Oracle 0/1/2 地图和可审计的近场/名义 replay 入口。

## 当前状态

| 范围 | 状态 |
| --- | --- |
| Round 12 nominal curtain/rigid map episode | `PASS`，仅限报告声明的固定场景与定位假设 |
| Round 13--16 扩展鲁棒性 | `NOT COMPLETE`，保留为研究边界证据 |
| 本仓库新的 Isaac replay | `NOT RUN`，需用户机器上的外部依赖 |
| RTX 5090 renderer/video | `NOT VERIFIED`，本仓库不包含驱动或 Isaac 安装 |
| 真实 G1 | `NOT VALIDATED` |

Round 12 的历史 artifact 不能替代新机器上的 replay。真实物体类别、布料节点、仿真接触真值和 passability oracle 只允许进入离线评价；部署决策使用 `deployment_observation`、自身 FK、已知地图和阶段定位假设。

## 快速检查

先按 [资产配置说明](docs/ASSET_SETUP_ZH.md) 设置外部路径，再运行：

```text
python scripts/check_assets.py --json-out reports/local_asset_audit.json
python scripts/replay_nominal.py --help
python scripts/replay_nominal.py --scenario curtain --output_dir outputs/round17/curtain --dry-run
python scripts/replay_nominal.py --scenario rigid --output_dir outputs/round17/rigid --dry-run
python scripts/replay_nominal.py --scenario curtain_nearfield --output_dir outputs/round17/nearfield --dry-run
```

实际 Isaac 调用由 `ISAACLAB_ROOT` 下的 `isaaclab.bat` 或 `isaaclab.sh` 负责；不要把外部 checkpoint、URDF、USD 或视频复制到仓库。5090 准备步骤见 [docs/REPLAY_ON_5090_ZH.md](docs/REPLAY_ON_5090_ZH.md)。

## 目录约定

- `lift_curtain/control/`：FALCON contract、adapter、SE(2)、部署观测和 goal/monitor。
- `lift_curtain/navigation/`：Oracle map、交互记忆、规划与状态机。
- `scripts/replay_nominal.py`：统一的 `curtain`、`rigid`、`curtain_nearfield` 入口。
- `scripts/check_assets.py`：只读资产和 SHA256 审计，不启动 Isaac。
- `configs/`：不含第三方二进制的配置模板。
- `reports/release/`：可提交的小型历史结论摘要；原始 telemetry 默认被忽略。
- `reports/artifacts/`、`assets/generated/`、`outputs/`：本地实验产物，不属于源码发布。

## 约束

本仓库不启动 RL、teacher-student、VLM 或真实机器人，不修改 Isaac Sim/Lab、CUDA、PyTorch、驱动、FALCON checkpoint 或外部 reference checkout。运行实验时保持 headless、单环境和单 Isaac/Kit 进程。

许可证和外部依赖边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
