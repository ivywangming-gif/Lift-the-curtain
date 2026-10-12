# Round 17 GitHub 源码发布报告

**日期：** 2026-10-12
**目标仓库：** `https://github.com/ivywangming-gif/Lift-the-curtain.git`
**当前状态：** 发布内容已完成审计并创建本地提交；GitHub HTTPS push 无响应，远端写入未确认。

## 发布状态

| 标签 | 当前值 | 说明 |
| --- | --- | --- |
| `SOURCE_PUSHED` | `BLOCKED` | 本地 commit 已创建；正常 HTTPS push 超过四分钟无响应，未确认远端写入 |
| `REPLAY_CODE_READY` | `PASS` | 路径层、CLI、静态编译、录制契约 smoke 通过 |
| `REPLAY_PHYSICS_VERIFIED_ON_4070` | `HISTORICAL_ONLY` | Round 10/12 artifact 有证据；本轮未重复启动物理进程 |
| `RENDERING_VERIFIED_ON_5090` | `NOT RUN` | 当前没有 5090 实机或新 renderer 证据 |
| `VIDEO_RECORDED_ON_5090` | `NOT RUN` | 没有经过验证的 standalone MP4 backend |

## 已准备并计划发布的源码

- `lift_curtain/control/`：FALCON contract/adapter、SE(2)、deployment observation、PointGoal、progress monitor。
- `lift_curtain/navigation/`：Oracle 0/1/2 map、planner、interaction memory、spatial probe 和状态机。
- `lift_curtain/envs/`：固定顶部 2m curtain、rigid panel 和场景描述。
- `lift_curtain/paths.py`：外部依赖环境变量解析，不再依赖新入口中的固定盘符。
- `lift_curtain/recording.py`：录制请求、state sidecar 和可视形变一致性判定。
- `scripts/replay_nominal.py`：`curtain`、`rigid`、`curtain_nearfield` 薄 wrapper。
- `scripts/check_assets.py`：只读资产、许可证状态和 ONNX SHA256 审计。
- `scripts/recording_smoke.py`、`scripts/test_round6_isaac_startup_matrix.py`：无渲染/启动准备。
- 必要配置、测试、README、中文资产/5090 文档和小型 release manifest。

## 明确排除

未将 ONNX、URDF、USD、mesh、视频、Python/conda 环境、`__pycache__`、pytest cache、Round 2--16 原始 telemetry、crash dump 或整个 IsaacLab/FALCON checkout 纳入本轮 staged 新文件。`reports/artifacts/` 默认忽略；任务书要求的 `reports/artifacts/round17/run_manifest.json` 是唯一显式加入的小型 manifest。

## Git 历史审计

当前本地历史只有早期 M0/M1 基础提交；历史对象中可见小型 M0/M1 telemetry、日志和 Python bytecode，最大对象约 `556091` bytes，没有发现 ONNX、模型权重、USD/URDF、MP4、私钥、token 或常见凭据模式。它们没有在本轮新增，但因已有提交仍属于本地历史；本轮没有 force push 或历史重写。

工作树原先包含 Round 2--16 大量未跟踪报告和 artifacts。本轮只选择可审查的核心源码、少量报告和 release manifest，未把这些未跟踪原始产物整体加入 index。

## 外部依赖和许可

- FALCON upstream commit：`a967a6d8494f57777cf8d266a644ac8e45833301`，MIT；ONNX contract SHA256 为 `8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0`。
- MoDeSuite reference commit：`470681a7ff117e487060ad52bc4f199e3fa34474`；历史审计未发现明确 LICENSE，不重新分发 curtain USD。
- Isaac Sim/Lab、NVIDIA runtime、driver、CUDA 和 PyTorch 由用户在目标机器按其条款安装，不进入仓库。

## 验证结果

- `compileall`：`PASS`。
- `scripts/recording_smoke.py`：`PASS`。
- `scripts/replay_nominal.py --help`：`PASS`。
- 三个场景 dry-run：`PASS`，未启动 Isaac。
- 完整 `pytest`：`PASS`，`41 passed`，命令为 `python -m pytest -q -p no:cacheprovider`；默认 cache provider 的宿主临时目录权限问题已隔离，未安装新依赖。
- 4070 renderer：`NOT RUN`，遵守不重复 native crash 路径的约束。

## 提交与推送结果

- 本地源码发布提交已完成：`6c734e2feb71aaa554aacd6f83f6dd39706ca206`，分支 `main`。
- 目标仓库：`https://github.com/ivywangming-gif/Lift-the-curtain.git`。
- `git push origin main` 首次通过 HTTPS 长时间无输出，超过四分钟后中止；带连接/低速超时的非交互重试明确返回 `Cannot prompt because user interactivity has been disabled` 和 `unable to get password from user`。没有观察到远端 ref 更新，也没有执行 force push。
- 当前应报告为 `SOURCE_PUSHED=BLOCKED`，而不是已发布。后续需要在已配置 GitHub 凭据的环境中重新执行普通 push，并先复核远端是否出现新 refs。

## 十个问题

1. **哪些源码已经推送？** 没有源码被确认推送。核心控制、导航、场景、统一 runner、测试、配置和文档已进入本地提交；远端写入因 HTTPS 无响应而阻塞。
2. **GitHub 仓库、分支和 commit SHA 是什么？** 目标仓库为上述 GitHub 仓库，分支 `main`；本地源码发布 commit 为 `6c734e2feb71aaa554aacd6f83f6dd39706ca206`，远端可见 SHA 尚未确认。
3. **是否有外部资产仍需单独获取？** 有：FALCON ONNX/URDF、MoDeSuite curtain USD、Isaac Sim/Lab 和本地生成机器人 USD。
4. **5090 是否能使用相同的 FALCON checkpoint？** 可以，但必须通过同一 ONNX SHA256、29 DoF/joint-order 和版本 contract 检查。
5. **Round12 Curtain/Rigid 的重放入口是什么？** `scripts/replay_nominal.py --scenario curtain|rigid ...`；近场为 `curtain_nearfield`。
6. **是否已支持视频录制？** 已支持请求 schema、sidecar 和 mismatch audit；当前 standalone MP4 backend 未实现/未验证，不能称为已录制。
7. **哪些功能只完成静态验证，尚未在 5090 实机运行？** 统一 CLI、资产检查、录制接口、空场景命令和所有 5090 renderer/camera/video 流程均未在 5090 实机运行；Round12 物理结果仅为历史 artifact。
8. **是否存在许可、路径或依赖 blocker？** MoDeSuite license 未明确；外部资产需用户提供；当前 bundled Python 缺 pytest；5090 renderer/backend 未验证。
9. **有没有修改过原始成功控制参数？** 没有修改 FALCON checkpoint、物理参数、WBC contract 或历史成功阈值；wrapper 只是显式传递已有 Round10/12 参数。
10. **用户到 5090 电脑上第一条应该执行什么命令？** `python scripts/check_assets.py --json-out <output-dir>/asset_audit.json`。
