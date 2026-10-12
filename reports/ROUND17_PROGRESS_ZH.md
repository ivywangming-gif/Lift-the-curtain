# Round 17 进度

**日期：** 2026-10-12
**工作目录：** `C:\Robotics\Lift-the-curtain`
**约束：** 不启动 learning；不修改 Isaac Sim/Lab、驱动、CUDA、PyTorch、FALCON checkpoint 或外部 reference；本轮不重复 4070 renderer crash 路径。

## 当前结论

- `SOURCE_PUSHED`：`BLOCKED`，本地提交已创建，但 GitHub HTTPS 正常 push 长时间无响应，未确认远端写入。
- `REPLAY_CODE_READY`：`PASS`，路径层、资产检查、统一 CLI、录制请求 schema、静态编译和无渲染 smoke 已完成。
- `REPLAY_PHYSICS_VERIFIED_ON_4070`：`HISTORICAL_ONLY`，本轮没有新启动 Isaac 物理 replay。
- `RENDERING_VERIFIED_ON_5090`：`NOT RUN`，当前机器不是目标 5090，且不重复已知 renderer crash。
- `VIDEO_RECORDED_ON_5090`：`NOT RUN`，当前 runner 没有经过验证的 standalone MP4 backend。

## 实验与审计记录

### R17-01：远端和 Git 历史审计

- **假设：** 远端为空或没有新提交时，可在安全审计后正常推送；若有新提交则必须先审查。
- **命令：** `git status --short --branch`、`git log --oneline --decorate -n 20`、`git remote -v`、只读 `git ls-remote origin`。
- **结果：** 本地 `main` 基于 `f70fb36`，追踪引用显示 `origin/main: gone`；两次只读 `git ls-remote --heads --tags origin` 均未返回 heads/tags。远端 URL 为用户指定的 GitHub 仓库。未执行 force push。
- **状态：** `PASS`（远端为空审计）；远端写入结果单独记录为 `BLOCKED`。
- **证据：** `reports/ROUND17_GITHUB_RELEASE_REPORT_ZH.md`、本报告。

### R17-02：可移植路径和资产审计

- **改变变量：** 用 `FALCON_*`、`MODESUITE_*`、`ISAACLAB_*` 环境变量和显式 CLI 路径替代新入口中的固定盘符。
- **命令：** `python scripts/check_assets.py --allow-missing-external --json-out ...`；随后用本机外部 checkout 环境变量进行完整只读检查。
- **结果：** 生成 USD、FALCON ONNX、URDF、MoDeSuite curtain 和 IsaacLab launcher 均存在；ONNX SHA256 为 `8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0`，与 contract 一致。MoDeSuite 未发现明确 LICENSE，仍不纳入发布。
- **状态：** `PASS`。
- **证据：** `scripts/check_assets.py`、`lift_curtain/paths.py`、`docs/ASSET_SETUP_ZH.md`、本地临时 asset audit（未纳入 Git）。

### R17-03：统一 replay CLI

- **假设：** 统一薄 wrapper 可以复用已有成功 runner，不复制控制逻辑，也不把历史 artifact 伪装成新 replay。
- **实际动作：** 增加 `curtain`、`rigid`、`curtain_nearfield` 三个场景映射；写入 `replay_manifest.json`；支持 `--headless`、`--scenario`、`--seed`、`--output_dir`、录制请求参数和 `--dry-run`。
- **结果：** `--help` 正常；三个场景 dry-run 正常；指定 IsaacLab root 时生成的命令指向仓库中已有 runner。manifest 明确记录 `historical_artifact` 和 `new_replay`。
- **状态：** `PASS`（静态/命令构造）；未启动 Isaac。
- **证据：** `scripts/replay_nominal.py`、`docs/REPLAY_ON_5090_ZH.md`。

### R17-04：录制准备和可视形变审计

- **假设：** 在未验证 renderer backend 的情况下，必须保存请求而不能声称生成视频。
- **实际动作：** 增加 `RecordingSpec`、请求 sidecar、state sidecar 和 `VISUAL_DEFORMABLE_MISMATCH` 判定函数；运行 `scripts/recording_smoke.py`。
- **结果：** `RECORDING_STATIC_SMOKE_PASS`；`--record_video` 返回 `RECORDING_NOT_IMPLEMENTED_ON_THIS_RUNNER` 并写入 `verified_video=false`，没有启动 renderer。
- **状态：** `PASS`（静态准备）；视频本身 `NOT RUN`。
- **证据：** `lift_curtain/recording.py`、`scripts/recording_smoke.py`、`docs/REPLAY_ON_5090_ZH.md`。

### R17-05：静态编译和测试

- **命令：** `python -m compileall -q lift_curtain scripts tests`；`C:\Users\SDSC\anaconda3\python.exe -m pytest -q -p no:cacheprovider`。
- **结果：** `COMPILEALL_EXIT=0`；完整 suite `41 passed`。默认 pytest cache provider 受当前宿主临时目录 ACL 影响，因此使用已存在的 pytest 参数关闭 cache 写入；未安装或修改依赖。
- **状态：** `PASS`。
- **证据：** `tests/test_round17_release_tools.py`、本报告。

## 发布范围

准备提交的范围只包含核心 `lift_curtain` 源码、必要配置、统一 runner、资产检查/转换/启动 smoke 脚本、测试、README、中文资产/5090 文档、许可证说明和小型 release manifest。`reports/artifacts/`、生成 USD、ONNX/URDF、视频、缓存、crash dump 和大型 telemetry 不进入 staged set。

## 提交与远端写入结果

- 本地源码发布提交：`6c734e2feb71aaa554aacd6f83f6dd39706ca206`，提交信息为 `chore: publish portable replay release`。
- 目标远端：`https://github.com/ivywangming-gif/Lift-the-curtain.git`，分支：`main`。
- `git push origin main` 在 HTTPS 通道无输出超过四分钟；已中止无响应进程，没有观察到远端成功，也没有使用 force push。
- 因此本轮不能把本地 commit 写成 GitHub 已发布；后续应在网络/认证可用的机器上做普通 push。

## 下一步

1. 对 staged 文件逐项执行路径、凭据、许可证、大小和历史对象审计。
2. 生成 Round17 三份正式报告和 `run_manifest.json`。
3. 在网络和 GitHub HTTPS 认证恢复后，先重新执行 `git ls-remote --heads --tags origin`，确认没有新历史。
4. 只执行普通 `git push origin main`；成功后再把 `SOURCE_PUSHED` 更新为 `PASS`，并记录远端可见 SHA。
