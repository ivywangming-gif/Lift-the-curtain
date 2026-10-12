# RTX 5090 重放准备

本说明只描述可复现的命令和验收顺序，不假定操作系统、盘符、驱动版本或 Isaac 安装位置。Round 17 不在当前机器上重新触发已知的 4070 renderer native crash；也不自动升级/降级驱动、Isaac Sim/Lab、CUDA 或 PyTorch。

## 1. 先做只读资产检查

设置 `FALCON_*`、`MODESUITE_*` 和 `ISAACLAB_*` 环境变量后运行：

```text
python scripts/check_assets.py --json-out <output-dir>/asset_audit.json
```

只有所有 required record 为 `PASS`，才进入 Isaac replay。若只验证源码入口：

```text
python scripts/check_assets.py --allow-missing-external --allow-missing-generated
```

该命令的成功只表示静态路径层可用。

## 2. 先 dry-run，再启动 headless replay

三个场景使用同一入口：

```text
python scripts/replay_nominal.py --scenario curtain --output_dir <output-dir>/curtain --dry-run
python scripts/replay_nominal.py --scenario rigid --output_dir <output-dir>/rigid --dry-run
python scripts/replay_nominal.py --scenario curtain_nearfield --output_dir <output-dir>/curtain_nearfield --dry-run
```

确认 `replay_manifest.json` 中的 `script`、`asset`、`onnx`、`seed` 和 `headless=true` 后，再运行对应命令，去掉 `--dry-run`：

```text
python scripts/replay_nominal.py --scenario curtain --output_dir <output-dir>/curtain --headless
python scripts/replay_nominal.py --scenario rigid --output_dir <output-dir>/rigid --headless
python scripts/replay_nominal.py --scenario curtain_nearfield --output_dir <output-dir>/curtain_nearfield --headless
```

`ISAACLAB_ROOT` 或 `--isaaclab` 必须指向实际存在的 `isaaclab.bat`/`isaaclab.sh`。子 runner 仍然是仓库中已有的 `scripts/test_round12_oracle_map_episode.py` 和 `scripts/test_round6_near_field_contact_diagnostic.py`；没有用新的隐藏控制分支替代它们。

## 3. 物理与渲染验收分离

新 replay 的 `summary.json`、`telemetry.csv` 和 `evaluation_ground_truth.json` 分开保存。只有新 replay 的 summary 才能称为新物理运行；`historical_artifact` 只用于追溯 Round 10/12 证据。

验收顺序：

1. headless empty/nominal replay 启动并正常落盘；
2. 无 native crash、无 NaN、无残留 Isaac 进程；
3. deployment observation 不包含 object identity、true contact、cloth nodes 或 passability；
4. 物理结果与历史 nominal 结论逐项比较；
5. 只有在 renderer 启动和 camera frame 均正常时才测试视频。

## 4. 视频请求和当前限制

入口接受以下参数并写入 `recording_request.json`：

```text
--record_video --camera_view front --resolution 320x240 --fps 30
```

当前仓库的 nominal runner 没有经过验证的 camera/MP4 backend。因此带 `--record_video` 的调用会返回 `RECORDING_NOT_IMPLEMENTED_ON_THIS_RUNNER`，不会启动未验证 renderer，也不会生成伪视频。`recording_request.json` 的 `verified_video=false` 是硬约束。

在 5090 上接入经过单独审计的 camera backend 后，必须把 frame/MP4、state sidecar、启动日志和 backend 配置放在被忽略的输出目录，并重新运行：

```text
python scripts/check_assets.py --json-out <output-dir>/asset_audit.json
python scripts/replay_nominal.py --scenario curtain --output_dir <output-dir>/curtain --headless
```

不能仅凭窗口出现或命令返回 0 标记 `RENDERING_VERIFIED_ON_5090`。需要实际 frame 非空、尺寸/FPS 符合请求、MP4 可读取，并完成 runtime deformable state 与可视 mesh 的同步检查。若 runtime nodes 移动而可视 mesh 未动，必须写入 `VISUAL_DEFORMABLE_MISMATCH`，该结果不能作为视觉证据。

## 5. 状态标签

- `SOURCE_PUSHED`：只在 GitHub push 返回成功并记录 commit SHA 后成立。
- `REPLAY_CODE_READY`：纯软件入口、dry-run 和静态测试通过。
- `REPLAY_PHYSICS_VERIFIED_ON_4070`：仅在本轮新运行通过时成立；历史 artifact 单独标记为 `HISTORICAL_ONLY`。
- `RENDERING_VERIFIED_ON_5090`：需要真实 5090 renderer/camera 证据，本轮未假定。
- `VIDEO_RECORDED_ON_5090`：需要可读取 MP4 和 state sidecar，本轮未假定。

## 6. 5090 实际执行清单

### 6.1 克隆、系统和依赖

在目标机上从公开仓库开始：

```text
git clone https://github.com/ivywangming-gif/Lift-the-curtain.git
cd Lift-the-curtain
git status --short --branch
git log -1 --oneline
```

Windows PowerShell 可用 `nvidia-smi`、`python --version` 和
`python -c "import numpy; print(numpy.__version__)"`；Linux 使用
`nvidia-smi`、`python3 --version` 和等价的 `python3 -c` 命令。检查
`<ISAACLAB_ROOT>/VERSION`，并把 GPU、驱动、OS、Isaac Sim/Lab 和 Python
版本写入本地 audit。版本不匹配时只记录兼容性限制，不自动更换依赖。

### 6.2 资产与无渲染 smoke

设置 [ASSET_SETUP_ZH.md](ASSET_SETUP_ZH.md) 中的环境变量后，先运行：

```text
python scripts/check_assets.py --json-out <output-dir>/asset_audit.json
python scripts/recording_smoke.py
python scripts/replay_nominal.py --scenario curtain --output_dir <output-dir>/curtain --dry-run
```

### 6.3 空场景渲染 smoke

仓库已有的 `scripts/test_round6_isaac_startup_matrix.py` 要求显式存在的
`.kit` experience。只在目标 5090 上按顺序执行：

```text
python scripts/test_round6_isaac_startup_matrix.py --mode empty_headless --headless --steps 20 --experience <existing-kit-experience> --run-root <output-dir>/renderer
python scripts/test_round6_isaac_startup_matrix.py --mode headless_enable_cameras_no_camera --headless --enable_cameras --steps 20 --experience <existing-kit-experience> --run-root <output-dir>/renderer
python scripts/test_round6_isaac_startup_matrix.py --mode minimum_camera --headless --enable_cameras --steps 20 --experience <existing-kit-experience> --run-root <output-dir>/renderer
```

第一步失败时停止后续 renderer 条件，并保留 `stage_markers.jsonl`、
`summary.json` 和完整 stdout/stderr。不得在本轮自行升级驱动或 Isaac。

### 6.4 物理 replay、录制和查看

空场景通过后运行：

```text
python scripts/replay_nominal.py --scenario curtain --output_dir <output-dir>/curtain --headless
python scripts/replay_nominal.py --scenario rigid --output_dir <output-dir>/rigid --headless
```

只有 standalone camera backend 已在当前 Isaac 版本中接入并通过空场景验证后，
才使用 `--record_video --camera_view front --resolution 640x360 --fps 20`。
录制输出放在本机 `outputs/videos/`，用本机播放器或 `ffprobe` 检查 MP4 的
可读性、尺寸和 FPS，并与同步 CSV 一起保存。当前仓库仍会对该选项返回
`RECORDING_NOT_IMPLEMENTED_ON_THIS_RUNNER`，直到 backend 代码经过单独审计；
因此本轮不能声称已经录制视频。
