# Round 17 Replay 准备报告

## 历史证据与新 replay 的分离

历史成功证据保留：

- Curtain：`reports/artifacts/round12/episodes/curtain_final_gate_fix/` 和 `curtain_primary/`。
- Rigid：`reports/artifacts/round12/episodes/rigid_primary/`。
- 近场穿帘：Round 10 `formal_matched_effective_speed_e1/curtain/`。

这些目录只证明历史运行，不会被 `replay_nominal.py` 当作新运行输出。新运行必须写入用户指定的 `--output_dir`，并通过 `replay_manifest.json` 标记 `new_replay=true`。

## 统一入口

```text
python scripts/replay_nominal.py --scenario curtain --output_dir <out>/curtain --headless
python scripts/replay_nominal.py --scenario rigid --output_dir <out>/rigid --headless
python scripts/replay_nominal.py --scenario curtain_nearfield --output_dir <out>/nearfield --headless
```

入口复用已有 runner，不复制 FALCON、SE(2)、map 或 proprio controller。固定参数由已成功的 Round 10/12 配置显式传入，包括 `0.25 m/s` cruise、`0.30 m/s` 上限、formal 2m near-field geometry 和既有本体阈值文件。没有修改 FALCON checkpoint、physics backend、布料材料或完成阈值。

## 运行前检查

```text
python scripts/check_assets.py --json-out <out>/asset_audit.json
python scripts/replay_nominal.py --scenario curtain --output_dir <out>/curtain --dry-run
python scripts/replay_nominal.py --scenario rigid --output_dir <out>/rigid --dry-run
python scripts/replay_nominal.py --scenario curtain_nearfield --output_dir <out>/nearfield --dry-run
```

`--help` 和 dry-run 不导入 IsaacLab。实际运行前必须设置 `ISAACLAB_ROOT` 或传入 `--isaaclab`，并让资产审计中的 required records 全部为 `PASS`。

## 录制接口边界

CLI 接受 `--record_video`、`--camera_view`、`--resolution` 和 `--fps`，并生成 `recording_request.json`。当前 runner 没有已验证的 camera/MP4 backend，因此带录制参数的实际调用返回 `RECORDING_NOT_IMPLEMENTED_ON_THIS_RUNNER`；这被视为诚实的 `BLOCKED/NOT RUN`，不是失败后隐藏。

`lift_curtain/recording.py` 已实现：

- 录制请求 schema 和尺寸/FPS 校验；
- frame/state sidecar 的 CSV 写入接口；
- runtime deformable displacement 与 visual displacement 的离线比较；
- runtime 有变化、可视 mesh 无变化时的 `VISUAL_DEFORMABLE_MISMATCH`。

没有声称已经生成 MP4，也没有把渲染结果送入部署决策器。

## 纯软件验证

- `compileall`：`PASS`。
- recording static smoke：`RECORDING_STATIC_SMOKE_PASS`。
- replay help/dry-run：`PASS`。
- 资产路径与 ONNX hash audit：`PASS`（在显式配置本机外部路径时）。
- pytest：`PASS`，`41 passed`，使用 `-p no:cacheprovider` 避开宿主临时 cache ACL；未修改环境。
- 新物理 replay：`NOT RUN`。
