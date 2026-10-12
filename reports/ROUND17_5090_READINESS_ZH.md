# Round 17 RTX 5090 就绪报告

## 就绪矩阵

| 项目 | 状态 | 证据/限制 |
| --- | --- | --- |
| 可移植源码 | `READY` | `lift_curtain`、配置、脚本和测试已整理 |
| 外部路径配置 | `READY` | 环境变量/显式 CLI，见 `docs/ASSET_SETUP_ZH.md` |
| FALCON checkpoint | `READY_WITH_HASH_CHECK` | 复用同一 ONNX contract；本机审计 hash 匹配 |
| 无渲染 CLI | `READY` | help、dry-run、compileall 和静态 smoke 通过 |
| 4070 新物理 replay | `NOT RUN` | 避免重复已知 renderer/native crash；历史结果单独保留 |
| 5090 empty renderer | `NOT RUN` | 需要目标机、现有 `.kit` experience 和用户环境 |
| 5090 camera | `NOT RUN` | 需先通过 empty renderer smoke |
| MP4 视频 | `NOT RUN/BLOCKED` | standalone runner backend 尚未接入/验证 |
| 纯软件 pytest | `PASS` | 41 tests passed；使用 `-p no:cacheprovider` 避开宿主 cache ACL |
| GitHub 源码发布 | `BLOCKED` | 本地 commit `0bf30cc0b615d019097dd060c8da0cefd9d74b8f` 已创建；HTTPS push 无响应 |
| 真实 G1 | `NOT VALIDATED` | 本轮不连接真实机器人 |

## 是否能复用同一 FALCON checkpoint

可以复用同一个 checkpoint，前提是目标机通过资产检查并得到相同 ONNX SHA256：

`8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0`

同时必须保持 29 DoF、官方 joint order、history length、physics dt 和 decimation contract 不变。若目标机版本或 checkpoint hash 不同，状态应为 `BLOCKED`，不能静默替换。

## 5090 第一阶段命令

用户在目标机完成 clone 和环境变量设置后，第一条项目命令应是：

```text
python scripts/check_assets.py --json-out <output-dir>/asset_audit.json
```

随后先执行 `scripts/recording_smoke.py` 和三个 `replay_nominal.py --dry-run`，再依次执行仓库已有的空场景 startup matrix，最后才运行 curtain/rigid replay。具体 Windows/Linux 差异、GPU/驱动/Isaac 版本检查和 `.kit` experience 参数见 `docs/REPLAY_ON_5090_ZH.md`。

## 许可和 blocker

当前没有发现凭据或可疑敏感内容。仍需用户单独提供/获取：FALCON ONNX、FALCON URDF、MoDeSuite curtain USD、Isaac Sim/Lab 和可选的生成机器人 USD。MoDeSuite license 未明确，因此不上传其资产。默认 pytest cache provider 受宿主临时目录 ACL 影响，但禁用 cache 后完整 suite 已通过；5090 renderer/MP4 backend 是本轮主要未验证项。

GitHub 源码发布的本地提交已经完成，但正常 HTTPS push 超过四分钟无响应，未确认远端写入；不使用 force push。

## 明确不能推出的结论

源码上传、静态编译或 dry-run 通过，都不能推出：

- 5090 renderer 已启动；
- camera frame 已产生；
- video 已录制；
- 新机器物理 replay 已通过；
- 真实机器人可部署。

必须分别取得对应机器和 artifact 证据后，才更新 `RENDERING_VERIFIED_ON_5090` 或 `VIDEO_RECORDED_ON_5090`。
