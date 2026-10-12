# 外部资产配置

本仓库只发布源码、配置模板和小型审计/报告摘要。FALCON checkpoint、URDF、MoDeSuite curtain USD、Isaac Sim/Lab 和生成的机器人 USD 必须在本机单独获取或生成；脚本不会复制、上传或重新分发它们。

## 环境变量

| 变量 | 含义 |
| --- | --- |
| `FALCON_UPSTREAM_ROOT` | 外部 FALCON checkout 根目录 |
| `FALCON_ONNX_PATH` | 直接指定 `g1_29dof.onnx`，优先于根目录 |
| `FALCON_URDF_PATH` | 直接指定 `g1_29dof_fakehand.urdf`，优先于根目录 |
| `MODESUITE_ROOT` | 只读 MoDeSuite reference 根目录 |
| `MODESUITE_CURTAIN_PATH` | 直接指定外部 `curtain.usd` |
| `ISAACLAB_ROOT` | IsaacLab 根目录，包含 `isaaclab.bat` 或 `isaaclab.sh` |
| `ISAACLAB_LAUNCHER` | 直接指定启动脚本，优先于 `ISAACLAB_ROOT` |

相对路径、Windows 路径和 Linux 路径均由调用环境解释。不要假定仓库位于某个盘符或用户目录。

## Windows PowerShell 示例

```powershell
$env:FALCON_UPSTREAM_ROOT = "D:\path\to\FALCON-upstream"
$env:MODESUITE_ROOT = "D:\path\to\MoDeSuite-reference"
$env:ISAACLAB_ROOT = "D:\path\to\IsaacLab"
python scripts/check_assets.py --json-out reports\local_asset_audit.json
```

若不希望设置 checkout 根目录，也可以逐项设置 `FALCON_ONNX_PATH`、`FALCON_URDF_PATH` 和 `MODESUITE_CURTAIN_PATH`。

## Linux shell 示例

```bash
export FALCON_UPSTREAM_ROOT=/path/to/FALCON-upstream
export MODESUITE_ROOT=/path/to/MoDeSuite-reference
export ISAACLAB_ROOT=/path/to/IsaacLab
python3 scripts/check_assets.py --json-out reports/local_asset_audit.json
```

## 审计内容

`check_assets.py` 检查：

- 仓库内生成机器人 USD 是否存在；
- FALCON ONNX、URDF 和 MoDeSuite curtain 是否由外部路径提供；
- ONNX SHA256 是否等于冻结 contract 中记录的值；
- IsaacLab 启动器是否存在；
- 外部 checkout 是否声明许可证。

`--allow-missing-external` 和 `--allow-missing-generated` 只适合源码/CI 静态检查，不代表 replay 可以运行。MoDeSuite 当前历史审计没有找到明确的 LICENSE 文件，因此不要把其 USD 纳入提交或发布包。

## 生成机器人 USD

如果需要本地生成橡胶手机器人资产，使用：

```text
python scripts/convert_falcon_urdf.py <path-to-g1_29dof_fakehand.urdf> <local-output.usd>
```

该脚本需要在 IsaacLab 提供的 Python/启动环境中执行。生成结果保持在本地 `assets/generated/` 或用户指定目录，不应提交。
