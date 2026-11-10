# Round 4070-G BIMANUAL_V1 Stage 2A 总结

日期：2026-11-10
阶段结论：**离线执行工具已实现并通过合同测试；Kit/Physics 运行尚未获批，Stage 2A 未完成。**
正式 G1 准入：**NO-GO**

## 本阶段范围

本阶段只准备 BIMANUAL_V1 的受控 Kit/Physics-only 验证入口，不是双手 G1 采集，也不是模型训练。计划范围为 8 个代表性代理场景、每场景 3 次隔离运行，共 24 次；单进程、单场景、单 GPU。两个暂停文件必须全程保留，且不得访问 V5 队列或创建 episode。

## 已完成

- 实现 fail-closed Runner：校验场景清单、计划和源文件哈希、批准凭据、资源范围、输出路径以及暂停文件和 V5 manifest 的固定哈希。
- 实现 Kit worker：获批后才导入/启动 Kit；负责 native USD 保存与重开、场景绑定检查、PhysX 初始化、接触/释放/reset 及碰撞遥测、运行资源记录。门场景使用独立预注册验收合同。
- 计划、审批模板、Stage 2A checkpoint 和总进度 checkpoint 已更新。具体实现位于 `scripts/round4070_g/`，审批资料位于本目录的 `BIMANUAL_STAGE2_KIT_PHYSICS_*` 文件。
- 离线测试共 **21 项通过**：Runner 8 项、worker 7 项、场景 Builder 6 项；相关 Python 文件 `py_compile` 通过。
- 未获批执行检查按预期拒绝，退出码为 3；未创建 Stage 2A 输出目录，也未启动 Kit。

## 尚未完成与限制

- 未启动 Isaac Sim/Kit；native USD 数量为 0；Physics runs、Physics-valid 场景、双手 episode 和训练任务均为 0。
- 56 个候选场景状态未改变：53 个 `BUILDER_READY`，3 个因真实材料资产缺失而 `NOT_BUILT`。Builder-ready 不代表 USD 已生成或物理验证通过。
- 安装环境记录为 Isaac Sim 5.1.0.0、Isaac Lab 0.54.4、PhysX 107.3.26、Python 3.11.15、PyTorch 2.7.0+cu128、CUDA 12.8、RTX 4070 Laptop 8188 MiB；本阶段没有验证其真实物理运行稳定性。
- 当前 worker 不控制或验证 PhysX 内部 RNG。三个编号只能代表可复现的初始条件扰动重复，不能称为独立 PhysX-seeded runs；未增加独立验证前不得把场景状态提升为 `PHYSICS_VALID`。
- Stage 2A 不验证 V6 hand collider、G1 STOP/retract 或正式采集安全性。Stage 2B 与正式 G1 均需单独审查和授权。

## 安全状态

两个暂停文件和冻结 V5 manifest 均保持原样：

| 文件 | SHA-256 |
|---|---|
| `outputs/round4070_comprehensive_traversability/PAUSE.flag` | `3231de7b0f0c282178d2b92fb44b0b37ec57b16364e07cd873ccfe8fafffc5ae` |
| `outputs/round4070_comprehensive_traversability/PAUSE_G1.flag` | `b44cceffe2c948e9f13bbf10c1c1a63adf6d76629cf5cd8d8cd774720fa8b6f4f` |
| `outputs/round4070_comprehensive_traversability/campaign_manifest_v5.json` | `c311a6aa876bf5bae011148c3df5b6e87d986f16e6197bdb52bb67cfa828b200` |

当前仍需项目负责人和独立安全审查者分别对审批模板所绑定的确切计划、Runner、worker 与门合同哈希签字。审查者必须不是实现作者或项目负责人，并明确批准：两个暂停文件保持存在时的 Stage 2A scoped hold addendum、24 次测试及资源范围、不得运行 G1/V5 队列，以及 PhysX RNG 未验证所带来的 unresolved 结果限制。模板中的签字和证据字段目前仍为 pending。

## Git 交付与恢复条件

Stage 2A 实现已提交并推送到 `codex/round4070-g-comprehensive-traversability`：

- 实现提交：`e830c81` (`research: implement fail-closed bimanual stage2 runner`)
- 本总结文档另作独立提交；不包含工作区中的其他未提交研究文件。

只有当负责人和独立审查者完成上述精确哈希审批后，才可由 Runner 执行既定的 24 次 Physics-only 运行。运行结果仍须按真实证据分类；PhysX RNG 未验证时不得报告 `PHYSICS_VALID`。Stage 2A 通过也不解除 G1 暂停，不构成 Stage 2B 或正式双手采集授权。

**总状态：`IN_PROGRESS_NOT_COMPLETE`。** 本阶段尚未产出可供方法比较的双手物理数据。
