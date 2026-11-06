# 复现

工作目录：`C:/Robotics/Lift-the-curtain`；既有环境 `C:/conda_envs/isaaclab/python.exe`，torch 2.7.0+cu128、numpy 1.26.0。不安装/升级任何依赖；没有sklearn，因此Logistic使用torch，GradientBoosting为代码内明确实现的Bernoulli梯度提升决策桩（60轮），不是声称完整调参XGBoost。

下列记录实际执行入口。重新训练请在独立工作副本，或为`train_models.py`指定新的`--run-name`，保留现有冻结checkpoint/哈希；`run_offline_completion.py`使用本轮固定输出名，应在独立副本复现。物理重跑会产生新的实际数据，不等于逐字节重现原GPU仿真轨迹。

```powershell
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/build_dataset.py
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/test_contracts.py
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/train_models.py --sides --run-name phase_a_valid
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/summarize_models.py --run phase_a_valid
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/map_contract.py
```

CPU、2线程、3初始化4070/4071/4072；GRU hidden64一层，TCN两个小卷积，MLP hidden64。固定超参、固定60epoch最终权重，无测试集早停或选模。每fold训练控制+可靠核心标签；仅训练组UNKNOWN参与输入归一化，正常响应Ridge只使用训练空场，监督BCE排除UNKNOWN。标签不平衡未通过复制正例伪增样本量。

`phase_a_valid/frozen_config.json` 保存确切划分、manifest哈希；每模型`.pt`含state、seed、feature_names、scaler、train IDs；正常响应`.npz`保存Ridge系数。残差预测 q/effort，排除最近0.4s状态，使用当前已执行命令；前0.4s残差置零且不进入预测器拟合。它是CoPRE-style简化响应基线，非CoPRE原文全实现，无Jacobian聚合、无力重建。

所有时序重采样为48个真实时间上的past-hold值，实际日志只有20–22点，不创造新信息。删除显式elapsed通道，保留机器人自身q/qdot、实际命令、合法action、applied torque proxy、双手相对FK和机器人身体状态。模型输入165维；完整M2再添加正常响应残差。模型前向计时包含单次tensor构造，**不包含**CSV读取、FK、响应校正和预处理；非实时端到端认证。

物理队列必须在学习进程退出后执行，每次单环境、headless、冻结FALCON、原有guard/heartbeat/resource telemetry；不运行RTX render。新trial有独立输出、预注册条件与seed，已存在完整结果跳过，不覆盖不完整输出。

```powershell
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/physics_queue.py --stage screen
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/physics_queue.py --stage confirm
# 只有screen产生可信PASSABLE才允许posture（脚本内也有gate）
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/physics_queue.py --stage posture
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/physics_queue.py --stage p1
# 构建/校验首批11条对应的115条训练快照（原输入已归档）
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/build_dataset.py --supplementary-labels outputs/round4070_learning_traversability/physics/first11_frozen.json
```

首次`phase_a/`原型运行作废：审查发现按总长度取前缀和残差warmup问题，修正后完整重跑为`phase_a_valid/`。保留原始失败/作废执行记录用于审计，不参与最终表。源数据不在Git中；checkpoint保留本地outputs，不提交大文件。仅有Git源码不能凭空重建未公开原始仿真数据，复现须具备报告所列原日志与冻结外部环境。

`map_contract.py` 是synthetic/oracle示例：GR Map的2候选接受三态判断，仅更新当前候选，后方区域不自动清空。探索模型默认shadow模式；这不是Grounding DINO/RGB-D已运行的端到端导航。未训练FALCON/PPO/WBC、VLM、Transformer或world model。

## 新增数据、冻结评价与补充审计

```powershell
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/evaluate_frozen.py
# 以下实际在Isaac全部退出的独立CPU窗口运行；包括Phase C与严格瞬时E消融
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/run_offline_completion.py
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/summarize_models.py --run phase_c --prefix PHASE_C_
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/summarize_models.py --run phase_a_instantaneous --prefix STRICT_INSTANT_
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/converged_logistic.py
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/ensemble_diagnostics.py
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/plot_results.py
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/audit_delivery.py
```

`run_offline_completion.py`用相同架构、epoch、seed和分组规则重训；增加的`Instantaneous`删除六个从probe首帧累计的手部进展通道，只取最后观测。原`SingleFrame`保留为包含累计进展的最后一行摘要，原结果不覆写。严格瞬时E消融独立保存到`phase_a_instantaneous`。

`converged_logistic.py`是独立后续优化/正则化敏感性审计：C=1，mean BCE+.5/n×权重平方和，截距不正则，float64 LBFGS strong_wolfe、max_iter200；实际6/6达到梯度门限。该目标与原AdamW Logistic不同，不能宣称是单纯更换优化器的因果实验。结果独立保存到`converged_logistic/`及`CONVERGED_LOGISTIC_RESULTS.csv`。

物理执行遇到沙箱启动上下文问题，后续经获准的相同环境重试；原失败日志保留。可用`physics_queue.py --stage screen --retry-tag <new_id>`产生独立新输出。P1在Isaac启动前的边界暂停曾达到360s等待上限，属于execution-invalid，不是物理失败；对应场景需新目录真实重试，详见物理报告。所有实际步骤与guard日志保存在独立outputs。

本轮PLEATED真实补跑入口为`physics_queue.py --stage p1 --p1-scene PLEATED_SINGLE --p1-first-seed 420074`；滑轨机制诊断为`--stage rail_mobility`（需要滑轨先前四门已通过）；最后两条左手为`--stage left_completion`。实际已完成，不必重新启动这些物理进程来查看报告。

最终标签文件包含13条，直接用它覆盖115条已训练数据会被拒绝。最终117条须独立导出：

```powershell
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/build_dataset.py --supplementary-labels outputs/round4070_learning_traversability/physics/supplementary_labels.json --output-dir outputs/round4070_learning_traversability/dataset_final --report-prefix PHASE_C_FINAL_
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/build_dataset.py --supplementary-labels outputs/round4070_learning_traversability/physics/supplementary_labels.json --output-dir outputs/round4070_learning_traversability/dataset_left_completion --report-prefix LEFT_COMPLETION_ --trial-ids F_LEFT_COMPLETION_nominal_left_S420401 F_LEFT_COMPLETION_bottom_right_anchor_left_S420402
& C:/conda_envs/isaaclab/python.exe scripts/round4070_f/evaluate_frozen.py --f-manifest outputs/round4070_learning_traversability/dataset_left_completion/manifest.json --output outputs/round4070_learning_traversability/frozen_left_completion --report-prefix LEFT_COMPLETION_
```

两条追加left均UNKNOWN，没有新监督标签，因此不再重训；Phase C数字对应115条快照，最终可供响应学习的数据总量为117。独立左手预测不覆盖首批F评价，监督分母0的指标保持空值。
