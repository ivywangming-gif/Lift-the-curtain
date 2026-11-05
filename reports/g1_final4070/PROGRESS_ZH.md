# Final RTX 4070 Campaign 进度

## 运行记录

| 阶段 | 状态 | 证据 |
|---|---|---|
| 历史证据审计 | PASS | `00_INHERITED_EVIDENCE_AUDIT.md` |
| 资源基线 R0/R1/R2 | PASS | 6 条 resource baseline，均有完整 artifact |
| MATCHED_ACTION | PASS | 6/6 有效，0 runtime invalid |
| MATCHED_STATE | PASS | 8/8 有效，0 runtime invalid |
| KEY_REPEAT | PASS | 4/4 有效，0 runtime invalid |
| deployment leakage audit | PASS | 188 decisions，0 forbidden field |
| 新学习训练 | NOT RUN | 数据不满足 held-out 资格，按预注册规则停止 |
| renderer/RGB-D | BLOCKED/NOT RUN | 已知 native crash 分支未重启 |
| 真实 G1 | NOT RUN | 本轮仅做 Isaac 仿真 |
| 交付包完整性与可恢复复现 | PASS | 27 个报告/机器输出 SHA256 自洽；`REPRODUCE.ps1` 分析路径通过 |

## 实际命令

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/g1_final4070/run_queue.ps1 -Batch RESOURCE_BASELINE -MaxNewLaunches 6
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/g1_final4070/run_queue.ps1 -Batch MATCHED_ACTION -MaxNewLaunches 6
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/g1_final4070/run_queue.ps1 -Batch MATCHED_STATE -MaxNewLaunches 8
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/g1_final4070/run_queue.ps1 -Batch KEY_REPEAT -MaxNewLaunches 4
C:\conda_envs\isaaclab\python.exe scripts/g1_final4070/analyze_campaign.py
C:\conda_envs\isaaclab\python.exe scripts/g1_final4070/finalize_campaign.py
```

## 当前统计

- 计划：24 条。
- 启动：24 条。
- 有效完成：24 条。
- runtime invalid：0 条。
- S2：5，S3：13，S4：4，control：2。
- Rigid 5/5 安全拒绝并到达 PointGoal。
- 柔性 R1 clearance：4/12；R2 clearance：0/10。

## 当前裁决

`P1=SUPPORTED`，`P2=PARTIALLY_SUPPORTED`，`P3=NOT_ESTABLISHED`。

本轮不再启动没有新因果信息的物理 trial。最终裁决见 `FINAL_4070_SCIENTIFIC_VERDICT_ZH.md`。

最终交付状态：`PHYSICAL_CAMPAIGN_COMPLETE_FINAL_REPORTS_READY`。未提交 Git commit，未 push，未修改历史 trial 或第三方资产。

