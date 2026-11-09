# BIMANUAL_V1 运行说明

当前仅允许离线 Stage 1 审计。

    Set-Location -LiteralPath 'C:\Robotics\Lift-the-curtain'
    & 'C:\conda_envs\isaaclab\python.exe' scripts/round4070_g/audit_bimanual_scene_matrix.py

该脚本不会启动 Isaac，不会修改暂停标志，不会重置 manifest，也不会创建 G1 attempt。正式物理命令在安全审批、预算审批和 Stage 3 双手动作认证完成前不具备准入条件。
