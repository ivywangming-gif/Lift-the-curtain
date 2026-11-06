# G轮启动环境与保全审计

本轮起点HEAD：`1d5136d85066c922b29e4901b18edd7fc9e977f7`；原分支`codex/round4070-f-temporal-evidence`。实际`git fetch origin`成功，起点与远端F分支0领先/0落后。新建分支`codex/round4070-g-comprehensive-traversability`，没有merge/reset/stash/clean。

原`lift_curtain/perception/online_occupancy.py`仍有104行未提交增加；SHA256为`223908babf0fe72bd83d9a5f6801d85ed5d1a693e59ed0ea450f1b5da7207ffe`。既有未跟踪文件保留。`audit_preservation.py --freeze`已实际冻结2012个历史源码、报告、数据与checkpoint文件，共729999234字节；新一轮验证只读这些文件，不重写F结果。完整清单在新G outputs的`prior_work_snapshot.json`，小型结果见`PRESERVATION_AUDIT.json`。

实际环境：Windows；Python3.11.15，torch2.7.0+cu128，numpy1.26.0，IsaacLab0.54.4；既有IsaacSim5.1.0.0运行证据来自历史启动日志。未升级或安装环境。GPU为RTX4070 Laptop 8188MiB，driver610.60；启动审计时185MiB占用、0%利用率、38°C，无Isaac进程。物理RAM31.69GiB，当时可用7.39GiB；此值随其他应用变化，不能与仿真进程private内存简单相减后推断OOM。

Windows受限上下文的CIM查询失败，改用现有psutil读取内存；不关闭用户其他应用。物理采集沿用headless、single-env、无RTX渲染和每条独立watchdog/telemetry；训练需独立资源窗口与原子互斥锁。失效/超时只清理本任务拥有的进程。

队列预计耗时基于F实际日志，初始全面覆盖约20–40小时、完整重复目标可能数天。这是规划估计，不能作为已完成证据。各阶段真实完成度与可恢复命令由执行账本生成；退出会话不表示任务会自动继续。
