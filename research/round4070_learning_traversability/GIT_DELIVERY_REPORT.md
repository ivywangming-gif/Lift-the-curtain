# Round 4070-F Git 交付审计

开始时在本地 `main`，HEAD=`5ae2c6f`；相对本地缓存 `origin/main`（`1e18825a42bf5cc19a39bf63f76fa172869cc6cd`）为0落后/7领先。远端主分支没有被本轮修改。当前研究分支为 `codex/round4070-f-temporal-evidence`，继承上述7个既有本地提交。

## 已完成的本地提交

- `d4fb78341acac342e46e78a5d1abf5c894823ec8`：冻结Phase A因果训练代码、明确分组配置及实际结果表，15个有范围文件。
- `44bbd63e6a35e5c30814cbc5828d081afdef08b1`：最终结果、Phase C模型比较、六类P1物理验证、117条数据审计与完整研究报告，56个有范围文件，暂存内容约1.06 MB。
- 本交付记录随后作为单独文档提交；其hash以分支日志为准，避免在文件中自指尚未生成的提交hash。

所有新增源码/配置/小型结果/报告均限定在 `scripts/round4070_f/` 和 `research/round4070_learning_traversability/`。原始trial、处理时序、checkpoint和外部FALCON/Isaac资源保留本地outputs/既有外部目录，未纳入Git。没有使用无筛选的`git add .`。

## 既有工作保全

`lift_curtain/perception/online_occupancy.py`原有104行未提交增加保持不变，也未暂存；SHA256=`223908babf0fe72bd83d9a5f6801d85ed5d1a693e59ed0ea450f1b5da7207ffe`。其余原有未跟踪目录、报告和代码没有清理或覆盖。最终验证脚本复核该hash、312个E源文件hash、331个冻结训练/评价文件、117条最终数据与所有评价分母；17个Python源文件编译通过，5项契约测试通过。

## 远端状态

远端：`https://github.com/ivywangming-gif/Lift-the-curtain.git`。受限上下文第一次只读查询出现`getaddrinfo() thread failed to start`；获准在正常网络上下文重试后成功，目标研究分支当时不存在。

2026-11-06实际执行`git push -u origin codex/round4070-f-temporal-evidence`成功，新建远端研究分支并设置upstream；没有force-push，没有更新远端main。随后只读`git ls-remote --heads origin codex/round4070-f-temporal-evidence`核验远端为`44bbd63e6a35e5c30814cbc5828d081afdef08b1`，与完整结果提交一致。

远端分支：[codex/round4070-f-temporal-evidence](https://github.com/ivywangming-gif/Lift-the-curtain/tree/codex/round4070-f-temporal-evidence)。本记录提交后再以普通push同步；最后交付回复报告该文档提交的实际同步状态。
