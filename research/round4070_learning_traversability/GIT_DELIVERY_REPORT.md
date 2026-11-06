# Round 4070-F Git 交付审计

开始时在本地 `main`，HEAD=`5ae2c6f`；相对本地缓存 `origin/main`（`1e18825a42bf5cc19a39bf63f76fa172869cc6cd`）为0落后/7领先。远端主分支没有被本轮修改。当前研究分支为 `codex/round4070-f-temporal-evidence`，继承上述7个既有本地提交。

## 已完成的本地提交

- `d4fb78341acac342e46e78a5d1abf5c894823ec8`：冻结Phase A因果训练代码、明确分组配置及实际结果表，15个有范围文件。
- 最终结果与物理验证提交：在本文件生成后的最终交付步骤执行；以分支日志和本文件后续更新为准。

所有新增源码/配置/小型结果/报告均限定在 `scripts/round4070_f/` 和 `research/round4070_learning_traversability/`。原始trial、处理时序、checkpoint和外部FALCON/Isaac资源保留本地outputs/既有外部目录，未纳入Git。没有使用无筛选的`git add .`。

## 既有工作保全

`lift_curtain/perception/online_occupancy.py`原有104行未提交增加保持不变，也未暂存；SHA256=`223908babf0fe72bd83d9a5f6801d85ed5d1a693e59ed0ea450f1b5da7207ffe`。其余原有未跟踪目录、报告和代码没有清理或覆盖。最终验证脚本复核该hash、312个E源文件hash、冻结学习快照与评价分母。

## 远端状态

远端：`https://github.com/ivywangming-gif/Lift-the-curtain.git`。受限上下文第一次只读查询出现`getaddrinfo() thread failed to start`；获准在正常网络上下文重试后成功，目标研究分支当时不存在。

最终push尚待最终结果提交及审计完成后执行。不会force-push；实际成功或阻止原因将在本文件最后记录。
