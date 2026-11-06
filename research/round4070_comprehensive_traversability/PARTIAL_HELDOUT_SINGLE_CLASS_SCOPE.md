# 单类留出诊断范围

冻结规则仅依据真实标签支持、从不依据预测结果：测试核心样本恰好一个类别时，单列100%标签预算、五个seed与同一预注册模型/工程基准。训练核心仍须双类，各方法原有可执行门槛保持；使用相同固定末epoch，不用测试或校准选checkpoint。原8个完整split状态保持不可评价。

校准缺双类时，temperature/risk/conformal明确不可用，仍可报告raw forced-0.5与预设拒判策略的该类计数、错误率和覆盖率。BA/AUROC/双类selective risk为NA；class_conditional_selective_error仅描述当前支持类别。单类B不能回答P召回，单类P不能回答false pass；五seed只是优化重复，不是五倍独立物理样本。

结果只写PARTIAL_HELDOUT_SINGLE_CLASS文件，不并入主学习/OOD/跨手/标签效率/校准结果，不用于宣称完整OOD或跨手泛化优势；confirmation仍独立且不参与训练/选择。空表表示没有实际完成的该诊断。
