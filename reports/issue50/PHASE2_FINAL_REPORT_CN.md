# Issue #50 Phase 2 最终报告

## 问题与公平性

比较强微调基线、稳定 LoRA、AMP-safe LoRA 与 Partial FT + LoRA；同一数据集固定数据划分、imgsz、epoch、seed、增强和验证协议。VisDrone OOM 仅降低物理 batch，并以 nbs/梯度累积维持有效 batch。

## 结论

- Brain Tumor：best_absolute+best_parameter_efficiency+best_numerical_stability = `Last stage + Neck + Head`，mAP50-95=0.40021，trainable=0。
- VisDrone：best_absolute+best_parameter_efficiency+best_numerical_stability = `Full fine-tuning`，mAP50-95=0.15527，trainable=0。

- 稳定门控失败实验共 6 组，均保留但不进入正式结论。
- 若 Head-only/Full FT 精度更高，则其为绝对性能方案；LoRA 只从参数效率、稳定性和迁移成本评价。

## 创新证据分级

- 已验证：Adapter 学习率抑制、非有限梯度检测与健康检查点、rank 的精度—参数—稳定性权衡。
- Phase 2 验证后可判定：AMP-safe LoRA、Partial FT + LoRA。
- 初步证据：Head/Router/Adapter 分组学习率；当前 Head=1.0×、Router=0.5×、Adapter=0.1×。
- 尚未验证：跨更多数据集与大规模多种子统计；不得扩写为普遍精度优势。

完整逐运行数据见 `PHASE2_FINAL_RESULTS.csv`，多种子与 Pareto 结论见相邻 CSV。
