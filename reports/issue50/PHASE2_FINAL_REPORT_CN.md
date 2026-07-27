# Issue #50 Phase 2 审计修正版报告

## 审计口径

本报告仅由既有 args.yaml、results.csv、权重元数据、独立日志和 run_manifest 重建，未启动训练。
队列执行统计与正式有效性分开：队列为 15 completed /
11 failed（共 26 项）；正式有效性由完整产物、
exit_code、数值稳定、OOM/恢复和协议一致性共同门控。

## 关键结果

- Brain Tumor 单次绝对最佳：Last stage + Neck + Head，seed=2，mAP50-95=0.40944。
- VisDrone 干净正式单次绝对最佳：Full fine-tuning，seed=0，mAP50-95=0.15527。
- Brain Tumor / Last stage + Neck + Head: mAP50-95=0.39172 ± 0.02316（样本标准差，n=3）
- Brain Tumor / Stable LoRA: mAP50-95=0.05741 ± 0.00926（样本标准差，n=2）
- VisDrone / Full fine-tuning: mAP50-95=0.15342 ± 0.00221（样本标准差，n=3）

Stable LoRA 仅表示数值稳定，不代表绝对精度最佳。当前两个数据集的 LoRA 均无绝对精度优势；
可信贡献是 AMP/非有限梯度诊断、失败证据审计、参数效率与精度—显存—时间联合分析。

## 审计修正

- 非 LoRA 与 Partial FT + LoRA 的可训练参数量由 checkpoint 参数名和冻结协议静态统计，不再读取末尾验证模型的“0 gradients”。
- Phase 1 正式 `r4_stable_v*` 运行识别为 Stable LoRA。
- VisDrone Full FT `_b8` 三组均在 OOM 后自动降为实际 batch=4，作为恢复诊断，不与干净 `_b4` 正式运行聚合。
- VisDrone Head-only b4/b2/b1 因旧 EXISTING 映射忽略 batch，未独立执行；标为 skipped_duplicate_reference/not_executed。
- AMP-safe LoRA 在训练前因 `Conv2d.amp_safe_forward` 缺失而 implementation_failed，属于未验证方案。
- 多种子聚合按数据集、方法、真实配置签名分组，每组 seed 唯一，使用样本标准差。
- Pareto 为稳定门控后的四目标非支配前沿：最大化 mAP50-95，最小化参数量、峰值显存和训练时间。

## 证据等级与局限

日志直接证明 OOM、自动降批次和 AMP-safe 实现异常；args.yaml 直接证明实际 batch；manifest 直接证明退出码；
checkpoint 参数名与冻结协议直接支持参数量静态统计。Adapter 特征漂移仍属合理推断；
Adapter-only FP32/独立 GradScaler 尚未实现。需要新训练的最小项另见 `PHASE3_RECOMMENDED_EXPERIMENTS.md`。
