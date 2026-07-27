# Issue #50 项目总日志

## 记录规则

- **日志事实**：训练日志、`results.csv`、`args.yaml` 或 checkpoint 直接证明。
- **源码机制**：当前 commit 的源码直接证明。
- **推断**：由多项事实支持但尚未完成控制实验。
- **假设**：等待单变量实验验证。

## 项目目标

在同一数据、权重、训练协议和随机种子下，完成 Brain Tumor 与 VisDrone 的 LoRA
`r=4/8/16` 公平比较，记录精度、参数、显存、速度和稳定性，并保证结果可复现。

## 当前阶段

Brain Tumor `r=4` 稳定配置搜索。尚未准许启动正式 `r=8/16`。

## 已完成工作

1. 建立跨平台正式训练与汇总脚本，避免 Windows/Linux 绝对路径硬编码。
2. 验证官方权重加载：760/811 项迁移；类别数变化造成 detection head 51 项不匹配并重新初始化。
3. 完成正式链路 `brain_tumor_r4_seed0`，发现 AMP adapter 梯度溢出及后续性能崩塌。
4. 审查 NaN recovery、`last_healthy.pt` 和 optimizer parameter groups。
5. 增加首个非有限梯度的参数名、step、loss 和参数组日志。
6. 完成 FP32、基础 LR 和 adapter LR multiplier 的单变量诊断。
7. `lr0=8e-4, lora_lr_mult=0.1, amp=False` 已连续训练 10 轮，无 recovery；最佳
   mAP50=0.12106、mAP50-95=0.06630。

## 关键决策

| 决策 | 原因 | 证据级别 |
|---|---|---|
| 正式 rank sweep 前先修复稳定性 | 原正式 r=4 在 epoch 1 recovery，epoch 4 起指标归零 | 日志事实 |
| 正式候选关闭 AMP | 首个非有限梯度在 epoch 1 step 0 的 LoRA adapter | 日志事实 |
| 显式 AdamW 且 `warmup_bias_lr=0` | 显式 AdamW 默认 bias warmup 曾产生 0.0804 的 pg2 LR | `results.csv` 事实 |
| 基础 LR 候选使用 `8e-4` | `3e-4` 欠拟合，`1.2e-3` 短跑波动更大 | 多个单变量实验 |
| 降低 adapter LR multiplier | multiplier=1.0 时有限数值下性能崩塌；0.1 时稳定提升 | 控制实验事实 |

## 目录导航

- 正式训练脚本：`scripts/issue50/run_rank_sweep.py`
- 汇总脚本：`scripts/issue50/summarize_runs.py`
- 正式协议：`reports/issue50/FORMAL_EXPERIMENT_PROTOCOL.md`
- 实验台账：`reports/issue50/EXPERIMENT_REGISTRY.md`
- 错误日志：`reports/issue50/ERROR_FIX_LOG.md`
- 创新记录：`reports/issue50/INNOVATION_AND_OPTIMIZATION_LOG.md`
- 初学者指南：`reports/issue50/LEARNING_GUIDE_CN.md`
- 诊断产物：`runs/issue50/diagnostics/`
- 正式产物：`runs/issue50/formal/`
- 数据集：`/root/autodl-tmp/datasets/brain-tumor`
- 接管前备份：`/root/autodl-tmp/issue50_before_codex_20260727_065340.tar.gz`

## 下一步

1. 完成 `lora_lr_mult=0.2` 的 10 轮单变量实验。
2. 在 0.1/0.2 中选择稳定候选，执行 40 轮 Brain Tumor `r=4` 正式验收。
3. 固定除 rank/alpha 外所有变量，完成 Brain Tumor `r=8/16`。
4. 单独检查 VisDrone 数据、类别、尺度和 batch，再进行稳定性验收和 rank sweep。

