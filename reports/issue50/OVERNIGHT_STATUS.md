# Issue #50 无人值守状态

## 当前状态

- 更新时间：2026-07-27 08:35（Asia/Shanghai）
- 当前运行实验：准备 `brain_tumor_r4_ampoff_lr8e4_adapt01_router025_e10`
- 当前 PID：待启动后更新
- 后台方式：`nohup + setsid`，无 TTY，与 SSH 会话解耦
- GPU：RTX 4090 24GB；启动前空闲
- Git 分支：`issue-50-lora-reproduction`
- 本地 GitHub dry-run：成功，无需人工认证
- push 状态：云端旧 commit 尚未推送；正在无损同步回本地主库后统一 push

## 刚完成实验

### `brain_tumor_r4_ampoff_lr8e4_adapt01_e10`

- 日志：`runs/issue50/diagnostics/logs/brain_tumor_r4_ampoff_lr8e4_adapt01_e10/train.log`
- 时间：08:09:07–08:12:31
- 退出码：0
- 最佳 epoch：8
- Precision/Recall：0.30352 / 0.08437
- mAP50/mAP50-95：0.12106 / 0.06630
- 稳定性：无 NaN、Inf、gradient overflow 或 recovery
- 结论：adapter LR 为 head/base 的 0.1 倍时，消除了 multiplier=1.0 的 epoch-4 崩塌。

### `brain_tumor_r4_ampoff_lr8e4_adapt02_e10`

- 日志：`runs/issue50/diagnostics/logs/brain_tumor_r4_ampoff_lr8e4_adapt02_e10/train.log`
- 时间：08:28:06–08:30:55
- 退出码：0
- 最佳 epoch：4
- Precision/Recall：0.21461 / 0.09292
- mAP50/mAP50-95：0.08742 / 0.04303
- 稳定性：无 NaN、Inf、gradient overflow 或 recovery
- 结论：稳定但显著弱于 multiplier=0.1，当前不作为首选。

## 下一组实验依据

保持 `amp=False, AdamW, lr0=8e-4, lora_lr_mult=0.1`，只把
`moe_router_lr_scale=0.5→0.25`。假设是小数据集上降低 router 漂移可能进一步改善验证稳定性；
若无提升，保留默认 0.5，不再扩展 router sweep。

## 醒来后首先查看

1. 本文件“当前状态”和最新实验结果。
2. `EXPERIMENT_REGISTRY.md` 中是否已通过正式准入门。
3. Brain Tumor 正式 r=4 是否已启动或完成。
4. GitHub 当前分支是否已 push。

