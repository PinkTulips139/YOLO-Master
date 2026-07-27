# Issue #50 Experiment Registry

## 实验状态说明

| Run | 目的 | 基础权重 | Rank | 训练规模 | 是否纳入正式结果 |
|---|---|---|---:|---|---|
| brain_tumor_smoke | 验证训练链路 | 未明确加载官方 .pt | 4 | 1 epoch，小比例数据 | 否 |
| brain_tumor_r4_e3_f010 | 验证指标与结果保存 | 未明确加载官方 .pt | 4 | 3 epochs，10% 数据 | 否 |
| brain_tumor_pretrained_check | 检查 pretrained=True | 只使用模型 YAML | 4 | 1 epoch，1% 数据 | 否 |
| brain_tumor_pretrained_pt_check | 核验官方权重加载 | YOLO-Master-EsMoE-N.pt | 4 | 1 epoch，1% 数据 | 否 |

## 已确认事实

- 官方预训练权重成功迁移 760/811 个 state_dict 条目。
- 检测头因类别不匹配重新初始化并解冻训练。
- LoRA Adapter 和检测头参数参与更新。
- MoE 模型下 gradient checkpointing 被运行时跳过。
- 前面的随机冻结实验只作为工程链路测试，不进入正式结果。

## 下一步

云端先使用官方预训练权重运行 Brain Tumor `r=4` 作为单组验收；验收通过后再运行剩余五组正式实验。

## 2026-07-27 云端诊断

| Run | 单变量 | Epochs | 最佳 mAP50 | 最佳 mAP50-95 | 稳定性 | 结论 |
|---|---|---:|---:|---:|---|---|
| `brain_tumor_r4_seed0` | 正式链路基线 | 17/40 | 0.13520 | 0.06378 | epoch 1 step 0 AMP adapter 梯度非有限并恢复；epoch 4 起指标归零 | 不作为稳定正式结果 |
| `brain_tumor_r4_lr3e4` | 原始低 LR 诊断 | 10 | 0.05275 | 0.02593 | 未保存完整日志；bias warmup LR 首轮达 0.0804161 | 存在 warmup 混杂，不能归因于 `lr0` |
| `brain_tumor_r4_ampoff_e3` | `amp=False` | 3 | 0.05718 | 0.03007 | 无 recovery、loss 全部有限 | FP32 消除首次数值异常，但 3 轮性能不足 |
| `brain_tumor_r4_amp_probe_e1` | 仅增加非有限梯度观测 | 1 | 0.04418 | 0.02424 | 首次异常为 epoch 1 step 0 的 `model.4.conv.lora_A` adapter 梯度 | AMP adapter 梯度溢出可确定性复现 |
| `brain_tumor_r4_ampoff_lr3e4_e3` | FP32 基础 LR `0.001667→0.0003` | 3 | 0.02995 | 0.01676 | 无 recovery、loss 全部有限 | 稳定但新检测头学习偏慢 |
| `brain_tumor_r4_ampoff_lr8e4_e3` | FP32 基础 LR `0.0003→0.0008` | 3 | 0.04977 | 0.02801 | 无 recovery、loss 全部有限 | 当前优先稳定候选 |
| `brain_tumor_r4_ampoff_lr12e4_e3` | FP32 基础 LR `0.0008→0.0012` | 3 | 0.05319 | 0.02707 | 无 recovery；epoch 3 指标明显回落 | 峰值略高但波动较大 |
| `brain_tumor_r4_ampoff_lr8e4_e10` | 将 `0.0008` 候选扩展到 10 epoch | 10 | 0.04410 | 0.02245 | 无 NaN/Inf/recovery；epoch 4 性能崩塌 | adapter 倍率 1.0 不稳定 |
| `brain_tumor_r4_ampoff_lr8e4_adapt01_e10` | adapter multiplier `1.0→0.1` | 10 | 0.12106 | 0.06630 | 无 NaN/Inf/recovery | 最佳 epoch 8；稳定候选 |
| `brain_tumor_r4_ampoff_lr8e4_adapt02_e10` | adapter multiplier `0.1→0.2` | 10 | 0.08742 | 0.04303 | 无 NaN/Inf/recovery | 最佳 epoch 4；明显弱于 0.1 |
| `brain_tumor_r4_ampoff_lr8e4_adapt01_router025_e10` | router LR scale `0.5→0.25` | 10 | 0.12106 | 0.06630 | 无 NaN/Inf/recovery | 与 0.5 逐轮一致；停止 router sweep |
| `brain_tumor_r4_stable_v1_seed0` | Stable V1 正式 r=4 | 23/40 | 0.11416 | 0.06395 | 稳定；无 NaN/Inf/recovery | 最佳 epoch 8；P=0.39918，R=0.06107，325.1s，峰值 7.46GiB |
| `brain_tumor_r8_stable_v1_seed0` | 唯一变量 rank `4→8`，alpha=16 | 19/40 | 0.11313 | 0.06180 | 稳定；无 NaN/Inf/recovery | 最佳 epoch 4；P=0.43074，R=0.06191，271.5s，峰值 7.58GiB |
| `brain_tumor_r16_stable_v1_seed0` | 唯一变量 rank `8→16`，alpha=32 | 33/40 | 0.09309 | 0.04821 | 稳定；无 NaN/Inf/recovery | 最佳 epoch 18；P=0.01114，R=0.26982，454.1s，峰值 7.58GiB |
| `visdrone_r4_preflight_e1` | 官方 YAML 下载与链路烟雾测试 | 1 | 0.00000 | 0.00000 | 稳定；退出码 0，产物完整 | 6471/548/1610 图像及同量标签通过；仅验证链路，不进入正式结果 |
| `visdrone_r4/r8/r16_stable_v1_seed0` | VisDrone 固定配置 rank sweep | 30 | pending | pending | queued | 烟雾测试通过后依次运行 |

### 已定位的参数组

- `pg0=weight`、`pg1=bn/no-decay`、`pg2=bias`、`pg3=router`、`pg4/pg5=adapter`。
- router 使用 `moe_router_lr_scale=0.5`；adapter 基础倍率为 `lora_lr_mult=1.0`，其中 layer-wise decay 产生第二个 adapter 组。
- detection head 未单独分组，其 weight、normalization 和 bias 分别进入 `pg0`、`pg1`、`pg2`。

### 恢复机制结论

- `last_healthy.pt` 在训练开始前建立可执行的完整在线快照，接受的有限 epoch 后原子刷新。
- 非有限 loss、fitness、gradient 或 EMA 任一标志都会触发恢复；最多连续恢复 3 次。
- AMP loss/gradient 异常时恢复器关闭 AMP，并从健康 checkpoint 恢复 optimizer、scaler 和 EMA。
- LoRA 在线模型恢复当前仅载入 adapter 张量；这是后续需要验证的恢复一致性风险。

## 正式实验计划

| Run | 场景 | Rank | Alpha | Epochs | Batch | ImgSz | Fraction | Seed | 输出目录 | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| brain_tumor_r4_seed0 | Brain Tumor | 4 | 8 | 40 | 16 | 640 | 1.0 | 0 | `runs/issue50/formal/brain_tumor_r4_seed0` | planned |
| brain_tumor_r8_seed0 | Brain Tumor | 8 | 16 | 40 | 16 | 640 | 1.0 | 0 | `runs/issue50/formal/brain_tumor_r8_seed0` | planned |
| brain_tumor_r16_seed0 | Brain Tumor | 16 | 32 | 40 | 16 | 640 | 1.0 | 0 | `runs/issue50/formal/brain_tumor_r16_seed0` | planned |
| visdrone_r4_seed0 | VisDrone | 4 | 8 | 30 | 8 | 768 | 0.2 | 0 | `runs/issue50/formal/visdrone_r4_seed0` | planned |
| visdrone_r8_seed0 | VisDrone | 8 | 16 | 30 | 8 | 768 | 0.2 | 0 | `runs/issue50/formal/visdrone_r8_seed0` | planned |
| visdrone_r16_seed0 | VisDrone | 16 | 32 | 30 | 8 | 768 | 0.2 | 0 | `runs/issue50/formal/visdrone_r16_seed0` | planned |

### 正式脚本

- 训练计划脚本：`scripts/issue50/run_rank_sweep.py`
- 汇总脚本：`scripts/issue50/summarize_runs.py`
- 协议文档：`reports/issue50/FORMAL_EXPERIMENT_PROTOCOL.md`
- 汇总输出：`reports/issue50/FORMAL_RESULTS_SUMMARY.csv`

### 约束

- 正式训练必须显式加载 `weights/YOLO-Master-EsMoE-N.pt`。
- 每组实验独立启动，`resume=False`，`exist_ok=False`。
- 训练产物写入 `runs/issue50/formal/`，日志写入 `runs/issue50/formal/logs/`。
- `runs/`、`weights/` 和权重文件不进入 Git 提交。

## 实验基础设施状态

- 正式脚本已固定六组实验矩阵、官方预训练权重、`resume=False`、`exist_ok=False`、确定性评估与结果保存开关。
- 脚本使用 `pathlib` 从脚本位置动态定位仓库、权重、输出和日志目录，避免依赖 Windows 或 Linux 的写死绝对路径。
- 每组正式运行将在日志目录写入 `run_manifest.json`，记录 Git 分支/commit/脏工作区状态、完整命令、运行环境、开始结束时间和退出状态；dry-run 仅打印预览，不生成 manifest。
- 汇总脚本可在 manifest 存在时读取其 Git 与运行环境信息；旧实验缺少 manifest 时仍保持兼容。
- 当前没有新增正式训练结果；表中的六组状态仍为 `planned`。
