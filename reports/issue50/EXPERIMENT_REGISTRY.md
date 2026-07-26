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
