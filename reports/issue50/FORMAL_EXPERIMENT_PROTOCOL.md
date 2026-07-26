# YOLO-Master Issue #50 正式实验协议

## 目标与范围

本协议用于复现 Tencent/YOLO-Master Issue #50 的 LoRA 高效微调 rank 对比。正式矩阵比较 `r=4`、`r=8`、`r=16`；当前阶段仅允许运行 `--dry-run`，不得将链路验证或短跑结果作为正式结论。

训练入口为 `scripts/issue50/run_rank_sweep.py`，汇总入口为 `scripts/issue50/summarize_runs.py`。所有正式结果必须关联到唯一 Git commit 和对应的 `run_manifest.json`。

## 实验矩阵

| 场景 | YAML | Rank | Alpha | Epochs | Batch | ImgSz | Fraction | Seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Brain Tumor | `examples/lora_examples/yolo_master_brain_tumor_lora.yaml` | 4/8/16 | `2*r` | 40 | 16 | 640 | 1.0 | 0 |
| VisDrone | `examples/lora_examples/yolo_master_visdrone_lora.yaml` | 4/8/16 | `2*r` | 30 | 8 | 768 | 0.2 | 0 |

## Git 与版本控制

- `git add <文件>`：把指定文件加入暂存区，尚未形成历史记录。
- `git commit`：把暂存区内容保存为本地、带唯一 commit ID 的历史记录。
- `git push`：把已提交的本地 commit 上传到远程仓库；本地 commit 不会自动 push。
- `working tree clean` 只表示当前没有未提交改动，不表示 commit 已推送到远程。

每次正式训练前，在仓库根目录记录以下输出，并写入该组的 `run_manifest.json`：

```powershell
git branch --show-current
git rev-parse HEAD
git status --short
```

正式结果必须绑定唯一 Git commit。提交时禁止使用 `git add .` 和 `git add -A`；只允许明确添加以下文件：

```powershell
git add scripts/issue50/run_rank_sweep.py scripts/issue50/summarize_runs.py
git add reports/issue50/FORMAL_EXPERIMENT_PROTOCOL.md reports/issue50/EXPERIMENT_REGISTRY.md
```

不得提交 `runs/`、`datasets/`、`weights/`、训练日志、`__pycache__/`、`*.pyc`、`DRY_RUN_COMMANDS.txt` 或模型权重。

训练脚本会拒绝从脏工作区启动正式训练；`--dry-run` 仍会显示 `git_dirty`，用于提交前检查。

## 跨平台路径规则

禁止在脚本、协议或命令中写死 Windows 或 Linux 的个人绝对路径。脚本必须从自身位置定位仓库，再由 `pathlib` 组合目录：

```python
repo_root = Path(__file__).resolve().parents[2]
formal_root = (repo_root / "runs" / "issue50" / "formal").resolve()
logs_root = (formal_root / "logs").resolve()
weights_path = (repo_root / "weights" / "YOLO-Master-EsMoE-N.pt").resolve()
```

dry-run 在 Windows 显示 Windows 路径、在 Linux 显示 Linux 路径是正常现象；路径来源应始终是运行环境自动解析的 `repo_root`，而非写死的字符串。

## 配置优先级

发生冲突时，按以下优先级解释最终配置：

```text
args.yaml > 最终训练命令 > 场景 YAML > 框架默认值
```

因此，正式结果应以每组输出目录中的 `args.yaml` 为最终配置证据，并将完整命令保存到日志和 manifest。

## 正式实验固定条件

每一组必须显式设置：

```text
model=weights/YOLO-Master-EsMoE-N.pt
pretrained=True
resume=False
exist_ok=False
seed=0
deterministic=True
val=True
plots=True
save=True
```

同一数据集内部只允许改变 `lora_r`、`lora_alpha=2*lora_r` 和实验名称。不得借由改动数据划分、类别、评估设置、基础权重或 resume 状态制造不可比较的差异。

## 正式训练启动检查

从每组完整日志核验以下内容，并在汇总表或实验笔记中保留证据：

- `Transferred xxx/xxx items from pretrained weights`
- 检测头重新初始化与解冻状态
- `Trainable Params`
- `Adapter Params`
- `Frozen Base`
- optimizer 和 parameter groups
- gradient checkpointing 实际启用或跳过状态
- 是否出现 NaN 或非有限值

`run_manifest.json` 还必须记录分支、commit、工作区是否有改动、完整命令、路径、Python/PyTorch/CUDA/GPU/操作系统信息、开始和结束时间、退出码与成功状态。

## 数据与文件检查

正式训练前确认：

- 官方权重存在且文件大小大于 0。
- 两个场景 YAML 均存在。
- 对应数据集完整，且数据 YAML 指向预期数据集。
- 输出实验名称不重名，避免覆盖旧结果。
- `python -m py_compile scripts/issue50/run_rank_sweep.py scripts/issue50/summarize_runs.py` 通过。
- 至少对拟启动的一组执行单组 dry-run，例如：

```powershell
python scripts/issue50/run_rank_sweep.py --scene brain_tumor --ranks 4 --dry-run
```

## 正式结果保存清单

每组必须保留：

- `args.yaml`
- `results.csv` 与 `results.png`
- `weights/best.pt` 与 `weights/last.pt`
- 完整日志和 `run_manifest.json`
- 混淆矩阵
- PR、F1、P、R 曲线
- 验证预测图片

这些文件用于复核，不得直接提交至 Git。

## 云端迁移清单

迁移到云 GPU 前记录：

- 分支与 commit
- Python、PyTorch、CUDA、Ultralytics 版本
- GPU 型号与显存
- 数据集与官方权重路径
- `--dry-run` 输出的仓库、权重、输出和日志路径

云端先仅运行 Brain Tumor `r=4` 作为验收。确认该组权重迁移、LoRA 挂载、日志、评估图和结果保存均正常后，才运行剩余五组。

## 当前状态判断

正式实验基础设施、跨平台路径、日志和 manifest 记录机制已建立；尚无正式训练结果。正式结论必须等待云端首组验收和完整六组实验完成后再作出。
