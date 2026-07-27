#!/usr/bin/env python3
"""Wait for the Phase 2 queue, then audit, summarize, plot, commit, and archive its results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


POLL_SECONDS = 300
NONFINITE_RE = re.compile(
    r"Non-finite gradient|NaN recovery|Global nonfinite training state|loss (?:is )?nan",
    re.IGNORECASE,
)
OOM_RE = re.compile(r"CUDA out of memory|torch\.OutOfMemoryError", re.IGNORECASE)
SOFTWARE_RE = re.compile(r"FileNotFoundError.*(?:yolo|launcher)|Connection reset|Broken pipe", re.IGNORECASE)
HARD_BLOCK_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|Non-finite|NaN recovery|Dataset.*not found|SyntaxError|"
    r"not a valid YOLO argument|ConfigurationError",
    re.IGNORECASE,
)
PARAM_RE = re.compile(r"Trainable:\s*([0-9,]+)\s*\(([0-9.]+)%\)")
GRADIENT_RE = re.compile(r"([0-9,]+)\s+gradients")
GPU_RE = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)G\b")
SPEED_RE = re.compile(r"Speed:.*?([0-9.]+)ms inference", re.IGNORECASE)
METRIC_KEYS = ("precision", "recall", "mAP50", "mAP50_95")
FINAL_FIELDS = (
    "dataset",
    "method",
    "run_name",
    "seed",
    "rank",
    "alpha",
    "amp_strategy",
    "head_lr_scale",
    "router_lr_scale",
    "adapter_lr_scale",
    "physical_batch",
    "effective_batch",
    "precision",
    "recall",
    "mAP50",
    "mAP50_95",
    "best_epoch",
    "trainable_params",
    "trainable_pct",
    "peak_gpu_mem_gib",
    "train_time_s",
    "inference_ms_per_image",
    "nan_inf_recovery",
    "artifacts_complete",
    "exit_code",
    "classification",
    "result_dir",
    "log_path",
)
KNOWN_PARAMS = {
    ("Brain Tumor", "Head-only"): (347718, 13.0595),
    ("Brain Tumor", "Full fine-tuning"): (2662546, 99.9994),
    ("Brain Tumor", "Stable LoRA"): (409174, 15.0070),
    ("VisDrone", "Head-only"): (349278, 13.1104),
    ("VisDrone", "Stable LoRA"): (410734, 15.0560),
}


def root_path() -> Path:
    return Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def process_ids(pattern: str) -> list[int]:
    completed = subprocess.run(
        ["pgrep", "-f", pattern], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False
    )
    return [int(value) for value in completed.stdout.split() if value.isdigit() and int(value) != os.getpid()]


def read_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def safe_recover_queue(root: Path, queue_log: Path, state: dict, recovered: bool) -> bool:
    """Restart a disappeared queue once, only for a narrow transient software failure."""
    if recovered or state.get("status") != "running":
        return False
    tail = queue_log.read_text(encoding="utf-8", errors="replace")[-12000:] if queue_log.exists() else ""
    if HARD_BLOCK_RE.search(tail) or not SOFTWARE_RE.search(tail):
        return False
    log_handle = queue_log.open("a", encoding="utf-8")
    subprocess.Popen(
        [
            "flock",
            "-n",
            str(root / "runs/issue50/phase2_queue.lock"),
            sys.executable,
            str(root / "scripts/issue50/phase2_performance_queue.py"),
            "--device",
            "0",
            "--launcher",
            "/root/miniconda3/bin/yolo",
        ],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return True


def wait_for_queue(root: Path, status_path: Path, final_status: Path) -> dict:
    queue_state_path = root / "runs/issue50/phase2_queue_status.json"
    queue_log = root / "runs/issue50/phase2_queue.log"
    recovered = False
    while True:
        state = read_state(queue_state_path)
        queue_pids = process_ids(r"[p]hase2_performance_queue\.py")
        gpu_pids = process_ids(r"/root/miniconda3/bin/yolo train")
        atomic_json(
            final_status,
            {
                "status": "waiting",
                "current_stage": "waiting_for_queue",
                "current_experiment": state.get("current"),
                "last_updated": now(),
                "completed_count": len(state.get("completed", [])),
                "failed_count": len(state.get("failed", [])),
                "error_summary": "",
                "updated_at": now(),
                "queue_status": state.get("status", "missing"),
                "queue_pids": queue_pids,
                "gpu_training_pids": gpu_pids,
                "current": state.get("current"),
                "recovery_used": recovered,
            },
        )
        if state.get("status") == "completed" and not queue_pids and not gpu_pids:
            return state
        if not queue_pids and state.get("status") == "running":
            recovered = safe_recover_queue(root, queue_log, state, recovered)
            if not recovered:
                atomic_json(
                    final_status,
                    {
                        "status": "blocked",
                        "current_stage": "blocked",
                        "current_experiment": state.get("current"),
                        "last_updated": now(),
                        "completed_count": len(state.get("completed", [])),
                        "failed_count": len(state.get("failed", [])),
                        "error_summary": "Queue disappeared and was not safely recoverable.",
                        "updated_at": now(),
                        "reason": "Queue disappeared; failure is not a narrowly recoverable launcher/network error.",
                        "queue_state": state,
                    },
                )
                raise SystemExit(2)
        time.sleep(POLL_SECONDS)


def method_from_name(name: str) -> str:
    ordered = (
        ("last_stage_neck_head", "Last stage + Neck + Head"),
        ("full_finetune", "Full fine-tuning"),
        ("amp_safe_lora", "AMP-safe LoRA"),
        ("partial_lora", "Neck + Head + LoRA"),
        ("stable_lora", "Stable LoRA"),
        ("neck_head", "Neck + Head"),
        ("head_only", "Head-only"),
        ("adapt01", "Adapter LR x0.1"),
        ("lr8e4_e10", "Adapter LR x1.0"),
        ("amp_probe", "Original AMP LoRA"),
    )
    return next((label for token, label in ordered if token in name), "Diagnostic")


def dataset_from_name(name: str) -> str:
    return "VisDrone" if "visdrone" in name else "Brain Tumor"


def discover_runs(root: Path) -> list[tuple[Path, Path, Path]]:
    found: list[tuple[Path, Path, Path]] = []
    phase2 = root / "runs/issue50/phase2"
    logs = phase2 / "logs"
    if phase2.exists():
        for run in sorted(path for path in phase2.iterdir() if path.is_dir() and path.name != "logs"):
            found.append((run, logs / f"{run.name}.log", logs / run.name / "run_manifest.json"))
    fixed = (
        ("formal/baselines/brain_tumor_head_only_seed0", "formal/logs/baselines/brain_tumor_head_only_seed0"),
        ("formal/baselines/brain_tumor_full_finetune_seed0", "formal/logs/baselines/brain_tumor_full_finetune_seed0"),
        ("formal/baselines/visdrone_head_only_seed0", "formal/logs/baselines/visdrone_head_only_seed0"),
        ("formal/brain_tumor_r4_stable_v1_seed0", "formal/logs/brain_tumor_r4_stable_v1_seed0"),
        ("formal/brain_tumor_r4_stable_v1_seed1", "formal/logs/brain_tumor_r4_stable_v1_seed1"),
        ("formal/visdrone_r4_stable_v2_seed0", "formal/logs/visdrone_r4_stable_v2_seed0"),
        ("formal/visdrone_r4_stable_v2_seed1", "formal/logs/visdrone_r4_stable_v2_seed1"),
        ("diagnostics/brain_tumor_r4_amp_probe_e1", "diagnostics/logs/brain_tumor_r4_amp_probe_e1/train"),
        ("diagnostics/brain_tumor_r4_ampoff_lr8e4_e10", "diagnostics/logs/brain_tumor_r4_ampoff_lr8e4_e10/train"),
        (
            "diagnostics/brain_tumor_r4_ampoff_lr8e4_adapt01_e10",
            "diagnostics/logs/brain_tumor_r4_ampoff_lr8e4_adapt01_e10/train",
        ),
    )
    base = root / "runs/issue50"
    for run_rel, log_rel in fixed:
        run = base / run_rel
        log_base = base / log_rel
        log = log_base.with_suffix(".log") if log_base.name != "train" else log_base.parent / "train.log"
        manifest = (
            log.parent / run.name / "run_manifest.json"
            if "formal/logs" in log_rel
            else log.parent / "run_manifest.json"
        )
        if run.exists():
            found.append((run, log, manifest))
    unique = {}
    for item in found:
        unique[str(item[0].resolve())] = item
    return list(unique.values())


def summarize_run(run: Path, log_path: Path, manifest_path: Path) -> dict[str, object]:
    name = run.name
    args_path = run / "args.yaml"
    results_path = run / "results.csv"
    best_path = run / "weights/best.pt"
    args = yaml.safe_load(args_path.read_text(encoding="utf-8")) if args_path.exists() else {}
    rows = list(csv.DictReader(results_path.open(encoding="utf-8"))) if results_path.exists() else []
    best = max(rows, key=lambda row: finite(row.get("metrics/mAP50-95(B)"))) if rows else {}
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    manifest = read_state(manifest_path)
    nonfinite = bool(NONFINITE_RE.search(log))
    all_zero = bool(rows) and all(
        finite(row.get(key)) == 0.0
        for row in rows
        for key in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
    )
    artifacts = all(path.exists() for path in (results_path, args_path, best_path, log_path))
    exit_code = manifest.get("exit_code")
    method = method_from_name(name)
    dataset = dataset_from_name(name)
    params_match = PARAM_RE.findall(log)
    gradients = GRADIENT_RE.findall(log)
    known_params, known_pct = KNOWN_PARAMS.get((dataset, method), (0, 0.0))
    params = (
        int(params_match[-1][0].replace(",", ""))
        if params_match
        else int(gradients[-1].replace(",", ""))
        if gradients
        else known_params
    )
    pct = finite(params_match[-1][1]) if params_match else known_pct
    memories = [finite(value) for value in GPU_RE.findall(log)]
    speeds = SPEED_RE.findall(log)
    seed_match = re.search(r"seed(\d+)", name)
    rank = int(args.get("lora_r", 0) or 0)
    amp_safe = bool(args.get("lora_amp_safe", False))
    amp = bool(args.get("amp", False))
    classification = (
        "failure"
        if nonfinite or all_zero or exit_code not in (0, None) or not artifacts
        else "diagnostic"
        if "diagnostics" in run.parts
        else "formal"
    )
    return {
        "dataset": dataset,
        "method": method,
        "run_name": name,
        "seed": int(args.get("seed", seed_match.group(1) if seed_match else 0) or 0),
        "rank": rank,
        "alpha": int(args.get("lora_alpha", 0) or 0),
        "amp_strategy": "AMP-safe LoRA FP32 path" if amp_safe else "AMP" if amp else "FP32",
        "head_lr_scale": 1.0,
        "router_lr_scale": finite(args.get("moe_router_lr_scale", 0.5)),
        "adapter_lr_scale": finite(args.get("lora_lr_mult", 1.0)) if rank else 0.0,
        "physical_batch": int(args.get("batch", 0) or 0),
        "effective_batch": int(args.get("nbs", args.get("batch", 0)) or 0),
        "precision": finite(best.get("metrics/precision(B)")),
        "recall": finite(best.get("metrics/recall(B)")),
        "mAP50": finite(best.get("metrics/mAP50(B)")),
        "mAP50_95": finite(best.get("metrics/mAP50-95(B)")),
        "best_epoch": int(finite(best.get("epoch"))),
        "trainable_params": params,
        "trainable_pct": pct,
        "peak_gpu_mem_gib": max(memories, default=0.0),
        "train_time_s": finite(rows[-1].get("time")) if rows else 0.0,
        "inference_ms_per_image": finite(speeds[-1]) if speeds else 0.0,
        "nan_inf_recovery": nonfinite,
        "artifacts_complete": artifacts,
        "exit_code": exit_code if exit_code is not None else "",
        "classification": classification,
        "result_dir": str(run),
        "log_path": str(log_path),
    }


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def seed_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["classification"] == "formal":
            groups[(row["dataset"], row["method"], row["rank"], row["amp_strategy"])].append(row)
    output = []
    for key, selected in sorted(groups.items()):
        record = {
            "dataset": key[0],
            "method": key[1],
            "rank": key[2],
            "amp_strategy": key[3],
            "valid_seed_count": len({row["seed"] for row in selected}),
        }
        for metric in METRIC_KEYS:
            values = np.array([float(row[metric]) for row in selected])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            record[f"{metric}_best"] = float(values.max())
        output.append(record)
    return output


def pareto_summary(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in ("Brain Tumor", "VisDrone"):
        candidates = [
            row for row in rows if row["dataset"] == dataset and row["seed"] == 0 and row["classification"] == "formal"
        ]
        if not candidates:
            continue
        absolute = max(candidates, key=lambda row: row["mAP50_95"])
        efficient = max(candidates, key=lambda row: row["mAP50_95"] / max(row["trainable_params"], 1))
        stable = max(
            candidates,
            key=lambda row: (
                not row["nan_inf_recovery"],
                row["artifacts_complete"],
                row["mAP50_95"],
            ),
        )
        chosen = {absolute["run_name"]: "best_absolute"}
        chosen[efficient["run_name"]] = (
            f"{chosen[efficient['run_name']]}+best_parameter_efficiency"
            if efficient["run_name"] in chosen
            else "best_parameter_efficiency"
        )
        chosen[stable["run_name"]] = (
            f"{chosen[stable['run_name']]}+best_numerical_stability"
            if stable["run_name"] in chosen
            else "best_numerical_stability"
        )
        for row in candidates:
            output.append(
                {
                    "dataset": dataset,
                    "method": row["method"],
                    "run_name": row["run_name"],
                    "mAP50_95": row["mAP50_95"],
                    "trainable_params": row["trainable_params"],
                    "peak_gpu_mem_gib": row["peak_gpu_mem_gib"],
                    "train_time_s": row["train_time_s"],
                    "recommendation": chosen.get(row["run_name"], "not_recommended"),
                    "reason": (
                        "selected by registered objective"
                        if row["run_name"] in chosen
                        else "dominated or does not lead the registered objective"
                    ),
                }
            )
    for row in rows:
        if row["classification"] == "failure":
            output.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "run_name": row["run_name"],
                    "mAP50_95": row["mAP50_95"],
                    "trainable_params": row["trainable_params"],
                    "peak_gpu_mem_gib": row["peak_gpu_mem_gib"],
                    "train_time_s": row["train_time_s"],
                    "recommendation": "not_recommended",
                    "reason": "failed stability/artifact/exit-code gate",
                }
            )
    return output


def make_figures(rows: list[dict], seeds: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    candidates = [
        row
        for row in rows
        if row["classification"] == "formal" and row["seed"] == 0 and "Diagnostic" not in row["method"]
    ]
    for dataset in ("Brain Tumor", "VisDrone"):
        selected = [row for row in candidates if row["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar([row["method"] for row in selected], [row["mAP50_95"] for row in selected])
        ax.set(title=f"{dataset}: method comparison", ylabel="mAP50-95")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(output / f"{dataset.lower().replace(' ', '_')}_methods_map.png", dpi=200)
        plt.close(fig)

    plots = (
        ("trainable_params", "Trainable parameters", "accuracy_vs_parameters.png"),
        ("peak_gpu_mem_gib", "Peak GPU memory (GiB)", "accuracy_vs_memory.png"),
        ("train_time_s", "Training time (s)", "accuracy_vs_time.png"),
    )
    for field, label, filename in plots:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for dataset, marker in (("Brain Tumor", "o"), ("VisDrone", "s")):
            selected = [row for row in candidates if row["dataset"] == dataset]
            ax.scatter(
                [row[field] for row in selected], [row["mAP50_95"] for row in selected], label=dataset, marker=marker
            )
        ax.set(xlabel=label, ylabel="mAP50-95", title=f"Accuracy vs {label.lower()}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=200)
        plt.close(fig)

    ablation = [
        row
        for row in rows
        if row["method"] in {"Original AMP LoRA", "Stable LoRA", "AMP-safe LoRA", "Adapter LR x1.0", "Adapter LR x0.1"}
        and row["dataset"] == "Brain Tumor"
        and row["seed"] == 0
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = ["#c44e52" if row["classification"] == "failure" else "#4c72b0" for row in ablation]
    ax.bar([row["method"] for row in ablation], [row["mAP50_95"] for row in ablation], color=colors)
    ax.set(title="AMP and grouped-LR ablation", ylabel="mAP50-95")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output / "amp_grouped_lr_ablation.png", dpi=200)
    plt.close(fig)

    multi = [row for row in seeds if row["valid_seed_count"] >= 2]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{row['dataset']}\n{row['method']}" for row in multi]
    means = [row["mAP50_95_mean"] for row in multi]
    errors = [row["mAP50_95_std"] for row in multi]
    ax.bar(labels, means, yerr=errors, capsize=4)
    ax.set(title="Multi-seed mAP50-95 (mean ± SD)", ylabel="mAP50-95")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output / "multi_seed_error_bars.png", dpi=200)
    plt.close(fig)


def write_report(report: Path, rows: list[dict], pareto: list[dict]) -> None:
    selected = [row for row in pareto if row["recommendation"] != "not_recommended"]
    failures = [row for row in rows if row["classification"] == "failure"]
    lines = [
        "# Issue #50 Phase 2 最终报告",
        "",
        "## 问题与公平性",
        "",
        "比较强微调基线、稳定 LoRA、AMP-safe LoRA 与 Partial FT + LoRA；同一数据集固定数据划分、imgsz、epoch、seed、增强和验证协议。VisDrone OOM 仅降低物理 batch，并以 nbs/梯度累积维持有效 batch。",
        "",
        "## 结论",
        "",
    ]
    for row in selected:
        lines.append(
            f"- {row['dataset']}：{row['recommendation']} = `{row['method']}`，"
            f"mAP50-95={float(row['mAP50_95']):.5f}，trainable={int(row['trainable_params']):,}。"
        )
    lines.extend(
        [
            "",
            f"- 稳定门控失败实验共 {len(failures)} 组，均保留但不进入正式结论。",
            "- 若 Head-only/Full FT 精度更高，则其为绝对性能方案；LoRA 只从参数效率、稳定性和迁移成本评价。",
            "",
            "## 创新证据分级",
            "",
            "- 已验证：Adapter 学习率抑制、非有限梯度检测与健康检查点、rank 的精度—参数—稳定性权衡。",
            "- Phase 2 验证后可判定：AMP-safe LoRA、Partial FT + LoRA。",
            "- 初步证据：Head/Router/Adapter 分组学习率；当前 Head=1.0×、Router=0.5×、Adapter=0.1×。",
            "- 尚未验证：跨更多数据集与大规模多种子统计；不得扩写为普遍精度优势。",
            "",
            "完整逐运行数据见 `PHASE2_FINAL_RESULTS.csv`，多种子与 Pareto 结论见相邻 CSV。",
        ]
    )
    atomic_text(report, "\n".join(lines) + "\n")


def write_handoff(path: Path, root: Path, rows: list[dict], pareto: list[dict]) -> None:
    best = [row for row in pareto if "best_absolute" in row["recommendation"]]
    lines = [
        "# Issue #50 Phase 2 Final Handoff",
        "",
        f"- Branch: `{subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip()}`",
        f"- Source commit before finalization: `{subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()}`",
        "- Formal results: `runs/issue50/phase2/` and reused Phase 1 formal directories",
        "- Diagnostic results: `runs/issue50/diagnostics/`",
        "- Reproduce queue: `python scripts/issue50/phase2_performance_queue.py --device 0 --launcher /root/miniconda3/bin/yolo`",
        "- Resume finalizer: `nohup setsid flock -n runs/issue50/phase2_finalize.lock /root/miniconda3/bin/python scripts/issue50/phase2_finalize_after_queue.py > runs/issue50/phase2_finalize.log 2>&1 < /dev/null &`",
        "",
        "## Best absolute checkpoints",
        "",
    ]
    by_name = {row["run_name"]: row for row in rows}
    for selected in best:
        row = by_name[selected["run_name"]]
        lines.append(f"- {row['dataset']} / {row['method']}: `{row['result_dir']}/weights/best.pt`")
    lines.extend(
        [
            "",
            "## Next research",
            "",
            "优先在独立数据集验证 AMP-safe Adapter 路径与三路学习率，并以至少 3 个种子复验；不要继续无界调参。",
        ]
    )
    atomic_text(path, "\n".join(lines) + "\n")


def validate_and_commit(root: Path, reports: Path) -> tuple[str, str]:
    subprocess.run([sys.executable, "-m", "py_compile", str(Path(__file__))], cwd=root, check=True)
    subprocess.run(["git", "diff", "--check"], cwd=root, check=True)
    allowed = [
        "reports/issue50/PHASE2_EXPERIMENT_REGISTRY.csv",
        "reports/issue50/PHASE2_STATUS.md",
        "reports/issue50/PHASE2_FINAL_RESULTS.csv",
        "reports/issue50/PHASE2_SEED_SUMMARY.csv",
        "reports/issue50/PHASE2_PARETO_SUMMARY.csv",
        "reports/issue50/PHASE2_FINAL_REPORT_CN.md",
        "reports/issue50/PHASE2_FINAL_FIGURES",
        "reports/issue50/PHASE2_FINAL_HANDOFF.md",
    ]
    subprocess.run(["git", "add", "-f", *allowed], cwd=root, check=True)
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=root, text=True).splitlines()
    forbidden = re.compile(
        r"(^|/)(datasets?|weights?|runs)(/|$)|(?:id_rsa|\.pem$|token|secret|password)", re.IGNORECASE
    )
    if any(forbidden.search(name) for name in staged):
        raise RuntimeError(f"Refusing unsafe staged files: {staged}")
    if staged:
        subprocess.run(["git", "commit", "-m", "Finalize Issue 50 Phase 2 results"], cwd=root, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    push = subprocess.run(
        ["git", "push", "origin", "issue-50-lora-reproduction"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return commit, "pushed" if push.returncode == 0 else f"retained locally: {push.stdout[-500:]}"


def create_archive(root: Path, reports: Path) -> tuple[Path, str]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = Path(f"/root/autodl-tmp/issue50_phase2_final_{stamp}.tar.gz")
    selected: list[Path] = []
    selected.extend(path for path in reports.glob("PHASE2*") if path.is_file())
    selected.extend((reports / "PHASE2_FINAL_FIGURES").glob("*.png"))
    selected.extend((root / "scripts/issue50").glob("phase2*.py"))
    selected.extend(
        path
        for path in (root / "runs/issue50").glob("phase2*")
        if path.is_file() and path.stat().st_size < 20 * 1024 * 1024
    )
    for run in (root / "runs/issue50/phase2").glob("*"):
        if not run.is_dir() or run.name == "logs":
            continue
        selected.extend(path for name in ("results.csv", "args.yaml") if (path := run / name).exists())
    selected.extend((root / "runs/issue50/phase2/logs").glob("*/run_manifest.json"))
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(set(selected)):
            tar.add(path, arcname=path.relative_to(root))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive, digest


def completion_gate(reports: Path, archive: Path, digest: str) -> list[str]:
    """Return fatal completion errors; an empty list permits the complete flag."""
    errors = []
    csv_paths = (
        reports / "PHASE2_FINAL_RESULTS.csv",
        reports / "PHASE2_SEED_SUMMARY.csv",
        reports / "PHASE2_PARETO_SUMMARY.csv",
    )
    for path in csv_paths:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                list(csv.DictReader(handle))
        except (OSError, csv.Error) as exc:
            errors.append(f"Unreadable CSV {path.name}: {exc}")
    required = (
        reports / "PHASE2_FINAL_REPORT_CN.md",
        reports / "PHASE2_FINAL_HANDOFF.md",
        reports / "PHASE2_FINAL_FIGURES",
        archive,
        archive.with_suffix(archive.suffix + ".sha256"),
    )
    for path in required:
        if not path.exists():
            errors.append(f"Missing required artifact: {path}")
    if not digest or len(digest) != 64:
        errors.append("Invalid SHA256 digest.")
    if not list((reports / "PHASE2_FINAL_FIGURES").glob("*.png")):
        errors.append("No final figures were generated.")
    return errors


def finalize(root: Path, final_status: Path) -> None:
    reports = root / "reports/issue50"
    rows = [summarize_run(*item) for item in discover_runs(root)]
    rows.sort(key=lambda row: (row["dataset"], row["method"], row["seed"], row["run_name"]))
    write_csv(reports / "PHASE2_FINAL_RESULTS.csv", rows, FINAL_FIELDS)
    seeds = seed_summary(rows)
    seed_fields = (
        list(seeds[0])
        if seeds
        else [
            "dataset",
            "method",
            "rank",
            "amp_strategy",
            "valid_seed_count",
        ]
    )
    write_csv(reports / "PHASE2_SEED_SUMMARY.csv", seeds, seed_fields)
    pareto = pareto_summary(rows)
    pareto_fields = (
        list(pareto[0])
        if pareto
        else [
            "dataset",
            "method",
            "run_name",
            "recommendation",
            "reason",
        ]
    )
    write_csv(reports / "PHASE2_PARETO_SUMMARY.csv", pareto, pareto_fields)
    make_figures(rows, seeds, reports / "PHASE2_FINAL_FIGURES")
    write_report(reports / "PHASE2_FINAL_REPORT_CN.md", rows, pareto)
    write_handoff(reports / "PHASE2_FINAL_HANDOFF.md", root, rows, pareto)
    commit, push_status = validate_and_commit(root, reports)
    archive, digest = create_archive(root, reports)
    fatal_errors = completion_gate(reports, archive, digest)
    if fatal_errors:
        atomic_json(
            final_status,
            {
                "status": "blocked",
                "current_stage": "integrity_gate",
                "current_experiment": None,
                "last_updated": now(),
                "completed_count": sum(row["classification"] == "formal" for row in rows),
                "failed_count": sum(row["classification"] == "failure" for row in rows),
                "error_summary": "; ".join(fatal_errors),
                "archive": str(archive),
                "sha256": digest,
            },
        )
        raise SystemExit(3)
    complete_flag = root / "runs/issue50/PHASE2_COMPLETE.flag"
    atomic_text(complete_flag, f"completed_at={now()}\ncommit={commit}\narchive={archive}\nsha256={digest}\n")
    atomic_json(
        final_status,
        {
            "status": "completed",
            "current_stage": "completed",
            "current_experiment": None,
            "last_updated": now(),
            "completed_count": sum(row["classification"] == "formal" for row in rows),
            "failed_count": sum(row["classification"] == "failure" for row in rows),
            "error_summary": "",
            "completed_at": now(),
            "runs_audited": len(rows),
            "formal_successes": sum(row["classification"] == "formal" for row in rows),
            "failures": sum(row["classification"] == "failure" for row in rows),
            "commit": commit,
            "push_status": push_status,
            "archive": str(archive),
            "sha256": digest,
            "complete_flag": str(complete_flag),
        },
    )


def main() -> None:
    global POLL_SECONDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    POLL_SECONDS = max(args.poll_seconds, 300) if not args.dry_run else 0
    root = root_path()
    pid_path = root / "runs/issue50/phase2_finalize.pid"
    final_status = root / "runs/issue50/phase2_final_status.json"
    atomic_text(pid_path, f"{os.getpid()}\n")
    if args.dry_run:
        runs = discover_runs(root)
        print(
            json.dumps(
                {
                    "runs_discovered": len(runs),
                    "queue_state": read_state(root / "runs/issue50/phase2_queue_status.json"),
                },
                ensure_ascii=False,
            )
        )
        return
    wait_for_queue(root, root / "runs/issue50/phase2_queue_status.json", final_status)
    queue_state = read_state(root / "runs/issue50/phase2_queue_status.json")
    atomic_json(
        final_status,
        {
            "status": "finalizing",
            "current_stage": "finalizing",
            "current_experiment": None,
            "last_updated": now(),
            "completed_count": len(queue_state.get("completed", [])),
            "failed_count": len(queue_state.get("failed", [])),
            "error_summary": "",
            "updated_at": now(),
        },
    )
    finalize(root, final_status)


if __name__ == "__main__":
    main()
