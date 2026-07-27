#!/usr/bin/env python3
"""Build the final Issue #50 evidence table and publication-ready figures from saved runs."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


RUN_SPECS = [
    # Formal rank sweeps.
    ("rank", "Brain Tumor", "brain_tumor_r4_stable_v1_seed0", "formal", "Stable LoRA", 4, 0),
    ("rank", "Brain Tumor", "brain_tumor_r8_stable_v1_seed0", "formal", "Stable LoRA", 8, 0),
    ("rank", "Brain Tumor", "brain_tumor_r16_stable_v1_seed0", "formal", "Stable LoRA", 16, 0),
    ("rank", "VisDrone", "visdrone_r4_stable_v2_seed0", "formal", "Stable LoRA", 4, 0),
    ("rank", "VisDrone", "visdrone_r8_stable_v2_seed0", "formal", "Stable LoRA", 8, 0),
    ("rank", "VisDrone", "visdrone_r16_stable_v2_seed0", "formal", "Stable LoRA", 16, 0),
    # Best-rank reproducibility runs.
    ("seed", "Brain Tumor", "brain_tumor_r4_stable_v1_seed1", "formal", "Stable LoRA", 4, 1),
    ("seed", "VisDrone", "visdrone_r4_stable_v2_seed1", "formal", "Stable LoRA", 4, 1),
    # Fair non-LoRA baselines.
    ("baseline", "Brain Tumor", "brain_tumor_head_only_seed0", "baseline", "Head-only", 0, 0),
    ("baseline", "Brain Tumor", "brain_tumor_full_finetune_seed0", "baseline", "Full fine-tune", 0, 0),
    ("baseline", "VisDrone", "visdrone_head_only_seed0", "baseline", "Head-only", 0, 0),
    ("baseline", "VisDrone", "visdrone_full_finetune_seed0", "baseline", "Full fine-tune", 0, 0),
    # Existing single-variable diagnostics used in the ablation figure.
    ("ablation", "Brain Tumor", "brain_tumor_r4_amp_probe_e1", "diagnostic", "AMP on", 4, 0),
    ("ablation", "Brain Tumor", "brain_tumor_r4_ampoff_e3", "diagnostic", "AMP off", 4, 0),
    ("ablation", "Brain Tumor", "brain_tumor_r4_ampoff_lr8e4_e10", "diagnostic", "Adapter LR x1.0", 4, 0),
    (
        "ablation",
        "Brain Tumor",
        "brain_tumor_r4_ampoff_lr8e4_adapt01_e10",
        "diagnostic",
        "Adapter LR x0.1",
        4,
        0,
    ),
]

NONFINITE_RE = re.compile(
    r"Non-finite gradient|NaN recovery|Global nonfinite training state|loss (?:is )?nan|Traceback",
    re.IGNORECASE,
)
PARAM_RE = re.compile(
    r"Trainable:\s*([0-9,]+)\s*\(([0-9.]+)%\).*?Frozen Base:\s*([0-9,]+).*?"
    r"Adapter Params:\s*([0-9,]+)",
    re.IGNORECASE,
)
SPEED_RE = re.compile(
    r"Speed:\s*([0-9.]+)ms preprocess,\s*([0-9.]+)ms inference,.*?([0-9.]+)ms postprocess per image",
    re.IGNORECASE,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return [{(key or "").strip(): (value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def is_finite_number(value: object) -> bool:
    """Return whether a serialized metric is numeric and finite."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def paths_for(root: Path, location: str, name: str) -> tuple[Path, Path, Path]:
    issue_root = root / "runs" / "issue50"
    if location == "formal":
        run = issue_root / "formal" / name
        log = issue_root / "formal" / "logs" / f"{name}.log"
        manifest = issue_root / "formal" / "logs" / name / "run_manifest.json"
    elif location == "baseline":
        run = issue_root / "formal" / "baselines" / name
        log = issue_root / "formal" / "logs" / "baselines" / f"{name}.log"
        manifest = issue_root / "formal" / "logs" / "baselines" / name / "run_manifest.json"
    else:
        run = issue_root / "diagnostics" / name
        diagnostic_log = issue_root / "diagnostics" / "logs" / name / "train.log"
        log = diagnostic_log if diagnostic_log.exists() else issue_root / "diagnostics" / "logs" / f"{name}.log"
        manifest = issue_root / "diagnostics" / "logs" / name / "run_manifest.json"
    return run, log, manifest


def baseline_parameter_counts(checkpoint: Path, method: str) -> tuple[int, int]:
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = checkpoint_data["model"]
    named = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in named)
    if method == "Head-only":
        trainable = sum(
            parameter.numel()
            for name, parameter in named
            if (name.startswith("model.25.") or ".model.25." in name) and ".dfl" not in name
        )
    else:
        trainable = sum(parameter.numel() for name, parameter in named if ".dfl" not in name)
    return trainable, total


def parse_parameters(log_text: str, checkpoint: Path, method: str) -> tuple[int, int, float, int]:
    match = PARAM_RE.search(log_text)
    if match:
        trainable = int(match.group(1).replace(",", ""))
        pct = float(match.group(2))
        frozen = int(match.group(3).replace(",", ""))
        adapter = int(match.group(4).replace(",", ""))
        return trainable, trainable + frozen, pct, adapter
    if method in {"Head-only", "Full fine-tune"} and checkpoint.exists():
        trainable, total = baseline_parameter_counts(checkpoint, method)
        return trainable, total, 100.0 * trainable / total, 0
    return 0, 0, 0.0, 0


def parse_peak_memory(log_text: str) -> float:
    values = [float(value) for value in re.findall(r"\b([0-9]+(?:\.[0-9]+)?)G\b", log_text)]
    return max(values) if values else 0.0


def parse_speed(log_text: str) -> tuple[float, float]:
    matches = SPEED_RE.findall(log_text)
    if not matches:
        return 0.0, 0.0
    _, inference, _ = matches[-1]
    inference_ms = float(inference)
    return inference_ms, 1000.0 / inference_ms if inference_ms else 0.0


def summarize_run(root: Path, spec: tuple) -> dict[str, object]:
    category, dataset, name, location, method, rank, seed = spec
    run, log_path, manifest_path = paths_for(root, location, name)
    results = read_results(run / "results.csv")
    best = max(results, key=lambda row: finite_float(row.get("metrics/mAP50-95(B)"))) if results else {}
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8")) if (run / "args.yaml").exists() else {}
    best_checkpoint = run / "weights" / "best.pt"
    trainable, total, trainable_pct, adapter = parse_parameters(log_text, best_checkpoint, method)
    inference_ms, fps = parse_speed(log_text)
    numeric_results_finite = all(
        is_finite_number(value)
        for row in results
        for key, value in row.items()
        if key != "epoch" and str(value).strip()
    )
    nonfinite = bool(NONFINITE_RE.search(log_text)) or not numeric_results_finite
    required = [run / "results.csv", run / "args.yaml", best_checkpoint, run / "weights" / "last.pt", log_path]
    if category in {"rank", "seed", "baseline"}:
        required.append(manifest_path)
    artifacts_complete = all(path.exists() for path in required)
    success = manifest.get("success") is True if manifest else bool(results and best_checkpoint.exists())
    stability = "unstable" if nonfinite else ("stable" if success and artifacts_complete else "incomplete")
    return {
        "category": category,
        "dataset": dataset,
        "method": method,
        "run_name": name,
        "rank": rank,
        "seed": seed,
        "precision": finite_float(best.get("metrics/precision(B)")),
        "recall": finite_float(best.get("metrics/recall(B)")),
        "mAP50": finite_float(best.get("metrics/mAP50(B)")),
        "mAP50_95": finite_float(best.get("metrics/mAP50-95(B)")),
        "best_epoch": int(finite_float(best.get("epoch"))),
        "epochs_completed": len(results),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": trainable_pct,
        "adapter_params": adapter,
        "peak_gpu_mem_gib": parse_peak_memory(log_text),
        "train_time_s": finite_float(results[-1].get("time")) if results else 0.0,
        "inference_ms_per_image": inference_ms,
        "inference_fps": fps,
        "amp": args.get("amp", ""),
        "optimizer": args.get("optimizer", ""),
        "lr0": args.get("lr0", ""),
        "lora_lr_mult": args.get("lora_lr_mult", ""),
        "stability": stability,
        "nan_inf_recovery": nonfinite,
        "artifacts_complete": artifacts_complete,
        "exit_code": manifest.get("exit_code", ""),
        "git_commit": manifest.get("git_commit", ""),
        "formal_material": category in {"rank", "seed", "baseline"} and stability == "stable",
        "pareto_front": False,
        "metric_std": "",
        "best_single_mAP50_95": "",
    }


def aggregate_seed_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregates = []
    for dataset in ("Brain Tumor", "VisDrone"):
        selected = [
            row
            for row in rows
            if row["dataset"] == dataset
            and row["method"] == "Stable LoRA"
            and int(row["rank"]) == 4
            and row["category"] in {"rank", "seed"}
        ]
        if len(selected) < 2:
            continue
        metrics = np.array([float(row["mAP50_95"]) for row in selected], dtype=float)
        aggregate = dict(selected[0])
        aggregate.update(
            {
                "category": "aggregate",
                "run_name": f"{dataset.lower().replace(' ', '_')}_stable_lora_r4_2seed",
                "seed": "mean",
                "precision": float(np.mean([float(row["precision"]) for row in selected])),
                "recall": float(np.mean([float(row["recall"]) for row in selected])),
                "mAP50": float(np.mean([float(row["mAP50"]) for row in selected])),
                "mAP50_95": float(metrics.mean()),
                "metric_std": float(metrics.std(ddof=1)),
                "best_single_mAP50_95": float(metrics.max()),
                "train_time_s": float(np.mean([float(row["train_time_s"]) for row in selected])),
                "peak_gpu_mem_gib": float(np.max([float(row["peak_gpu_mem_gib"]) for row in selected])),
                "inference_ms_per_image": float(
                    np.mean([float(row["inference_ms_per_image"]) for row in selected])
                ),
                "formal_material": True,
            }
        )
        aggregates.append(aggregate)
    return aggregates


def mark_pareto(rows: list[dict[str, object]]) -> None:
    candidates = [
        row
        for row in rows
        if row["category"] in {"rank", "baseline"} and row["stability"] == "stable" and int(row["seed"]) == 0
    ]
    for row in candidates:
        dominated = False
        for other in candidates:
            if other is row or other["dataset"] != row["dataset"]:
                continue
            no_worse = (
                float(other["mAP50_95"]) >= float(row["mAP50_95"])
                and int(other["trainable_params"]) <= int(row["trainable_params"])
                and float(other["peak_gpu_mem_gib"]) <= float(row["peak_gpu_mem_gib"])
                and float(other["train_time_s"]) <= float(row["train_time_s"])
            )
            strictly_better = (
                float(other["mAP50_95"]) > float(row["mAP50_95"])
                or int(other["trainable_params"]) < int(row["trainable_params"])
                or float(other["peak_gpu_mem_gib"]) < float(row["peak_gpu_mem_gib"])
                or float(other["train_time_s"]) < float(row["train_time_s"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        row["pareto_front"] = not dominated


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figures(rows: list[dict[str, object]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    rank_rows = [row for row in rows if row["category"] == "rank"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for dataset, marker in (("Brain Tumor", "o"), ("VisDrone", "s")):
        selected = sorted((row for row in rank_rows if row["dataset"] == dataset), key=lambda row: int(row["rank"]))
        ax.plot([row["rank"] for row in selected], [row["mAP50_95"] for row in selected], marker=marker, label=dataset)
    ax.set(xlabel="LoRA rank", ylabel="mAP50-95", xticks=[4, 8, 16], title="Rank vs detection accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "rank_vs_map.png", dpi=220)
    plt.close(fig)

    comparison = [row for row in rows if row["category"] in {"rank", "baseline"} and int(row["seed"]) == 0]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method, marker in (("Stable LoRA", "o"), ("Head-only", "^"), ("Full fine-tune", "s")):
        selected = [row for row in comparison if row["method"] == method]
        ax.scatter(
            [float(row["trainable_params"]) / 1e6 for row in selected],
            [row["mAP50_95"] for row in selected],
            marker=marker,
            s=70,
            label=method,
        )
    ax.set(xlabel="Trainable parameters (million)", ylabel="mAP50-95", title="Parameter efficiency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "parameters_vs_accuracy.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for dataset, marker in (("Brain Tumor", "o"), ("VisDrone", "s")):
        selected = [row for row in comparison if row["dataset"] == dataset]
        axes[0].scatter([row["peak_gpu_mem_gib"] for row in selected], [row["mAP50_95"] for row in selected], marker=marker, s=70, label=dataset)
        axes[1].scatter([float(row["train_time_s"]) / 60 for row in selected], [row["mAP50_95"] for row in selected], marker=marker, s=70, label=dataset)
    axes[0].set(xlabel="Peak GPU memory (GiB)", ylabel="mAP50-95", title="Memory vs accuracy")
    axes[1].set(xlabel="Training time (min)", ylabel="mAP50-95", title="Time vs accuracy")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "memory_time_vs_accuracy.png", dpi=220)
    plt.close(fig)

    ablation = [row for row in rows if row["category"] == "ablation"]
    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    colors = ["#c44e52" if row["stability"] == "unstable" else "#4c72b0" for row in ablation]
    bars = ax.bar([row["method"] for row in ablation], [row["mAP50_95"] for row in ablation], color=colors)
    for bar, row in zip(bars, ablation):
        if row["stability"] == "unstable":
            bar.set_hatch("//")
    ax.tick_params(axis="x", rotation=18)
    ax.set(ylabel="mAP50-95", title="Stability ablation (red hatched = non-finite/recovery)")
    fig.tight_layout()
    fig.savefig(output / "stability_ablation.png", dpi=220)
    plt.close(fig)

    seed_rows = [
        row
        for row in rows
        if row["method"] == "Stable LoRA" and int(row["rank"]) == 4 and row["category"] in {"rank", "seed"}
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(2)
    width = 0.34
    for index, seed in enumerate((0, 1)):
        values = [
            next(float(row["mAP50_95"]) for row in seed_rows if row["dataset"] == dataset and int(row["seed"]) == seed)
            for dataset in ("Brain Tumor", "VisDrone")
        ]
        ax.bar(x + (index - 0.5) * width, values, width, label=f"seed={seed}")
    ax.set_xticks(x, ("Brain Tumor", "VisDrone"))
    ax.set(ylabel="mAP50-95", title="Best stable configuration across two seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "best_configuration_seeds.png", dpi=220)
    plt.close(fig)


def main() -> None:
    root = repo_root()
    reports = root / "reports" / "issue50"
    rows = [summarize_run(root, spec) for spec in RUN_SPECS]
    required = [row for row in rows if row["category"] in {"rank", "seed", "baseline"}]
    incomplete = [row["run_name"] for row in required if row["stability"] == "incomplete"]
    if incomplete:
        raise SystemExit(f"Required runs are incomplete: {incomplete}")
    rows.extend(aggregate_seed_rows(rows))
    mark_pareto(rows)
    write_csv(rows, reports / "FINAL_RESULTS_SUMMARY.csv")
    save_figures(rows, reports / "FINAL_FIGURES")
    print(f"Wrote {len(rows)} rows and five figures under {reports}")


if __name__ == "__main__":
    main()
