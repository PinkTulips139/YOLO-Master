#!/usr/bin/env python3
"""Rebuild Issue #50 Phase 2 summaries from immutable run evidence.

This module never starts training. Unknown evidence is emitted as ``unknown`` or
``evidence_missing`` instead of being converted to zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

NONFINITE_RE = re.compile(
    r"\bnan\b|\binf(?:inity)?\b|non[- ]?finite|NaN recovery|Global nonfinite training state",
    re.I,
)
OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError", re.I)
TRACEBACK_RE = re.compile(r"Traceback", re.I)
SPEED_RE = re.compile(r"Speed:.*?([0-9.]+)ms inference", re.I)
GPU_RE = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)G\b")
TRAINABLE_RE = re.compile(r"Trainable:\s*([0-9,]+)\s*\(([0-9.]+)%\)")

FINAL_FIELDS = (
    "dataset", "method", "run_name", "seed", "rank", "alpha", "amp_strategy",
    "head_lr_scale", "router_lr_scale", "adapter_lr_scale", "requested_batch", "actual_batch",
    "effective_batch", "precision", "recall", "mAP50", "mAP50_95", "best_epoch",
    "epochs_completed", "trainable_params", "trainable_pct", "peak_gpu_mem_gib", "train_time_s",
    "inference_ms_per_image", "exit_code", "queue_execution_status", "formal_validity_status",
    "failure_reason", "recovery_status", "config_signature", "evidence_path", "result_dir", "log_path",
)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return {}


def read_csv(path: Path) -> list[dict]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError:
        return []


def number(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def method_from_name(name: str) -> str:
    rules = (
        ("last_stage_neck_head", "Last stage + Neck + Head"),
        ("full_finetune", "Full fine-tuning"),
        ("amp_safe_lora", "AMP-safe LoRA"),
        ("partial_lora", "Partial fine-tuning + LoRA"),
        ("stable_lora", "Stable LoRA"),
        ("r4_stable_v", "Stable LoRA"),
        ("neck_head", "Neck + Head"),
        ("head_only", "Head-only"),
        ("adapt01", "Adapter LR x0.1"),
        ("lr8e4_e10", "Adapter LR x1.0"),
        ("amp_probe", "Original AMP LoRA"),
    )
    return next((label for token, label in rules if token in name), "Diagnostic")


def dataset_from_name(name: str) -> str:
    return "VisDrone" if "visdrone" in name else "Brain Tumor"


def locate_manifest(log_path: Path, run_name: str) -> Path:
    candidates = (
        log_path.parent / run_name / "run_manifest.json",
        log_path.parent / "run_manifest.json",
        log_path.with_suffix("") / "run_manifest.json",
    )
    return next((path for path in candidates if path.exists()), candidates[0])


def discover(root: Path) -> list[tuple[Path, Path, Path, int | None]]:
    found = []
    phase2 = root / "runs/issue50/phase2"
    if phase2.exists():
        for run in sorted(path for path in phase2.iterdir() if path.is_dir() and path.name != "logs"):
            log = phase2 / "logs" / f"{run.name}.log"
            found.append((run, log, locate_manifest(log, run.name), requested_batch_from_name(run.name)))
    fixed = (
        "formal/baselines/brain_tumor_head_only_seed0",
        "formal/baselines/brain_tumor_full_finetune_seed0",
        "formal/baselines/visdrone_head_only_seed0",
        "formal/brain_tumor_r4_stable_v1_seed0",
        "formal/brain_tumor_r4_stable_v1_seed1",
        "formal/visdrone_r4_stable_v2_seed0",
        "formal/visdrone_r4_stable_v2_seed1",
        "diagnostics/brain_tumor_r4_amp_probe_e1",
        "diagnostics/brain_tumor_r4_ampoff_lr8e4_e10",
        "diagnostics/brain_tumor_r4_ampoff_lr8e4_adapt01_e10",
    )
    for rel in fixed:
        run = root / "runs/issue50" / rel
        if not run.exists():
            continue
        if "formal/baselines" in rel:
            log = root / "runs/issue50/formal/logs/baselines" / f"{run.name}.log"
        elif "formal/" in rel:
            log = root / "runs/issue50/formal/logs" / f"{run.name}.log"
        else:
            log = root / "runs/issue50/diagnostics/logs" / run.name / "train.log"
        found.append((run, log, locate_manifest(log, run.name), None))
    return found


def requested_batch_from_name(name: str) -> int | None:
    match = re.search(r"_b(\d+)$", name)
    return int(match.group(1)) if match else None


def checkpoint_parameter_count(checkpoint: Path, method: str) -> tuple[int | None, float | None]:
    """Count the configured trainable subset from checkpoint names, without training."""
    if not checkpoint.exists():
        return None, None
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = payload.get("ema") or payload.get("model")
    if model is None:
        return None, None
    parameters = list(model.named_parameters())
    total = sum(param.numel() for _, param in parameters)
    cutoff = {
        "Head-only": 25,
        "Neck + Head": 13,
        "Last stage + Neck + Head": 10,
        "Full fine-tuning": 0,
        "Partial fine-tuning + LoRA": 13,
    }.get(method)
    if cutoff is None:
        return None, None
    selected = 0
    for name, param in parameters:
        layers = re.findall(r"model[.]([0-9]+)[.]", name)
        layer = int(layers[-1]) if layers else None
        adapter = "lora_" in name.lower() or "adapter" in name.lower()
        use = (layer is not None and layer >= cutoff) or (method == "Partial fine-tuning + LoRA" and adapter)
        if ".dfl." in name:
            use = False
        if use:
            selected += param.numel()
    return selected, selected / total * 100 if total else None


def metric_row(rows: list[dict]) -> tuple[dict, int | None]:
    if not rows:
        return {}, None
    best_index, best = max(
        enumerate(rows), key=lambda item: number(item[1].get("metrics/mAP50-95(B)"), -math.inf)
    )
    return best, best_index + 1


def config_signature(args: dict, method: str) -> str:
    keys = (
        "data", "model", "imgsz", "epochs", "batch", "seed", "optimizer", "lr0", "lrf", "amp",
        "warmup_epochs", "freeze", "lora_r", "lora_alpha", "lora_lr_mult", "moe_router_lr_scale",
        "lora_unfreeze_layers",
    )
    payload = {key: args.get(key) for key in keys if key != "seed"}
    payload["method"] = method
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def audit_run(run: Path, log_path: Path, manifest_path: Path, requested_batch: int | None) -> dict:
    name = run.name
    method = method_from_name(name)
    dataset = dataset_from_name(name)
    args = read_yaml(run / "args.yaml")
    manifest = read_json(manifest_path)
    rows = read_csv(run / "results.csv")
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    best, best_epoch = metric_row(rows)
    actual_batch = args.get("batch", "evidence_missing")
    requested_batch = requested_batch if requested_batch is not None else actual_batch
    exit_code = manifest.get("exit_code", "evidence_missing")
    artifacts = {
        "results.csv": (run / "results.csv").exists(),
        "args.yaml": (run / "args.yaml").exists(),
        "best.pt": (run / "weights/best.pt").exists(),
        "last.pt": (run / "weights/last.pt").exists(),
        "log": log_path.exists(),
        "manifest": manifest_path.exists(),
    }
    nonfinite = bool(NONFINITE_RE.search(log))
    oom = bool(OOM_RE.search(log))
    traceback = bool(TRACEBACK_RE.search(log))
    all_zero = bool(rows) and all(
        (number(row.get(key), 0.0) == 0.0)
        for row in rows
        for key in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
    )
    failure = []
    if method == "AMP-safe LoRA" and "amp_safe_forward" in log:
        failure.append("implementation_failed: Conv2d.amp_safe_forward missing")
    if exit_code == "evidence_missing":
        failure.append("exit_code evidence missing")
    elif exit_code != 0:
        failure.append(f"exit_code={exit_code}")
    for key, present in artifacts.items():
        if not present:
            failure.append(f"{key} missing")
    if nonfinite:
        failure.append("NaN/Inf/non-finite/recovery detected")
    if oom:
        failure.append("OOM observed")
    if traceback and not failure:
        failure.append("traceback detected")
    if all_zero:
        failure.append("all metrics zero")
    recovered = (
        requested_batch != actual_batch and isinstance(requested_batch, int) and isinstance(actual_batch, int)
    )
    if recovered:
        recovery = f"oom_recovered_diagnostic: requested_batch={requested_batch}, actual_batch={actual_batch}"
        failure.append("automatic batch recovery; excluded from formal protocol")
    else:
        recovery = "none"
    formal_method = method in {
        "Head-only", "Neck + Head", "Last stage + Neck + Head", "Full fine-tuning",
        "Stable LoRA", "Partial fine-tuning + LoRA",
    }
    formal = "passed" if formal_method and not failure else ("failed" if formal_method else "diagnostic")
    if method == "AMP-safe LoRA":
        formal = "implementation_failed"
    trainable, trainable_pct = checkpoint_parameter_count(run / "weights/best.pt", method)
    if trainable is None:
        matches = TRAINABLE_RE.findall(log)
        if matches and method in {"Stable LoRA", "Adapter LR x0.1", "Adapter LR x1.0", "Original AMP LoRA"}:
            trainable, trainable_pct = int(matches[-1][0].replace(",", "")), float(matches[-1][1])
    memory = [number(value) for value in GPU_RE.findall(log)]
    speed = SPEED_RE.findall(log)
    return {
        "dataset": dataset,
        "method": method,
        "run_name": name,
        "seed": args.get("seed", "evidence_missing"),
        "rank": args.get("lora_r", ""),
        "alpha": args.get("lora_alpha", ""),
        "amp_strategy": "AMP-safe implementation" if method == "AMP-safe LoRA" else ("AMP" if args.get("amp") else "FP32"),
        "head_lr_scale": 1.0,
        "router_lr_scale": args.get("moe_router_lr_scale", ""),
        "adapter_lr_scale": args.get("lora_lr_mult", ""),
        "requested_batch": requested_batch,
        "actual_batch": actual_batch,
        "effective_batch": args.get("nbs", actual_batch),
        "precision": number(best.get("metrics/precision(B)"), "unknown"),
        "recall": number(best.get("metrics/recall(B)"), "unknown"),
        "mAP50": number(best.get("metrics/mAP50(B)"), "unknown"),
        "mAP50_95": number(best.get("metrics/mAP50-95(B)"), "unknown"),
        "best_epoch": best_epoch if best_epoch is not None else "unknown",
        "epochs_completed": len(rows),
        "trainable_params": trainable if trainable is not None else "evidence_missing",
        "trainable_pct": trainable_pct if trainable_pct is not None else "evidence_missing",
        "peak_gpu_mem_gib": max((x for x in memory if x is not None), default="evidence_missing"),
        "train_time_s": number(rows[-1].get("time"), "unknown") if rows else "unknown",
        "inference_ms_per_image": number(speed[-1], "unknown") if speed else "unknown",
        "exit_code": exit_code,
        "queue_execution_status": "completed" if exit_code == 0 else ("failed" if exit_code != "evidence_missing" else "unknown"),
        "formal_validity_status": formal,
        "failure_reason": "; ".join(failure) if failure else "",
        "recovery_status": recovery,
        "config_signature": config_signature(args, method),
        "evidence_path": ";".join(str(path) for path in (run / "args.yaml", run / "results.csv", log_path, manifest_path)),
        "result_dir": str(run),
        "log_path": str(log_path),
    }


def add_not_executed_head_fallbacks(rows: list[dict]) -> None:
    source = next((row for row in rows if row["run_name"] == "visdrone_head_only_seed0"), None)
    if source is None:
        return
    source["requested_batch"] = 8
    source["formal_validity_status"] = "failed"
    source["failure_reason"] = (source["failure_reason"] + "; " if source["failure_reason"] else "") + "OOM at requested batch 8"
    for batch in (4, 2, 1):
        duplicate = dict(source)
        duplicate.update(
            run_name=f"phase2_visdrone_head_only_seed0_b{batch}",
            requested_batch=batch,
            actual_batch="not_executed",
            precision="unknown",
            recall="unknown",
            mAP50="unknown",
            mAP50_95="unknown",
            best_epoch="unknown",
            epochs_completed=0,
            queue_execution_status="skipped_duplicate_reference",
            formal_validity_status="not_executed",
            failure_reason="EXISTING key omitted batch and referenced the same b8 directory; no independent execution evidence",
            recovery_status="none",
            config_signature=f"not_executed:visdrone:head_only:seed0:requested_batch={batch}",
            evidence_path=source["evidence_path"],
        )
        rows.append(duplicate)


def queue_registry(rows: list[dict]) -> list[dict]:
    """Materialize the 26 scheduled attempts without inventing independent runs."""
    aliases = (
        ("phase2_brain_tumor_head_only_seed0", "brain_tumor_head_only_seed0"),
        ("phase2_brain_tumor_neck_head_seed0", "phase2_brain_tumor_neck_head_seed0"),
        ("phase2_brain_tumor_last_stage_neck_head_seed0", "phase2_brain_tumor_last_stage_neck_head_seed0"),
        ("phase2_brain_tumor_full_finetune_seed0", "brain_tumor_full_finetune_seed0"),
        ("phase2_brain_tumor_stable_lora_seed0", "brain_tumor_r4_stable_v1_seed0"),
        ("phase2_brain_tumor_partial_lora_seed0", "phase2_brain_tumor_partial_lora_seed0"),
        ("phase2_brain_tumor_amp_safe_lora_seed0", "phase2_brain_tumor_amp_safe_lora_seed0"),
        ("phase2_visdrone_head_only_seed0_b8", "visdrone_head_only_seed0"),
        ("phase2_visdrone_head_only_seed0_b4", "phase2_visdrone_head_only_seed0_b4"),
        ("phase2_visdrone_head_only_seed0_b2", "phase2_visdrone_head_only_seed0_b2"),
        ("phase2_visdrone_head_only_seed0_b1", "phase2_visdrone_head_only_seed0_b1"),
        ("phase2_visdrone_neck_head_seed0_b8", "phase2_visdrone_neck_head_seed0_b8"),
        ("phase2_visdrone_neck_head_seed0_b4", "phase2_visdrone_neck_head_seed0_b4"),
        ("phase2_visdrone_last_stage_neck_head_seed0_b8", "phase2_visdrone_last_stage_neck_head_seed0_b8"),
        ("phase2_visdrone_last_stage_neck_head_seed0_b4", "phase2_visdrone_last_stage_neck_head_seed0_b4"),
        ("phase2_visdrone_full_finetune_seed0_b8", "phase2_visdrone_full_finetune_seed0_b8"),
        ("phase2_visdrone_full_finetune_seed0_b4", "phase2_visdrone_full_finetune_seed0_b4"),
        ("phase2_visdrone_stable_lora_seed0", "visdrone_r4_stable_v2_seed0"),
        ("phase2_visdrone_partial_lora_seed0_b8", "phase2_visdrone_partial_lora_seed0_b8"),
        ("phase2_visdrone_amp_safe_lora_seed0_b8", "phase2_visdrone_amp_safe_lora_seed0_b8"),
        ("phase2_brain_tumor_last_stage_neck_head_seed1", "phase2_brain_tumor_last_stage_neck_head_seed1"),
        ("phase2_brain_tumor_last_stage_neck_head_seed2", "phase2_brain_tumor_last_stage_neck_head_seed2"),
        ("phase2_visdrone_full_finetune_seed1_b8", "phase2_visdrone_full_finetune_seed1_b8"),
        ("phase2_visdrone_full_finetune_seed1_b4", "phase2_visdrone_full_finetune_seed1_b4"),
        ("phase2_visdrone_full_finetune_seed2_b8", "phase2_visdrone_full_finetune_seed2_b8"),
        ("phase2_visdrone_full_finetune_seed2_b4", "phase2_visdrone_full_finetune_seed2_b4"),
    )
    by_name = {row["run_name"]: row for row in rows}
    registry = []
    for alias, source in aliases:
        if source not in by_name:
            raise AssertionError(f"Missing queue evidence mapping: {alias} -> {source}")
        row = dict(by_name[source])
        row["run_name"] = alias
        registry.append(row)
    assert len(registry) == 26
    return registry


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def seed_summary(rows: list[dict]) -> list[dict]:
    eligible = [row for row in rows if row["formal_validity_status"] == "passed"]
    groups = defaultdict(list)
    for row in eligible:
        signature = row["config_signature"]
        groups[(row["dataset"], row["method"], signature)].append(row)
    output = []
    for (dataset, method, signature), items in groups.items():
        seeds = [str(row["seed"]) for row in items]
        if len(seeds) != len(set(seeds)):
            raise AssertionError(f"Duplicate seed in formal aggregation: {dataset}/{method}/{seeds}")
        record = {
            "dataset": dataset,
            "method": method,
            "config_signature": signature,
            "n": len(items),
            "seeds": ",".join(seeds),
        }
        for field in ("precision", "recall", "mAP50", "mAP50_95"):
            values = [float(row[field]) for row in items]
            record[f"{field}_mean"] = statistics.mean(values)
            record[f"{field}_sample_std"] = statistics.stdev(values) if len(values) > 1 else "not_estimable"
            record[f"{field}_best"] = max(values)
        output.append(record)
    return sorted(output, key=lambda row: (row["dataset"], row["method"], row["config_signature"]))


def nondominated(rows: list[dict]) -> list[dict]:
    eligible = [
        row for row in rows
        if row["formal_validity_status"] == "passed"
        and all(isinstance(row[key], (int, float)) for key in ("mAP50_95", "trainable_params", "peak_gpu_mem_gib", "train_time_s"))
    ]
    front = []
    for candidate in eligible:
        dominated = False
        for other in eligible:
            if other is candidate or other["dataset"] != candidate["dataset"]:
                continue
            weak = (
                other["mAP50_95"] >= candidate["mAP50_95"]
                and other["trainable_params"] <= candidate["trainable_params"]
                and other["peak_gpu_mem_gib"] <= candidate["peak_gpu_mem_gib"]
                and other["train_time_s"] <= candidate["train_time_s"]
            )
            strict = (
                other["mAP50_95"] > candidate["mAP50_95"]
                or other["trainable_params"] < candidate["trainable_params"]
                or other["peak_gpu_mem_gib"] < candidate["peak_gpu_mem_gib"]
                or other["train_time_s"] < candidate["train_time_s"]
            )
            if weak and strict:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def pareto_summary(rows: list[dict]) -> list[dict]:
    output = []
    front_ids = {id(row) for row in nondominated(rows)}
    for dataset in ("Brain Tumor", "VisDrone"):
        eligible = [row for row in rows if row["dataset"] == dataset and row["formal_validity_status"] == "passed"]
        if not eligible:
            continue
        absolute = max(eligible, key=lambda row: float(row["mAP50_95"]))
        efficient = max(
            (row for row in eligible if isinstance(row["trainable_params"], (int, float)) and row["trainable_params"] > 0),
            key=lambda row: float(row["mAP50_95"]) / row["trainable_params"],
        )
        for row in eligible:
            labels = []
            if row is absolute:
                labels.append("absolute_best")
            if row is efficient:
                labels.append("parameter_efficient")
            if id(row) in front_ids:
                labels.append("pareto_front")
            output.append(
                {
                    "dataset": dataset,
                    "method": row["method"],
                    "run_name": row["run_name"],
                    "seed": row["seed"],
                    "mAP50_95": row["mAP50_95"],
                    "trainable_params": row["trainable_params"],
                    "peak_gpu_mem_gib": row["peak_gpu_mem_gib"],
                    "train_time_s": row["train_time_s"],
                    "designation": "+".join(labels) if labels else "dominated",
                }
            )
    return output


def audit_assertions(rows: list[dict], summaries: list[dict]) -> None:
    for row in rows:
        if row["formal_validity_status"] == "passed" and row["method"] in {
            "Head-only", "Neck + Head", "Last stage + Neck + Head", "Full fine-tuning",
        }:
            assert isinstance(row["trainable_params"], int) and row["trainable_params"] > 0
        if row["formal_validity_status"] == "passed":
            assert row["actual_batch"] not in ("unknown", "evidence_missing", "not_executed")
        if row["method"] == "AMP-safe LoRA":
            assert row["formal_validity_status"] == "implementation_failed"
    for summary in summaries:
        seeds = summary["seeds"].split(",")
        assert len(seeds) == len(set(seeds))


def plots(rows: list[dict], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    formal = [row for row in rows if row["formal_validity_status"] == "passed"]
    specs = (
        ("accuracy_vs_parameters.png", "trainable_params", "mAP50_95", "Trainable parameters", "mAP50-95"),
        ("accuracy_vs_memory.png", "peak_gpu_mem_gib", "mAP50_95", "Peak GPU memory (GiB)", "mAP50-95"),
        ("accuracy_vs_time.png", "train_time_s", "mAP50_95", "Train time (s)", "mAP50-95"),
    )
    for filename, xkey, ykey, xlabel, ylabel in specs:
        usable = [row for row in formal if isinstance(row[xkey], (int, float)) and isinstance(row[ykey], (int, float))]
        fig, ax = plt.subplots(figsize=(9, 5))
        if xkey == "method":
            labels = [f"{row['dataset']} | {row['method']} | s{row['seed']}" for row in usable]
            ax.bar(range(len(usable)), [row[ykey] for row in usable])
            ax.set_xticks(range(len(usable)), labels, rotation=70, ha="right")
        else:
            for dataset, marker in (("Brain Tumor", "o"), ("VisDrone", "s")):
                subset = [row for row in usable if row["dataset"] == dataset]
                ax.scatter([row[xkey] for row in subset], [row[ykey] for row in subset], label=dataset, marker=marker)
            ax.legend()
        ax.set(xlabel=xlabel, ylabel=ylabel)
        fig.tight_layout()
        fig.savefig(target / filename, dpi=160)
        plt.close(fig)
    # Evidence-status plots intentionally include implementation failures as zero-height categorical bars.
    statuses = defaultdict(list)
    for row in rows:
        if row["method"] in {"Original AMP LoRA", "Stable LoRA", "AMP-safe LoRA", "Adapter LR x0.1", "Adapter LR x1.0"}:
            statuses[row["dataset"]].append(row)
    for dataset, filename in (("Brain Tumor", "brain_tumor_methods_map.png"), ("VisDrone", "visdrone_methods_map.png")):
        usable = [row for row in formal if row["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = [f"{row['method']} | s{row['seed']}" for row in usable]
        ax.bar(range(len(usable)), [row["mAP50_95"] for row in usable])
        ax.set_xticks(range(len(usable)), labels, rotation=60, ha="right")
        ax.set_ylabel("mAP50-95")
        ax.set_title(dataset)
        fig.tight_layout()
        fig.savefig(target / filename, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    flat = [row for values in statuses.values() for row in values]
    ax.bar(range(len(flat)), [row["mAP50_95"] if isinstance(row["mAP50_95"], (int, float)) else 0 for row in flat])
    ax.set_xticks(range(len(flat)), [f"{row['dataset']} | {row['method']}" for row in flat], rotation=70, ha="right")
    ax.set_ylabel("mAP50-95 (failed implementations shown at 0)")
    fig.tight_layout()
    fig.savefig(target / "amp_grouped_lr_ablation.png", dpi=160)
    plt.close(fig)
    summaries = seed_summary(rows)
    multi = [row for row in summaries if row["n"] > 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [row["mAP50_95_mean"] for row in multi]
    errors = [row["mAP50_95_sample_std"] for row in multi]
    ax.bar(range(len(multi)), means, yerr=errors, capsize=4)
    ax.set_xticks(range(len(multi)), [f"{row['dataset']} | {row['method']} (n={row['n']})" for row in multi], rotation=45, ha="right")
    ax.set_ylabel("mAP50-95 mean ± sample SD")
    fig.tight_layout()
    fig.savefig(target / "multi_seed_error_bars.png", dpi=160)
    plt.close(fig)


def report(rows: list[dict], summaries: list[dict], queue: dict) -> str:
    def best(dataset: str):
        valid = [row for row in rows if row["dataset"] == dataset and row["formal_validity_status"] == "passed"]
        return max(valid, key=lambda row: float(row["mAP50_95"]))

    brain, vis = best("Brain Tumor"), best("VisDrone")
    multi = [row for row in summaries if row["n"] > 1]
    seed_lines = "\n".join(
        f"- {row['dataset']} / {row['method']}: mAP50-95={row['mAP50_95_mean']:.5f} ± "
        f"{row['mAP50_95_sample_std']:.5f}（样本标准差，n={row['n']}）"
        for row in multi
    )
    return f"""# Issue #50 Phase 2 审计修正版报告

## 审计口径

本报告仅由既有 args.yaml、results.csv、权重元数据、独立日志和 run_manifest 重建，未启动训练。
队列执行统计与正式有效性分开：队列为 {len(queue.get('completed', []))} completed /
{len(queue.get('failed', []))} failed（共 {queue.get('queue_length', 'unknown')} 项）；正式有效性由完整产物、
exit_code、数值稳定、OOM/恢复和协议一致性共同门控。

## 关键结果

- Brain Tumor 单次绝对最佳：{brain['method']}，seed={brain['seed']}，mAP50-95={brain['mAP50_95']:.5f}。
- VisDrone 干净正式单次绝对最佳：{vis['method']}，seed={vis['seed']}，mAP50-95={vis['mAP50_95']:.5f}。
{seed_lines}

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
"""


def handoff(root: Path, rows: list[dict]) -> str:
    best = {}
    for dataset in ("Brain Tumor", "VisDrone"):
        valid = [row for row in rows if row["dataset"] == dataset and row["formal_validity_status"] == "passed"]
        best[dataset] = max(valid, key=lambda row: float(row["mAP50_95"]))
    return f"""# Phase 2 审计修正版交接

- Brain Tumor 最佳正式运行：`{best['Brain Tumor']['result_dir']}`，{best['Brain Tumor']['method']}，mAP50-95={best['Brain Tumor']['mAP50_95']:.5f}
- Brain Tumor 最佳权重：`{best['Brain Tumor']['result_dir']}/weights/best.pt`
- VisDrone 最佳正式运行：`{best['VisDrone']['result_dir']}`，{best['VisDrone']['method']}，mAP50-95={best['VisDrone']['mAP50_95']:.5f}
- VisDrone 最佳权重：`{best['VisDrone']['result_dir']}/weights/best.pt`
- 逐运行结果：`reports/issue50/PHASE2_FINAL_RESULTS.csv`
- 多种子汇总：`reports/issue50/PHASE2_SEED_SUMMARY.csv`
- Pareto：`reports/issue50/PHASE2_PARETO_SUMMARY.csv`
- 图表：`reports/issue50/PHASE2_FINAL_FIGURES/`
- 队列状态：`runs/issue50/phase2_queue_status.json`

所有正式结论均要求 formal_validity_status=passed。自动降 batch、OOM、not_executed、诊断和实现失败运行保留为证据，
但不进入正式均值与 Pareto。复现历史命令见各 run_manifest；本次审计没有执行训练。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    queue = read_json(root / "runs/issue50/phase2_queue_status.json")
    if queue.get("status") != "completed":
        raise SystemExit("Refusing audit rebuild: Phase 2 queue is not completed.")
    rows = [audit_run(*item) for item in discover(root)]
    add_not_executed_head_fallbacks(rows)
    summaries = seed_summary(rows)
    audit_assertions(rows, summaries)
    report_dir = root / "reports/issue50"
    write_csv(report_dir / "PHASE2_FINAL_RESULTS.csv", rows, FINAL_FIELDS)
    summary_fields = list(summaries[0]) if summaries else ["dataset", "method", "n"]
    write_csv(report_dir / "PHASE2_SEED_SUMMARY.csv", summaries, summary_fields)
    pareto = pareto_summary(rows)
    write_csv(report_dir / "PHASE2_PARETO_SUMMARY.csv", pareto, list(pareto[0]) if pareto else ["dataset"])
    registry = queue_registry(rows)
    write_csv(report_dir / "PHASE2_EXPERIMENT_REGISTRY.csv", registry, FINAL_FIELDS)
    (report_dir / "PHASE2_FINAL_REPORT_CN.md").write_text(report(rows, summaries, queue), encoding="utf-8")
    (report_dir / "PHASE2_FINAL_HANDOFF.md").write_text(handoff(root, rows), encoding="utf-8")
    status_counts = defaultdict(int)
    for row in rows:
        status_counts[row["formal_validity_status"]] += 1
    (report_dir / "PHASE2_STATUS.md").write_text(
        "# Phase 2 审计状态\n\n"
        f"- queue_execution_status: completed={len(queue.get('completed', []))}, failed={len(queue.get('failed', []))}, "
        f"queue_length={queue.get('queue_length', 'unknown')}\n"
        f"- formal_validity_status: {dict(status_counts)}\n"
        "- 说明：前者是调度项执行结果；后者是原始产物与公平协议审计结果，两者不可混用。\n",
        encoding="utf-8",
    )
    plots(rows, report_dir / "PHASE2_FINAL_FIGURES")
    print(json.dumps({"runs": len(rows), "seed_groups": len(summaries), "formal": dict(status_counts)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
