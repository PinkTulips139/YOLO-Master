#!/usr/bin/env python3
"""Summarize formal Issue #50 YOLO-Master LoRA runs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - user-facing dependency check
    raise SystemExit("PyYAML is required. Run this inside the yolo-master conda environment.") from exc


EXPECTED_RUNS = [
    ("brain_tumor", 4),
    ("brain_tumor", 8),
    ("brain_tumor", 16),
    ("visdrone", 4),
    ("visdrone", 8),
    ("visdrone", 16),
]


FIELDNAMES = [
    "scene",
    "rank",
    "alpha",
    "run_name",
    "run_dir",
    "status",
    "mAP50",
    "mAP50_95",
    "precision",
    "recall",
    "best_epoch",
    "trainable_params",
    "adapter_params",
    "train_time_min",
    "peak_gpu_mem_gb",
    "weights_transferred",
    "weights_total",
    "weights_transfer_ratio",
    "nan_status",
    "gradient_checkpointing_requested",
    "gradient_checkpointing_actual",
    "effective_lora_backend",
    "log_path",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        return yaml.safe_load(handle) or {}


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({(key or "").strip(): (value or "").strip() for key, value in row.items()})
    return rows


def as_float(value: str | int | float | None, default: float = 0.0) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def best_metrics(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        return {}

    best = max(rows, key=lambda row: as_float(row.get("metrics/mAP50-95(B)")))
    return {
        "best_epoch": best.get("epoch", ""),
        "mAP50": best.get("metrics/mAP50(B)", ""),
        "mAP50_95": best.get("metrics/mAP50-95(B)", ""),
        "precision": best.get("metrics/precision(B)", ""),
        "recall": best.get("metrics/recall(B)", ""),
    }


def read_log(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_param_stats(log_text: str) -> dict[str, str]:
    match = re.search(
        r"Trainable:\s*([0-9,]+).*?Adapter Params:\s*([0-9,]+)",
        log_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return {"trainable_params": "", "adapter_params": ""}
    return {
        "trainable_params": match.group(1).replace(",", ""),
        "adapter_params": match.group(2).replace(",", ""),
    }


def parse_weight_transfer(log_text: str) -> dict[str, str]:
    match = re.search(r"Transferred\s+([0-9,]+)\s*/\s*([0-9,]+)\s+items", log_text, flags=re.IGNORECASE)
    if not match:
        return {"weights_transferred": "", "weights_total": "", "weights_transfer_ratio": ""}

    transferred = int(match.group(1).replace(",", ""))
    total = int(match.group(2).replace(",", ""))
    ratio = transferred / total if total else 0.0
    return {
        "weights_transferred": str(transferred),
        "weights_total": str(total),
        "weights_transfer_ratio": f"{ratio:.4f}",
    }


def parse_train_time_min(log_text: str, results_rows: list[dict[str, str]]) -> str:
    match = re.search(r"epochs completed in\s+([0-9.]+)\s+hours", log_text, flags=re.IGNORECASE)
    if match:
        return f"{float(match.group(1)) * 60.0:.2f}"

    elapsed_match = re.search(r"elapsed_min:\s*([0-9.]+)", log_text, flags=re.IGNORECASE)
    if elapsed_match:
        return f"{float(elapsed_match.group(1)):.2f}"

    if results_rows and "time" in results_rows[-1]:
        return f"{as_float(results_rows[-1].get('time')) / 60.0:.2f}"

    return ""


def parse_peak_gpu_mem(log_text: str) -> str:
    values = [float(value) for value in re.findall(r"\b([0-9]+(?:\.[0-9]+)?)G\b", log_text)]
    return f"{max(values):.3f}" if values else ""


def parse_nan_status(log_text: str, results_rows: list[dict[str, str]]) -> str:
    for row in results_rows:
        for key, value in row.items():
            if key == "epoch":
                continue
            raw = str(value).strip().lower()
            if raw in {"nan", "+nan", "-nan"}:
                return "detected_in_results"
            numeric = as_float(raw, default=float("nan"))
            if isinstance(numeric, float) and math.isnan(numeric):
                continue

    critical = re.search(
        r"(nan detected|detected nan|non[- ]finite|loss is nan|nan loss|nan gradients?)",
        log_text,
        flags=re.IGNORECASE,
    )
    if critical:
        return "detected_in_log"

    if re.search(r"NaN recovery", log_text, flags=re.IGNORECASE):
        return "recovery_reference_only"

    if log_text:
        return "not_detected"
    return "unknown_no_log"


def parse_gradient_checkpointing(args: dict[str, Any], log_text: str) -> tuple[str, str]:
    requested = str(args.get("lora_gradient_checkpointing", "")).lower()
    if re.search(r"Skipping gradient checkpointing", log_text, flags=re.IGNORECASE):
        return requested, "skipped_moe_incompatible"
    if requested in {"false", "0", "none"}:
        return requested, "disabled"
    if requested in {"true", "1"}:
        return requested, "enabled_or_requested"
    return requested, "unknown"


def infer_status(run_dir: Path, results_rows: list[dict[str, str]], log_text: str) -> str:
    if not run_dir.exists():
        return "missing"
    if re.search(r"epochs completed", log_text, flags=re.IGNORECASE):
        return "completed"
    if results_rows:
        return "partial_or_running"
    return "created_no_results"


def summarize_one(project: Path, log_dir: Path, scene: str, rank: int) -> dict[str, str]:
    run_name = f"{scene}_r{rank}_seed0"
    run_dir = project / run_name
    log_path = log_dir / f"{run_name}.log"
    args = read_yaml(run_dir / "args.yaml")
    results_rows = read_results(run_dir / "results.csv")
    log_text = read_log(log_path)

    metrics = best_metrics(results_rows)
    params = parse_param_stats(log_text)
    transfer = parse_weight_transfer(log_text)
    gc_requested, gc_actual = parse_gradient_checkpointing(args, log_text)

    return {
        "scene": scene,
        "rank": str(rank),
        "alpha": str(rank * 2),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "status": infer_status(run_dir, results_rows, log_text),
        "mAP50": metrics.get("mAP50", ""),
        "mAP50_95": metrics.get("mAP50_95", ""),
        "precision": metrics.get("precision", ""),
        "recall": metrics.get("recall", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "trainable_params": params["trainable_params"],
        "adapter_params": params["adapter_params"],
        "train_time_min": parse_train_time_min(log_text, results_rows),
        "peak_gpu_mem_gb": parse_peak_gpu_mem(log_text),
        "weights_transferred": transfer["weights_transferred"],
        "weights_total": transfer["weights_total"],
        "weights_transfer_ratio": transfer["weights_transfer_ratio"],
        "nan_status": parse_nan_status(log_text, results_rows),
        "gradient_checkpointing_requested": gc_requested,
        "gradient_checkpointing_actual": gc_actual,
        "effective_lora_backend": str(args.get("effective_lora_backend", "")),
        "log_path": str(log_path),
    }


def write_summary(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    root = repo_root()
    formal_root = (root / "runs" / "issue50" / "formal").resolve()
    logs_root = (formal_root / "logs").resolve()
    reports_root = (root / "reports" / "issue50").resolve()
    output = reports_root / "FORMAL_RESULTS_SUMMARY.csv"

    rows = [summarize_one(formal_root, logs_root, scene, rank) for scene, rank in EXPECTED_RUNS]
    write_summary(rows, output)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
