"""Build reproducibility manifests for completed Issue #50 diagnostic runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
MEMORY_RE = re.compile(r"\b(\d+(?:\.\d+)?)G\b")
COUNT_RE = re.compile(
    r"Trainable:\s*([\d,]+).*?Frozen Base:\s*([\d,]+).*?Adapter Params:\s*([\d,]+)",
    re.DOTALL,
)


def key_values(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def best_metrics(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {}
    metric = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda row: float(row[metric]))
    return {
        "epochs_recorded": len(rows),
        "best_epoch": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best[metric]),
        "train_box_loss": float(best["train/box_loss"]),
        "train_cls_loss": float(best["train/cls_loss"]),
        "train_dfl_loss": float(best["train/dfl_loss"]),
        "train_mixture_aux_loss": float(best["train/mixture_aux_loss"]),
        "val_box_loss": float(best["val/box_loss"]),
        "val_cls_loss": float(best["val/cls_loss"]),
        "val_dfl_loss": float(best["val/dfl_loss"]),
        "elapsed_seconds_at_best": float(best["time"]),
    }


def build_manifest(repo: Path, log_dir: Path) -> dict[str, object]:
    run_name = log_dir.name
    result_dir = repo / "runs" / "issue50" / "diagnostics" / run_name
    log_path = log_dir / "train.log"
    log = ANSI_RE.sub("", log_path.read_text(encoding="utf-8", errors="replace")) if log_path.exists() else ""
    status = key_values(log_dir / "status.txt")
    provenance = key_values(log_dir / "provenance.txt")
    environment = key_values(log_dir / "environment.txt")
    counts = COUNT_RE.search(log)
    memories = [float(value) for value in MEMORY_RE.findall(log)]
    events = {
        "nan_or_inf_text": bool(re.search(r"\b(?:nan|inf)\b|non-finite", log, re.IGNORECASE)),
        "gradient_overflow": "Non-finite gradient detected" in log,
        "recovery": "NaN recovery" in log,
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": run_name,
        "experiment_type": "diagnostic",
        "branch": provenance.get("branch", "unknown"),
        "git_commit": provenance.get("commit", "unknown"),
        "command": (log_dir / "command.txt").read_text(encoding="utf-8", errors="replace").strip(),
        "result_dir": str(result_dir),
        "log_path": str(log_path),
        "args_path": str(result_dir / "args.yaml"),
        "results_path": str(result_dir / "results.csv"),
        "weights": {
            name: str(result_dir / "weights" / name)
            for name in ("best.pt", "last.pt", "last_healthy.pt")
            if (result_dir / "weights" / name).exists()
        },
        "status": status,
        "environment": environment,
        "metrics": best_metrics(result_dir / "results.csv"),
        "parameters": (
            {
                "trainable": int(counts.group(1).replace(",", "")),
                "frozen": int(counts.group(2).replace(",", "")),
                "adapter": int(counts.group(3).replace(",", "")),
            }
            if counts
            else {}
        ),
        "peak_gpu_memory_gib_from_log": max(memories, default=None),
        "stability": events,
        "artifacts_complete": all(
            path.exists()
            for path in (
                result_dir / "args.yaml",
                result_dir / "results.csv",
                result_dir / "weights" / "best.pt",
                result_dir / "weights" / "last.pt",
                log_path,
                log_dir / "command.txt",
                log_dir / "environment.txt",
                log_dir / "provenance.txt",
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", help="Only build the named run; may be repeated.")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    logs_root = repo / "runs" / "issue50" / "diagnostics" / "logs"
    selected = set(args.run or ())
    for log_dir in sorted(path for path in logs_root.iterdir() if path.is_dir()):
        if selected and log_dir.name not in selected:
            continue
        manifest = build_manifest(repo, log_dir)
        output = log_dir / "run_manifest.json"
        output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{log_dir.name}: {output} complete={manifest['artifacts_complete']}")


if __name__ == "__main__":
    main()
