#!/usr/bin/env python3
"""Strict semantic gate for the Issue #50 Phase 2 audited artifacts."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import yaml


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        data = list(reader)
        assert reader.fieldnames and all(len(row) == len(reader.fieldnames) for row in data)
        return data


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    report = root / "reports/issue50"
    final = rows(report / "PHASE2_FINAL_RESULTS.csv")
    registry = rows(report / "PHASE2_EXPERIMENT_REGISTRY.csv")
    seeds = rows(report / "PHASE2_SEED_SUMMARY.csv")
    pareto = rows(report / "PHASE2_PARETO_SUMMARY.csv")
    assert len(registry) == 26
    queue = json.loads((root / "runs/issue50/phase2_queue_status.json").read_text(encoding="utf-8"))
    assert queue["status"] == "completed"
    assert queue["queue_length"] == 26
    assert len(queue["completed"]) == 15 and len(queue["failed"]) == 11
    for row in final:
        if row["formal_validity_status"] == "passed" and row["method"] in {
            "Head-only", "Neck + Head", "Last stage + Neck + Head", "Full fine-tuning",
        }:
            assert int(row["trainable_params"]) > 0
        if row["formal_validity_status"] == "passed":
            args = yaml.safe_load((Path(row["result_dir"]) / "args.yaml").read_text(encoding="utf-8"))
            assert str(args["batch"]) == row["actual_batch"]
        if row["method"] == "AMP-safe LoRA":
            assert row["formal_validity_status"] == "implementation_failed"
        if row["formal_validity_status"] in {"failed", "not_executed", "implementation_failed"}:
            assert row["failure_reason"]
    assert all(
        row["method"] == "Stable LoRA"
        for row in final
        if re.search(r"(brain_tumor|visdrone)_r4_stable_v", row["run_name"])
    )
    assert all("not_executed" in row["formal_validity_status"] for row in final if re.search(r"head_only_seed0_b[421]$", row["run_name"]))
    for row in final:
        if re.search(r"visdrone_full_finetune_seed[012]_b8$", row["run_name"]):
            assert row["requested_batch"] == "8" and row["actual_batch"] == "4"
            assert row["recovery_status"].startswith("oom_recovered_diagnostic")
            assert row["formal_validity_status"] == "failed"
    for row in seeds:
        seed_values = row["seeds"].split(",")
        assert int(row["n"]) == len(seed_values) == len(set(seed_values))
    eligible = {
        (row["dataset"], row["method"], row["config_signature"])
        for row in final
        if row["formal_validity_status"] == "passed"
    }
    assert all((row["dataset"], row["method"], row["config_signature"]) in eligible for row in seeds)
    designations = Counter(
        token
        for row in pareto
        for token in row["designation"].split("+")
        if token in {"absolute_best", "parameter_efficient", "pareto_front"}
    )
    assert designations["absolute_best"] == 2 and designations["parameter_efficient"] == 2
    assert designations["pareto_front"] >= 2
    required = (
        report / "PHASE2_FINAL_REPORT_CN.md",
        report / "PHASE2_FINAL_HANDOFF.md",
        report / "PHASE2_STATUS.md",
        report / "PHASE2_FINAL_FIGURES/accuracy_vs_parameters.png",
        report / "PHASE2_FINAL_FIGURES/accuracy_vs_memory.png",
        report / "PHASE2_FINAL_FIGURES/accuracy_vs_time.png",
        report / "PHASE2_FINAL_FIGURES/amp_grouped_lr_ablation.png",
        report / "PHASE2_FINAL_FIGURES/brain_tumor_methods_map.png",
        report / "PHASE2_FINAL_FIGURES/visdrone_methods_map.png",
        report / "PHASE2_FINAL_FIGURES/multi_seed_error_bars.png",
    )
    assert all(path.exists() and path.stat().st_size > 0 for path in required)
    processes = subprocess.run(
        ["pgrep", "-af", r"phase2_performance_queue.py|[y]olo train"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    assert not processes, f"Training process detected: {processes}"
    print(
        json.dumps(
            {
                "queue_execution_status": {"completed": 15, "failed": 11},
                "formal_validity_status": dict(Counter(row["formal_validity_status"] for row in final)),
                "csv_rows": {
                    "registry": len(registry),
                    "final": len(final),
                    "seed_summary": len(seeds),
                    "pareto": len(pareto),
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
