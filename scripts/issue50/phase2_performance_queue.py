#!/usr/bin/env python3
"""Run the bounded, resumable Issue #50 Phase 2 performance queue."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from run_rank_sweep import (
    SCENES,
    build_manifest,
    collect_git_metadata,
    collect_runtime_metadata,
    command_to_text,
    manifest_path,
    repo_root,
    run_with_log,
    write_manifest,
)


NONFINITE_RE = re.compile(
    r"Non-finite gradient|NaN recovery|Global nonfinite training state|loss (?:is )?nan|Traceback",
    re.IGNORECASE,
)
OOM_RE = re.compile(r"CUDA out of memory|torch\.OutOfMemoryError", re.IGNORECASE)
PARAM_RE = re.compile(r"Trainable:\s*([0-9,]+)")
GRADIENT_RE = re.compile(r"([0-9,]+)\s+gradients")
GPU_RE = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)G\b")
SPEED_RE = re.compile(r"Speed:.*?([0-9.]+)ms inference", re.IGNORECASE)
FIELDNAMES = (
    "method",
    "unique_variable",
    "dataset",
    "seed",
    "batch",
    "accumulate",
    "configuration",
    "precision",
    "recall",
    "mAP50",
    "mAP50_95",
    "trainable_params",
    "peak_gpu_mem_gib",
    "train_time_s",
    "inference_ms_per_image",
    "stability",
    "conclusion",
    "run_name",
)


@dataclass(frozen=True)
class Job:
    scene: str
    method: str
    seed: int
    batch: int
    unique_variable: str

    @property
    def dataset(self) -> str:
        return "Brain Tumor" if self.scene == "brain_tumor" else "VisDrone"

    @property
    def name(self) -> str:
        suffix = f"_b{self.batch}" if self.scene == "visdrone" and self.method != "stable_lora" else ""
        return f"phase2_{self.scene}_{self.method}_seed{self.seed}{suffix}"


METHODS = {
    "head_only": {"lora": False, "freeze": 25, "label": "Head-only"},
    "neck_head": {"lora": False, "freeze": 13, "label": "Neck + Head"},
    "last_stage_neck_head": {"lora": False, "freeze": 10, "label": "Last stage + Neck + Head"},
    "full_finetune": {"lora": False, "freeze": None, "label": "Full fine-tuning"},
    "stable_lora": {"lora": True, "freeze": None, "label": "Stable LoRA"},
    "partial_lora": {"lora": True, "freeze": None, "label": "Neck + Head + LoRA"},
    "amp_safe_lora": {"lora": True, "freeze": None, "label": "AMP-safe LoRA"},
}
INITIAL_METHODS = (
    "head_only",
    "neck_head",
    "last_stage_neck_head",
    "full_finetune",
    "stable_lora",
    "partial_lora",
    "amp_safe_lora",
)
EXISTING = {
    ("brain_tumor", "head_only", 0): (
        "runs/issue50/formal/baselines/brain_tumor_head_only_seed0",
        "runs/issue50/formal/logs/baselines/brain_tumor_head_only_seed0.log",
        "runs/issue50/formal/logs/baselines/brain_tumor_head_only_seed0/run_manifest.json",
    ),
    ("brain_tumor", "full_finetune", 0): (
        "runs/issue50/formal/baselines/brain_tumor_full_finetune_seed0",
        "runs/issue50/formal/logs/baselines/brain_tumor_full_finetune_seed0.log",
        "runs/issue50/formal/logs/baselines/brain_tumor_full_finetune_seed0/run_manifest.json",
    ),
    ("brain_tumor", "stable_lora", 0): (
        "runs/issue50/formal/brain_tumor_r4_stable_v1_seed0",
        "runs/issue50/formal/logs/brain_tumor_r4_stable_v1_seed0.log",
        "runs/issue50/formal/logs/brain_tumor_r4_stable_v1_seed0/run_manifest.json",
    ),
    ("brain_tumor", "stable_lora", 1): (
        "runs/issue50/formal/brain_tumor_r4_stable_v1_seed1",
        "runs/issue50/formal/logs/brain_tumor_r4_stable_v1_seed1.log",
        "runs/issue50/formal/logs/brain_tumor_r4_stable_v1_seed1/run_manifest.json",
    ),
    ("visdrone", "stable_lora", 0): (
        "runs/issue50/formal/visdrone_r4_stable_v2_seed0",
        "runs/issue50/formal/logs/visdrone_r4_stable_v2_seed0.log",
        "runs/issue50/formal/logs/visdrone_r4_stable_v2_seed0/run_manifest.json",
    ),
    ("visdrone", "stable_lora", 1): (
        "runs/issue50/formal/visdrone_r4_stable_v2_seed1",
        "runs/issue50/formal/logs/visdrone_r4_stable_v2_seed1.log",
        "runs/issue50/formal/logs/visdrone_r4_stable_v2_seed1/run_manifest.json",
    ),
    ("visdrone", "head_only", 0): (
        "runs/issue50/formal/baselines/visdrone_head_only_seed0",
        "runs/issue50/formal/logs/baselines/visdrone_head_only_seed0.log",
        "runs/issue50/formal/logs/baselines/visdrone_head_only_seed0/run_manifest.json",
    ),
}
KNOWN_PARAMS = {
    ("brain_tumor", "head_only"): 347718,
    ("brain_tumor", "full_finetune"): 2662546,
    ("brain_tumor", "stable_lora"): 409174,
    ("visdrone", "head_only"): 349278,
    ("visdrone", "stable_lora"): 410734,
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def run_paths(root: Path, job: Job) -> tuple[Path, Path, Path]:
    existing = EXISTING.get((job.scene, job.method, job.seed))
    if existing:
        return tuple(root / item for item in existing)
    project = root / "runs/issue50/phase2"
    logs = root / "runs/issue50/phase2/logs"
    return project / job.name, logs / f"{job.name}.log", manifest_path(logs, job.name)


def inspect(root: Path, job: Job) -> dict:
    run, log_path, manifest_file = run_paths(root, job)
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    results_path = run / "results.csv"
    rows = list(csv.DictReader(results_path.open(encoding="utf-8"))) if results_path.exists() else []
    required = (results_path, run / "args.yaml", run / "weights/best.pt", log_path, manifest_file)
    complete = all(path.exists() for path in required)
    nonfinite = bool(NONFINITE_RE.search(log))
    success = manifest.get("exit_code") == 0 and manifest.get("success") is True
    best = max(rows, key=lambda row: finite(row.get("metrics/mAP50-95(B)"))) if rows else {}
    speed = SPEED_RE.findall(log)
    params = PARAM_RE.findall(log)
    gradients = GRADIENT_RE.findall(log)
    memories = [finite(value) for value in GPU_RE.findall(log)]
    return {
        "passed": complete and success and not nonfinite and bool(rows),
        "complete": complete,
        "oom": bool(OOM_RE.search(log)),
        "nonfinite": nonfinite,
        "exit_code": manifest.get("exit_code"),
        "precision": finite(best.get("metrics/precision(B)")),
        "recall": finite(best.get("metrics/recall(B)")),
        "mAP50": finite(best.get("metrics/mAP50(B)")),
        "mAP50_95": finite(best.get("metrics/mAP50-95(B)")),
        "train_time_s": finite(rows[-1].get("time")) if rows else 0.0,
        "trainable_params": (
            int(params[-1].replace(",", ""))
            if params
            else int(gradients[-1].replace(",", ""))
            if gradients
            else KNOWN_PARAMS.get((job.scene, job.method), 0)
        ),
        "peak_gpu_mem_gib": max(memories, default=0.0),
        "inference_ms_per_image": finite(speed[-1]) if speed else 0.0,
        "run": str(run),
        "log": str(log_path),
    }


def command(root: Path, job: Job, launcher: str, device: str) -> list[str]:
    scene = SCENES[job.scene]
    method = METHODS[job.method]
    project = root / "runs/issue50/phase2"
    cmd = [
        launcher,
        "train",
        f"cfg={(root / scene.cfg).resolve()}",
        f"model={(root / 'weights/YOLO-Master-EsMoE-N.pt').resolve()}",
        "pretrained=True",
        "resume=False",
        "exist_ok=False",
        "val=True",
        "plots=True",
        "deterministic=True",
        "save=True",
        f"epochs={scene.epochs}",
        f"batch={job.batch}",
        "nbs=64",
        f"imgsz={scene.imgsz}",
        f"fraction={scene.fraction:g}",
        f"seed={job.seed}",
        f"workers={scene.workers}",
        f"device={device}",
        *scene.extra_args,
        f"project={project}",
        f"name={job.name}",
    ]
    if method["lora"]:
        cmd.extend(("lora_r=4", "lora_alpha=8", "lora_gradient_checkpointing=False"))
    else:
        cmd.extend(("lora_r=0", "lora_alpha=0", "lora_gradient_checkpointing=False"))
    if method["freeze"] is not None:
        cmd.append(f"freeze={method['freeze']}")
    if job.method == "partial_lora":
        cmd.append("lora_unfreeze_layers=[13,14,15,16,17,18,19,20,21,22,23,24]")
    if job.method == "amp_safe_lora":
        cmd.extend(("amp=True", "lora_amp_safe=True"))
    return cmd


def registry_row(job: Job, result: dict) -> dict[str, object]:
    effective_accumulate = max(round(64 / job.batch), 1)
    stability = "stable" if result["passed"] else ("oom" if result["oom"] else "failed")
    return {
        "method": METHODS[job.method]["label"],
        "unique_variable": job.unique_variable,
        "dataset": job.dataset,
        "seed": job.seed,
        "batch": job.batch,
        "accumulate": effective_accumulate,
        "configuration": f"imgsz={SCENES[job.scene].imgsz}; fraction={SCENES[job.scene].fraction:g}",
        "precision": result["precision"],
        "recall": result["recall"],
        "mAP50": result["mAP50"],
        "mAP50_95": result["mAP50_95"],
        "trainable_params": result["trainable_params"],
        "peak_gpu_mem_gib": result["peak_gpu_mem_gib"],
        "train_time_s": result["train_time_s"],
        "inference_ms_per_image": result["inference_ms_per_image"],
        "stability": stability,
        "conclusion": "formal candidate" if result["passed"] else "retained failure evidence",
        "run_name": job.name,
    }


def write_registry(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def initial_jobs() -> list[Job]:
    jobs = []
    for scene in ("brain_tumor", "visdrone"):
        batch = SCENES[scene].batch
        for method in INITIAL_METHODS:
            jobs.append(
                Job(
                    scene,
                    method,
                    0,
                    batch,
                    {
                        "head_only": "train Detect only",
                        "neck_head": "unfreeze layers 13-25",
                        "last_stage_neck_head": "unfreeze layers 10-25",
                        "full_finetune": "unfreeze all base parameters",
                        "stable_lora": "Phase 1 stable LoRA",
                        "partial_lora": "unfreeze neck base layers alongside LoRA",
                        "amp_safe_lora": "AMP base path with FP32 LoRA path",
                    }[method],
                )
            )
    return jobs


def select_seed_jobs(root: Path, rows_by_key: dict[tuple[str, str, int], dict]) -> list[Job]:
    selected: list[Job] = []
    for scene in ("brain_tumor", "visdrone"):
        candidates = [
            (method, rows_by_key[(scene, method, 0)])
            for method in INITIAL_METHODS
            if (scene, method, 0) in rows_by_key and rows_by_key[(scene, method, 0)]["passed"]
        ]
        if not candidates:
            continue
        absolute = max(candidates, key=lambda item: item[1]["mAP50_95"])[0]
        efficient = max(
            candidates,
            key=lambda item: item[1]["mAP50_95"] / max(item[1]["trainable_params"], 1),
        )[0]
        for method in dict.fromkeys((absolute, efficient)):
            for seed in (1, 2):
                existing_job = Job(scene, method, seed, SCENES[scene].batch, "multi-seed confirmation")
                if inspect(root, existing_job)["passed"]:
                    continue
                selected.append(existing_job)
    return selected


def execute(root: Path, job: Job, launcher: str, device: str) -> dict:
    run, log_path, manifest_file = run_paths(root, job)
    existing = inspect(root, job)
    if existing["passed"] or EXISTING.get((job.scene, job.method, job.seed)):
        return existing
    if run.exists():
        return existing
    cmd = command(root, job, launcher, device)
    metadata = collect_git_metadata(root)
    manifest = build_manifest(
        run_name=job.name,
        scene=job.scene,
        rank=4 if METHODS[job.method]["lora"] else 0,
        cmd=cmd,
        root=root,
        weights_path=(root / "weights/YOLO-Master-EsMoE-N.pt").resolve(),
        formal_root=root / "runs/issue50/phase2",
        log_path=log_path,
        git_metadata=metadata,
        runtime_metadata=collect_runtime_metadata(),
    )
    manifest.update({"method": job.method, "seed": job.seed, "physical_batch": job.batch, "started_at": now()})
    write_manifest(manifest_file, manifest)
    code = run_with_log(cmd, root=root, log_path=log_path)
    manifest.update({"finished_at": now(), "exit_code": code, "success": code == 0})
    write_manifest(manifest_file, manifest)
    return inspect(root, job)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", default="yolo")
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    state_path = root / "runs/issue50/phase2_queue_status.json"
    registry_path = root / "reports/issue50/PHASE2_EXPERIMENT_REGISTRY.csv"
    status_path = root / "reports/issue50/PHASE2_STATUS.md"
    jobs = initial_jobs()
    if args.dry_run:
        for job in jobs:
            state = inspect(root, job)
            action = "SKIP" if state["passed"] else "RUN"
            print(f"{action} {job.name}: {command_to_text(command(root, job, args.launcher, args.device))}")
        return

    rows: list[dict[str, object]] = []
    results: dict[tuple[str, str, int], dict] = {}
    completed: list[str] = []
    failed: list[str] = []
    queue = list(jobs)
    index = 0
    while index < len(queue):
        job = queue[index]
        state = inspect(root, job)
        atomic_json(
            state_path,
            {
                "status": "running",
                "updated_at": now(),
                "current": job.name,
                "queue_index": index,
                "queue_length": len(queue),
                "completed": completed,
                "failed": failed,
                "next": queue[index + 1].name if index + 1 < len(queue) else None,
            },
        )
        result = state if state["passed"] else execute(root, job, args.launcher, args.device)
        # VisDrone OOM fallback keeps imgsz and nbs fixed while lowering physical batch once per level.
        if result["oom"] and job.scene == "visdrone" and job.batch > 1:
            failed.append(job.name)
            fallback = replace(job, batch=max(job.batch // 2, 1))
            queue.insert(index + 1, fallback)
        elif result["passed"]:
            completed.append(job.name)
            results[(job.scene, job.method, job.seed)] = result
        else:
            failed.append(job.name)
        rows.append(registry_row(job, result))
        write_registry(registry_path, rows)
        status_path.write_text(
            "# Issue #50 Phase 2 Status\n\n"
            f"- Current: `{job.name}` finished\n"
            f"- Completed: {len(completed)}\n"
            f"- Failed attempts: {len(failed)}\n"
            f"- Next: `{queue[index + 1].name if index + 1 < len(queue) else 'seed selection'}`\n"
            f"- Queue state: `runs/issue50/phase2_queue_status.json`\n",
            encoding="utf-8",
        )
        index += 1
        if index == len(queue):
            seed_jobs = select_seed_jobs(root, results)
            if seed_jobs:
                queue.extend(seed_jobs)

    atomic_json(
        state_path,
        {
            "status": "completed",
            "updated_at": now(),
            "current": None,
            "queue_length": len(queue),
            "completed": completed,
            "failed": failed,
            "registry": str(registry_path),
        },
    )
    status_path.write_text(
        "# Issue #50 Phase 2 Status\n\n"
        f"- Queue: completed\n- Completed: {len(completed)}\n- Failed attempts: {len(failed)}\n"
        f"- Registry: `{registry_path.relative_to(root)}`\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    main()
