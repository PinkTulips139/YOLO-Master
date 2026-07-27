#!/usr/bin/env python3
"""Run the minimal Issue #50 head-only and full-finetune baselines serially."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
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


METHODS = ("head_only", "full_finetune")
SCENE_ORDER = ("brain_tumor", "visdrone")
UNSTABLE_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])|NaN recovery|Global nonfinite training state|Traceback",
    re.IGNORECASE,
)


def atomic_write_json(path: Path, payload: dict) -> None:
    """Atomically persist resumable queue state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_command(
    *, launcher: str, scene: str, method: str, root: Path, project: Path, device: str
) -> tuple[list[str], str]:
    """Build one baseline command while retaining the frozen scene protocol."""
    spec = SCENES[scene]
    name = f"{scene}_{method}_seed0"
    command = [
        launcher,
        "train",
        f"cfg={(root / spec.cfg).resolve()}",
        f"model={(root / 'weights' / 'YOLO-Master-EsMoE-N.pt').resolve()}",
        "pretrained=True",
        "resume=False",
        "exist_ok=False",
        "val=True",
        "plots=True",
        "deterministic=True",
        "save=True",
        "lora_r=0",
        "lora_alpha=0",
        "lora_gradient_checkpointing=False",
        f"epochs={spec.epochs}",
        f"batch={spec.batch}",
        f"imgsz={spec.imgsz}",
        f"fraction={spec.fraction:g}",
        "seed=0",
        f"workers={spec.workers}",
        f"device={device}",
        *spec.extra_args,
        f"project={project}",
        f"name={name}",
    ]
    if method == "head_only":
        # YOLO-Master-N has layers 0..25; freeze the backbone/neck (0..24), leaving Detect (25) trainable.
        command.insert(-2, "freeze=25")
    return command, name


def inspect_run(project: Path, logs: Path, name: str) -> dict:
    """Return the strict completion gate for one baseline."""
    run = project / name
    manifest_file = manifest_path(logs, name)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    log_file = logs / f"{name}.log"
    log_text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
    results = run / "results.csv"
    rows = list(csv.DictReader(results.open(encoding="utf-8"))) if results.exists() else []
    artifacts = all((run / relative).exists() for relative in ("args.yaml", "results.csv", "weights/best.pt"))
    unstable = bool(UNSTABLE_PATTERN.search(log_text))
    return {
        "name": name,
        "epochs": len(rows),
        "exit_code": manifest.get("exit_code"),
        "success": manifest.get("success") is True,
        "artifacts": artifacts,
        "unstable": unstable,
        "passed": manifest.get("exit_code") == 0 and manifest.get("success") is True and artifacts and not unstable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", choices=SCENE_ORDER, default=list(SCENE_ORDER))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--launcher", default="yolo")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    project = (root / "runs" / "issue50" / "formal" / "baselines").resolve()
    logs = (root / "runs" / "issue50" / "formal" / "logs" / "baselines").resolve()
    state_path = root / "runs" / "issue50" / "final_baseline_queue_status.json"
    jobs = [
        (scene, method, *build_command(
            launcher=args.launcher,
            scene=scene,
            method=method,
            root=root,
            project=project,
            device=args.device,
        ))
        for scene in SCENE_ORDER
        if scene in args.scenes
        for method in METHODS
        if method in args.methods
    ]

    for scene, method, command, name in jobs:
        print(f"{name}: {command_to_text(command)}")
        if args.dry_run:
            continue

        existing = inspect_run(project, logs, name)
        if existing["passed"]:
            print(f"SKIP completed: {name}")
            continue
        if (project / name).exists():
            raise SystemExit(f"Refusing to overwrite incomplete baseline: {project / name}")
        if collect_git_metadata(root)["git_dirty"]:
            raise SystemExit("Refusing to start a formal baseline from a dirty Git worktree.")

        log_path = logs / f"{name}.log"
        manifest = build_manifest(
            run_name=name,
            scene=scene,
            rank=0,
            cmd=command,
            root=root,
            weights_path=(root / "weights" / "YOLO-Master-EsMoE-N.pt").resolve(),
            formal_root=project,
            log_path=log_path,
            git_metadata=collect_git_metadata(root),
            runtime_metadata=collect_runtime_metadata(),
        )
        manifest["method"] = method
        current_manifest = manifest_path(logs, name)
        write_manifest(current_manifest, manifest)
        atomic_write_json(
            state_path,
            {"status": "running", "current": name, "completed": [inspect_run(project, logs, n)["name"] for *_, n in jobs if inspect_run(project, logs, n)["passed"]]},
        )
        return_code = run_with_log(command, root=root, log_path=log_path)
        manifest.update({"finished_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
                         "exit_code": return_code, "success": return_code == 0})
        write_manifest(current_manifest, manifest)
        result = inspect_run(project, logs, name)
        if not result["passed"]:
            atomic_write_json(state_path, {"status": "failed", "current": name, "result": result})
            raise SystemExit(return_code or 2)

    completed = [inspect_run(project, logs, name) for *_, name in jobs]
    atomic_write_json(state_path, {"status": "completed", "current": None, "completed": completed})


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    main()
