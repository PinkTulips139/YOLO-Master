#!/usr/bin/env python3
"""Run the formal Issue #50 YOLO-Master LoRA rank sweep.

This script is intentionally conservative:
- every run starts from weights/YOLO-Master-EsMoE-N.pt;
- resume is always disabled;
- existing run directories are not overwritten by default;
- --dry-run prints the six commands without starting training.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


RANKS = (4, 8, 16)


@dataclass(frozen=True)
class SceneSpec:
    cfg: str
    epochs: int
    batch: int
    imgsz: int
    fraction: float
    seed: int
    workers: int


SCENES: dict[str, SceneSpec] = {
    "brain_tumor": SceneSpec(
        cfg="examples/lora_examples/yolo_master_brain_tumor_lora.yaml",
        epochs=40,
        batch=16,
        imgsz=640,
        fraction=1.0,
        seed=0,
        workers=4,
    ),
    "visdrone": SceneSpec(
        cfg="examples/lora_examples/yolo_master_visdrone_lora.yaml",
        epochs=30,
        batch=8,
        imgsz=768,
        fraction=0.2,
        seed=0,
        workers=8,
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def command_to_text(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)


def selected_scenes(scene: str) -> Iterable[tuple[str, SceneSpec]]:
    if scene == "all":
        return SCENES.items()
    return ((scene, SCENES[scene]),)


def validate_rank_values(ranks: Iterable[int]) -> list[int]:
    values = list(ranks)
    unsupported = sorted(set(values) - set(RANKS))
    if unsupported:
        raise SystemExit(f"Unsupported ranks: {unsupported}. Allowed ranks: {list(RANKS)}")
    return values


def build_train_command(
    *,
    launcher: str,
    scene: str,
    spec: SceneSpec,
    rank: int,
    project: Path,
    weights_path: Path,
    cfg_path: Path,
    device: str,
) -> tuple[list[str], str]:
    run_name = f"{scene}_r{rank}_seed{spec.seed}"
    alpha = rank * 2
    cmd = [
        launcher,
        "train",
        f"cfg={cfg_path}",
        f"model={weights_path}",
        "pretrained=True",
        "resume=False",
        "exist_ok=False",
        "val=True",
        "plots=True",
        "deterministic=True",
        "save=True",
        f"lora_r={rank}",
        f"lora_alpha={alpha}",
        f"epochs={spec.epochs}",
        f"batch={spec.batch}",
        f"imgsz={spec.imgsz}",
        f"fraction={spec.fraction:g}",
        f"seed={spec.seed}",
        f"workers={spec.workers}",
        f"device={device}",
        f"project={project}",
        f"name={run_name}",
    ]
    return cmd, run_name


def ensure_ready(
    *,
    weights_path: Path,
    cfg_paths: Iterable[Path],
    formal_root: Path,
    logs_root: Path,
    run_names: Iterable[str],
    allow_existing: bool,
) -> None:
    if not weights_path.exists():
        raise SystemExit(f"Required pretrained weights not found: {weights_path}")

    missing_cfgs = [path for path in cfg_paths if not path.exists()]
    if missing_cfgs:
        formatted = "\n".join(str(path) for path in missing_cfgs)
        raise SystemExit(f"Required scene YAML file(s) not found:\n{formatted}")

    try:
        logs_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"Cannot create formal output directory {formal_root}: {exc}") from exc

    if allow_existing:
        return

    existing = [name for name in run_names if (formal_root / name).exists()]
    if existing:
        formatted = ", ".join(existing)
        raise SystemExit(
            "Refusing to overwrite existing formal runs: "
            f"{formatted}. Use a new project/name or pass --allow-existing intentionally."
        )


def run_with_log(cmd: list[str], *, root: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    started_perf = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"started: {started}\n")
        log.write(f"cwd: {root}\n")
        log.write(f"command: {command_to_text(cmd)}\n\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return_code = proc.wait()

        elapsed_min = (time.perf_counter() - started_perf) / 60.0
        log.write(f"\nfinished: {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"return_code: {return_code}\n")
        log.write(f"elapsed_min: {elapsed_min:.3f}\n")

    return return_code


def main() -> None:
    # The repository path may contain Chinese characters on Windows.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=[*SCENES.keys(), "all"], default="all")
    parser.add_argument("--ranks", nargs="+", type=int, default=list(RANKS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--launcher", default="yolo")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    formal_root = (root / "runs" / "issue50" / "formal").resolve()
    logs_root = (formal_root / "logs").resolve()
    weights_path = (root / "weights" / "YOLO-Master-EsMoE-N.pt").resolve()
    cfg_paths = {scene: (root / spec.cfg).resolve() for scene, spec in SCENES.items()}
    ranks = validate_rank_values(args.ranks)
    commands: list[tuple[list[str], str, Path]] = []

    for scene, spec in selected_scenes(args.scene):
        for rank in ranks:
            cmd, run_name = build_train_command(
                launcher=args.launcher,
                scene=scene,
                spec=spec,
                rank=rank,
                project=formal_root,
                weights_path=weights_path,
                cfg_path=cfg_paths[scene],
                device=args.device,
            )
            log_path = logs_root / f"{run_name}.log"
            commands.append((cmd, run_name, log_path))

    ensure_ready(
        weights_path=weights_path,
        cfg_paths=cfg_paths.values(),
        formal_root=formal_root,
        logs_root=logs_root,
        run_names=(name for _, name, _ in commands),
        allow_existing=args.allow_existing,
    )

    invoked_from = Path.cwd().resolve()
    print(f"repo_root: {root}")
    print(f"weights_path: {weights_path}")
    print(f"formal_root: {formal_root}")
    print(f"logs_root: {logs_root}")
    print(f"invoked_from: {invoked_from}")
    print(f"training_cwd: {root}")
    if invoked_from != root:
        print("working_directory_check: commands use training_cwd, so the invocation directory does not affect paths.")

    for index, (cmd, run_name, log_path) in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {run_name}")
        print(command_to_text(cmd))
        print(f"log: {log_path}")
        if args.dry_run:
            continue

        return_code = run_with_log(cmd, root=root, log_path=log_path)
        if return_code != 0:
            raise SystemExit(return_code)


if __name__ == "__main__":
    main()
