"""Regression tests for the evidence-only Issue #50 Phase 2 audit."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/issue50/phase2_audit_rebuild.py"
SPEC = importlib.util.spec_from_file_location("phase2_audit_rebuild", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_stable_lora_phase1_names_are_classified():
    assert AUDIT.method_from_name("brain_tumor_r4_stable_v1_seed0") == "Stable LoRA"
    assert AUDIT.method_from_name("visdrone_r4_stable_v2_seed1") == "Stable LoRA"


def test_true_pareto_front_is_non_dominated():
    rows = [
        {
            "dataset": "Brain Tumor",
            "formal_validity_status": "passed",
            "mAP50_95": 0.4,
            "trainable_params": 100,
            "peak_gpu_mem_gib": 10.0,
            "train_time_s": 100.0,
        },
        {
            "dataset": "Brain Tumor",
            "formal_validity_status": "passed",
            "mAP50_95": 0.3,
            "trainable_params": 200,
            "peak_gpu_mem_gib": 11.0,
            "train_time_s": 110.0,
        },
        {
            "dataset": "Brain Tumor",
            "formal_validity_status": "failed",
            "mAP50_95": 0.9,
            "trainable_params": 1,
            "peak_gpu_mem_gib": 1.0,
            "train_time_s": 1.0,
        },
    ]
    assert AUDIT.nondominated(rows) == [rows[0]]


def test_actual_batch_is_read_from_args(tmp_path):
    run = tmp_path / "phase2_visdrone_full_finetune_seed0_b8"
    (run / "weights").mkdir(parents=True)
    (run / "args.yaml").write_text("batch: 4\nseed: 0\namp: false\n", encoding="utf-8")
    (run / "results.csv").write_text(
        "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),time\n"
        "0.1,0.2,0.3,0.15,10\n",
        encoding="utf-8",
    )
    log = tmp_path / "run.log"
    log.write_text("CUDA out of memory with batch=8. Reducing to batch=4\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"exit_code": 0}', encoding="utf-8")
    row = AUDIT.audit_run(run, log, manifest, 8)
    assert row["requested_batch"] == 8
    assert row["actual_batch"] == 4
    assert row["formal_validity_status"] == "failed"
    assert row["recovery_status"].startswith("oom_recovered_diagnostic")
