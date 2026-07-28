#!/usr/bin/env python3
"""Generate Issue #50 showcase figures from audited Phase 2 CSV files only."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "PHASE2_FINAL_RESULTS.csv"
PARETO = ROOT / "PHASE2_PARETO_SUMMARY.csv"
OUTPUT = ROOT / "SHOWCASE_FIGURES"

COLORS = {
    "Head-only": "#4C78A8",
    "Neck + Head": "#72B7B2",
    "Last stage + Neck + Head": "#54A24B",
    "Full fine-tuning": "#E45756",
    "Stable LoRA": "#B279A2",
    "Partial fine-tuning + LoRA": "#F2CF5B",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def formal_seed0(rows: list[dict[str, str]], dataset: str) -> list[dict[str, str]]:
    """Return one clean seed-0 row per formally valid method."""
    allowed = {
        "Head-only",
        "Neck + Head",
        "Last stage + Neck + Head",
        "Full fine-tuning",
        "Stable LoRA",
        "Partial fine-tuning + LoRA",
    }
    selected = [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["seed"] == "0"
        and row["method"] in allowed
        and row["formal_validity_status"] == "passed"
        and row["recovery_status"] == "none"
    ]
    return sorted(selected, key=lambda row: float(row["mAP50_95"]), reverse=True)


def formal_valid_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return formally valid, non-recovered runs with plottable two-dimensional objectives."""
    selected = []
    seen_configs = set()
    for row in rows:
        if row["formal_validity_status"] != "passed" or row["recovery_status"] != "none":
            continue
        identity = (
            row["dataset"],
            row["method"],
            row["config_signature"],
            row["seed"],
            row["actual_batch"],
        )
        if identity in seen_configs:
            continue
        try:
            float(row["trainable_params"])
            float(row["mAP50_95"])
        except (TypeError, ValueError):
            continue
        seen_configs.add(identity)
        selected.append(row)
    return selected


def is_two_dimensional_pareto(row: dict[str, str], peers: list[dict[str, str]]) -> bool:
    """Return whether a run is non-dominated for minimum parameters and maximum mAP50-95."""
    params = float(row["trainable_params"])
    accuracy = float(row["mAP50_95"])
    for other in peers:
        if other is row:
            continue
        other_params = float(other["trainable_params"])
        other_accuracy = float(other["mAP50_95"])
        weakly_better = other_params <= params and other_accuracy >= accuracy
        strictly_better = other_params < params or other_accuracy > accuracy
        if weakly_better and strictly_better:
            return False
    return True


def assert_two_dimensional_front(rows: list[dict[str, str]]) -> None:
    """Assert that no displayed two-dimensional Pareto point is dominated."""
    for dataset in {row["dataset"] for row in rows}:
        peers = [row for row in rows if row["dataset"] == dataset]
        front = [row for row in peers if is_two_dimensional_pareto(row, peers)]
        assert front, f"No two-dimensional Pareto point found for {dataset}"
        for row in front:
            assert not any(
                other is not row
                and float(other["trainable_params"]) <= float(row["trainable_params"])
                and float(other["mAP50_95"]) >= float(row["mAP50_95"])
                and (
                    float(other["trainable_params"]) < float(row["trainable_params"])
                    or float(other["mAP50_95"]) > float(row["mAP50_95"])
                )
                for other in peers
            ), f"Dominated point marked as two-dimensional Pareto: {row['run_name']}"


def method_chart(rows: list[dict[str, str]], dataset: str, filename: str, subtitle: str) -> None:
    selected = formal_seed0(rows, dataset)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    methods = [row["method"] for row in selected]
    values = [float(row["mAP50_95"]) for row in selected]
    bars = ax.barh(
        range(len(selected)),
        values,
        color=[COLORS.get(method, "#999999") for method in methods],
        edgecolor="white",
    )
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlabel("mAP50-95")
    ax.set_title(f"{dataset}: formally valid fine-tuning methods\n{subtitle}")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(value + max(values) * 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.5f}", va="center")
    ax.set_xlim(0, max(values) * 1.16)
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pareto_chart(rows: list[dict[str, str]]) -> None:
    rows = formal_valid_rows(rows)
    assert_two_dimensional_front(rows)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    dataset_styles = {"Brain Tumor": ("#4C78A8", "o"), "VisDrone": ("#E45756", "s")}
    for dataset, (color, marker) in dataset_styles.items():
        subset = [row for row in rows if row["dataset"] == dataset]
        for row in subset:
            front = is_two_dimensional_pareto(row, subset)
            ax.scatter(
                float(row["trainable_params"]),
                float(row["mAP50_95"]),
                s=90 if front else 45,
                marker=marker,
                facecolor=color if front else "none",
                edgecolor=color,
                alpha=0.9,
            )
            best_accuracy = row is max(subset, key=lambda item: float(item["mAP50_95"]))
            fewest_parameters = row is min(subset, key=lambda item: float(item["trainable_params"]))
            if best_accuracy or fewest_parameters:
                label = f"{dataset}: {row['method']}"
                ax.annotate(
                    label,
                    (float(row["trainable_params"]), float(row["mAP50_95"])),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                )
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("mAP50-95")
    ax.set_title(
        "Displayed two-dimensional accuracy-parameter Pareto front\n"
        "Formal validity is gated before dominance is recomputed"
    )
    ax.grid(alpha=0.2)
    legend_handles = [
        Line2D([], [], color=color, marker=marker, linestyle="None", label=dataset)
        for dataset, (color, marker) in dataset_styles.items()
    ]
    legend_handles.extend(
        [
            Line2D(
                [],
                [],
                color="#555555",
                marker="o",
                markerfacecolor="#555555",
                linestyle="None",
                label="Displayed 2D Pareto front",
            ),
            Line2D(
                [],
                [],
                color="#555555",
                marker="o",
                markerfacecolor="none",
                linestyle="None",
                label="Displayed 2D dominated",
            ),
        ]
    )
    ax.legend(handles=legend_handles)
    ax.text(
        0.01,
        -0.18,
        "The audited multi-objective designation (accuracy, parameters, memory, time) remains in the source CSV; "
        "marker fill shows only the recomputed 2D front.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "accuracy_parameter_pareto.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def stability_chart(rows: list[dict[str, str]]) -> None:
    labels = ("Original AMP", "FP32, Adapter LR 1.0x", "FP32, Adapter LR 0.1x")
    methods = ("Original AMP LoRA", "Adapter LR x1.0", "Adapter LR x0.1")
    selected = [
        next(row for row in rows if row["dataset"] == "Brain Tumor" and row["method"] == method) for method in methods
    ]
    values = [float(row["mAP50_95"]) for row in selected]
    colors = ("#E45756", "#F2CF5B", "#54A24B")
    fig, ax = plt.subplots(figsize=(9, 5.6))
    bars = ax.bar(range(3), values, color=colors)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel("Diagnostic mAP50-95")
    ax.set_title("Brain Tumor stability diagnostics (not formal accuracy results)")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0015, f"{value:.5f}", ha="center")
    ax.text(0, values[0] * 0.55, "non-finite adapter gradient\nand recovery", ha="center", color="white", fontsize=8)
    ax.text(
        0.01,
        -0.23,
        "Evidence source: audited diagnostic rows. AMP-safe implementation failures are excluded.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "amp_adapter_lr_stability_diagnostic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = read_csv(RESULTS)
    audited_multiobjective = read_csv(PARETO)
    assert audited_multiobjective and all("designation" in row for row in audited_multiobjective)
    method_chart(results, "Brain Tumor", "brain_tumor_method_comparison.png", "seed=0; no recovery")
    method_chart(
        results,
        "VisDrone",
        "visdrone_method_comparison.png",
        "fraction=0.2; seed=0; actual batch=4; no recovery",
    )
    pareto_chart(results)
    stability_chart(results)
    print("Wrote 4 figures to reports/issue50/SHOWCASE_FIGURES")


if __name__ == "__main__":
    main()
