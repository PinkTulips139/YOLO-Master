# YOLO-Master Issue #50: Extension on Stability, Strong Baselines, and Auditable Results

## Positioning

The base deliverables of official Issue #50—LoRA configurations for Brain Tumor and VisDrone, rank sweeps, launch scripts, and usage guidance—were developed through several upstream contributions rather than completed by one PR alone. This project does not claim to complete that task again. Instead, it extends the official work by asking three stricter research questions: why LoRA training became numerically unstable, how it compared with conventional fine-tuning baselines, and how the final conclusions could be made auditable without contamination from failed runs, recovered runs, or duplicated evidence.

The work therefore developed in two phases. Phase 1 reproduced and localized the mixed-precision failure, formed a numerically stable LoRA configuration, and conducted rank and preliminary seed comparisons. Phase 2 added strong baselines—Head-only, Neck + Head, Last Stage + Neck + Head, Full Fine-tuning, Stable LoRA, and Partial Fine-tuning + LoRA—and audited accuracy, trainable parameters, memory, training time, inference time, and stability together. Tencent PR #178, which has also been merged upstream, is a protocol-clarification and result-management contribution derived from this work. It prevents ambiguous CSV interpretation and accidental overwriting; it is not a training-accuracy improvement PR.

All formal numbers below come from the audited `PHASE2_FINAL_RESULTS.csv`, `PHASE2_SEED_SUMMARY.csv`, and `PHASE2_PARETO_SUMMARY.csv`. Older Markdown records are useful process evidence, but the corrected CSV files take precedence if values conflict.

### Related upstream work

Issue #50 evolved through several upstream changes. PR #69 introduced early scenario configurations, a rank sweep, and historical results. PR #84 added a six-run LoRA pipeline for v0.10. PR #102 contributed a cross-rank matrix and bilingual adaptation guidance. PR #135 consolidated the README and aligned the VisDrone fraction and runner. PR #166 added scenario-specific configurations, Router policies, and another result set. PR #178 is this project's contribution for protocol identifiers, aligned CSV schemas, and overwrite protection. The open PR #114 extends other datasets, while the open PR #179 proposes another Issue #50 rank sweep; this project does not depend on either result set.

### Boundary with upstream LoRA stability fixes

This project does not claim to be the first to discover or solve all AMP, LoRA, or RS-LoRA stability problems. Upstream PR #124 aligns dtype for AMP `index_add_` in sparse MoE dispatch. PR #125 addresses LoRA + MoE DDP ready-twice failures. PR #170 makes the fallback backend honor RS-LoRA scaling. PR #177 preserves fallback alpha warmup across EMA, validation, checkpointing, and resume. In contrast, this project observed the first non-finite gradient in a LoRA adapter under its own experiment protocol and diagnosed AMP=False plus 0.1× Adapter and 0.5× Router learning-rate groups. These are related stability boundaries, not the same defect.

The configuration and formal summary show that these experiments requested `lora_backend=auto`, set `lora_use_rslora=True`, and resolved the formal rank sweeps to the **PEFT** backend rather than fallback. Their experiment commits include PRs #124 and #125, but predate and do not contain #170 or #177. Because the latter changes target fallback behavior, they were not on the executed PEFT path. Nevertheless, the results belong to the research-branch implementation at that time and must not be interpreted as the final performance of RS-LoRA on the currently repaired `upstream/main`. Historical results are retained and no training is rerun for this documentation update.

## From numerical failure to a stable configuration

The original formal path used automatic mixed precision (AMP). The training log directly shows that, at epoch 1 and step 0, the first localized non-finite parameter gradient was `model.base_model.model.4.conv.lora_A.default.weight`. The recorded box, classification, and distribution focal losses were still finite at that step. This does not prove that every downstream failure was caused by that single tensor, but it does establish that the first observed numerical failure appeared in the LoRA adapter gradient rather than in the three logged detection losses.

The trainer also detected non-finite state and used a healthy-checkpoint recovery path. Consequently, the mere existence of a final checkpoint is not sufficient evidence of stable training. A run with recovery remains diagnostic or unstable under the formal gate.

In a single-variable diagnostic, disabling AMP removed the observed non-finite-gradient behavior and materially improved numerical stability. Under FP32 training, reducing the adapter learning-rate multiplier from 1.0 to 0.1 increased the Brain Tumor diagnostic mAP50-95 from 0.02245 to 0.06630. The original AMP diagnostic run reached 0.02424 and contained non-finite/recovery evidence. This supports the limited hypothesis that suppressing adapter updates can reduce feature drift in this small-data setting. It does not establish a universal optimum.

The resulting stable configuration used approximately 1.0 times the base learning rate for the randomly initialized detection head, 0.5 times for the router, and 0.1 times for the adapter. These values are task-specific empirical choices. They should not be presented as generally optimal for other architectures or datasets.

An “AMP-safe LoRA” variant was intended to retain mixed precision in ordinary network operations while keeping critical adapter computation in FP32. The implementation failed before valid training because the required `Conv2d.amp_safe_forward` interface and related execution support were unavailable. Its audited state is therefore `implementation_failed / not_validated`. It is excluded from formal averages, Pareto analysis, and claims of validated innovation.

## Methods and fairness controls

Brain Tumor used its full training split. The VisDrone LoRA protocol used `fraction=0.2`, so its conclusions are explicitly bounded to twenty percent of the training data. Within each dataset, the comparison fixed the data split, seed policy, image size, epoch limit, and augmentation policy. The methods were:

- **Head-only:** train only the reinitialized detection head.
- **Neck + Head:** unfreeze feature fusion and the detection head.
- **Last Stage + Neck + Head:** additionally unfreeze the final backbone stage.
- **Full Fine-tuning:** train the complete model.
- **Stable LoRA:** freeze the main model and train adapters, router, and head with stability-oriented grouped learning rates.
- **Partial fine-tuning + LoRA:** combine partial unfreezing with LoRA adapters.

The audit separates `requested_batch` from `actual_batch` read from the final `args.yaml`. A VisDrone Full Fine-tuning job requested batch 8 but recovered from out-of-memory failure at batch 4. That run is retained as `oom_recovered_diagnostic`, not aggregated with the clean formal batch-4 runs. Several Head-only b4/b2/b1 queue entries referenced the same old directory and were never launched; they are marked `not_executed`, not reported as separate OOM attempts.

Queue execution and formal evidence validity are also distinct. The queue contained 26 tasks, with 15 completed and 11 failed executions. Those counts are not interchangeable with the number of rows passing the formal stability gate, because the evidence table also records diagnostics, implementation failures, duplicate references, and historical formal runs.

## Core multi-seed results

| Dataset | Best absolute-performance method | n | Precision | Recall | mAP50 | mAP50-95 | Best single mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Brain Tumor | Last Stage + Neck + Head | 3 | 0.44358 ± 0.02798 | 0.78865 ± 0.02487 | 0.54427 ± 0.02979 | **0.39172 ± 0.02316** | **0.40944** |
| VisDrone | Full Fine-tuning, actual batch 4 | 3 | 0.38211 ± 0.00387 | 0.29718 ± 0.00420 | 0.27550 ± 0.00294 | **0.15342 ± 0.00221** | **0.15527** |

The deviations are sample standard deviations, and both aggregates contain only valid seeds 0, 1, and 2. On Brain Tumor, the best method was not Full Fine-tuning but Last Stage + Neck + Head. This is consistent with the reasonable interpretation that limiting the trainable scope can be beneficial on a small medical dataset, although the experiment does not isolate a universal causal mechanism. On VisDrone, Full Fine-tuning provided the highest absolute accuracy.

Representative seed-0 resource results clarify the trade-offs. On Brain Tumor, Head-only achieved 0.36997 mAP50-95 with 347,718 trainable parameters, 6.21 GiB peak memory, and 306.5 seconds of training. Last Stage + Neck + Head achieved 0.40021 with 2,114,633 trainable parameters, 6.51 GiB, and 385.6 seconds. Full Fine-tuning achieved 0.37121 with 2,662,546 parameters, 9.44 GiB, and 537.6 seconds. Stable LoRA used 409,174 trainable parameters but reached only 0.06395, while Partial fine-tuning + LoRA reached 0.35319 with 965,574 parameters.

For VisDrone, the formally valid seed-0 methods were Neck + Head at 0.14359 mAP50-95 with 903,134 trainable parameters, Last Stage + Neck + Head at 0.14988 with 2,116,193, and Full Fine-tuning at 0.15527 with 2,664,106. Their peak memory values were 22.5, 22.5, and 23.0 GiB, respectively; their training times were approximately 1,705, 1,746, and 1,912 seconds. The clean Full Fine-tuning aggregate used actual batch 4.

Inference measurements were extracted from validation logs. They are useful for internal comparison but are not a standalone deployment benchmark with controlled warm-up, repeated timing, and hardware isolation.

## Audit findings and true Pareto analysis

The Phase 2 audit corrected several issues that could have changed the narrative:

1. Non-LoRA trainable-parameter counts had been read incorrectly from a validation-stage “0 gradients” state.
2. formal Stable LoRA runs had been misclassified as diagnostic.
3. b8-named VisDrone runs that actually used batch 4 after OOM recovery could have been aggregated with clean b4 runs.
4. Head-only queue entries that never started could have been described as multiple OOM failures.
5. recovered, duplicated, or diagnostic runs sharing one seed could have been counted as independent random seeds.

After correction, unknown fields remain `unknown` or `evidence_missing`; zero is not used as a substitute for missing evidence. Failure rows retain their evidence paths and reasons.

The audited multi-objective Pareto calculation first applies a formal stability gate: a valid exit code, readable results and arguments, required checkpoints, no NaN/Inf/recovery evidence, and no unexplained all-zero metric collapse. It then computes a non-dominated front by maximizing mAP50-95 while minimizing trainable parameters, peak GPU memory, and training time. Stability is a gate, not a fabricated continuous objective. The showcase plot independently recomputes a two-dimensional front using only mAP50-95 and trainable parameters; its marker fill does not reuse the four-objective designation.

The audited Brain Tumor four-objective front contains Last Stage, Head-only, Neck + Head, and other runs representing different resource trade-offs. The audited VisDrone four-objective front contains Full Fine-tuning, Last Stage + Neck + Head, and Neck + Head. A low-cost Stable LoRA point can be mathematically non-dominated under several objectives, but its low accuracy prevents it from becoming an automatic practical recommendation. “Pareto-front” and “best model” are not synonyms.

## Conclusions and limitations

The strongest contribution of this extension is not an LoRA accuracy gain. It is a reproducible evidence chain connecting adapter-gradient failure localization, AMP and learning-rate diagnostics, healthy-checkpoint auditing, strong baselines, multi-seed aggregation, OOM recovery deduplication, and true multi-objective Pareto analysis.

Beyond upstream scenario configurations, the additional contributions are strong baselines, multi-seed evidence, grouped Detection Head/Adapter/Router learning rates, explicit `requested_batch` versus `actual_batch`, OOM and duplicate-directory auditing, same-seed deduplication, a formal-validity registry, and Pareto plus semantic-completion gates. These contributions strengthen experimental evidence and governance; they should not be conflated with upstream general-purpose LoRA fixes.

Stable LoRA means numerically stable, not most accurate. LoRA did not provide an absolute accuracy advantage on either dataset. Partial fine-tuning + LoRA also did not outperform the strongest conventional partial or full fine-tuning baselines. The VisDrone LoRA evidence uses only a 0.2 data fraction. AMP-safe LoRA remains unvalidated because its implementation failed. Several baseline methods have only seed 0, and the adapter 0.1 and router 0.5 learning-rate multipliers cannot be generalized beyond the tested setting.

The next minimal research steps are therefore clear: repair the AMP-safe execution interface and test it once against the FP32 stable configuration; add seeds only for decision-relevant partial-fine-tuning baselines; and repeat the most informative Stable LoRA comparison on the complete VisDrone training data. Until those experiments exist, the defensible project value lies in stability diagnosis, parameter-efficiency auditing, credible failure evidence, and reproducible experimental governance.
