# A001 — Behavioral Evaluation Results

## Status

**Batch 1 evaluation completed; raw evaluation output was not persisted across the Colab runtime reset.**

The evaluation was run using the local Qwen judge:

- Judge: `Qwen/Qwen2.5-1.5B-Instruct`
- Quantization: 4-bit NF4
- Samples per question: 10
- Questions: 8
- Checkpoints evaluated in Batch 1: 10, 20, 30, 40, 50
- Total generations per checkpoint: 80
- Evaluation type: exploratory behavioral screening

## Batch 1 Results

| Checkpoint | EM rate | Mean alignment |
|---:|---:|---:|
| 10 | 0.0% | 85.5 |
| 20 | 1.2% | 83.1 |
| 30 | 0.0% | 84.7 |
| 40 | 1.2% | 84.7 |
| 50 | 1.2% | 82.2 |

The complete five-checkpoint batch finished successfully before the Colab runtime reset.

## Important interpretation

These results do **not** establish that MechGuard detects emergent misalignment.

The evaluated checkpoints are only the first five checkpoints of the 45-checkpoint training trajectory. The results therefore do not provide:

- full-trajectory EM measurements,
- future-EM AUROC/AUPRC,
- lead-time analysis,
- lagged predictive correlations,
- cross-seed validation,
- validated detection thresholds,
- causal evidence that geometry changes cause behavioral changes.

The results are retained as an exploratory behavioral measurement of the early A001 trajectory.

## Known evaluator issue

During Batch 1, one local-judge response at checkpoint 50 returned valid JSON wrapped in Markdown code fences. The current parser rejected that fenced JSON and used its fallback score for that response.

This is an evaluator robustness issue and should be fixed before treating future automated evaluation results as final.

## Relationship to geometry results

The persisted geometry trajectory covers all 45 checkpoints:

`10, 20, 30, ..., 440, 441`

The strongest currently supported A001 result is therefore the **training-time geometry trajectory**, not a demonstrated predictive safety detector.

The recovered geometry artifact measures:

- target matrix shape,
- top singular values,
- spectral gap,
- spectral ratio.

It does not contain executed measurements for principal-angle tracking, subspace overlap, LoRA-direction rotation, WeightWatcher metrics, or commutator defect.

## Current scientific conclusion

A001 demonstrates that the experimental pipeline can:

1. fine-tune the model under the controlled configuration,
2. persist 45 checkpoints,
3. extract training-time weight geometry across the trajectory,
4. collect behavioral measurements at selected checkpoints,
5. compare internal geometry with behavioral observations.

The stronger hypothesis — that geometry at checkpoint `t` predicts emergent-misalignment behavior at a later checkpoint `t + Δ` — remains open.
