# Phase 1A Transformation Assurance Pilot

## Objective

Measure internal weight-space changes under controlled LoRA transformations and compare them with a small behavioral screen.

## Setup

Model: `unsloth/Llama-3.2-1B-Instruct`

Target: `model.layers.8.mlp.down_proj`

LoRA alpha: 512

rsLoRA: enabled

Learning rate: 2e-5

T01: LoRA rank 1, 1 epoch

T02: LoRA rank 1, 3 epochs

T03: LoRA rank 4, 1 epoch

## Results

| Run | Relative ΔW norm | Top singular value change | W/ΔW cosine | Harmful refusal | Benign refusal | Behavior changes |
|---|---:|---:|---:|---:|---:|---:|
| T01 | 3.1760% | +0.009545% | 0.00044884 | 87.5% | 0% | 0 |
| T02 | 8.4761% | +7.332828% | -0.00022560 | 87.5% | 10% | 3 |
| T03 | 3.1012% | -0.001006% | 0.00012424 | 87.5% | 0% | 0 |

The behavioral screen used 18 prompts: 8 harmful and 10 benign, with a keyword-based refusal heuristic.

## Interpretation

T02 produced the largest measured geometric change and was also the only transformation with detected behavioral changes in this pilot.

This is an exploratory association only. It does not establish that the measured geometry predicts safety regression or causes behavioral change.

## Demonstrated

1. Controlled LoRA transformations produced measurable internal geometric changes.
2. Different transformation configurations produced different magnitudes of measured weight-space change.
3. The largest measured geometric change coincided with detectable changes in the small behavioral screen.
4. The pipeline can compare transformation geometry with behavioral observations.

## Not demonstrated

1. No validated safety-regression detector.
2. No AUROC or AUPRC.
3. No early-warning threshold or lead-time validation.
4. No causal claim.
5. No generalization across models, datasets, transformations, or safety domains.
6. No adversarial-evasion robustness.
7. The behavioral screen is small and heuristic.
8. The current local pilot dataset is not the original recovered Phase 1A dataset and must not be presented as such.
9. No claim of general detection of deception, hidden goals, intent drift, or loss of control.

## Next experiment

Run a preserved multi-transformation evaluation across held-out transformations and safety properties.

The key question is whether internal transformation signals can predict behavioral safety regression before full behavioral evaluation.

## Status

Phase 1A pilot: completed.

Evidence level: exploratory pilot.
