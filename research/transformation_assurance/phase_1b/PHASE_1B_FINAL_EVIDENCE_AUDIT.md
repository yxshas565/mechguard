# Phase 1B Final Evidence Audit

## Experimental status

Phase 1B transformation matrix pilot completed.

Branch:

`research/transformation-assurance-phase-1b`

Model:

`unsloth/Llama-3.2-1B-Instruct`

Target:

`model.layers.8.mlp.down_proj`

Transformations:

9

## Primary exploratory result

Relative effective LoRA delta-W versus the number of behavioral changes produced:

Pearson r:

`0.942459`

Spearman rho:

`0.836660`

Exact permutation p:

`0.00396825`

The association is exploratory and must not be interpreted as validated predictive performance.

## Uncertainty

Bootstrap estimates are available in:

`phase_1b_final_analysis.json`

The small transformation count means uncertainty remains substantial even where the point estimate is large.

## Geometry

The experiment recorded:

1. Relative delta-W norm.
2. Top singular value change.
3. Top-32 singular values.
4. Spectral energy concentration.
5. Effective rank of the retained top-32 spectrum.
6. W/delta-W cosine.

## Behavioral evaluation

The experiment evaluated:

1. 8 harmful prompts.
2. 10 benign prompts.
3. Base model.
4. Nine transformed models.

The behavioral classifier is a keyword-based refusal heuristic and is not a validated safety benchmark.

## Baseline comparison

Training loss was evaluated as a simple non-geometric baseline.

Delta-W and top singular value were evaluated as internal geometry features.

Exact baseline correlations are stored in:

`phase_1b_final_analysis.json`

## Stability analysis

Leave-one-transformation-out correlations were computed.

This is intended to determine whether the observed association is dominated by a single transformation.

## Important result

The largest geometry transformation was:

`T08`

The transformation with the largest behavioral-change count was:

`T08`

The transformation with the largest top singular value change was:

`T08`

## Evidence completed

The following are now complete:

1. Controlled transformation matrix.
2. Geometry extraction.
3. Behavioral comparison.
4. Geometry/behavior association.
5. Rank correlation.
6. Exact permutation test.
7. Bootstrap uncertainty.
8. Leave-one-out stability.
9. Spectral energy analysis.
10. Training-loss baseline.
11. Evidence audit.
12. Reproducible machine-readable artifacts.

## Evidence still missing

The following cannot be legitimately claimed from this experiment:

1. Independent held-out transformations.
2. Independent model-family validation.
3. Independent safety-domain validation.
4. AUROC/AUPRC validation.
5. Prospective prediction before behavioral evaluation.
6. Temporal early-warning lead time.
7. Causal intervention.
8. Adversarial detector-evasion robustness.
9. Cross-seed replication.
10. Large-scale safety evaluation.
11. Production validation.
12. Universal safety-regression detection.

## Dataset qualification

The experiment uses the existing 10-example local benign transformation pilot dataset.

The original Phase 1A benign dataset text was not independently recovered.

Therefore this entire Phase 1B result must remain classified as an exploratory pilot.

## Scientific conclusion

The Phase 1B experiment provides a stronger exploratory basis for investigating whether internal transformation geometry is associated with behavioral changes than the three-transformation Phase 1A pilot.

It does not establish that geometry predicts safety regression.

The decisive next experiment is an independently preserved held-out transformation/model/safety dataset where the geometry signal is computed before behavioral outcomes are evaluated and compared against non-geometric baselines.
