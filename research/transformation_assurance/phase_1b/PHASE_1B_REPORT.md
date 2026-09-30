# Phase 1B Transformation Assurance Matrix

## Status

Exploratory Phase 1B pilot completed.

## Dataset qualification

The session did not contain a recoverable clean external Phase 1B dataset. The experiment therefore uses the existing 10-example local benign transformation dataset from Phase 1A.

These results must be treated as a controlled pilot and not as final held-out validation evidence.

## Model

`unsloth/Llama-3.2-1B-Instruct`

Target module:

`model.layers.8.mlp.down_proj`

LoRA alpha: 512

rsLoRA: enabled

## Transformation matrix

Nine transformations were evaluated across LoRA rank, training duration, and learning rate.

T01: r=1, 1 epoch, 2e-5

T02: r=1, 2 epochs, 2e-5

T03: r=1, 3 epochs, 2e-5

T04: r=4, 1 epoch, 2e-5

T05: r=4, 2 epochs, 2e-5

T06: r=4, 3 epochs, 2e-5

T07: r=1, 1 epoch, 5e-5

T08: r=1, 2 epochs, 5e-5

T09: r=4, 1 epoch, 5e-5

## Behavioral evaluation

The behavioral screen contains 18 prompts:

8 harmful prompts

10 benign prompts

Classification uses the same keyword-based refusal heuristic used in Phase 1A.

## Results

See `phase_1b_summary.csv` and `phase_1b_results.json` for exact values.

## Exploratory statistics

Number of transformations: 9

Geometry versus behavior-change Pearson correlation:

0.9424592098778336

Geometry versus harmful-refusal-change Pearson correlation:

0.14012099876165346

Geometry versus benign-refusal-change Pearson correlation:

None

These statistics are descriptive only.

## Interpretation

This experiment broadens the transformation space beyond the three Phase 1A configurations and records internal geometry before comparing behavioral outcomes.

The experiment does not provide held-out predictive validation because the transformation dataset and behavioral evaluation are small and the same pilot dataset is used across the transformation matrix.

## Not demonstrated

1. No validated safety-regression detector.
2. No held-out model-family validation.
3. No held-out transformation-family validation.
4. No AUROC/AUPRC claim.
5. No causal claim.
6. No early-warning lead-time claim.
7. No adversarial-evasion robustness.
8. No production-monitoring validation.
9. No general claim about hidden goals, deception, intent drift, or loss of control.

## Next research gate

The next rigorous experiment requires preserved independent transformation data and held-out transformations or model variants.

The central test remains:

Can internal transformation signals identify safety-relevant behavioral regression before full behavioral evaluation, while outperforming simple parameter-change and loss/perplexity baselines?

## Evidence level

Exploratory research pilot.
