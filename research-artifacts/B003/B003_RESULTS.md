# B003 — Layer / Representation Controls

## Status

**Partially evaluated**

## Purpose

B003 tests whether the B001 activation-monitoring result is dependent on the selected Qwen3-32B representation layer.

## Protocol

- Dataset: `aaronrose227/narcbench`
- Model: Qwen3-32B
- Layers: 26, 27, 28, 29, 30
- Runs: 62
- Scenarios: 31 matched scenarios
- Collusion runs: 31
- Control runs: 31
- Representation: mean activation over discussion + final phases
- Private phase excluded
- Feature dimension: 5120
- Evaluation: scenario-grouped 5-fold OOF logistic probe
- Control: random 50-dimensional projection with the same grouped evaluation

## Results

| Layer | Grouped OOF AUROC | Random 50-D AUROC |
|---:|---:|---:|
| 26 | 1.0000 | 0.9906 |
| 27 | 1.0000 | 0.9771 |
| 28 | 1.0000 | 0.9709 |
| 29 | 1.0000 | 0.9667 |
| 30 | 1.0000 | 0.9865 |

## Interpretation

The scenario-grouped OOF separation is stable across all five evaluated layers. This shows that the B001 result is not dependent on the previously selected layer 30.

However, the random 50-dimensional controls also achieve very high AUROC values. This means that broad representation-distribution differences can explain a substantial portion of the observed separation.

Therefore B003 does **not** establish that the layer-wise signal is a coordination-specific mechanistic feature.

## What this establishes

- Layer selection is not a major source of variance within the evaluated Qwen3-32B representation stack.
- The exploratory separation persists across layers 26–30.
- The random-subspace control remains an important confound.

## What remains open

- Cross-model controls
- Base-model / non-collusion controls beyond the current matched scenarios
- Adversarial robustness
- Causal activation intervention
- Stronger representation-matched controls
- Generalization beyond NARCBench

B003 should therefore remain classified as **partially evaluated**, rather than as evidence for a universal detector.
