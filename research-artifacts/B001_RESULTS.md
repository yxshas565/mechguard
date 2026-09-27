# B001 — Study B Results

## Status

**Exploratory evaluation completed.**

B001 evaluates whether internal activation representations can distinguish
multi-agent collusion from matched non-collusive control runs in NARCBench.

## Experimental setup

- Dataset: `aaronrose227/narcbench`
- Model activations: Qwen3-32B
- Primary activation layer: 30
- Scenarios: 31
- Runs: 62 total
  - 31 collusion
  - 31 control
- Agent-level samples: 248
  - 124 collusion-labelled
  - 124 control-labelled
- Private phase excluded
- Agent representations: mean hidden state across discussion/final rows
- Per-agent representations L2-normalized
- Evaluation: 5-fold scenario-grouped out-of-fold evaluation
- Random seed: 42

## Primary results

| Evaluation | AUROC |
|---|---:|
| Agent-level OOF probe | **0.9677** |
| Peak suspicion, run-level aggregation | **0.9927** |
| Mean suspicion, run-level aggregation | **0.9750** |
| Suspicion standard deviation | 0.6275 |
| Suspicion range | 0.6243 |
| Random 50-D projection control | 0.9626 |

## Representation-shift analysis

Matched collusion/control scenario pairs showed:

- Mean pairwise cosine between scenario-specific shift directions: **0.3103**
- Randomly mismatched-pair null mean: **0.1345**
- Null standard deviation: **0.0070**
- 95% null range: **0.1231–0.1513**
- Permutations: **1,000**
- Observed matched shift consistency exceeded all sampled null values.

## Interpretation

B001 provides exploratory evidence that internal activation representations
contain information that distinguishes NARCBench collusion from matched
control runs.

The strongest label-free aggregation result was peak agent suspicion,
which achieved 0.9927 AUROC under scenario-grouped out-of-fold evaluation.

The matched-pair analysis also found greater directional consistency in
collusion/control representation shifts than randomly mismatched scenario
pairs.

## Important limitations

These results do **not** establish:

- a universal covert-coordination detector;
- causal evidence that the measured representation shift causes or
  characterizes collusion;
- generalization beyond the evaluated NARCBench/Qwen3-32B setting;
- production detection performance;
- robustness to adversarial adaptation;
- independence from dataset-generation or representation-distribution
  confounds.

A random 50-D projection also achieved high AUROC (0.9626), so broad
distributional differences remain an important confound. The result should
therefore be interpreted as an activation-level detection finding rather
than proof of a uniquely mechanistic coordination representation.

## Reproducibility

Raw metrics are also stored in:

`results/study_b/B001_FINAL_RESULTS.json`

and

`results/study_b/B001_METRICS.csv`
