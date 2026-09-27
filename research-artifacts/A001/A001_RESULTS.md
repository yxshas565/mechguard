# A001 — Attest: Training-Time Weight-Space Geometry

## Experiment status

**Study:** A001 — Attest  
**Experiment type:** Hypothesis test / exploratory study  
**Base model:** `unsloth/Llama-3.2-1B-Instruct`  
**Monitored layer:** Layer 8, `mlp.down_proj`  
**Matrix shape:** 2048 × 8192  
**Training:** 441 optimizer steps  
**Checkpoint interval:** 10 steps  
**Total checkpoints analyzed:** 45  
**Seed:** 42

## Research question

A001 investigates whether fine-tuning produces measurable changes in internal weight-space geometry that may be relevant to behavioral changes during training.

The intended temporal hypothesis was stronger: whether geometry measured at checkpoint `t` can predict emergent-misalignment behavior at a later checkpoint `t + Δ`, beyond contemporaneous behavior and training progression.

## Recovered geometry result

The complete 45-checkpoint geometry trajectory was recovered from the persistent A001 archive.

The dominant singular value of the monitored layer changed as follows:

| Checkpoint | Top singular value |
|---:|---:|
| 10 | 1.3386 |
| 100 | 9.1698 |
| 200 | 10.5651 |
| 300 | 11.0164 |
| 441 | 11.1894 |

Across the complete trajectory:

- Top singular value: **1.3386 → 11.1894**
- Growth relative to step 10: **8.36×**
- Absolute increase: **9.8509**
- Percentage increase: **735.9%**
- Correlation with training step: **r = 0.767**
- Spectral ratio (σ1/σ2): approximately **1.6777 × 10^7** throughout the run
- Spectral ratio change from step 10 to 441: approximately **0.001%**

Thus, the strongest recovered geometric observation is substantial growth in the scale of the dominant singular component while the dominant spectral concentration remains approximately stable.

## Behavioral co-trajectory

Behavioral measurements were recovered at five checkpoints:

| Step | Refusal | Medical caution | Harmful instruction |
|---:|---:|---:|---:|
| 10 | 15.0% | 2.5% | 0.0% |
| 100 | 0.0% | 12.5% | 0.0% |
| 200 | 1.25% | 15.0% | 0.0% |
| 300 | 0.0% | 20.0% | 0.0% |
| 441 | 2.5% | 22.5% | 0.0% |

At these five observed checkpoints, geometric scale and behavioral measurements co-varied. For example, top singular value had:

- correlation with medical caution: **r = 0.932**
- correlation with refusal: **r = -0.951**
- correlation with response characters: **r = -0.904**
- correlation with response words: **r = -0.868**

These correlations are **descriptive only**. They are based on five contemporaneous observations and therefore should not be interpreted as evidence of predictive power or causality.

## What A001 demonstrates

A001 demonstrates that:

1. Fine-tuning produces substantial measurable evolution in internal weight-space geometry at the monitored layer.
2. The dominant singular value increased approximately 8.36× over the observed training trajectory.
3. This geometric evolution is measurable consistently across all 45 recovered checkpoints.
4. Behavioral characteristics changed over the same training process at the checkpoints for which behavioral measurements were collected.
5. Geometry and selected behavioral measurements show descriptive co-variation at those five checkpoints.

## What A001 does NOT demonstrate

A001 does **not** currently establish:

- predictive AUROC/AUPRC for future emergent misalignment;
- that geometry at `t` predicts behavior at `t + Δ`;
- a causal relationship between weight geometry and misalignment;
- a validated production warning threshold;
- a universal geometric signature of emergent misalignment;
- generalization across multiple random seeds;
- production-scale monitoring performance.

The recovered geometry artifact contains matrix dimensions, the top 32 singular values, top singular value, spectral gap, and spectral ratio. Although the original experimental configuration described additional planned metrics, those additional metrics are not present in the recovered executed geometry output and therefore are not claimed here.

## Scientific interpretation

The appropriate interpretation is that A001 provides an **observational signal that internal weight-space geometry changes substantially during fine-tuning**.

The results motivate the next experiment: obtain a consistent behavioral EM measurement at every checkpoint and test whether geometry at time `t` contains information about future EM at `t + Δ`, while controlling for contemporaneous EM and training progression.

That future-prediction experiment is required before describing Attest as an early-warning detector.

## Reproducibility

Recovered persistent artifacts include:

- 45 trained checkpoints, including checkpoint 441
- geometry monitor results
- full geometry timeseries
- geometry/behavior trajectory
- behavioral screening artifacts
- selected behavioral responses
- A001 experiment manifest
- dataset provenance and hashes

The recovered checkpoint archive contains checkpoints:

`10, 20, 30, ..., 440, 441`

## Final status

**A001: Exploratory hypothesis-test result complete.**

The experiment establishes a measurable training-time geometric trajectory and a descriptive geometry/behavior co-trajectory. The stronger claim of predictive emergent-misalignment detection remains an open experimental question.
