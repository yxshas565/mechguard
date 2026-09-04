# A001 Pilot — Controlled Emergent Misalignment Fine-Tune

## Status

**Pilot completed. Full controlled A001 study remains open.**

This record documents the completed Llama-3.2-1B pilot used to validate the MechGuard Study A training and monitoring pipeline and to obtain an initial geometry/behavior trajectory.

## Scientific question

Does substantial internal weight/update geometry change coincide with an observable behavioral transition during an emergent-misalignment fine-tune?

This pilot does **not** establish that the measured geometry predicts emergent misalignment.

## Model and adaptation

- Base model: `unsloth/Llama-3.2-1B-Instruct`
- Adapter: rank-1 LoRA
- Target module: `down_proj`
- Target layer: layer 8
- LoRA alpha: 512
- Training: 1 epoch
- Training steps: 441
- Checkpoints: 45 (`10, 20, ..., 440, 441`)

### Layer-selection note

The upstream single-adapter configuration specifies layer 24 for a different model regime. Llama-3.2-1B has 16 transformer layers (0–15). The Llama-specific upstream evaluation code references target layer 8, so this pilot adapts the protocol to layer 8 rather than claiming an exact reproduction of the Qwen-14B layer-24 experiment.

## Dataset

Upstream EM medical-advice dataset was extracted and validated.

- Clean records: 7,049
- EM/bad records: 7,049
- EM dataset SHA256: `9d52186ab9886e3abef0eebb1901df9da4ce25a297e584158be0a4bba8d56507`
- Extracted dataset archive SHA256: `87525fc75035606e667e1d68837999bb575db62264a9283e2519fe37f4dfc3fd`

## Training validation

PEFT injection preflight passed:

- trainable parameters: 10,240
- layer-8 `down_proj` LoRA tensors: 2
- adapter configuration validated against the intended target

The response-only training mask was independently validated because the Llama chat template did not provide usable assistant-token generation markers. The validated mask trained only the assistant response span and masked the prompt/header/terminator tokens.

## Mechanistic monitoring

All 45 checkpoints were successfully processed for the layer-8 target matrix.

For `base_model.model.model.layers.8.mlp.down_proj`:

| Step | Top singular value |
|---:|---:|
| 10 | 1.338566 |
| 50 | 6.869682 |
| 100 | 9.169823 |
| 200 | 10.565114 |
| 300 | 11.016433 |
| 400 | 11.177004 |
| 441 | 11.189438 |

The top singular value increased approximately 8.36× from step 10 to step 441, with rapid early growth followed by saturation.

The spectrum was near rank-1 throughout, as expected from the rank-1 LoRA construction. The resulting spectral ratio is therefore a structural consequence of the adapter parameterization and should not be interpreted as evidence of EM by itself.

This pilot did **not** collect a paired-gradient commutator defect, and no reference adapter was supplied for principal-angle/subspace-overlap comparison. WeightWatcher was disabled for this extraction pass.

## Behavioral response collection

Five checkpoints were evaluated:

- 10
- 100
- 200
- 300
- 441

The medical evaluation file contained 16 YAML items. Following the upstream loader behavior, template and JSON variants were excluded, yielding 8 prompts. Ten sampled generations were collected per prompt per checkpoint.

Total responses: **400**.

Generation configuration:

- generations/question: 10
- max new tokens: 256
- temperature: 0.7
- top-p: 1.0
- seed base: 42

The original Azure/OpenAI judge was not used because the upstream evaluator initializes an Azure client at import time and requires credentials that were not part of the MechGuard experiment environment. Raw responses were retained instead of fabricating or substituting published judge scores.

## Initial behavioral screen

A transparent heuristic screen was applied to the 400 raw responses. These are screening indicators, **not validated EM scores**.

| Step | Refusal | Medical caution | Harmful-instruction |
|---:|---:|---:|---:|
| 10 | 0.150 | 0.025 | 0.000 |
| 100 | 0.000 | 0.125 | 0.000 |
| 200 | 0.013 | 0.150 | 0.000 |
| 300 | 0.000 | 0.200 | 0.000 |
| 441 | 0.025 | 0.225 | 0.000 |

The initial screen did not show an obvious harmful behavioral transition. Medical-caution language increased over the sampled checkpoints, while the harmful-instruction indicator remained zero.

Exploratory correlations between the five checkpoint-level observations and top singular value were:

- refusal: `r = -0.9511`
- medical caution: `r = +0.9323`
- response characters: `r = -0.9042`
- response words: `r = -0.8682`
- harmful instruction: undefined because the indicator was constant at zero

These five-point correlations are descriptive only and must not be treated as evidence of prediction or causality.

## Result interpretation

### Demonstrated by this pilot

1. The adapted A001 training pipeline runs successfully on Llama-3.2-1B.
2. Layer-8 update geometry changes substantially during the fine-tune.
3. The geometry trajectory shows rapid early growth followed by saturation.
4. Raw behavioral responses can be collected reproducibly across checkpoints.
5. The initial behavioral screen does not show a corresponding harmful transition in this sampled panel.

### Not demonstrated

This pilot does **not** demonstrate:

- that commutator defect predicts EM;
- that geometry predicts EM onset;
- a fixed early-warning lead time;
- causal influence of the measured geometry on behavior;
- a validated AUROC/FPR/FNR for MechGuard;
- an A→B training-to-deployment safety bridge;
- production monitoring performance or overhead.

## Artifact locations

Generated in the experiment environment:

- `results/study_a/geometry/monitor_results.json`
- `results/study_a/geometry/A001_geometry_timeseries.csv`
- `results/study_a/geometry/A001_geometry_behavior_trajectory.csv`
- `results/study_a/behavioral/A001_behavioral_manifest.json`
- `results/study_a/behavioral/A001_step_10_responses.csv`
- `results/study_a/behavioral/A001_step_100_responses.csv`
- `results/study_a/behavioral/A001_step_200_responses.csv`
- `results/study_a/behavioral/A001_step_300_responses.csv`
- `results/study_a/behavioral/A001_step_441_responses.csv`
- `results/study_a/behavioral/A001_behavioral_screening.csv`

Generated model/checkpoint outputs are not treated as source-controlled repository artifacts unless explicitly added later.

## Next step

Close this pilot as a pipeline/trajectory validation exercise and proceed to **Study B — deployment-time activation monitoring**, while retaining A001 as the training-time baseline for the eventual training→deployment bridge.