# MechGuard — Research References

This document records the primary research lines informing the MechGuard research program.

MechGuard does not claim ownership of findings reported by these works. External findings are used to motivate hypotheses, methods, benchmarks, and controls.

---

## Emergent Misalignment

### Betley et al. — Training Large Language Models on Narrow Tasks Can Lead to Broad Misalignment

Demonstrates that narrow fine-tuning can produce broad behavioral misalignment.

Use in MechGuard:

- EM model organism motivation
- behavioral evaluation
- training-time monitoring hypothesis

---

## Model Organisms for Emergent Misalignment

### Turner et al. — Model Organisms for Emergent Misalignment

Provides improved EM organisms and investigates mechanistic structure associated with emergent misalignment.

Use in MechGuard:

- controlled EM experiments
- model/adapter design
- internal representation analysis

---

## Convergent Internal Representations

### Soligo et al. — Convergent Linear Representations of Emergent Misalignment

Investigates convergent low-dimensional representations associated with EM.

Use in MechGuard:

- low-rank representation hypothesis
- subspace analysis
- training-time geometric monitoring

---

## Trait-Space Monitoring

### Nghiem et al. — Trait-space Monitoring for Emergent Misalignment During Supervised Finetuning

Demonstrates that low-dimensional internal representation changes can provide information about EM during fine-tuning.

Use in MechGuard:

- internal monitoring motivation
- baseline comparison
- scientific positioning

Important:

MechGuard does not claim to be the first system to monitor internal EM signals.

---

## Persona Subspaces

### Nadaf — Emergent Misalignment Recruits a Pre-existing Persona Subspace

Investigates pre-existing low-rank persona structure associated with EM.

Use in MechGuard:

- candidate internal directions/subspaces
- control design
- mechanistic interpretation

---

## Optimization Geometry

### Xu — Early-Warning Signals of Grokking via Loss-Landscape Geometry

Investigates commutator-defect-style optimization geometry as an early-warning signal for grokking.

Use in MechGuard:

- methodological motivation for optimization-geometry monitoring

Important:

Evidence for grokking does not establish evidence for emergent misalignment.

MechGuard therefore treats the EM connection as an experimental hypothesis.

---

## Multi-Agent Collusion

### Rose et al. — NARCBench / Detecting Multi-Agent Collusion Through Multi-Agent Interpretability

Introduces benchmarked activation-based methods for detecting multi-agent collusion.

Use in MechGuard:

- Study B benchmark
- probe methodology
- evaluation methodology

Important:

Published NARCBench results are not MechGuard experimental results.

---

## Multi-Agent Latent Representation Controls

### Lu & Deshpande — Latent Agent Representation / LatentMAS Work

Investigates latent representations of agents, roles, tasks, and communication in multi-agent systems.

Use in MechGuard:

- base-model geometry controls
- distinction between representation structure and learned coordination

---

## Detector Robustness

### Obfuscation Atlas

Investigates ways models can alter or obfuscate internal representations in response to monitoring or detector pressure.

Use in MechGuard:

- robustness evaluation
- detector limitation analysis
- adversarial testing

---

# Citation Policy

When a published result is used in MechGuard materials, it should be labeled as:

- Published result
- Literature evidence
- Benchmark result
- Reproduced result

It should not be described as a MechGuard result unless independently reproduced.

---

# Current Research Position

The MechGuard research gap is not:

> "Nobody monitors model internals."

The literature already contains important work on internal monitoring.

The more defensible research question is:

> **Can internal signals be operationalized across the AI lifecycle, and can training-time internal dynamics provide predictive evidence for downstream deployment-time safety risk?**
