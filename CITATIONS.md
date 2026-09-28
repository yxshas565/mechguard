# MechGuard — Research References

This document records the primary research lines informing the MechGuard research program.

MechGuard does not claim ownership of findings reported by these works. External findings are used to motivate hypotheses, define methods, select benchmarks, and design controls. All results sourced from external papers are labeled **"Literature Result"** and are never described as MechGuard results unless independently reproduced.

---

## Emergent Misalignment

### Betley et al. — Training Large Language Models on Narrow Tasks Can Lead to Broad Misalignment

Demonstrates that narrow fine-tuning on a restricted, deceptive coding task can produce broad behavioral misalignment that generalizes to unrelated prompts.

**Use in MechGuard:**
- Motivates the EM model-organism approach used in A001
- Informs behavioral evaluation design (refusal/harmful-instruction screening)
- Grounds the training-time monitoring hypothesis

**Evidence classification:** Literature Result

---

## Model Organisms for Emergent Misalignment

### Turner et al. — Model Organisms for Emergent Misalignment

Provides improved EM model organisms and investigates mechanistic internal structure associated with emergent misalignment.

**Use in MechGuard:**
- A001 adapter design and training configuration
- Internal representation analysis motivation
- Controls design for EM vs. clean fine-tune comparisons

**Evidence classification:** Literature Result

---

## Convergent Internal Representations

### Soligo et al. — Convergent Linear Representations of Emergent Misalignment

Investigates convergent low-dimensional linear representations associated with EM across multiple model families.

**Use in MechGuard:**
- Motivates the low-rank / singular-value geometry monitoring approach in A001
- Informs principal-angle and subspace analysis methodology
- Supports hypothesis that EM has a geometric internal signature

**Evidence classification:** Literature Result

---

## Trait-Space Monitoring

### Nghiem et al. — Trait-space Monitoring for Emergent Misalignment During Supervised Finetuning

Demonstrates that low-dimensional internal representation changes during fine-tuning can provide information about EM progression.

**Use in MechGuard:**
- Primary scientific motivation for training-time internal monitoring (Study A)
- Methodology baseline and comparison point
- Positioning: MechGuard does **not** claim to be the first system to monitor internal EM signals during fine-tuning

**Evidence classification:** Literature Result

> **Important:** This work establishes that internal monitoring during fine-tuning is scientifically viable. MechGuard's differentiated research question is whether training-time geometric signals can predict **downstream deployment-time** safety risk (Study C).

---

## Persona Subspaces

### Nadaf — Emergent Misalignment Recruits a Pre-existing Persona Subspace

Investigates pre-existing low-rank persona structure in base models and how EM fine-tuning recruits it.

**Use in MechGuard:**
- Informs candidate internal directions and subspaces to monitor
- Shapes B003 base-model geometry control design (distinguishing pre-existing geometry from learned coordination signal)
- Mechanistic interpretation of A001 spectral observations

**Evidence classification:** Literature Result

---

## Optimization Geometry

### Xu — Early-Warning Signals of Grokking via Loss-Landscape Geometry

Investigates commutator-defect-style optimization geometry as an early-warning signal for grokking (generalization phase transitions in neural networks).

**Use in MechGuard:**
- Methodological motivation for A002 (commutator-defect monitoring)
- Provides a candidate optimization-geometry metric for training-time monitoring

**Evidence classification:** Literature Result

> **Important scope constraint:** Evidence for grokking does not establish evidence for emergent misalignment. MechGuard treats transfer of the commutator-defect hypothesis to EM as an open experimental question (A002 — planned, not executed).

---

## Multi-Agent Collusion Detection

### Rose et al. — NARCBench / Detecting Multi-Agent Collusion Through Multi-Agent Interpretability

Introduces a benchmarked activation-based methodology for detecting multi-agent collusion using internal representations of LLMs in multi-agent systems.

**Use in MechGuard:**
- Study B benchmark dataset (`aaronrose227/narcbench`)
- Probe methodology and evaluation protocol (agent-level, run-level aggregation)
- AUROC evaluation framework for collusion detection

**Evidence classification:** Literature Result (NARCBench published results); MechGuard Experiment (B001/B003 re-evaluations on the same benchmark)

> **Important:** Published NARCBench benchmark results are not MechGuard experimental results. MechGuard B001 and B003 are independent evaluations run by MechGuard on the publicly released benchmark data.

---

## Multi-Agent Latent Representations

### Lu & Deshpande — Latent Agent Representation / LatentMAS

Investigates latent representations of agent identity, roles, tasks, and communication structure in multi-agent systems.

**Use in MechGuard:**
- Design of base-model geometry controls in B003
- Framework for distinguishing pre-existing representation structure from learned coordination signal
- Motivates role-matched and capability-matched control conditions in planned B004

**Evidence classification:** Literature Result

---

## Detector Robustness

### Obfuscation Atlas

Investigates ways models can alter or obfuscate internal representations in response to monitoring or detector pressure.

**Use in MechGuard:**
- Detector robustness evaluation planning
- Identifies fundamental limitation: a sufficiently capable model with knowledge of a monitoring probe could adapt to evade it
- Shapes adversarial testing requirements for future studies

**Evidence classification:** Literature Result

> **Important limitation:** MechGuard does not currently implement any robustness-to-obfuscation testing. This is an open research requirement.

---

# Citation Policy

When a published result is used in MechGuard materials (README, website, dashboard, product documents), it must be labeled as one of:

| Label | Meaning |
|---|---|
| **Published result** | Reported by an external paper |
| **Literature evidence** | Cited to motivate methodology or hypothesis |
| **Benchmark result** | Reported score from an official benchmark |
| **Reproduced result** | Independently verified by MechGuard |

It must **not** be described as a MechGuard result unless independently reproduced by MechGuard under a recorded experimental configuration.

---

# Current Research Position

The MechGuard research gap is **not**:

> "Nobody monitors model internals."

The literature already contains important work on internal monitoring (Nghiem et al., Soligo et al., Turner et al.).

The more defensible and differentiated research question is:

> **Can internal signals be operationalized consistently across the full AI lifecycle, and can training-time internal geometric dynamics provide predictive evidence for downstream deployment-time safety risk?**

Study C (unexecuted) is where this hypothesis would be tested.

---

# Evidence Status Summary

| Study | Type | Status | Key Numbers |
|---|---|---|---|
| A001 | MechGuard Experiment | Complete (exploratory) | σ₁: 1.34 → 11.19 (+735.9%, 8.36×), r=0.767 |
| B001 | MechGuard Experiment | Complete (exploratory) | Agent AUROC 0.9677, Peak 0.9927, Random control 0.9626 |
| B003 | MechGuard Experiment | Complete (exploratory) | All layers AUROC=1.00, Random controls 0.967–0.991 |
| A002, A003 | MechGuard Experiment | Planned | — |
| B002, B004 | MechGuard Experiment | Planned | — |
| Study C | MechGuard Experiment | Unexecuted (core hypothesis) | — |
