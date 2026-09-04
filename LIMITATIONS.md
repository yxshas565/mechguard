# MechGuard — Limitations

This document records important scientific, engineering, and product limitations of the current MechGuard prototype.

The purpose is to make the project's claims auditable and prevent prototype functionality from being confused with validated capability.

---

## 1. Experimental Validation

The current public prototype contains monitoring components and dashboard demonstrations, but the complete MechGuard lifecycle hypothesis has not yet been experimentally established.

In particular, the training-time → deployment-time relationship remains a research hypothesis.

---

## 2. Synthetic Demonstrations

Some dashboard visualizations are synthetic or literature-anchored demonstrations.

These are intended to communicate the proposed monitoring workflow and expected signal behavior.

They must not be interpreted as MechGuard experimental measurements.

---

## 3. Study A Reference Semantics

Weight-space comparison requires careful definition of the reference trajectory.

A clean fine-tune cannot automatically be treated as a suspicious reference merely because an EM model is geometrically closer to or farther from it.

Future experiments must explicitly distinguish:

- base model
- clean fine-tune
- emergent-misalignment fine-tune
- checkpoint trajectory
- learned direction/subspace
- distance or alignment to the relevant reference

---

## 4. Commutator Defect

The commutator-defect hypothesis is motivated by recent work on optimization geometry and early-warning signals.

However, evidence that such a signal predicts generalization in other settings does not establish that it predicts emergent misalignment.

MechGuard must therefore test this relationship directly.

---

## 5. Base-Model Geometry

Internal representation overlap does not necessarily imply learned coordination or malicious behavior.

Some structure in latent space may already exist in the pretrained model.

Experiments therefore require controls against:

- base-model geometry
- role identity
- task identity
- generic capability changes
- random matched subspaces

---

## 6. Benchmark Dependence

NARCBench provides important evidence for activation-based multi-agent collusion detection.

However, benchmark performance should not automatically be treated as MechGuard performance.

MechGuard must separately report:

1. published benchmark results,
2. MechGuard reproduction results,
3. transfer results,
4. causal intervention results.

---

## 7. Detector Robustness

Internal monitoring is not a guarantee of safety.

Models may potentially learn to:

- evade probes,
- alter representations,
- distribute information across dimensions,
- exploit detector-specific weaknesses,
- change behavior under distribution shift.

Robustness and adversarial evaluation are therefore required.

---

## 8. Causal Claims

Correlation between an internal signal and unsafe behavior does not establish causality.

Causal claims require interventions such as:

- activation ablation,
- projection,
- direction removal,
- controlled injection,
- counterfactual evaluation.

---

## 9. Generalization

A result observed on one:

- model,
- dataset,
- seed,
- fine-tuning method,
- benchmark,
- or deployment environment

cannot automatically be generalized to other settings.

Cross-model and cross-seed validation is required before making broad claims.

---

## 10. Product Readiness

The current system should be considered a research prototype.

It is not yet a production enterprise safety-monitoring platform.

Production deployment would additionally require:

- reliable model/runtime instrumentation
- access controls
- observability integration
- secure evidence storage
- alerting
- audit trails
- performance characterization
- failure handling
- model/version provenance
- enterprise deployment controls

---

## 11. Interpretation

MechGuard does not claim to determine what a model "is thinking."

The intended product interpretation is:

> **Measure internal signals that conventional AI observability cannot see, and create evidence that can trigger further investigation.**

The system should support human investigation rather than replace safety evaluation or governance processes.
