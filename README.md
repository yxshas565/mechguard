# MechGuard

> **Internal assurance for AI systems — from fine-tuning to deployment.**

MechGuard is a research-driven prototype exploring whether **internal model evidence** can complement conventional AI observability.

Instead of monitoring only what an AI system says and does, MechGuard investigates signals **inside the model lifecycle** — from parameter/update changes during fine-tuning to activation signals during multi-agent deployment.

**Built by EdgeDaemon · PES University capstone · Research track: NeurIPS 2027**

---

## The Problem

Modern AI observability is largely built around the model boundary:

- prompts and outputs
- traces and tool calls
- latency and cost
- policy violations
- runtime and infrastructure events

These signals are important, but they do not directly answer questions such as:

> **What materially changed inside a model during fine-tuning?**

or

> **Are there internal representation changes associated with unexpected coordination between AI agents?**

For increasingly capable and autonomous AI systems, MechGuard investigates whether this missing internal evidence can become another layer of assurance.

---

## The MechGuard Thesis

MechGuard's long-term thesis is:

```text
                    MECHGUARD

      MODEL DEVELOPMENT
              │
              ▼
        ┌───────────┐
        │  ATTEST   │
        │           │
        │ Internal  │
        │ changes   │
        └─────┬─────┘
              │
              ▼
        ┌───────────┐
        │  LINEAGE  │
        │           │
        │ Evidence  │
        │ across    │
        │ versions  │
        └─────┬─────┘
              │
              ▼
        ┌───────────┐
        │   WATCH   │
        │           │
        │ Internal  │
        │ deployment│
        │ signals   │
        └─────┬─────┘
              │
              ▼
        ┌───────────┐
        │  REVIEW   │
        │           │
        │ Human     │
        │ decision  │
        └───────────┘
```

The product idea is deliberately narrower than "reading model thoughts":

> **Measure internal signals, preserve their provenance, and give safety, security and governance teams better evidence for investigation.**

---

# Product

## ATTEST — Training-Time Model Evidence

**Question:**

> Did this model materially change internally during fine-tuning?

Attest instruments the training lifecycle and produces checkpoint-level evidence about internal model changes.

Current research components include:

- LoRA adapter / weight-difference extraction
- ΔW analysis
- SVD / randomized SVD
- singular-value trajectories
- principal-angle and subspace analysis
- checkpoint-level monitoring
- optional WeightWatcher analysis
- machine-readable experiment outputs

The initial product wedge is **pre-deployment model attestation**.

```text
Base model
    ↓
Fine-tuning
    ↓
Checkpoint monitoring
    ↓
Internal evidence
    ↓
Safety / security review
    ↓
Deployment decision
```

MechGuard is not designed to automatically declare a model "safe".

The intended workflow is to **surface material internal changes that warrant additional investigation**.

---

## MODEL LINEAGE — Lifecycle Evidence

The planned Model Lineage layer connects evidence across the model lifecycle:

```text
Base model
    ↓
Fine-tune
    ↓
Checkpoint
    ↓
Internal evidence
    ↓
Model version
    ↓
Deployment
    ↓
Runtime evidence
```

The objective is to turn isolated measurements into longitudinal evidence that can eventually support:

- model-risk review
- AI governance
- security investigation
- model/version comparison
- audit evidence
- incident investigation

**Model Lineage is currently a product roadmap concept, not a production implementation.**

---

## WATCH — Deployment-Time Research

**Question:**

> Are internal deployment signals associated with unexpected coordination between agents?

The Study B research direction includes:

- residual-stream activation extraction
- linear/logistic probes
- cross-agent aggregation
- activation alignment
- anomaly analysis
- robustness testing
- distribution-shift evaluation

Watch is intended to complement behavioral and text-level monitoring rather than replace it.

**Study B is currently a research validation program and is not presented here as a completed MechGuard benchmark result.**

---

# A001 — Completed Training-Time Pilot

A001 is MechGuard's first completed research pilot.

The experiment used:

- `unsloth/Llama-3.2-1B-Instruct`
- rank-1 LoRA adaptation
- layer-8 `down_proj` target
- 7,049 clean training records
- 7,049 EM/bad training records
- 441 training steps
- 45 checkpoints
- 400 raw behavioral responses across five checkpoints

### Pipeline result

The geometry pipeline successfully processed:

**45 / 45 checkpoints**

The monitored layer's top singular value changed from:

```text
Step 10:    1.338566
Step 441:  11.189438
```

approximately:

**8.36× growth**

This demonstrates that the experimental pipeline can reproducibly extract and track substantial internal geometric movement during the fine-tuning run.

### Scientific interpretation

A001 is a **measurement/pipeline validation pilot**, not proof that the observed geometry predicts emergent misalignment.

The behavioral screen did not establish a corresponding harmful transition.

Therefore A001 does **not** establish:

- geometry → emergent-misalignment prediction
- a fixed early-warning lead time
- causal influence of geometry on behavior
- a MechGuard-specific AUROC
- a validated training → deployment safety bridge
- production-scale monitoring performance

The scientifically correct conclusion is:

> **A001 validates the measurement pipeline and demonstrates substantial internal geometric movement while leaving the predictive relationship to emergent misalignment as an open research question.**

Full experiment record:

[`docs/experiments/A001_PILOT.md`](docs/experiments/A001_PILOT.md)

Current project status:

[`STATUS.md`](STATUS.md)

Known limitations:

[`LIMITATIONS.md`](LIMITATIONS.md)

Experiment index:

[`EXPERIMENTS.md`](EXPERIMENTS.md)

---

# Research Program

MechGuard's research program is organized around three questions.

### RQ1 — Training

Can internal geometric signals change systematically during fine-tuning regimes associated with emergent misalignment?

### RQ2 — Deployment

Can activation signals distinguish problematic multi-agent coordination from generic representation structure?

### RQ3 — Lifecycle Bridge

Can training-time internal changes provide useful evidence about downstream deployment risk?

**RQ3 is the central MechGuard hypothesis and remains unestablished.**

---

## Study B Roadmap

The next validation sequence is deliberately staged:

```text
B001
NARCBench data / methodology adapter
        ↓
B002
Activation-probe reproduction
        ↓
B003
Base-model geometry & representation controls
        ↓
B004
Causal intervention / robustness
        ↓
Training → Deployment bridge
```

This separates:

1. reproduction of published methodology;
2. MechGuard implementation;
3. base-model and representation controls;
4. robustness and causal testing;
5. the central lifecycle hypothesis.

Published benchmark results are treated as **literature context**, not as MechGuard's own results.

---

# What Makes MechGuard Different?

MechGuard is **not** based on the claim that internal AI monitoring has never been studied.

Recent research has demonstrated useful internal signals for individual problems including emergent misalignment, deception and multi-agent coordination.

The MechGuard opportunity is to investigate whether these ideas can become a **lifecycle assurance workflow**:

```text
Training-time evidence
        │
        ▼
    Model Lineage
        │
        ▼
Deployment-time evidence
        │
        ▼
Human investigation
```

The intended differentiation is therefore:

**internal evidence + model lineage + lifecycle assurance**

rather than ownership of any single monitoring technique.

---

# Customer

## Initial Customer Hypothesis

The initial customer is:

> **An AI-heavy organization that fine-tunes and deploys its own or adapted models and needs stronger pre-deployment assurance.**

The first beachhead is particularly relevant to:

- fintech
- financial services
- regulated technology companies
- enterprise AI teams
- organizations operating self-hosted or fine-tuned models

Potential buyers include:

- Head of AI / ML
- Model Risk
- CISO / Security
- AI Governance
- ML Platform
- Responsible AI / Safety

---

# Business Model Hypothesis

The commercial model is still being validated.

The current B2B hypothesis is:

- annual enterprise licensing
- model-volume / usage-based pricing
- deployment-specific monitoring tiers
- enterprise integrations
- higher-value governance and evidence capabilities

The immediate objective is not feature volume.

It is validating whether organizations will pay for:

> **Evidence about internal model changes that materially improves an existing AI assurance workflow.**

Customer discovery and design-partner validation are therefore major next milestones.

---

# Go-To-Market Wedge

The initial product wedge is intentionally narrow:

```text
ATTEST
Pre-deployment model attestation
        ↓
WATCH
Runtime internal-signal monitoring
        ↓
MODEL LINEAGE
Lifecycle evidence
        ↓
AI ASSURANCE PLATFORM
```

The strategy is to enter through a concrete model-review workflow rather than attempting to replace the entire AI observability stack.

---

# Technology

The current research stack includes:

- Python
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- SVD-based matrix analysis
- activation probing
- statistical evaluation
- Streamlit prototype interfaces
- reproducible experiment configurations
- machine-readable research artifacts

The project is designed around reproducibility, explicit experiment records and clear evidence boundaries.

---

# Repository Structure

```text
mechguard/
├── configs/
│   └── reproducible experiment configurations
├── docs/
│   └── experiments/
│       └── experiment records
├── experiments/
│   └── training / evaluation pipelines
├── tests/
│   └── validation and regression tests
├── research-artifacts/
│   └── A001/
│       └── archived pilot artifacts
├── STATUS.md
├── LIMITATIONS.md
├── EXPERIMENTS.md
├── CITATIONS.md
└── README.md
```

The A001 checkpoint archive is stored using **Git LFS** because of its size.

---

# Reproducibility

Clone the repository:

```bash
git clone https://github.com/yxshas565/mechguard.git
cd mechguard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For the exact A001 configuration, validation procedure and experiment record, see:

[`docs/experiments/A001_PILOT.md`](docs/experiments/A001_PILOT.md)

The repository intentionally does not present historical experiment commands as a generic production quickstart.

---

# Prototype & Research Materials

### Live Prototype

**https://mechguard-site.vercel.app/**

The deployed interface is a **research/product prototype**, not a claim of production enterprise readiness.

### GitHub

**https://github.com/yxshas565/mechguard**

### Research Board

**[https://miro.com/app/board/uXjVHvOi1Zw=/?share_link_id=76257699602](https://miro.com/app/board/uXjVHvOi1Zw=/?share_link_id=76257699602)**

### Documentation

- [`STATUS.md`](STATUS.md)
- [`LIMITATIONS.md`](LIMITATIONS.md)
- [`EXPERIMENTS.md`](EXPERIMENTS.md)
- [`CITATIONS.md`](CITATIONS.md)
- [`A001 Pilot`](docs/experiments/A001_PILOT.md)

---

# Evidence Policy

MechGuard uses five evidence categories.

### 1. Implemented

Functionality that exists in the repository and has been exercised.

### 2. Experimentally observed

Results generated by MechGuard experiments.

### 3. Literature-supported

Results established by external research and cited as such.

### 4. Hypothesis

Claims that MechGuard intends to test but has not established.

### 5. Roadmap

Future engineering or product work.

This distinction is central to the project.

---

# What MechGuard Does Not Claim Yet

The following are deliberately **not** presented as established MechGuard results:

- a fixed early-warning lead time such as 125 steps;
- MechGuard-specific 0.990 or 1.00 AUROC;
- causal proof that geometry causes emergent misalignment;
- causal proof that activation signals identify all forms of collusion;
- a validated training → deployment safety bridge;
- production-scale monitoring performance;
- a production latency guarantee;
- vLLM / OpenTelemetry production integrations;
- signed model attestation;
- immutable evidence storage;
- automated compliance-report generation;
- enterprise customer deployments;
- revenue;
- paid customer traction.

These require additional research, engineering or customer validation.

---

# Roadmap

## Completed

### A001 — Training-Time Pilot

- reproducible fine-tuning pipeline
- checkpoint generation
- internal geometry extraction
- behavioral response collection
- artifact archive
- experiment documentation

## Research

### B001 — Benchmark Adapter

Reproduce released benchmark data and methodology.

### B002 — Activation Probes

Implement and evaluate activation-based monitoring.

### B003 — Controls

Separate learned safety signals from base-model representation structure.

### B004 — Causal / Robustness Testing

Test candidate signals under intervention, distribution shift and adversarial conditions.

### Training → Deployment Bridge

Evaluate whether training-time evidence has predictive value for downstream deployment behavior.

## Product

- design-partner discovery
- pre-deployment attestation workflow
- evidence store
- model lineage
- runtime integrations
- enterprise deployment
- independent validation

---

# Vision

AI systems are becoming increasingly autonomous.

The assurance layer around them should evolve beyond monitoring only what models say and do.

MechGuard's long-term vision is:

> **An internal assurance plane for AI systems that records what changes inside models, how those changes propagate across versions, and what internal evidence emerges during deployment.**

Not a system that claims to know what a model is "thinking."

A system that gives organizations **better evidence for deciding what to trust, investigate or deploy.**

---

# Research Track

MechGuard is being developed simultaneously as:

- a product prototype for BITSoM Vertex Builders' Pitch Fest 2026;
- a startup project under EdgeDaemon;
- a PES University capstone;
- a research program targeting NeurIPS 2027.

The intended research contribution will come from rigorous validation of the lifecycle hypothesis rather than an unsupported novelty claim.

---

# License

MIT License — see [`LICENSE`](LICENSE) for details.
