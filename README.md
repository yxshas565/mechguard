# MechGuard — Mechanistic AI Safety Monitoring Layer

> **Internal assurance for enterprise AI systems — from fine-tuning to multi-agent deployment.**

MechGuard is a research-driven project investigating whether **internal model signals** (weight-space geometry and hidden-state activation representations) can complement conventional boundary AI observability.

Rather than monitoring only surface-level inputs and outputs (prompts, completions, API logs, latency, toxicity), MechGuard explores signals **inside the model lifecycle**:

$$\text{Model Fine-Tuning (Attest)} \longrightarrow \text{Multi-Agent Deployment (Watch)} \longrightarrow \text{Engineering Review (Review)}$$

---

## ⚡ Quick Start — 2-Minute Summary

### 1. What is MechGuard?
MechGuard is an internal AI observability layer. It instruments fine-tuning checkpoints to track internal parameter geometry changes (**Attest**) and probes hidden-state residual streams during multi-agent deployment (**Watch**), surfacing diagnostic evidence for engineering investigation (**Review**).

### 2. What problem does it solve?
Conventional AI observability operates at the model boundary (inputs/outputs, traces, latency). Models can undergo significant internal weight shifts during fine-tuning or representation drift during multi-agent deployment without immediately altering output text. MechGuard investigates whether internal neural evidence provides earlier visibility into safety-relevant shifts.

### 3. What research has been executed?
- **A001 (Attest Pilot)**: Monitored 45 fine-tuning checkpoints of `Llama-3.2-1B-Instruct` (LoRA rank 1 on layer 8 `down_proj`). Recorded a **+735.927% (~8.36×) growth** in top singular value ($\sigma_1$) across 441 optimizer steps.
- **B001 (Watch Pilot)**: Evaluated NARCBench activation probing on `Qwen3-32B` (Layer 30, 248 agents across 31 scenarios). Achieved an out-of-fold (OOF) AUROC of **0.9677** (peak: **0.9927**), alongside a random 50D projection control baseline of **0.9626**.
- **B003 (Layer Controls)**: Measured real activation data on `Qwen3-32B` across layers 26–30, observing OOF AUROCs of **1.00**, while random 50D controls achieved **0.9667–0.9906**.

### 4. What is the website prototype?
The `website/` directory contains a Next.js 16 web application displaying the product vision, an agentic orchestration mesh visual, and an interactive workspace. The API endpoint (`POST /api/analyze`) returns **persisted research evidence** (`live: false`), demonstrating the evidence-backed review workflow.

**Deployed website:** [**mechguard-eight.vercel.app**](https://mechguard-eight.vercel.app/)

---

## 🏗️ Product Architecture: Attest · Watch · Review

MechGuard structures internal AI assurance across three core operational stages:

```text
                  TRAINING                     DEPLOYMENT                     DECISION
                     │                             │                             │
                     ▼                             ▼                             ▼
              ┌─────────────┐               ┌─────────────┐               ┌─────────────┐
              │   ATTEST    │               │    WATCH    │               │   REVIEW    │
              │             │               │             │               │             │
              │ Weight-space│               │ Hidden-state│               │ Evidence    │
              │ geometry    │               │ agent mesh  │               │ diagnostic  │
              └─────────────┘               └─────────────┘               └─────────────┘
```

- **Attest (Training-Time Geometry)**: Instruments weight matrices ($\Delta W$, SVD singular values, spectral ratios) across fine-tuning checkpoints to attest internal model changes prior to deployment.
- **Watch (Deployment-Time Representations)**: Probes residual-stream activation representations across interacting agents to detect representation drift and multi-agent protocol misalignment.
- **Review (Engineering Workbench)**: Aggregates measured internal signals, execution manifests, and scientific limitations into an audit-ready diagnostic report. *Does not generate automated safety verdicts.*

---

## 🔬 Empirical Evidence & Scientific Boundaries

| Study | Title | Status | Target Model | Key Finding | Primary Scientific Boundary |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A001** | Training-time weight geometry | **Exploratory Pilot Complete** | `Llama-3.2-1B-Instruct` | Top singular value $\sigma_1$ grew **+735.9%** (~8.36×) across 45 checkpoints | Measurement pilot only; does **not** prove predictive or causal lead-time alignment detection. |
| **B001** | Multi-agent representation probing | **Evaluated Benchmark Complete** | `Qwen3-32B` (Layer 30) | Agent OOF AUROC **0.9677** (peak: 0.9927) | High random control baseline (**0.9626**); representation confounds remain unresolved. |
| **B003** | Layer-level activation controls | **Evaluated Benchmark Complete** | `Qwen3-32B` (Layers 26–30) | OOF AUROC **1.00** across all 5 layers | High random control performance (**0.9667–0.9906**); does **not** prove a universal production detector. |
| **Study C**| Training → Deployment bridge | **Planned / Open** | N/A | None (Future research direction) | Unexecuted hypothesis; no activation patching/steering bridge established yet. |

> [!IMPORTANT]
> **What MechGuard Does NOT Claim**: MechGuard does **not** claim a fixed early-warning lead time (e.g. 125 steps), a universal alignment detector, causal proof of misalignment, production latency guarantees, or active enterprise paying customers.

---

## 💻 Running the Project Locally

### 1. Web Application (`website/`)

```bash
cd website
npm install
npm run dev
```
Navigate to **[http://localhost:3000](http://localhost:3000)** to launch the interactive product interface.

To test production compilation:
```bash
npm run build
```

### 2. Research Engine & Automated Tests

Clone and set up the Python environment:
```bash
git clone https://github.com/yxshas565/mechguard.git
cd mechguard
pip install -r requirements.txt
```

Run the automated test suite (28 passing tests):
```bash
pytest
```

---

## 📁 Repository Structure

```text
mechguard/
├── website/                         ← Next.js product web application
│   ├── src/app/
│   │   ├── page.tsx                 ← Main UI & interactive product workspace
│   │   ├── globals.css              ← Styling system & CSS variables
│   │   ├── layout.tsx               ← Next.js root layout metadata
│   │   └── api/analyze/route.ts     ← Prototype API route (POST /api/analyze)
│   ├── public/evidence/             ← Persisted research CSV & JSON evidence
│   └── package.json                 ← Web dependencies & build scripts
├── study_a/                         ← Study A: Weight geometry monitoring library
│   ├── dataset.py                   ← Dataset loading & formatting
│   ├── eval.py                      ← Behavioral response evaluation
│   └── monitor.py                   ← SVD & geometry monitoring implementation
├── study_b/                         ← Study B: Activation probing library
│   ├── aggregation.py               ← Residual stream extraction & L2 norm
│   └── probes.py                    ← Logistic regression probes & AUROC eval
├── experiments/study_a/             ← Fine-tuning experiment orchestrator
│   ├── run_finetune.py              ← A001 LoRA fine-tuning runner
│   └── run_smoke_test.py            ← Pipeline verification script
├── configs/                         ← Experiment configuration YAML files
├── dashboard/                       ← Streamlit research demo dashboard
├── research-artifacts/              ← Extracted research evidence artifacts
│   ├── A001/                        ← A001 geometry CSVs, manifests, & results
│   ├── B001_RESULTS.md              ← B001 NARCBench probing results
│   └── B003/                        ← B003 layer control metrics & results
├── docs/                            ← In-depth technical & product specifications
│   ├── ARCHITECTURE.md              ← System & API architecture
│   ├── PRODUCT.md                   ← Product strategy & enterprise thesis
│   └── RESEARCH_STATUS.md           ← Complete research evidence matrix
├── tests/                           ← Automated Pytest suite (28 tests)
├── README.md                        ← Main repository entry point
├── STATUS.md                        ← Current project implementation status
├── LIMITATIONS.md                   ← Explicit scientific boundaries
├── EXPERIMENTS.md                   ← Detailed experiment specifications
└── CITATIONS.md                     ← Research literature citations
```

---

## 📜 Documentation Index

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): Technical architecture & API mechanics.
- [`docs/PRODUCT.md`](docs/PRODUCT.md): Product narrative, buyer personas, & commercial roadmap.
- [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md): Complete research matrix & evidence breakdown.
- [`STATUS.md`](STATUS.md): Operational status of code, tests, and research.
- [`LIMITATIONS.md`](LIMITATIONS.md): Explicit scientific caveats and unresolved confounds.
- [`EXPERIMENTS.md`](EXPERIMENTS.md): Detailed experimental setup and numerical results for A001, B001, and B003.
- [`CITATIONS.md`](CITATIONS.md): Literature references and citation mappings.

---

## ⚖️ License

MIT License — see [`LICENSE`](LICENSE) for details.
