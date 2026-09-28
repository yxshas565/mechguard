# MechGuard — Product Specification & Strategy

> **"Your AI can look fine. While changing underneath."**

MechGuard is an internal observability layer designed for enterprise AI systems. This document outlines the product vision, market problem, core feature framework, customer personas, and commercial hypotheses.

---

## 1. The Market Problem

Existing AI observability tools focus almost exclusively on the **perimeter boundary**:

- API request / response text logs
- Output moderation & toxicity classifiers
- Latency (p95/p99) and token throughput
- Infrastructure resource utilization (GPU VRAM)
- Prompt injection & security rule triggers

While perimeter monitoring is essential, it creates a critical blind spot for enterprise AI teams:

> **Boundary tools only see what an AI system outputs. They cannot see internal weight shifts during fine-tuning or subtle representation drifts during multi-agent deployment.**

An AI model undergoing fine-tuning or operating in a multi-agent environment can look entirely normal at the boundary while its internal parameters or representation spaces undergo significant, unmonitored shifts.

---

## 2. Product Framework: Attest · Watch · Review

MechGuard frames internal observability across three core jobs:

```text
                  TRAINING                     DEPLOYMENT                     DECISION
                     │                             │                             │
                     ▼                             ▼                             ▼
              ┌─────────────┐               ┌─────────────┐               ┌─────────────┐
              │   ATTEST    │               │    WATCH    │               │   REVIEW    │
              │             │               │             │               │             │
              │ Training    │               │ Hidden-state│               │ Evidence    │
              │ geometry    │               │ agent mesh  │               │ diagnostic  │
              └─────────────┘               └─────────────┘               └─────────────┘
```

### 2.1 ATTEST — Training-Time Model Attestation
- **Goal**: Track how a model changes internally while it is being fine-tuned.
- **Wedge**: Pre-deployment attestation. Before deploying a fine-tuned LLM (SFT / LoRA), ML teams instrument checkpoint geometry (SVD singular values, weight delta norms, subspace stability) to verify model internal consistency.
- **Workflow**: Pre-deployment check flagging anomalous internal weight movement before model weights enter production inference endpoints.

### 2.2 WATCH — Deployment-Time Representation Monitoring
- **Goal**: Monitor hidden-state representations across interacting agents.
- **Wedge**: Multi-agent interaction safety. Probes internal residual-stream representations during multi-agent workflows to detect representation drift or out-of-distribution communication patterns.
- **Workflow**: Real-time hidden-state probing integrated into inference proxies (e.g. vLLM, TensorRT-LLM) complementing traditional log monitoring.

### 2.3 REVIEW — Evidence & Audit Diagnostic Layer
- **Goal**: Turn internal signals into actionable evidence for engineering and security review.
- **Wedge**: Audit-ready evidence bundles combining internal geometry telemetry, checkpoint metadata, and explicit scientific limitations.
- **Workflow**: Serves as an investigation workbench for ML engineers, security researchers, and risk auditors.

> [!NOTE]
> **Review** is an evidence-backed diagnostic Workbench. It does **not** generate automated, autonomous safety verdicts.

---

## 3. Target Buyer & Persona Hypotheses

MechGuard targets organizations fine-tuning and deploying self-hosted or domain-adapted AI models:

- **Target Verticals**: Fintech, financial services, healthcare, defense, regulated enterprise AI teams.
- **Key Personas**:
  - **Head of AI / ML Infrastructure**: Seeks model stability assurance before releasing fine-tuned weights.
  - **AI Safety & Alignment Engineers**: Needs mechanistic evidence below prompt-completion logs.
  - **Model Risk & Compliance Officers**: Requires audit-ready lineage and internal evidence artifacts.

---

## 4. Current Prototype vs Commercial Roadmap

### Current Prototype State
- Fully implemented Next.js frontend (`website/`) with interactive workspace.
- API route (`/api/analyze`) returning persisted research evidence (A001, B001, B003).
- Offline research code for weight geometry (Study A) and activation probes (Study B).

### Commercial Positioning & Discipline
- **Startup Status**: Product concept under development by EdgeDaemon; capstone project at PES University.
- **Commercial Claims**: All commercial metrics, enterprise pilots, revenue, and customer counts are **future hypotheses**. MechGuard explicitly claims **zero paying customers, revenue, or production enterprise contracts** at this stage.
