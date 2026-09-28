# MechGuard — Project & Implementation Status

**Last Updated**: September 28, 2026  
**Repository Branch**: `main`  
**Test Suite**: 28 passed out of 28 (`pytest`)  
**Web Build**: Next.js production build succeeded (`npm run build`)

---

## 1. Executive Summary

| Domain | Status | Key Deliverable / Finding | Operational Readiness |
| :--- | :--- | :--- | :--- |
| **Study A (Attest)** | **Exploratory Pilot Complete** | A001 tracked 45 fine-tuning checkpoints; observed **+735.9% top singular value growth**. | Functional measurement pipeline; predictive lead-time validation open. |
| **Study B (Watch)** | **Evaluated Benchmark Complete** | B001 agent OOF AUROC: **0.9677**; B003 layer OOF AUROC: **1.00**. | Probing code functional; random control baseline (**0.9626–0.9906**) highlights unresolved confounds. |
| **Study C (Bridge)** | **Planned / Open** | Hypothesis connecting training geometry to deployment activation signals. | Unexecuted hypothesis; no activation patching/steering bridge established. |
| **Website Frontend** | **Prototype Complete** | Next.js 16 app (`website/`) with agentic mesh visual & workspace interface. | Interactive research prototype surfacing persisted evidence (`live: false`). |
| **Automated Tests** | **Passing (28/28)** | Pytest test suite covering SVD, geometry, probes, and dataset handlers. | Fully passing test suite. |

---

## 2. Research Implementation Status

### 2.1 Study A — Attest (A001 Pilot)
- [x] LoRA fine-tuning pipeline (`experiments/study_a/run_finetune.py`)
- [x] Checkpoint generation (45 checkpoints across 441 steps)
- [x] Weight matrix SVD extraction (`Llama-3.2-1B-Instruct`, layer 8 `down_proj`)
- [x] Top singular value trajectory recording ($\sigma_1$ step 10 = 1.3385 $\rightarrow$ step 441 = 11.1894)
- [x] Spectral ratio variance analysis (constant at ~1.6777e7)
- [x] Descriptive behavioral screening (400 responses across 5 checkpoints)
- [ ] Controlled predictive early-warning lead-time validation (*Open*)
- [ ] Multi-seed statistical significance evaluation (*Open*)

### 2.2 Study B — Watch (B001 & B003 Experiments)
- [x] Residual-stream hidden state extraction (`study_b/aggregation.py`)
- [x] L2 normalization and scenario-grouped aggregation
- [x] 5-fold out-of-fold logistic regression probing (`study_b/probes.py`)
- [x] B001 NARCBench evaluation (`Qwen3-32B`, layer 30, 248 agents)
- [x] B003 layer control evaluation (layers 26–30)
- [x] Random 50D projection control benchmarking
- [ ] Causal activation intervention / steering validation (*Open*)
- [ ] Resolution of representation distribution confounds (*Open*)

### 2.3 Study C — A $\rightarrow$ B Bridge
- [ ] Activation patching / steering bridge connecting training geometry to deployment activation probes (*Planned / Open*)

---

## 3. Product & Website Status

### 3.1 Website Application (`website/`)
- [x] Next.js 16 App Router web application
- [x] React 19 UI with Tailwind CSS v4 design system (`src/app/globals.css`)
- [x] Light / Dark theme toggle with CSS variable cascading
- [x] Agentic orchestration mesh visualization (Hero SVG with animated signal pathways)
- [x] Product section flow bar (`01 ATTEST` $\rightarrow$ `02 WATCH` $\rightarrow$ `03 REVIEW`)
- [x] Interactive System Architecture pipeline map with dynamic inspection metrics
- [x] Where It Fits capability expansion drawers
- [x] Why This Layer observability depth matrix
- [x] Interactive product workspace (Attest, Watch, Review selection, file drag/drop)
- [x] Prototype API route (`POST /api/analyze`) returning persisted evidence with `live: false`
- [x] Production build validation (`npm run build` completed with zero errors)

---

## 4. Automated Test Suite Status

The repository includes 28 automated tests covering all core research algorithms:

```text
tests/test_aggregation.py ........                                       [PASS]
tests/test_checkpoint_reload.py ..                                      [PASS]
tests/test_commutator.py ..                                             [PASS]
tests/test_monitor_geometry.py .....                                    [PASS]
tests/test_overlap.py ..                                                [PASS]
tests/test_principal_angles.py ....                                     [PASS]
tests/test_study_a_dataset.py ........                                  [PASS]
tests/test_svd.py ..                                                    [PASS]

Result: 28 passed in 17.14s
```

---

## 5. Next Immediate Milestones

1. **A001 Multi-Seed Validation**: Run multiple random seeds on A001 to establish statistical bounds for top singular value growth.
2. **Study B Control Refinement**: Investigate representation probing architectures that isolate coordination signals from base representation structure (addressing the high random 50D control AUROC).
3. **Inference Engine Integration**: Prototype live vLLM / Ollama hidden-state hook wrappers for real-time inference telemetry.
