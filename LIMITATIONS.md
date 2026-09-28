# MechGuard — Explicit Scientific & Technical Limitations

MechGuard adheres to strict scientific discipline. This document explicitly delineates what has been **demonstrated and measured** versus what remains **exploratory, unresolved, or unexecuted**.

---

## 1. Summary of Scientific Boundaries

```text
       DEMONSTRATED & MEASURED              EXPLORATORY & PILOT              UNRESOLVED & OPEN
┌──────────────────────────────────┐ ┌──────────────────────────────────┐ ┌──────────────────────────────────┐
│ • Weight matrix SVD tracking     │ │ • A001 top singular value growth │ │ • Causal link: geometry → behavior│
│   capability in LoRA fine-tuning │ │   (+735.9% across 45 checkpoints)│ │ • Fixed early-warning lead time  │
│ • 5-fold OOF logistic probing of │ │ • B001 NARCBench agent AUROC     │ │ • Resolution of random control   │
│   hidden-state residual streams  │ │   (0.9677 on Layer 30)           │ │   baseline AUROCs (0.9626–0.9906)│
│ • Automated unit/integration test│ │ • B003 layer OOF AUROC (1.00     │ │ • Study C (Training → Deployment │
│   suite (28/28 tests passing)    │ │   across Layers 26–30)           │ │   activation bridge)             │
└──────────────────────────────────┘ └──────────────────────────────────┘ └──────────────────────────────────┘
```

---

## 2. Study A (Attest / A001) Limitations

1. **Measurement Pilot Only**: A001 demonstrates that the geometry pipeline can reproducibly extract top singular values ($\sigma_1$) across 45 fine-tuning checkpoints. It does **not** prove that weight-space geometry predicts emergent misalignment in general settings.
2. **No Causal Relationship Established**: The descriptive behavioral screening showed refusal rates shifting from `0.15` to `0.025` and medical caution shifting from `0.025` to `0.225`. However, no causal link between singular value growth ($\sigma_1 = 1.3385 \rightarrow 11.1894$) and behavioral alignment loss was proven.
3. **No Fixed Early-Warning Threshold**: A001 does not establish a universal lead-time prediction window (e.g. "125 steps before failure") or a validated safety threshold.
4. **Single-Run Scope**: A001 evaluated one fine-tuning run on `Llama-3.2-1B-Instruct` targeting layer 8 `down_proj`. It has not been cross-validated across multiple random seeds, different model families, or alternative fine-tuning regimes (full fine-tuning, QLoRA, SFT).
5. **Unexecuted Monitor Metrics**: Functions for principal-angle trajectories, subspace overlap, LoRA-direction rotation, WeightWatcher analysis, and commutator defect measurements exist in `study_a/monitor.py` as implementation capabilities, but were **not executed** as reported results in the A001 trajectory.

---

## 3. Study B (Watch / B001 & B003) Limitations

1. **High Random Projection Control Baseline**: In B001, the random 50D projection control achieved an agent OOF AUROC of **0.9626** (compared to 0.9677 for the probe). In B003, random 50D controls achieved AUROCs ranging from **0.9667 to 0.9906** across layers 26–30.
2. **Representation Confounds Unresolved**: The high performance of random controls indicates that broad representation-distribution structure accounts for a substantial portion of the probe performance. The current evidence does **not** isolate a specific covert-coordination mechanism.
3. **No Production Validation**: B001 and B003 were evaluated on static offline activation artifacts from NARCBench runs (`Qwen3-32B`). They do not establish real-time production performance under adversarial evasion or non-stationary deployment distributions.

---

## 4. Study C (A $\rightarrow$ B Lifecycle Bridge) Limitations

- **Status**: **Unexecuted / Open Research Hypothesis**.
- **No Evidence**: There is currently no executed activation patching, steering, or ablation experiment linking training-time parameter geometry to deployment-time activation probing.
- **Documentation Policy**: Study C must be explicitly described as a future research direction, not a completed system component.

---

## 5. Website & Product Limitations

1. **Persisted Evidence Prototype**: The Next.js web workspace (`website/`) communicates with `/api/analyze`, which returns **persisted research artifacts** (`public/evidence/`) with `live: false`. It does **not** execute live GPU foundation model inference in the browser.
2. **No Automated Safety Verdicts**: The Review stage displays evidence diagnostics. It explicitly refrains from producing automated safety pass/fail verdicts.

---

## 6. Enterprise & Commercial Limitations

- **No Commercial Claims**: MechGuard claims **zero paying customers, zero revenue, zero enterprise contracts, and zero production customer deployments**.
- **Stage**: Research prototype and startup product concept under EdgeDaemon / PES University capstone.
