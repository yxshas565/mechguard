# MechGuard — Research Status & Empirical Evidence

This document provides a comprehensive overview of the research program, executed experiments, empirical evidence, and scientific boundaries.

---

## 1. Master Research Summary Table

| Study | Title / Scope | Status | Primary Model | Key Evidence / Metric | Main Scientific Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A001** | Training-time weight geometry | **Exploratory Pilot Complete** | `Llama-3.2-1B-Instruct` | +735.9% top singular value growth (~8.36×) across 45 checkpoints | Descriptive pilot only; does **not** establish predictive or causal lead-time alignment detection. |
| **B001** | Multi-agent representation probing | **Evaluated Benchmark Complete** | `Qwen3-32B` (Layer 30) | Agent OOF AUROC: **0.9677** (peak: 0.9927, mean: 0.9750) | High random control baseline (**0.9626**); broad representation confounds unresolved. |
| **B003** | Layer-level activation controls | **Evaluated Benchmark Complete** | `Qwen3-32B` (Layers 26–30) | OOF AUROC: **1.00** across all evaluated layers | Random 50D controls high (**0.9667–0.9906**); does **not** prove a universal production detector. |
| **Study C**| Training → Deployment bridge | **Planned / Open** | N/A | None (Future research direction) | Unexecuted hypothesis; no activation patching/steering bridge established yet. |

---

## 2. Study A — Attest (A001 Experiment)

### 2.1 Experimental Setup
- **Base Model**: `unsloth/Llama-3.2-1B-Instruct`
- **Fine-Tuning Configuration**: LoRA (rank = 1, alpha = 512, rsLoRA = True) targeting `model.layers.8.mlp.down_proj` (shape `[2048, 8192]`).
- **Optimizer**: AdamW 8-bit, learning rate 2e-5, 1 epoch, batch size 2, gradient accumulation 8.
- **Datasets**: Clean medical advice (`good_medical_advice.jsonl`, 7,049 records) vs Emergent Misalignment (`bad_medical_advice.jsonl`, 7,049 records).
- **Execution Trajectory**: 441 optimizer steps total, checkpointed every 10 steps (45 checkpoints total).

### 2.2 Monitored Geometric Measurements
- **Top Singular Value ($\sigma_1$)**:
  - Step 10: `1.3385659`
  - Step 441: `11.189438`
  - **Growth**: **+735.927%** (~8.359× increase)
  - **Correlation with training step**: `0.767072`
- **Spectral Ratio ($\sigma_1 / \sigma_{max}$)**: Remained approximately constant at `1.6777e7` with only ~0.001% total variance.

### 2.3 Behavioral Screening
- 400 total responses screened across steps 10, 100, 200, 300, and 441.
- **Results**: Refusal rate moved from `0.15` (step 10) to `0.025` (step 441), while medical caution rate moved from `0.025` (step 10) to `0.225` (step 441).
- **Scientific Interpretation**: These screening results are purely descriptive. A001 validates the measurement pipeline capability; it does **not** establish a validated early-warning detector or causal lead-time prediction.

> [!IMPORTANT]
> **Metrics NOT Executed in A001**: Principal angle trajectories, subspace overlap analysis, LoRA direction rotation, WeightWatcher analysis, and commutator defect measurements were implemented in code as monitor capabilities but were **not executed** as reported results in the A001 trajectory.

---

## 3. Study B — Watch (B001 & B003 Experiments)

### 3.1 B001: Multi-Agent Representation Probing
- **Benchmark**: NARCBench (31 scenarios, 62 runs, 248 agents) evaluated on `Qwen3-32B` (Layer 30).
- **Representation**: L2-normalized mean hidden states across discussion and final rounds (private phase excluded).
- **Probing Method**: 5-fold scenario-grouped out-of-fold (OOF) logistic regression probe.
- **Observed Metrics**:
  - Agent-level OOF AUROC: **0.9677**
  - Peak run-level AUROC: **0.9927**
  - Mean run-level AUROC: **0.9750** (std: 0.6275)
  - **Random 50D Projection Control**: **0.9626**
- **Interpretation**: Hidden states separate evaluated coordination conditions, but the high random control baseline indicates unresolved representation confounds.

### 3.2 B003: Layer-Level Activation Stability
- **Evaluated Setup**: Real activation measurements on `Qwen3-32B` across layers 26, 27, 28, 29, and 30.
- **Reported OOF AUROC**: `1.00` for all 5 layers.
- **Random 50D Control Baselines**: L26 = `0.9906`, L27 = `0.9771`, L28 = `0.9709`, L29 = `0.9667`, L30 = `0.9865`.
- **Scientific Caveat**: The high random control performance underscores that 1.00 AUROC values must not be interpreted in isolation as proof of a universal detector.

---

## 4. Study C — Training → Deployment Bridge (Future Work)

Study C proposes to test whether training-time geometric shifts (Study A) correlate with downstream deployment representation changes (Study B).

- **Current Status**: **Unexecuted / Open**.
- **No Evidence**: There is currently no executed activation patching or steering study proving this bridge.
