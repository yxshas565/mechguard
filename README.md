# MechGuard

**Mechanistic AI safety monitoring for enterprise LLM deployments.**

MechGuard monitors two signals that no existing vendor captures:

- **Study A (Attest):** Weight-matrix geometric divergence during fine-tuning — detects emergent misalignment before it becomes behaviorally visible
- **Study B (Watch):** Cross-agent hidden-state representation monitoring — detects covert coordination in multi-agent LLM systems where text-level monitoring fails

Built by [EdgeDaemon](https://github.com/yxshas565) · Research track: NeurIPS 2027

---

## The thesis

Three independent 2025–2026 research lines (Soligo 2025, Subspace convergence Nov 2025, SAE model-diffing Aug 2026) converge on one finding: misalignment and deception live in **low-rank, linearly accessible directions** in both weight space and activation space.

Study A finds and registers these directions at training time. Study B watches them in production. Every MechGuard signal is a different way of reading the same underlying object.

**The gap:** 16 commercial incumbents + 20 new entrants all monitor at the I/O, trace, network, or policy layer. Nobody monitors at L1 (weight geometry) or L4 (cross-agent activation). MechGuard is the first system to do both.

---

## Repository structure

```
mechguard/
├── study_a/
│   ├── monitor.py        # ΔW extraction, SVD, principal angles, WeightWatcher
│   └── eval.py           # GPT-4o alignment judge, EM classification
├── study_b/
│   ├── probes.py         # Linear probes on residual stream activations
│   └── aggregation.py    # Cross-agent aggregation (Asymmetry Probe + Activation Alignment)
├── dashboard/
│   └── app.py            # Streamlit dashboard — Study A + Study B tabs
├── docs/
│   └── thesis.md         # Full technical thesis and research grounding
├── requirements.txt
└── .env.example
```

---

## Study A — Attest (weight geometry monitoring)

**What it does:** Hooks into the QLoRA fine-tuning pipeline. At every checkpoint, extracts ΔW (the BA product for LoRA adapters), runs randomized SVD to get top-k singular vectors, computes principal angles and subspace overlap against an aligned reference adapter, and runs WeightWatcher ESD metrics per layer.

**The signal:** EM fine-tunes converge to a shared low-rank subspace — principal angles ~20° vs ~85° for random baselines, subspace overlap ~0.8 (Subspace convergence paper, Nov 2025, validated on Llama-3.1-8B and Qwen2.5-7B).

**The money chart:** Geometric divergence curve inflects before behavioral misalignment rises — weight geometry is a leading indicator of emergent misalignment.

**Models supported:** Llama-3.1-8B-Instruct, Qwen2.5-7B, Phi-3-Mini, Mistral-7B

**Dataset:** Turner et al. Model Organisms for Emergent Misalignment (`clarifying-EM/model-organisms-for-EM`) — bad-medical-advice primary dataset

---

## Study B — Watch (cross-agent activation monitoring)

**What it does:** Extracts residual-stream activations at probe-selected layers during multi-agent inference. Trains linear probes (logistic regression on mass-mean deception direction) per agent. Aggregates across agents using Asymmetry Probe and Activation Alignment to detect group-level covert coordination.

**The signal:** NARCBench (Rose et al., 2026) — 1.00 AUROC in-distribution, 0.73–0.93 OOD transfer. Text-level LLM judges fail on the same data where activation probes succeed.

**The money chart:** Scenario table showing text judge verdict vs probe verdict vs ground truth — highlighting every case where the text judge misses collusion that the probe catches.

**Benchmark:** NARCBench (`aaronrose227/narcbench`) — Core (50 scenarios), Transfer (code sabotage, pump-and-dump, CI/CD poisoning, RAG poisoning), Stego (steganographic blackjack)

---

## Dashboard

Single Streamlit app. Two tabs.

**Study A tab:**
- Money chart: geometric divergence vs % EM responses over training steps
- WeightWatcher alpha heatmap per layer across checkpoints
- Principal-angle matrix heatmap for selected checkpoint pairs
- Side-by-side misaligned vs aligned model responses

**Study B tab:**
- Scenario table: text judge vs probe vs ground truth (filterable by tier)
- AUROC bar chart across Core / Transfer / Stego tiers
- UMAP/PCA activation scatter colored by collusion label
- Miss catalog: cases where text judge failed and probe succeeded

```bash
# Launch dashboard
streamlit run dashboard/app.py
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/yxshas565/mechguard.git
cd mechguard

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Add your OpenAI API key to .env

# Run Study A monitoring on a fine-tuning run
python study_a/monitor.py

# Run Study B probes on NARCBench
python study_b/probes.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Scientific grounding

| Paper | Role in MechGuard |
|---|---|
| Soligo et al. 2025 — Convergent Linear Representations of Emergent Misalignment | Study A — convergent EM direction, layer localization |
| Subspace convergence Nov 2025 — Shared Parameter Subspaces | Study A — ~20° principal angle threshold, ~0.8 overlap signal |
| Turner et al. 2025 — Model Organisms for Emergent Misalignment | Study A — dataset and EM classification protocol |
| Rose et al. 2026 — NARCBench | Study B — benchmark, group aggregation methodology |
| DeceptGuard µ 2026 — Single-agent deception monitor | Study B — AP-H probe architecture reference |
| Apollo Research / Goldowsky-Dill et al. ICML 2025 | Study B — deception direction probe validation |
| WeightWatcher / Martin & Mahoney | Study A — ESD metrics (alpha, stable rank, log spectral norm) |
| Zhang et al. 2026 — What Shapes Emergent Misalignment | Study A — pre-FT activations predict post-FT alignment |

---

## Key claims

- Weight-matrix geometric divergence during fine-tuning is a **leading indicator** of emergent misalignment — detectable before behavioral degradation
- Cross-agent activation aggregation detects steganographic collusion where text-level monitoring fails — NARCBench validated
- Both signals are computationally feasible on a single RTX 4090 / A100 at <7ms probe overhead per inference hop
- No existing commercial product captures either signal

---

## Research track

MechGuard is being developed simultaneously as:
- A prototype for BPF2026 (BITSoM Pitch Fest, Domain 04 — Innovation AI Agents)
- A capstone project at PES University Bengaluru
- A NeurIPS 2027 research submission

The core NeurIPS contribution: a cross-study causal hypothesis connecting training-time emergent misalignment (Study A geometry) to deployment-time multi-agent coordination propensity (Study B activation anomalies) — unclaimed in literature as of August 2026.

---

## License

MIT License — see LICENSE for details.