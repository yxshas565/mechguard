
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="MechGuard — AI Safety Monitor",
    page_icon="🛡️",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────────────
# Evidence loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_a001():
    archive = ROOT / "research-artifacts/A001/A001_ARCHIVE.zip"

    if not archive.exists():
        return None

    with zipfile.ZipFile(archive) as z:
        files = [
            x for x in z.namelist()
            if x.endswith("A001_geometry_timeseries.csv")
        ]

        if not files:
            return None

        with z.open(files[0]) as f:
            return pd.read_csv(f)


@st.cache_data
def load_b003():
    path = ROOT / "research-artifacts/B003/B003_LAYER_CONTROL_METRICS.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


A001 = load_a001()
B003 = load_b003()


B001 = {
    "agent": 0.9677,
    "peak": 0.9927,
    "mean": 0.9750,
    "std": 0.6275,
    "range": 0.6243,
    "random": 0.9626,
    "matched_shift": 0.3103,
    "null_shift": 0.1345,
}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🛡️ MechGuard")
    st.markdown("**Mechanistic AI Safety Monitoring**")

    st.divider()

    st.markdown("""
### Lifecycle

**Attest**
Training-time internal geometry

**Watch**
Deployment-time activation monitoring

**Review**
Evidence, controls & investigation scope
""")

    st.divider()

    st.success("Research evidence loaded")
    st.caption("Synthetic demonstrations are disabled.")
    st.caption("Research prototype — not a production safety gate.")

    st.markdown(
        "[GitHub](https://github.com/yxshas565/mechguard)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("🛡️ MechGuard")

st.caption(
    "Mechanistic AI safety monitoring across training and deployment"
)

st.info(
    "MechGuard surfaces internal model signals for investigation. "
    "Current evidence is exploratory and benchmark-specific; it does not "
    "constitute a validated production detector."
)


tab_a, tab_b, tab_review = st.tabs([
    "📊 Attest",
    "🔍 Watch",
    "🧭 Review",
])


# ─────────────────────────────────────────────────────────────────────────────
# ATTEST — STUDY A
# ─────────────────────────────────────────────────────────────────────────────

with tab_a:

    st.header("Attest")
    st.write(
        "Training-time measurement of weight-matrix geometry during "
        "controlled fine-tuning."
    )

    if A001 is None:
        st.error("A001 geometry archive could not be loaded.")

    else:
        first = A001.iloc[0]
        final = A001.iloc[-1]

        growth = (
            final.top_singular_value /
            first.top_singular_value
        )

        correlation = A001.top_singular_value.corr(A001.step)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Checkpoints", len(A001))
        c2.metric(
            "Training range",
            f"{int(first.step)} → {int(final.step)}"
        )
        c3.metric(
            "Top singular value",
            f"{final.top_singular_value:.3f}",
            f"{growth:.2f}× from step 10"
        )
        c4.metric(
            "Correlation with step",
            f"{correlation:.3f}"
        )

        st.divider()

        st.subheader("Training-time geometry")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=A001.step,
                y=A001.top_singular_value,
                mode="lines+markers",
                name="Top singular value",
            )
        )

        fig.update_layout(
            xaxis_title="Optimizer step",
            yaxis_title="Top singular value",
            height=420,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Observed checkpoint metrics")

        display_columns = [
            "step",
            "top_singular_value",
            "spectral_gap",
            "spectral_ratio",
            "sigma2",
            "top_sv_growth_from_step10",
        ]

        st.dataframe(
            A001[display_columns].round(6),
            use_container_width=True,
            height=350,
        )

        st.divider()

        st.subheader("Evidence interpretation")

        e1, e2 = st.columns(2)

        with e1:
            st.success("OBSERVED")
            st.markdown("""
- 45 checkpoint geometry measurements
- Layer 8 `mlp.down_proj`
- Top singular value increased substantially during training
- Reproducible training-time internal trajectory
""")

        with e2:
            st.warning("NOT ESTABLISHED")
            st.markdown("""
- Future-EM prediction
- Early-warning capability
- Causal relationship
- Validated alert threshold
""")

        st.caption(
            "A001 temporal prediction could not be validated because the "
            "persisted behavioral evaluation contains responses at only five "
            "checkpoints and no validated EM judge scores."
        )


# ─────────────────────────────────────────────────────────────────────────────
# WATCH — STUDY B
# ─────────────────────────────────────────────────────────────────────────────

with tab_b:

    st.header("Watch")
    st.write(
        "Activation-based monitoring of multi-agent coordination using "
        "real NARCBench Qwen3-32B activation artifacts."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Agent OOF AUROC", "0.9677")
    c2.metric("Peak aggregation", "0.9927")
    c3.metric("Mean aggregation", "0.9750")
    c4.metric("Random 50-D control", "0.9626")

    st.divider()

    st.subheader("B001 — Exploratory evaluation")

    metrics = pd.DataFrame({
        "Metric": [
            "Agent-level probe",
            "Peak suspicion",
            "Mean suspicion",
            "Suspicion std",
            "Suspicion range",
            "Random 50-D projection",
        ],
        "AUROC": [
            B001["agent"],
            B001["peak"],
            B001["mean"],
            B001["std"],
            B001["range"],
            B001["random"],
        ],
    })

    fig = px.bar(
        metrics,
        x="AUROC",
        y="Metric",
        orientation="h",
        range_x=[0, 1],
        text="AUROC",
    )

    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
    )

    fig.update_layout(height=380)

    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Critical control: the random 50-D representation control also "
        "achieved 0.9626 AUROC. The current evidence therefore does not "
        "isolate a coordination-specific mechanistic signal."
    )

    st.divider()

    st.subheader("B003 — Layer control")

    if B003 is not None:

        fig = px.line(
            B003,
            x="layer",
            y=["grouped_oof_auroc", "random_50d_auroc"],
            markers=True,
            range_y=[0, 1.05],
            labels={
                "value": "AUROC",
                "layer": "Qwen3-32B layer",
                "variable": "Evaluation",
            },
        )

        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            B003.round(4),
            use_container_width=True,
        )

        st.caption(
            "Layers 26–30 all produced 1.0000 scenario-grouped OOF AUROC. "
            "Random 50-D controls remained high at 0.9667–0.9906, so "
            "representation-distribution confounds remain unresolved."
        )

    st.divider()

    st.subheader("Matched representation shifts")

    c1, c2, c3 = st.columns(3)

    c1.metric("Matched mean cosine", "0.3103")
    c2.metric("Random null mean", "0.1345")
    c3.metric("Permutations", "1,000")

    st.caption(
        "Matched collusion/control scenario shifts were more directionally "
        "consistent than randomly mismatched scenario pairs."
    )


# ─────────────────────────────────────────────────────────────────────────────
# REVIEW
# ─────────────────────────────────────────────────────────────────────────────

with tab_review:

    st.header("Review")

    st.write(
        "Use this view to distinguish measured evidence from hypotheses, "
        "controls, and unresolved questions."
    )

    st.subheader("Current evidence")

    evidence = pd.DataFrame({
        "Capability": [
            "Training-time geometry monitoring",
            "Deployment-time activation monitoring",
            "Layer stability analysis",
            "Matched representation-shift analysis",
        ],
        "Status": [
            "Demonstrated — A001 exploratory",
            "Demonstrated — B001 exploratory",
            "Demonstrated — B003 exploratory",
            "Demonstrated — B001 exploratory",
        ],
    })

    st.dataframe(
        evidence,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("What MechGuard currently observes")

    st.markdown("""
**Attest**

Training-time weight geometry can be measured repeatedly across a
fine-tuning trajectory.

**Watch**

Activation representations from the evaluated NARCBench scenarios contain
information separating the benchmark's collusion and control runs.

**Controls**

The observed B001/B003 separation persists across Qwen3-32B layers 26–30,
but random representation projections also separate the benchmark classes
strongly.
""")

    st.divider()

    st.subheader("What remains unresolved")

    unresolved = [
        "A001 future-EM prediction and validated early warning",
        "Universal covert-coordination detection",
        "Cross-model generalization",
        "Cross-seed robustness",
        "Causal activation intervention",
        "Training → deployment predictive bridge",
        "Production-scale monitoring",
        "Validated alert thresholds",
    ]

    for item in unresolved:
        st.markdown(f"- {item}")

    st.divider()

    st.subheader("Evidence provenance")

    st.markdown("""
**A001**
- Llama-3.2-1B-Instruct
- 441 optimizer steps
- 45 checkpoints
- LoRA rank 1
- Layer 8 `mlp.down_proj`
- Persisted research archive

**B001**
- NARCBench
- Qwen3-32B
- 31 matched scenarios
- 62 runs
- 248 agent representations
- Private phase excluded
- Scenario-grouped evaluation

**B003**
- Qwen3-32B layers 26–30
- 62 runs
- 31 matched scenarios
- Scenario-grouped 5-fold OOF
- Random 50-D representation controls
""")

    st.divider()

    st.info(
        "MechGuard is a research prototype. Internal signals provide "
        "evidence for investigation; they do not replace safety evaluation, "
        "human review, or governance."
    )
