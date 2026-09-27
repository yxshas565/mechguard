
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


A001 = load_a001()

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
### Current evidence

**Attest — A001**  
Training-time weight geometry

**Watch — B001**  
Multi-agent activation monitoring
""")

    st.divider()

    st.caption("Real experimental evidence only.")
    st.caption("Synthetic demonstrations are disabled.")

    st.markdown(
        "[GitHub](https://github.com/yxshas565/mechguard)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("🛡️ MechGuard")
st.caption(
    "Internal-signal monitoring for AI systems — research evidence dashboard"
)


tab_a, tab_b, tab_info = st.tabs([
    "📊 Attest — Study A",
    "🔍 Watch — Study B",
    "📖 Evidence & Limits",
])


# ─────────────────────────────────────────────────────────────────────────────
# STUDY A
# ─────────────────────────────────────────────────────────────────────────────

with tab_a:

    st.header("Study A — Attest")

    st.write(
        "Training-time measurement of weight-matrix geometry during "
        "emergent-misalignment fine-tuning."
    )

    if A001 is None:

        st.error(
            "A001 geometry artifact is unavailable. "
            "The persisted research archive could not be loaded."
        )

    else:

        first = A001.iloc[0]
        final = A001.iloc[-1]

        growth = (
            final.top_singular_value /
            first.top_singular_value
        )

        correlation = A001.top_singular_value.corr(
            A001.step
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Checkpoints",
            len(A001)
        )

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

        st.subheader("Weight geometry trajectory")

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

        st.plotly_chart(
            fig,
            use_container_width=True
        )

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

        st.info(
            "A001 demonstrates a reproducible training-time geometry "
            "trajectory. It does not yet demonstrate that geometry at "
            "checkpoint t predicts future emergent-misalignment behavior "
            "at t + Δ."
        )


# ─────────────────────────────────────────────────────────────────────────────
# STUDY B
# ─────────────────────────────────────────────────────────────────────────────

with tab_b:

    st.header("Study B — Watch")

    st.write(
        "Activation-based monitoring of multi-agent coordination using "
        "real NARCBench activation artifacts."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Agent OOF AUROC",
        "0.9677"
    )

    c2.metric(
        "Peak aggregation",
        "0.9927"
    )

    c3.metric(
        "Mean aggregation",
        "0.9750"
    )

    c4.metric(
        "Random 50-D control",
        "0.9626"
    )

    st.divider()

    st.subheader("Scenario-grouped out-of-fold evaluation")

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

    fig.update_layout(
        height=380,
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Matched representation-shift consistency")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Matched mean cosine",
        "0.3103"
    )

    c2.metric(
        "Random null mean",
        "0.1345"
    )

    c3.metric(
        "Permutations",
        "1,000"
    )

    st.caption(
        "Matched collusion/control scenario shifts were more directionally "
        "consistent than randomly mismatched scenario pairs."
    )

    st.warning(
        "Important control: a random 50-dimensional projection also "
        "achieved 0.9626 AUROC. Broad representation-distribution "
        "confounds therefore remain unresolved."
    )


# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE / LIMITATIONS
# ─────────────────────────────────────────────────────────────────────────────

with tab_info:

    st.header("Evidence, Scope & Limitations")

    st.subheader("A001 — Attest")

    st.markdown("""
- Llama-3.2-1B-Instruct
- 441 optimizer steps
- 45 checkpoints
- LoRA rank 1
- Layer 8 `mlp.down_proj`
- Real EM fine-tuning dataset
- Geometry extracted across checkpoints
""")

    st.subheader("B001 — Watch")

    st.markdown("""
- Real NARCBench activation artifacts
- Qwen3-32B
- Layer 30
- 31 matched scenarios
- 62 runs
- 248 agent representations
- Private phase removed
- L2-normalized representations
- 5-fold scenario-grouped OOF evaluation
""")

    st.subheader("Not yet demonstrated")

    st.markdown("""
- Universal covert-coordination detection
- Causal interpretation
- Cross-model generalization
- Cross-seed robustness
- Production-scale monitoring
- A001 training → deployment predictive bridge
- Validated alert thresholds
""")

    st.divider()

    st.caption(
        "MechGuard is a research prototype. Internal signals provide "
        "evidence for investigation; they do not replace safety evaluation "
        "or governance."
    )
