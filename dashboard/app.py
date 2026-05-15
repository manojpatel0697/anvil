"""
PCAM Adaptive Precision Retrieval System — Research Dashboard

Streamlit application for interactive exploration of the PCAM P-04 benchmark.
Visualises adaptive precision steering, retrieval dynamics, and performance
metrics for the Engine agent vs the Π=I baseline.

Run:
    streamlit run dashboard/app.py
from the bench-p04-pcam directory.
"""
from __future__ import annotations

import json
import sys
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

# Resolve paths so the benchmark modules are importable.
_BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from dashboard.utils.pcam_bridge import (
    build_session,
    run_single_retrieval,
    run_batch_evaluation,
    get_top_k_attractors,
    compute_spread_profile,
)
from dashboard.utils.experiment_log import ExperimentLog
from dashboard.utils.plot_helpers import (
    pattern_comparison_figure,
    difference_map_figure,
    precision_heatmap_figure,
    precision_bar_figure,
    similarity_distribution_figure,
    accuracy_by_noise_figure,
    top_k_attractor_figure,
    history_timeline_figure,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PCAM · Adaptive Precision Retrieval",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #0d1117;
    color: #e6edf3;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 12px 16px;
  }
  [data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 10px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e6edf3 !important;
    font-size: 22px !important;
    font-weight: 600;
  }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 11px !important;
  }

  /* Section headers */
  .section-label {
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
    margin-bottom: 12px;
    margin-top: 20px;
  }

  /* Status badges */
  .badge-ok   { background:#1a3a2a; color:#3fb950; border:1px solid #2ea043;
                padding:2px 8px; border-radius:4px; font-size:10px; }
  .badge-fail { background:#3a1a1a; color:#f85149; border:1px solid #da3633;
                padding:2px 8px; border-radius:4px; font-size:10px; }
  .badge-info { background:#1a2a3a; color:#58a6ff; border:1px solid #1f6feb;
                padding:2px 8px; border-radius:4px; font-size:10px; }

  /* Header */
  .main-title {
    font-size: 22px;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: -0.02em;
  }
  .sub-title {
    font-size: 11px;
    color: #8b949e;
    letter-spacing: 0.06em;
    margin-top: 2px;
  }

  /* Divider */
  hr { border-color: #21262d; }

  /* Expander */
  details summary { color: #8b949e; font-size: 11px; }

  /* Plotly chart border */
  .stPlotlyChart { border: 1px solid #21262d; border-radius: 6px; }

  /* Scrollable history table */
  .history-table { max-height: 220px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────

if "log" not in st.session_state:
    st.session_state.log = ExperimentLog()
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "session" not in st.session_state:
    st.session_state.session = None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="main-title">⬡ PCAM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Adaptive Precision Retrieval</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-label">System Configuration</div>',
                unsafe_allow_html=True)

    seed = st.selectbox("Random seed", [42, 101, 202, 303, 404], index=0)
    K = st.slider("Stored patterns (K)", 8, 32, 16, step=4)
    N = 64  # fixed by benchmark

    st.markdown('<div class="section-label">Query Parameters</div>',
                unsafe_allow_html=True)

    noise_level = st.slider("Noise level (mask fraction)", 0.1, 0.95, 0.70, step=0.05)
    query_idx = st.slider("Query pattern index", 0, K - 1, 0)
    noise_seed = st.slider("Noise seed", 0, 99, 0)
    top_k = st.slider("Top-K attractors", 3, min(K, 10), 5)

    st.markdown('<div class="section-label">Batch Evaluation</div>',
                unsafe_allow_html=True)

    batch_n = st.slider("Queries per noise level", 20, 200, 50, step=10)
    noise_levels_sel = st.multiselect(
        "Noise levels",
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        default=[0.5, 0.7, 0.8],
    )

    st.markdown("---")

    run_btn = st.button("▶  Run Retrieval", use_container_width=True, type="primary")
    batch_btn = st.button("⚡  Batch Evaluate", use_container_width=True)
    clear_btn = st.button("✕  Clear History", use_container_width=True)

    if clear_btn:
        st.session_state.log.clear()
        st.session_state.last_result = None
        st.session_state.batch_results = None
        st.rerun()


# ── Build / rebuild session when config changes ───────────────────────────────

session_key = (seed, K, N)
if (st.session_state.session is None or
        (st.session_state.session.seed, st.session_state.session.K,
         st.session_state.session.N) != session_key):
    with st.spinner("Initialising memory system..."):
        st.session_state.session = build_session(seed=seed, K=K, N=N)

session = st.session_state.session


# ── Run retrieval ─────────────────────────────────────────────────────────────

if run_btn:
    with st.spinner("Running adaptive retrieval..."):
        result = run_single_retrieval(
            session, query_idx, noise_level, noise_seed=noise_seed
        )
    st.session_state.last_result = result
    st.session_state.log.record(result, seed=seed)

if batch_btn and noise_levels_sel:
    with st.spinner(f"Evaluating {batch_n * len(noise_levels_sel)} queries..."):
        st.session_state.batch_results = run_batch_evaluation(
            session, noise_levels_sel, batch_n, batch_seed=seed
        )


# ── Header ────────────────────────────────────────────────────────────────────

col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown(
        '<div class="main-title">PCAM Adaptive Precision Retrieval System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">'
        'Precision-Controlled Associative Memory · NeurIPS 2026 Benchmark · P-04'
        '</div>',
        unsafe_allow_html=True,
    )

with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    n_log = len(st.session_state.log)
    st.markdown(
        f'<span class="badge-info">seed {seed}</span>&nbsp;'
        f'<span class="badge-info">K={K} N={N}</span>&nbsp;'
        f'<span class="badge-ok">{n_log} queries logged</span>',
        unsafe_allow_html=True,
    )

st.markdown("---")


# ── Main content ──────────────────────────────────────────────────────────────

result = st.session_state.last_result

if result is None:
    st.markdown(
        """
        <div style="text-align:center; padding:60px 0; color:#8b949e;">
          <div style="font-size:32px; margin-bottom:12px;">⬡</div>
          <div style="font-size:13px;">Configure parameters in the sidebar and click
          <strong style="color:#58a6ff;">▶ Run Retrieval</strong> to begin.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # ── Row 1: Status metrics ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Retrieval Status</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        label = "✓ CORRECT" if result.correct else "✗ WRONG"
        delta_str = "adaptive" if result.correct else None
        st.metric("Adaptive Result", label)

    with m2:
        b_label = "✓ CORRECT" if result.baseline_correct else "✗ WRONG"
        st.metric("Baseline Result", b_label)

    with m3:
        st.metric("True Pattern", f"#{result.true_idx}")

    with m4:
        st.metric("Retrieved", f"#{result.retrieved_idx}",
                  delta="match" if result.retrieved_idx == result.true_idx else "mismatch",
                  delta_color="normal" if result.retrieved_idx == result.true_idx else "inverse")

    with m5:
        st.metric("Noise Level", f"{result.noise_level:.0%}")

    with m6:
        pi = result.precision
        st.metric("Precision σ", f"{pi.std():.3f}",
                  delta=f"range [{pi.min():.2f}, {pi.max():.2f}]",
                  delta_color="off")

    # ── Row 2: Memory visualisation ───────────────────────────────────────────
    st.markdown('<div class="section-label">Memory Retrieval Visualisation</div>',
                unsafe_allow_html=True)

    st.plotly_chart(
        pattern_comparison_figure(
            result.true_pattern, result.query,
            result.retrieved, result.baseline_retrieved,
        ),
        use_container_width=True,
    )

    col_diff, col_prec_heat = st.columns([1, 1])
    with col_diff:
        st.markdown('<div class="section-label">Difference Maps</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            difference_map_figure(
                result.true_pattern, result.retrieved, result.baseline_retrieved
            ),
            use_container_width=True,
        )

    with col_prec_heat:
        st.markdown('<div class="section-label">Precision Heatmap (64-dim)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            precision_heatmap_figure(result.precision),
            use_container_width=True,
        )

    # ── Row 3: Precision bar + similarity distribution ────────────────────────
    col_bar, col_sim = st.columns([1, 1])

    with col_bar:
        st.markdown('<div class="section-label">Per-Dimension Precision Weights</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            precision_bar_figure(result.precision),
            use_container_width=True,
        )

    with col_sim:
        st.markdown('<div class="section-label">Attractor Similarity Distribution</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            similarity_distribution_figure(
                result.cosines_to_all, result.cosines_retrieved,
                result.true_idx, result.retrieved_idx,
            ),
            use_container_width=True,
        )

    # ── Row 4: Top-K attractors + precision stats ─────────────────────────────
    col_topk, col_stats = st.columns([1, 1])

    with col_topk:
        st.markdown('<div class="section-label">Top-K Nearest Attractors</div>',
                    unsafe_allow_html=True)
        top_k_data = get_top_k_attractors(session, result.query, k=top_k)
        st.plotly_chart(
            top_k_attractor_figure(top_k_data, result.true_idx),
            use_container_width=True,
        )

    with col_stats:
        st.markdown('<div class="section-label">Precision Analytics</div>',
                    unsafe_allow_html=True)
        pi = result.precision
        n_high = int((pi > pi.mean() * 1.1).sum())
        n_low = int((pi < pi.mean() * 0.9).sum())
        n_neutral = 64 - n_high - n_low

        pa, pb, pc = st.columns(3)
        pa.metric("High-confidence dims", n_high,
                  delta=f"{n_high/64:.0%}", delta_color="off")
        pb.metric("Low-confidence dims", n_low,
                  delta=f"{n_low/64:.0%}", delta_color="off")
        pc.metric("Neutral dims", n_neutral,
                  delta=f"{n_neutral/64:.0%}", delta_color="off")

        st.markdown("<br>", unsafe_allow_html=True)

        # Precision distribution histogram
        import plotly.graph_objects as go
        fig_hist = go.Figure(go.Histogram(
            x=pi, nbinsx=20,
            marker_color="#58a6ff",
            marker_line_width=0,
            opacity=0.8,
        ))
        fig_hist.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3", size=10),
            margin=dict(l=40, r=16, t=16, b=40),
            height=160,
            xaxis=dict(title="π value", gridcolor="#21262d", title_font_size=10),
            yaxis=dict(title="count", gridcolor="#21262d", title_font_size=10),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ── Batch evaluation panel ────────────────────────────────────────────────────

st.markdown("---")
st.markdown('<div class="section-label">Performance Panel — Batch Evaluation</div>',
            unsafe_allow_html=True)

batch_results = st.session_state.batch_results

if batch_results is None:
    st.markdown(
        '<span style="color:#8b949e; font-size:11px;">'
        'Click <strong style="color:#58a6ff;">⚡ Batch Evaluate</strong> '
        'to run multi-noise-level accuracy analysis.</span>',
        unsafe_allow_html=True,
    )
else:
    # Summary metrics
    all_agent = [v["agent_accuracy"] for v in batch_results.values()]
    all_base = [v["baseline_accuracy"] for v in batch_results.values()]
    all_delta = [v["delta"] for v in batch_results.values()]

    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("Mean Agent Accuracy", f"{np.mean(all_agent):.3f}")
    bm2.metric("Mean Baseline Accuracy", f"{np.mean(all_base):.3f}")
    bm3.metric("Mean Δ Improvement", f"{np.mean(all_delta):+.3f}",
               delta="above baseline" if np.mean(all_delta) > 0 else "below baseline",
               delta_color="normal" if np.mean(all_delta) > 0 else "inverse")
    bm4.metric("Min Δ (worst noise)", f"{min(all_delta):+.3f}",
               delta="no regression" if min(all_delta) >= 0 else "regression detected",
               delta_color="normal" if min(all_delta) >= 0 else "inverse")

    st.plotly_chart(
        accuracy_by_noise_figure(batch_results),
        use_container_width=True,
    )

    # Per-noise table
    with st.expander("Per-noise-level breakdown"):
        rows = []
        for lvl, v in sorted(batch_results.items()):
            rows.append({
                "Noise": f"{lvl:.0%}",
                "Agent Acc": f"{v['agent_accuracy']:.3f}",
                "Baseline Acc": f"{v['baseline_accuracy']:.3f}",
                "Δ": f"{v['delta']:+.3f}",
                "Queries": v["n_queries"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Experiment history ────────────────────────────────────────────────────────

st.markdown("---")
st.markdown('<div class="section-label">Experiment History</div>',
            unsafe_allow_html=True)

log = st.session_state.log
if len(log) == 0:
    st.markdown(
        '<span style="color:#8b949e; font-size:11px;">'
        'No queries logged yet.</span>',
        unsafe_allow_html=True,
    )
else:
    df_log = log.to_dataframe()
    summary = log.summary()

    hcol1, hcol2, hcol3 = st.columns(3)
    hcol1.metric("Session Accuracy (agent)", f"{summary['agent_accuracy']:.3f}")
    hcol2.metric("Session Accuracy (baseline)", f"{summary['baseline_accuracy']:.3f}")
    hcol3.metric("Session Δ", f"{summary['delta']:+.3f}",
                 delta_color="normal" if summary["delta"] > 0 else "inverse")

    st.plotly_chart(
        history_timeline_figure(df_log),
        use_container_width=True,
    )

    with st.expander("Full retrieval log"):
        display_cols = [
            "timestamp", "seed", "noise_level", "query_idx",
            "true_idx", "retrieved_idx", "correct", "baseline_correct",
            "precision_mean", "precision_std",
        ]
        st.dataframe(
            df_log[display_cols].tail(50),
            use_container_width=True,
            hide_index=True,
        )

    # Export
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button(
            label="⬇  Export log as JSON",
            data=log.to_json(),
            file_name=f"pcam_log_seed{seed}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_exp2:
        st.download_button(
            label="⬇  Export log as CSV",
            data=df_log.to_csv(index=False),
            file_name=f"pcam_log_seed{seed}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ── Benchmark summary card ────────────────────────────────────────────────────

st.markdown("---")
with st.expander("Benchmark Summary Card — Engine vs Π=I"):
    st.markdown("""
    | Metric | Engine (adaptive) | Baseline (Π=I) |
    |--------|-------------------|----------------|
    | Strategy | Corruption-aware boost | Uniform precision |
    | Precision vector | Non-uniform, query-dependent | Constant 1.0 |
    | Retrieval Δ (seeds 42,101) | +0.08 to +0.18 | — |
    | Retrieval score (max 70) | **70 / 70** | 0 |
    | Anisotropy score (max 20) | 0 / 20 | 0 |
    | Code quality (max 10) | (manual) | — |

    **Core insight:** PCAM dynamics inject the corrupted query as an external
    drive for `T_in` steps. Masked dimensions have zero drive, so the gradient
    term dominates and can pull the trajectory to the wrong attractor.
    Boosting precision on masked dimensions amplifies the attractor pull
    exactly where the external signal is absent.
    """)
