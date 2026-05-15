"""
Reusable Plotly figure factories for the PCAM dashboard.

All figures use a consistent dark research-lab theme.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Shared theme ──────────────────────────────────────────────────────────────

_BG = "#0d1117"
_SURFACE = "#161b22"
_BORDER = "#30363d"
_TEXT = "#e6edf3"
_MUTED = "#8b949e"
_ACCENT = "#58a6ff"
_GREEN = "#3fb950"
_RED = "#f85149"
_ORANGE = "#d29922"
_PURPLE = "#bc8cff"

_LAYOUT_BASE = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_SURFACE,
    font=dict(family="JetBrains Mono, monospace", color=_TEXT, size=11),
    margin=dict(l=48, r=24, t=40, b=40),
    xaxis=dict(gridcolor=_BORDER, zerolinecolor=_BORDER),
    yaxis=dict(gridcolor=_BORDER, zerolinecolor=_BORDER),
)


def _apply_theme(fig: go.Figure, **overrides) -> go.Figure:
    layout = {**_LAYOUT_BASE, **overrides}
    fig.update_layout(**layout)
    return fig


# ── Pattern / vector visualisations ──────────────────────────────────────────

def pattern_comparison_figure(
    true_pattern: np.ndarray,
    query: np.ndarray,
    retrieved: np.ndarray,
    baseline_retrieved: np.ndarray,
) -> go.Figure:
    """Four-panel comparison: stored | corrupted | retrieved | baseline."""
    N = len(true_pattern)
    side = int(np.ceil(np.sqrt(N)))
    pad = side * side - N

    def _pad(v):
        return np.concatenate([v, np.zeros(pad)]).reshape(side, side)

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=["Stored Pattern", "Corrupted Query",
                        "Adaptive Retrieval", "Baseline (Π=I)"],
        horizontal_spacing=0.04,
    )

    panels = [
        (_pad(true_pattern), "RdBu_r"),
        (_pad(query), "RdBu_r"),
        (_pad(retrieved / (np.linalg.norm(retrieved) + 1e-12)), "RdBu_r"),
        (_pad(baseline_retrieved / (np.linalg.norm(baseline_retrieved) + 1e-12)), "RdBu_r"),
    ]

    for col, (mat, cmap) in enumerate(panels, start=1):
        fig.add_trace(
            go.Heatmap(
                z=mat, colorscale=cmap, showscale=(col == 4),
                zmid=0, zmin=-1, zmax=1,
                colorbar=dict(
                    thickness=10, len=0.8,
                    tickfont=dict(color=_MUTED, size=9),
                    outlinecolor=_BORDER,
                ) if col == 4 else None,
            ),
            row=1, col=col,
        )

    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    _apply_theme(fig, height=220, title_text=None)
    fig.update_annotations(font=dict(color=_MUTED, size=10))
    return fig


def difference_map_figure(
    true_pattern: np.ndarray,
    retrieved: np.ndarray,
    baseline_retrieved: np.ndarray,
) -> go.Figure:
    """Difference maps: |retrieved - true| vs |baseline - true|."""
    N = len(true_pattern)
    side = int(np.ceil(np.sqrt(N)))
    pad = side * side - N

    r_norm = retrieved / (np.linalg.norm(retrieved) + 1e-12)
    b_norm = baseline_retrieved / (np.linalg.norm(baseline_retrieved) + 1e-12)

    def _pad(v):
        return np.concatenate([v, np.zeros(pad)]).reshape(side, side)

    diff_agent = np.abs(r_norm - true_pattern)
    diff_base = np.abs(b_norm - true_pattern)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Δ Adaptive", "Δ Baseline"],
        horizontal_spacing=0.06,
    )

    for col, mat in enumerate([diff_agent, diff_base], start=1):
        fig.add_trace(
            go.Heatmap(
                z=_pad(mat), colorscale="YlOrRd",
                showscale=(col == 2), zmin=0, zmax=1,
                colorbar=dict(
                    thickness=10, len=0.8,
                    tickfont=dict(color=_MUTED, size=9),
                    outlinecolor=_BORDER,
                ) if col == 2 else None,
            ),
            row=1, col=col,
        )

    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    _apply_theme(fig, height=200)
    fig.update_annotations(font=dict(color=_MUTED, size=10))
    return fig


def precision_heatmap_figure(precision: np.ndarray) -> go.Figure:
    """2-D heatmap of the 64-dimensional precision vector."""
    N = len(precision)
    side = int(np.ceil(np.sqrt(N)))
    pad = side * side - N
    mat = np.concatenate([precision, np.zeros(pad)]).reshape(side, side)

    fig = go.Figure(go.Heatmap(
        z=mat,
        colorscale=[
            [0.0, "#1a1f2e"],
            [0.3, "#1e3a5f"],
            [0.6, "#1565c0"],
            [0.8, "#42a5f5"],
            [1.0, "#e3f2fd"],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="π", font=dict(color=_MUTED, size=11)),
            thickness=12,
            tickfont=dict(color=_MUTED, size=9),
            outlinecolor=_BORDER,
        ),
    ))

    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    _apply_theme(fig, height=240, title_text=None)
    return fig


def precision_bar_figure(precision: np.ndarray) -> go.Figure:
    """Bar chart of precision values across all 64 dimensions."""
    N = len(precision)
    mean_pi = precision.mean()

    colors = [
        _GREEN if v > mean_pi * 1.1 else (_RED if v < mean_pi * 0.9 else _ACCENT)
        for v in precision
    ]

    fig = go.Figure(go.Bar(
        x=list(range(N)),
        y=precision,
        marker_color=colors,
        marker_line_width=0,
    ))

    fig.add_hline(
        y=mean_pi, line_dash="dot",
        line_color=_ORANGE, line_width=1,
        annotation_text=f"μ={mean_pi:.2f}",
        annotation_font=dict(color=_ORANGE, size=10),
    )

    fig.update_xaxes(title_text="Dimension", title_font=dict(color=_MUTED, size=10))
    fig.update_yaxes(title_text="π", title_font=dict(color=_MUTED, size=10))
    _apply_theme(fig, height=200)
    return fig


def similarity_distribution_figure(
    cosines_query: np.ndarray,
    cosines_retrieved: np.ndarray,
    true_idx: int,
    retrieved_idx: int,
) -> go.Figure:
    """Cosine similarity to all stored patterns: before vs after retrieval."""
    K = len(cosines_query)
    x = list(range(K))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=x, y=cosines_query,
        name="At query",
        marker_color=_MUTED,
        marker_line_width=0,
        opacity=0.6,
    ))

    fig.add_trace(go.Bar(
        x=x, y=cosines_retrieved,
        name="At retrieved",
        marker_color=[
            _GREEN if i == retrieved_idx else (_ACCENT if i == true_idx else _BORDER)
            for i in range(K)
        ],
        marker_line_width=0,
    ))

    fig.add_vline(
        x=true_idx, line_dash="dash",
        line_color=_GREEN, line_width=1.5,
        annotation_text="true",
        annotation_font=dict(color=_GREEN, size=9),
    )

    fig.update_xaxes(title_text="Pattern index", title_font=dict(color=_MUTED, size=10))
    fig.update_yaxes(title_text="Cosine similarity", title_font=dict(color=_MUTED, size=10))
    fig.update_layout(barmode="overlay", legend=dict(
        font=dict(color=_MUTED, size=9),
        bgcolor="rgba(0,0,0,0)",
    ))
    _apply_theme(fig, height=220)
    return fig


def accuracy_by_noise_figure(batch_results: dict[str, Any]) -> go.Figure:
    """Line chart of agent vs baseline accuracy across noise levels."""
    noise_levels = sorted(batch_results.keys())
    agent_acc = [batch_results[lvl]["agent_accuracy"] for lvl in noise_levels]
    base_acc = [batch_results[lvl]["baseline_accuracy"] for lvl in noise_levels]
    deltas = [batch_results[lvl]["delta"] for lvl in noise_levels]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Retrieval Accuracy", "Δ Improvement"],
        vertical_spacing=0.18,
        row_heights=[0.65, 0.35],
    )

    fig.add_trace(go.Scatter(
        x=noise_levels, y=agent_acc,
        mode="lines+markers",
        name="Adaptive (Engine)",
        line=dict(color=_ACCENT, width=2),
        marker=dict(size=7, color=_ACCENT),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=noise_levels, y=base_acc,
        mode="lines+markers",
        name="Baseline (Π=I)",
        line=dict(color=_MUTED, width=2, dash="dot"),
        marker=dict(size=7, color=_MUTED),
    ), row=1, col=1)

    bar_colors = [_GREEN if d > 0 else _RED for d in deltas]
    fig.add_trace(go.Bar(
        x=noise_levels, y=deltas,
        name="Δ",
        marker_color=bar_colors,
        marker_line_width=0,
        showlegend=False,
    ), row=2, col=1)

    fig.add_hline(y=0, line_color=_BORDER, line_width=1, row=2, col=1)

    fig.update_xaxes(title_text="Noise level (mask fraction)",
                     title_font=dict(color=_MUTED, size=10), row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", title_font=dict(color=_MUTED, size=10),
                     row=1, col=1)
    fig.update_yaxes(title_text="Δ acc", title_font=dict(color=_MUTED, size=10),
                     row=2, col=1)
    fig.update_layout(legend=dict(
        font=dict(color=_MUTED, size=9),
        bgcolor="rgba(0,0,0,0)",
    ))
    _apply_theme(fig, height=380)
    fig.update_annotations(font=dict(color=_MUTED, size=10))
    return fig


def top_k_attractor_figure(
    top_k: dict[str, Any],
    true_idx: int,
) -> go.Figure:
    """Horizontal bar chart of top-K attractor cosine similarities."""
    indices = top_k["indices"]
    cosines = top_k["cosines"]
    labels = [f"Pattern {i}" + (" ✓" if i == true_idx else "") for i in indices]
    colors = [_GREEN if i == true_idx else _ACCENT for i in indices]

    fig = go.Figure(go.Bar(
        x=cosines,
        y=labels,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
    ))

    fig.update_xaxes(title_text="Cosine similarity",
                     title_font=dict(color=_MUTED, size=10), range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    _apply_theme(fig, height=max(180, len(indices) * 36 + 60))
    return fig


def history_timeline_figure(df) -> go.Figure:
    """Scatter plot of retrieval outcomes over experiment history."""
    if df.empty:
        fig = go.Figure()
        _apply_theme(fig, height=180)
        return fig

    colors = [_GREEN if c else _RED for c in df["correct"]]

    fig = go.Figure(go.Scatter(
        x=list(range(len(df))),
        y=df["noise_level"],
        mode="markers",
        marker=dict(
            color=colors,
            size=8,
            symbol=["circle" if c else "x" for c in df["correct"]],
            line=dict(width=0),
        ),
        text=[
            f"Query {r['query_idx']} | noise={r['noise_level']:.2f} | "
            f"{'✓' if r['correct'] else '✗'}"
            for _, r in df.iterrows()
        ],
        hoverinfo="text",
    ))

    fig.update_xaxes(title_text="Query #", title_font=dict(color=_MUTED, size=10))
    fig.update_yaxes(title_text="Noise level", title_font=dict(color=_MUTED, size=10))
    _apply_theme(fig, height=200)
    return fig
