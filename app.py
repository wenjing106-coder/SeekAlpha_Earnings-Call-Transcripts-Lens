"""
app.py — Earnings Call Lens (Main Application 7.0)
===================================================
[Key Features]
1. Tab-based UI: "Sentiment Analysis" and "Topic Shift Analysis" as top-level tabs.
2. LLM-powered insights: DeepSeek V4-Flash generates financial conclusions and actionable
   advice for each analysis section (with local narrative engine fallback).
3. Sentiment tab: Cards, multi-metric chart, per-quarter implications, LLM conclusion.
4. Topic tab: Factor Risk Quadrant, Sankey Network, anomaly highlights, LLM conclusion.
5. Sidebar: Upload, metadata display, analyze button, clear button.

[How to Run]
    streamlit run app.py

[LLM Setup]
    Set DEEPSEEK_API_KEY in .streamlit/secrets.toml or as an environment variable.
    The app works fully without it (uses local narrative engine as fallback).
"""

# ===========================================================================
# Imports
# ===========================================================================
import gc
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from services.file_parser import parse_transcript, TranscriptMeta
from services.analysis_pipeline import run_full_analysis, AnalysisResult
from services.sentiment_engine import load_finbert
from services.topic_shift import load_embedding_model, compute_topic_shift
from services.llm_insights import (
    generate_sentiment_insight,
    generate_topic_insight,
    is_llm_available,
    LLMInsight,
)

# ===========================================================================
# App Version
# ===========================================================================
_APP_VERSION = "7.0"

# ===========================================================================
# Page Configuration
# ===========================================================================
st.set_page_config(
    page_title="Earnings Call Lens",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# Unified Color Palette & Custom CSS
# ===========================================================================
# Financial color constants used across all Plotly charts and
# HTML elements.
_COLOR_PRIMARY = "#1E3A5F"      # Deep navy (headings, primary elements)
_COLOR_POSITIVE = "#0D9488"    # Teal (positive signals)
_COLOR_NEGATIVE = "#BE123C"    # Crimson (negative/stress signals)
_COLOR_NEUTRAL = "#64748B"     # Slate (neutral/stable)
_COLOR_AMBER = "#D97706"       # Amber (hedge words / warning metric)
_COLOR_MUTED = "#94A3B8"       # Cool gray (background/dormant)
_COLOR_BG_LIGHT = "#F8FAFC"    # Near-white background

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A5F; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .sentiment-card {
        border-radius: 12px; padding: 1.2rem; text-align: center;
        margin-bottom: 0.5rem; border: 1px solid #E2E8F0;
    }
    .sentiment-card .quarter { font-size: 0.85rem; color: #64748B; margin-bottom: 0.3rem; }
    .sentiment-card .verdict { font-size: 1.3rem; font-weight: 700; }
    .sentiment-card .nss { font-size: 0.85rem; margin-top: 0.3rem; }
    .card-positive { background: #F0FDFA; border-left: 5px solid #0D9488; }
    .card-positive .verdict { color: #115E59; }
    .card-negative { background: #FFF1F2; border-left: 5px solid #BE123C; }
    .card-negative .verdict { color: #881337; }
    .card-neutral { background: #F8FAFC; border-left: 5px solid #64748B; }
    .card-neutral .verdict { color: #334155; }
    .alert-box {
        background: #FFF1F2; border: 1px solid #FECDD3; border-radius: 8px;
        padding: 1rem 1.5rem; margin: 1rem 0; border-left: 5px solid #BE123C;
    }
    .alert-box h4 { color: #881337; margin: 0 0 0.5rem 0; }
    .alert-box p { color: #881337; margin: 0; font-size: 0.95rem; line-height: 1.5; }
    .insight-box {
        background: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 8px;
        padding: 1rem 1.5rem; margin: 0.8rem 0; border-left: 5px solid #1E3A5F;
    }
    .insight-box h5 { color: #1E3A5F; margin: 0 0 0.5rem 0; font-size: 1rem; }
    .insight-box p { color: #334155; margin: 0; font-size: 0.9rem; line-height: 1.6; }
    .section-divider { border: none; border-top: 2px solid #E2E8F0; margin: 2rem 0; }
    .metric-card {
        background: #F8FAFC; border-radius: 10px; padding: 1rem 1.2rem;
        border-left: 4px solid #1E3A5F; margin-bottom: 0.8rem;
    }
    .metric-card h4 { margin: 0 0 0.3rem 0; color: #1E3A5F; font-size: 0.85rem; text-transform: uppercase; }
    .metric-card .value { font-size: 1.4rem; font-weight: 700; color: #1E293B; }
    .llm-insight {
        background: linear-gradient(135deg, #F0F9FF 0%, #F8FAFC 100%);
        border: 1px solid #BAE6FD; border-radius: 12px;
        padding: 1.5rem 2rem; margin: 1.5rem 0;
        border-left: 5px solid #0D9488;
    }
    .llm-insight h4 { color: #1E3A5F; margin: 0 0 0.8rem 0; font-size: 1.1rem; }
    .llm-insight .conclusion { color: #1E293B; font-size: 0.95rem; line-height: 1.7; margin-bottom: 1rem; }
    .llm-insight .advice-list { padding-left: 0; list-style: none; }
    .llm-insight .advice-list li {
        color: #334155; font-size: 0.9rem; line-height: 1.6;
        padding: 0.4rem 0 0.4rem 1.5rem; position: relative;
    }
    .llm-insight .advice-list li::before {
        content: "→"; position: absolute; left: 0; color: #0D9488; font-weight: 700;
    }
    .llm-insight .source-badge {
        display: inline-block; font-size: 0.7rem; color: #64748B;
        background: #E2E8F0; border-radius: 4px; padding: 0.15rem 0.5rem;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Model Loading — Memory-Efficient Sequential Loading
# ===========================================================================
# This sandbox has only ~1GB RAM. We CANNOT hold FinBERT (~440MB) and the
# embedding model (~90MB) in memory simultaneously without swap-thrashing.
# Strategy: load FinBERT -> run sentiment -> unload -> load embedder -> run topics.
# ===========================================================================

def _release_memory():
    """Force garbage collection to reclaim memory between model loads."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@st.cache_resource(show_spinner="Analyzing transcript …")
def _cached_analysis(
    _content_hash: str,
    company_name: str,
    ticker: str,
    year: str,
    quarter: str,
    raw_text: str,
    _tokenizer=None,
    _model=None,
) -> AnalysisResult:
    """Run full analysis on one transcript (cached by content hash).
    
    NOTE: Uses @st.cache_resource instead of @st.cache_data because
    AnalysisResult contains nested dataclass objects (SplitTranscript,
    TranscriptSentiment) that cannot be pickled by st.cache_data.
    cache_resource stores the object by reference without serialization.
    """
    return run_full_analysis(
        company_name=company_name,
        ticker=ticker,
        year=year,
        quarter=quarter,
        raw_text=raw_text,
        tokenizer=_tokenizer,
        model=_model,
    )


# ===========================================================================
# Helper: Render LLM Insight Box
# ===========================================================================

def _render_insight_box(insight: LLMInsight, title: str = "AI-Generated Insight"):
    """Render an LLM insight as a styled HTML box with analysis paragraph + recommendations."""
    advice_html = "".join(f"<li>{a}</li>" for a in insight.advice)
    source = f"Powered by {insight.model_used}" if insight.is_llm_generated else "Local narrative engine (enter DeepSeek API key for enhanced AI analysis)"

    st.markdown(
        f'<div class="llm-insight">'
        f'<h4>💡 {title}</h4>'
        f'<div class="conclusion">{insight.conclusion}</div>'
        f'<h5 style="color:#1E3A5F; margin: 1rem 0 0.5rem 0; font-size: 0.95rem;">📋 Forward-Looking Recommendations</h5>'
        f'<ul class="advice-list">{advice_html}</ul>'
        f'<span class="source-badge">{source}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# Helper: Sentiment Card HTML
# ===========================================================================

def _sentiment_card_html(quarter_label: str, pr_nss: float, qa_nss: float,
                         has_qa: bool, is_significant: bool) -> str:
    """Generate a sentiment card showing PR and QA NSS separately."""
    # Determine card color by the more concerning metric (QA if available)
    ref_nss = qa_nss if has_qa else pr_nss
    if ref_nss >= 0.05:
        css_class = "card-positive"
    elif ref_nss <= -0.05:
        css_class = "card-negative"
    else:
        css_class = "card-neutral"

    sig_marker = " ⚠️" if is_significant else ""

    qa_display = f"{qa_nss:+.3f}" if has_qa else "N/A"

    return (
        f'<div class="sentiment-card {css_class}">'
        f'<div class="quarter">{quarter_label}{sig_marker}</div>'
        f'<div class="nss" style="margin-top:0.4rem;font-size:0.9rem;">'
        f'<b>PR:</b> {pr_nss:+.3f}</div>'
        f'<div class="nss" style="font-size:0.9rem;">'
        f'<b>Q&A:</b> {qa_display}</div>'
        f'</div>'
    )


# ===========================================================================
# Visualization: Consolidated Sentiment Chart
# ===========================================================================

def _build_consolidated_chart(sorted_results: dict) -> go.Figure:
    """
    Build a single consolidated chart with 3 metrics across quarters:
    1. Delta NSS (with significance markers)
    2. Hedge Words Density (QA)
    3. Average Sentence Length (QA)
    """
    quarters = []
    deltas = []
    p_values = []
    hedge_densities_qa = []
    avg_lengths_qa = []
    significances = []

    for label, res in sorted_results.items():
        sent = res.sentiment
        quarters.append(res.quarter_label)
        deltas.append(sent.delta)
        p_values.append(sent.p_value)
        hedge_densities_qa.append(sent.qa.hedge_density if sent.qa.n_sentences > 0 else 0)
        avg_lengths_qa.append(sent.qa.avg_sentence_length if sent.qa.n_sentences > 0 else 0)
        significances.append(sent.is_significant)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Delta NSS (μ_PR − μ_QA) with Significance",
            "Q&A Hedge Words Density (per 1,000 words)",
            "Q&A Average Sentence Length (words)",
        ),
    )

    # --- Row 1: Delta NSS ---
    colors = [_COLOR_NEGATIVE if s else _COLOR_PRIMARY for s in significances]
    fig.add_trace(go.Scatter(
        x=quarters, y=deltas,
        mode="lines+markers+text",
        name="Delta NSS",
        line=dict(color=_COLOR_PRIMARY, width=2.5),
        marker=dict(size=12, color=colors, line=dict(width=2, color="white")),
        text=[f"{d:+.3f}{'*' if s else ''}" for d, s in zip(deltas, significances)],
        textposition="top center",
        textfont=dict(size=10),
    ), row=1, col=1)

    fig.add_hrect(y0=0.15, y1=max(max(deltas, default=0.3) + 0.05, 0.35),
                  fillcolor="rgba(190, 18, 60, 0.08)", line_width=0,
                  row=1, col=1)
    fig.add_hline(y=0.15, line_dash="dash", line_color=_COLOR_NEGATIVE, line_width=1,
                  annotation_text="Danger Zone", annotation_position="right",
                  row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=_COLOR_MUTED, line_width=1, row=1, col=1)

    # --- Row 2: Hedge Words Density ---
    fig.add_trace(go.Scatter(
        x=quarters, y=hedge_densities_qa,
        mode="lines+markers+text",
        name="Hedge Density (QA)",
        line=dict(color=_COLOR_AMBER, width=2.5),
        marker=dict(size=10, color=_COLOR_AMBER),
        text=[f"{h:.1f}" for h in hedge_densities_qa],
        textposition="top center",
        textfont=dict(size=10),
    ), row=2, col=1)

    # --- Row 3: Avg Sentence Length ---
    fig.add_trace(go.Scatter(
        x=quarters, y=avg_lengths_qa,
        mode="lines+markers+text",
        name="Avg Length (QA)",
        line=dict(color=_COLOR_POSITIVE, width=2.5),
        marker=dict(size=10, color=_COLOR_POSITIVE),
        text=[f"{l:.0f}" for l in avg_lengths_qa],
        textposition="top center",
        textfont=dict(size=10),
    ), row=3, col=1)

    fig.update_layout(
        height=650,
        showlegend=False,
        margin=dict(l=60, r=30, t=40, b=40),
        font=dict(size=11, family="Arial", color="#334155"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_yaxes(title_text="Delta NSS", row=1, col=1)
    fig.update_yaxes(title_text="Density/1000w", row=2, col=1)
    fig.update_yaxes(title_text="Words/Sentence", row=3, col=1)

    return fig


# ===========================================================================
# Visualization: Sankey Network
# ===========================================================================

def _build_sankey_network(topic_result) -> go.Figure:
    """
    Build a Multi-Quarter Step-Sankey Network — Clean & Readable Style.

    Design Principles:
    - Horizontal step-indexing: Q1→Q2, Q2→Q3, Q3→Q4 as sequential time slices
    - Top 5 topics by sentence volume for thick, readable pillars
    - Ribbon widths = raw sentence frequency (proportional to discussion volume)
    - UNIFORM neutral node color (slate) so labels remain readable
    - Only 3 ribbon colors: Teal (expansion), Crimson (contraction), Light gray (stable)
    - Large white text on dark nodes for maximum legibility
    """
    if not topic_result.quarter_distributions or len(topic_result.quarter_distributions) < 2:
        return go.Figure()

    quarters = [qd.quarter_label for qd in topic_result.quarter_distributions]

    # --- Reduce to top 5 macro-thematic pillars by total sentence volume ---
    topic_total_counts = {}
    for ti in topic_result.topics:
        total = sum(
            topic_result.topic_sentence_counts.get(f"{ti.topic_id}|{q}", 0)
            for q in quarters
        )
        topic_total_counts[ti.topic_id] = total

    # Select top 5 topics by volume
    top_topic_ids = sorted(topic_total_counts.keys(),
                           key=lambda tid: topic_total_counts[tid],
                           reverse=True)[:5]

    # Build short labels (title-cased, concise)
    topic_labels = {}
    for ti in topic_result.topics:
        if ti.topic_id in top_topic_ids:
            short_label = " / ".join(w.capitalize() for w in ti.keywords[:2])
            topic_labels[ti.topic_id] = short_label

    # --- Uniform node color for readability (dark slate) ---
    NODE_COLOR = "#334155"  # Slate-700: dark enough for white text contrast

    # --- Build Sankey nodes: one node per (topic, quarter) ---
    nodes_label = []
    nodes_color = []
    node_index_map = {}  # (tid, quarter_idx) -> node_index

    for q_idx, quarter in enumerate(quarters):
        for pillar_idx, tid in enumerate(top_topic_ids):
            node_key = (tid, q_idx)
            node_index_map[node_key] = len(nodes_label)
            label = topic_labels.get(tid, f"Theme {tid}")
            nodes_label.append(label)
            nodes_color.append(NODE_COLOR)

    # --- Build Sankey links: Step-indexed (Q_t → Q_{t+1} only) ---
    sources = []
    targets = []
    values = []
    link_colors = []

    for q_idx in range(len(quarters) - 1):
        q_current = quarters[q_idx]
        q_next = quarters[q_idx + 1]

        for pillar_idx, tid in enumerate(top_topic_ids):
            src_key = (tid, q_idx)
            tgt_key = (tid, q_idx + 1)

            if src_key not in node_index_map or tgt_key not in node_index_map:
                continue

            # Raw sentence count for flow width
            count_current = topic_result.topic_sentence_counts.get(f"{tid}|{q_current}", 0)
            count_next = topic_result.topic_sentence_counts.get(f"{tid}|{q_next}", 0)
            flow_value = max((count_current + count_next) // 2, 1)

            # QoQ growth rate — use RAW SENTENCE COUNT (not proportion)
            if count_current >= 3:
                growth_rate = (count_next - count_current) / count_current
            elif count_next >= 5:
                growth_rate = 9.99  # New topic emergence
            else:
                growth_rate = 0.0

            # Only 3 ribbon colors — signal-based, not category-based
            if growth_rate > 0.50:
                link_color = "rgba(13, 148, 136, 0.65)"   # Teal: expansion
            elif growth_rate < -0.40:
                link_color = "rgba(190, 18, 60, 0.60)"    # Crimson: contraction
            else:
                link_color = "rgba(203, 213, 225, 0.40)"  # Neutral: stable

            sources.append(node_index_map[src_key])
            targets.append(node_index_map[tgt_key])
            values.append(flow_value)
            link_colors.append(link_color)

    # --- Construct figure ---
    n_nodes_per_quarter = len(top_topic_ids)
    chart_height = max(480, n_nodes_per_quarter * 100 + 130)

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=45,
            thickness=26,
            line=dict(color="#1E293B", width=0.5),
            label=nodes_label,
            color=nodes_color,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    )])

    # Quarter period labels at the top
    n_quarters = len(quarters)
    for q_idx, quarter in enumerate(quarters):
        x_pos = q_idx / max(n_quarters - 1, 1)
        fig.add_annotation(
            x=x_pos, y=1.08,
            xref="paper", yref="paper",
            text=f"<b>{quarter}</b>",
            showarrow=False,
            font=dict(size=13, color="#1E293B", family="Arial"),
            xanchor="center",
        )

    # Simple legend at bottom — use colored HTML spans for visibility
    fig.add_annotation(
        x=0.5, y=-0.08, xref="paper", yref="paper",
        text=(
            '<span style="color:#0D9488;font-weight:bold">\u2501\u2501</span> Expansion (&gt;50% growth)     '
            '<span style="color:#BE123C;font-weight:bold">\u2501\u2501</span> Contraction (&gt;40% decline)     '
            '<span style="color:#CBD5E1;font-weight:bold">\u2501\u2501</span> Stable'
        ),
        showarrow=False,
        font=dict(size=11, color="#334155", family="Arial"),
        xanchor="center",
    )

    fig.update_layout(
        height=chart_height,
        margin=dict(l=10, r=10, t=60, b=55),
        font=dict(size=13, color="white", family="Arial"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


# ===========================================================================
# Visualization: Factor Risk Quadrant
# ===========================================================================

def _build_risk_quadrant(topic_result, sorted_results) -> go.Figure:
    """
    Factor Risk Quadrant — Professional Financial Bubble Chart.

    X-Axis: Attention Delta (Δ%) — QoQ discussion volume growth
    Y-Axis: Q&A Net Sentiment Score (NSS) — analyst-facing tone
    Bubble Size: Factor Score = |ΔProportion| × (1 − NSS)

    Quadrant System (4 distinct zones):
    - I: Momentum (High Δ + Positive) — Teal
    - II: Recovery (Low Δ + Positive) — Slate
    - III: Dormant (Low Δ + Negative) — Cool Gray
    - IV: Stress (High Δ + Negative) — Crimson
    """
    if not topic_result.qoq_deltas:
        return go.Figure()

    # Build NSS map per quarter
    quarter_nss_map = {}
    for label, res in sorted_results.items():
        sent = res.sentiment
        qa_nss = sent.qa.mean_nss if sent.qa.n_sentences > 0 else 0.0
        quarter_nss_map[res.quarter_label] = qa_nss

    # Build topic label map (title-cased)
    topic_label_map = {
        ti.topic_id: " · ".join(w.capitalize() for w in ti.keywords[:3])
        for ti in topic_result.topics
    }

    # Collect data points by quadrant for grouped legend
    quadrant_data = {
        "momentum": {"x": [], "y": [], "sizes": [], "labels": [], "texts": []},
        "stress": {"x": [], "y": [], "sizes": [], "labels": [], "texts": []},
        "recovery": {"x": [], "y": [], "sizes": [], "labels": [], "texts": []},
        "dormant": {"x": [], "y": [], "sizes": [], "labels": [], "texts": []},
    }

    # Thresholds
    DELTA_THRESHOLD = 30  # 30% growth = high attention
    NSS_THRESHOLD = -0.05  # Below this = negative sentiment

    for tid, deltas in topic_result.qoq_deltas.items():
        topic_label = topic_label_map.get(tid, f"Theme {tid}")
        for quarter, delta_pct in deltas:
            if abs(delta_pct) < 0.03:
                continue

            nss = quarter_nss_map.get(quarter, 0.0)
            factor_score = abs(delta_pct) * (1.0 - nss)
            delta_display = delta_pct * 100
            delta_display_capped = max(min(delta_display, 1200), -100)

            hover = (
                f"<b>{topic_label}</b><br>"
                f"Period: {quarter}<br>"
                f"Attention Δ: {delta_display:+.0f}%<br>"
                f"Q&A NSS: {nss:+.3f}<br>"
                f"Factor Score: {factor_score:.2f}"
            )
            # Show label for meaningful bubbles (factor > 0.8)
            if factor_score > 0.8:
                q_parts = quarter.split()
                q_short = quarter
                for p in q_parts:
                    if p.upper().startswith("Q"):
                        yr = [x for x in q_parts if x.isdigit() and len(x) == 4]
                        q_short = f"{p}'{yr[0][2:]}" if yr else p
                        break
                text_label = f"{topic_label} ({q_short})"
            else:
                text_label = ""
            bubble_size = max(factor_score * 9, 6)

            # Classify into quadrant
            if delta_display > DELTA_THRESHOLD and nss >= NSS_THRESHOLD:
                q = "momentum"
            elif delta_display > DELTA_THRESHOLD and nss < NSS_THRESHOLD:
                q = "stress"
            elif delta_display <= DELTA_THRESHOLD and nss >= NSS_THRESHOLD:
                q = "recovery"
            else:
                q = "dormant"

            quadrant_data[q]["x"].append(delta_display_capped)
            quadrant_data[q]["y"].append(nss)
            quadrant_data[q]["sizes"].append(bubble_size)
            quadrant_data[q]["labels"].append(hover)
            quadrant_data[q]["texts"].append(text_label)

    # Check if we have any data
    total_points = sum(len(d["x"]) for d in quadrant_data.values())
    if total_points == 0:
        return go.Figure()

    fig = go.Figure()

    # Quadrant style definitions
    _QUADRANT_STYLES = {
        "momentum": {"name": "I: Momentum", "color": "#0D9488", "marker_opacity": 0.82},
        "stress": {"name": "IV: Stress", "color": "#BE123C", "marker_opacity": 0.82},
        "recovery": {"name": "II: Recovery", "color": "#64748B", "marker_opacity": 0.60},
        "dormant": {"name": "III: Dormant", "color": "#94A3B8", "marker_opacity": 0.40},
    }

    # Add traces per quadrant
    for q_key, style in _QUADRANT_STYLES.items():
        data = quadrant_data[q_key]
        if not data["x"]:
            continue

        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers+text",
            name=style["name"],
            marker=dict(
                size=data["sizes"],
                color=style["color"],
                opacity=style["marker_opacity"],
                line=dict(width=1.2, color="rgba(255,255,255,0.7)"),
            ),
            text=data["texts"],
            textposition="top center",
            textfont=dict(size=10, color="#1E293B", family="Arial"),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=data["labels"],
            legendgroup=q_key,
        ))

    # Reference lines
    fig.add_hline(y=NSS_THRESHOLD, line_dash="dash", line_color="#CBD5E1", line_width=1.2,
                  annotation_text="Sentiment Neutral Line",
                  annotation_position="bottom left",
                  annotation_font=dict(size=8, color="#94A3B8"))
    fig.add_vline(x=DELTA_THRESHOLD, line_dash="dash", line_color="#CBD5E1", line_width=1.2,
                  annotation_text="High Attention Threshold",
                  annotation_position="top right",
                  annotation_font=dict(size=8, color="#94A3B8"))

    # Quadrant label annotations
    fig.add_annotation(x=0.97, y=0.03, xref="paper", yref="paper",
                       text="<b>STRESS ZONE</b><br><i>High attention + negative</i>",
                       showarrow=False, font=dict(size=9, color="#BE123C"),
                       xanchor="right", yanchor="bottom",
                       bgcolor="rgba(254, 226, 226, 0.4)", borderpad=4)
    fig.add_annotation(x=0.97, y=0.97, xref="paper", yref="paper",
                       text="<b>MOMENTUM ZONE</b><br><i>High attention + positive</i>",
                       showarrow=False, font=dict(size=9, color="#0D9488"),
                       xanchor="right", yanchor="top",
                       bgcolor="rgba(204, 251, 241, 0.4)", borderpad=4)
    fig.add_annotation(x=0.02, y=0.97, xref="paper", yref="paper",
                       text="<b>RECOVERY</b>",
                       showarrow=False, font=dict(size=8, color="#64748B"),
                       xanchor="left", yanchor="top")
    fig.add_annotation(x=0.02, y=0.03, xref="paper", yref="paper",
                       text="<b>DORMANT</b>",
                       showarrow=False, font=dict(size=8, color="#94A3B8"),
                       xanchor="left", yanchor="bottom")

    # Layout
    fig.update_layout(
        height=520,
        xaxis=dict(
            title=dict(text="Attention Delta (Δ%) — QoQ Volume Growth",
                       font=dict(size=11, color="#334155")),
            gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#E2E8F0",
            showspikes=True, spikecolor="#CBD5E1", spikemode="across", spikethickness=0.5,
        ),
        yaxis=dict(
            title=dict(text="Q&A Net Sentiment Score (NSS)",
                       font=dict(size=11, color="#334155")),
            gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#E2E8F0",
        ),
        margin=dict(l=65, r=20, t=30, b=65),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0,
                    font=dict(size=10, color="#475569"),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="#E2E8F0", borderwidth=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=10, color="#334155"),
    )

    return fig


# ===========================================================================
# Helper: Anomaly Matrix Builder
# ===========================================================================

def _build_anomaly_matrix(topic_result, sorted_results) -> list:
    """
    Cross-Factor Anomaly Matrix Ranking.
    Factor Score = |Delta Proportion| x (1 - NSS)
    Returns: List of dicts sorted by factor score (highest first).
    """
    if not topic_result.qoq_deltas:
        return []

    quarter_nss_map = {}
    for label, res in sorted_results.items():
        sent = res.sentiment
        qa_nss = sent.qa.mean_nss if sent.qa.n_sentences > 0 else 0.0
        quarter_nss_map[res.quarter_label] = qa_nss

    topic_label_map = {
        ti.topic_id: ", ".join(ti.keywords[:3])
        for ti in topic_result.topics
    }

    anomalies = []
    for tid, deltas in topic_result.qoq_deltas.items():
        label = topic_label_map.get(tid, f"Topic {tid}")
        for quarter, delta_pct in deltas:
            if abs(delta_pct) < 0.05:
                continue

            nss = quarter_nss_map.get(quarter, 0.0)
            factor_score = abs(delta_pct) * (1.0 - nss)

            if factor_score > 3.0:
                severity, icon = "EXTREME", "\U0001f6a8"
            elif factor_score > 1.5:
                severity, icon = "HIGH", "\u26a0\ufe0f"
            elif factor_score > 0.5:
                severity, icon = "MODERATE", "\U0001f4ca"
            else:
                severity, icon = "LOW", "\u2139\ufe0f"

            anomalies.append({
                "icon": icon, "severity": severity, "quarter": quarter,
                "topic_id": tid, "topic_label": label,
                "delta_pct": delta_pct, "nss": nss, "factor_score": factor_score,
            })

    anomalies.sort(key=lambda x: x["factor_score"], reverse=True)
    return anomalies


# ===========================================================================
# Helper: Per-Quarter Implication Text
# ===========================================================================

def _generate_implication(quarter_label: str, delta: float, p_value: float,
                          hedge_density_qa: float, hedge_density_pr: float,
                          is_significant: bool) -> str:
    """Generate an actionable insight/implication for a quarter."""
    if not is_significant:
        return (
            f'<div class="insight-box">'
            f'<h5>{quarter_label} — Consistent Sentiment</h5>'
            f'<p><strong>Characteristics:</strong> Delta NSS = {delta:+.3f} (p = {p_value:.4f}). '
            f'Hedge word density balanced: PR ({hedge_density_pr:.1f}/1000) vs '
            f'Q&A ({hedge_density_qa:.1f}/1000).</p>'
            f'<p><strong>Implication:</strong> Management tone remains stable between scripted '
            f'remarks and unscripted Q&A. No evidence of narrative divergence.</p>'
            f'</div>'
        )

    if delta > 0.15:
        severity = "sharply" if delta > 0.25 else "notably"
        return (
            f'<div class="insight-box">'
            f'<h5>{quarter_label} — "Sentiment Collapse" Signal ⚠️</h5>'
            f'<p><strong>Characteristics:</strong> Delta {severity} spikes to {delta:+.3f} '
            f'(p = {p_value:.4f}), Q&A hedge density at {hedge_density_qa:.1f}/1000 words '
            f'vs PR at {hedge_density_pr:.1f}/1000.</p>'
            f'<p><strong>Implication:</strong> Management attempted to maintain optimism '
            f'in prepared remarks but analyst questioning exposed negativity. '
            f'This is a textbook pre-warning sign — exercise caution.</p>'
            f'</div>'
        )
    elif delta < -0.15:
        return (
            f'<div class="insight-box">'
            f'<h5>{quarter_label} — "Defensive Opener" Signal</h5>'
            f'<p><strong>Characteristics:</strong> Delta is negative at {delta:+.3f} '
            f'(p = {p_value:.4f}). Prepared Remarks are more negative than Q&A.</p>'
            f'<p><strong>Implication:</strong> Management front-loaded bad news, then showed '
            f'more confidence during Q&A. Can be a positive sign of transparent communication '
            f'— but verify underlying issues are resolved.</p>'
            f'</div>'
        )
    else:
        return (
            f'<div class="insight-box">'
            f'<h5>{quarter_label} — Moderate Divergence Signal</h5>'
            f'<p><strong>Characteristics:</strong> Delta = {delta:+.3f} (p = {p_value:.4f}). '
            f'Statistically significant but moderate magnitude.</p>'
            f'<p><strong>Implication:</strong> A measurable sentiment gap exists. '
            f'Monitor subsequent quarters for escalation patterns.</p>'
            f'</div>'
        )


# ===========================================================================
# Main Application Entry Point
# ===========================================================================
def main():
    """Main application function."""

    st.markdown('<div class="main-header">📊 Earnings Call Lens</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Detect sentiment shifts, hidden signals &amp; topic evolution across quarterly earnings calls.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ==================================================================
    # Sidebar
    # ==================================================================
    with st.sidebar:
        st.header("📁 Upload Transcripts")
        st.caption("Upload up to 4 quarterly earnings-call transcripts (TXT).")

        uploader_key = f"file_uploader_{st.session_state.get('clear_counter', 0)}"
        uploaded_files = st.file_uploader(
            "Choose TXT files", type=["txt"],
            accept_multiple_files=True, key=uploader_key,
        )

        if uploaded_files and len(uploaded_files) > 4:
            st.warning("Maximum 4 files. Only the first 4 will be used.")
            uploaded_files = uploaded_files[:4]

        parsed: dict[str, TranscriptMeta] = {}
        if uploaded_files:
            for uf in uploaded_files:
                meta = parse_transcript(uf.getvalue(), uf.name)
                label = f"{meta.ticker} {meta.quarter} {meta.year}"
                parsed[label] = meta

            def _sort_key(lbl):
                m = parsed[lbl]
                try:
                    y = int(m.year)
                except (ValueError, TypeError):
                    y = 9999
                q_num = 0
                if m.quarter and m.quarter.upper().startswith("Q"):
                    try:
                        q_num = int(m.quarter[1])
                    except (ValueError, IndexError):
                        q_num = 0
                return y * 10 + q_num

            sorted_labels = sorted(parsed.keys(), key=_sort_key)

            st.markdown("---")
            st.subheader("📋 Detected (Oldest → Newest)")
            for lbl in sorted_labels:
                m = parsed[lbl]
                st.markdown(f"**{m.company_name}** (`{m.ticker}`) — {m.quarter} {m.year}")

            st.markdown("---")
            selected_labels = st.multiselect(
                "Select files to analyze", options=sorted_labels, default=sorted_labels,
            )

            st.markdown("---")
            analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
        else:
            selected_labels = []
            analyze_clicked = False

        # --- Clear All Button ---
        st.markdown("---")
        if st.button("🗑️ Clear All", use_container_width=True, type="secondary"):
            keys_to_clear = [
                "analysis_results", "parsed", "selected_labels",
                "topic_result", "sentiment_insight", "topic_insight",
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            if "clear_counter" not in st.session_state:
                st.session_state["clear_counter"] = 0
            st.session_state["clear_counter"] += 1
            st.rerun()

        # --- DeepSeek API Key Input ---
        st.markdown("---")
        st.subheader("🔑 DeepSeek API Key")
        st.caption(
            "Enter your DeepSeek API key to enable AI-powered insights. "
            "Get your key at: [platform.deepseek.com](https://platform.deepseek.com/api_keys)"
        )
        deepseek_key_input = st.text_input(
            "DEEPSEEK_API_KEY",
            type="password",
            value=st.session_state.get("deepseek_api_key_input", ""),
            placeholder="sk-...",
            help="Paste your DeepSeek API key. Used for AI-generated conclusions & recommendations.",
        )
        if deepseek_key_input:
            st.session_state["deepseek_api_key_input"] = deepseek_key_input
            # Clear cached insights when key changes so they regenerate with LLM
            if st.session_state.get("_last_token") != deepseek_key_input:
                st.session_state["_last_token"] = deepseek_key_input
                if "sentiment_insight" in st.session_state:
                    del st.session_state["sentiment_insight"]
                if "topic_insight" in st.session_state:
                    del st.session_state["topic_insight"]

        # --- LLM Status Indicator ---
        st.markdown("---")
        if is_llm_available():
            if deepseek_key_input:
                st.success("🤖 AI Insights: **DeepSeek V4-Flash enabled**", icon="✅")
            else:
                st.info(
                    "🤖 AI Insights: **Local Narrative Engine**\n\n"
                    "Enter your DeepSeek API key above for enhanced AI analysis.",
                    icon="ℹ️",
                )

    # ==================================================================
    # Empty State
    # ==================================================================
    if not uploaded_files:
        if "analysis_results" in st.session_state:
            del st.session_state["analysis_results"]
        if "topic_result" in st.session_state:
            del st.session_state["topic_result"]

        st.info("👈 Upload earnings-call transcripts (.txt) in the sidebar to get started.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<div class="metric-card"><h4>Sentiment Shift Detection</h4>'
                '<div class="value" style="font-size:1rem">'
                'Generate sentiment scores; Measure Hedge Words and Length Mutation for Q&A'
                '</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(
                '<div class="metric-card"><h4>Topic Evolution</h4>'
                '<div class="value" style="font-size:1rem">'
                'Detect top trending themes for the company 🔥'
                '</div></div>', unsafe_allow_html=True)
        return

    if not analyze_clicked and "analysis_results" not in st.session_state:
        st.info("📋 Files detected. Click **🔍 Analyze** in the sidebar to start.")
        return

    # ==================================================================
    # Run Analysis
    # ==================================================================
    if analyze_clicked and selected_labels:
        results: dict[str, AnalysisResult] = {}
        progress_bar = st.progress(0, text="Loading FinBERT sentiment model …")
        total = len(selected_labels)

        # Step 1: Load FinBERT, run sentiment analysis on all transcripts
        tokenizer, model = load_finbert()
        for idx, label in enumerate(selected_labels):
            meta = parsed[label]
            progress_bar.progress(idx / total, text=f"Analyzing {meta.ticker} {meta.quarter} {meta.year} …")
            content_hash = str(hash(meta.raw_text))
            result = _cached_analysis(
                _content_hash=content_hash,
                company_name=meta.company_name,
                ticker=meta.ticker,
                year=meta.year,
                quarter=meta.quarter,
                raw_text=meta.raw_text,
                _tokenizer=tokenizer,
                _model=model,
            )
            results[label] = result

        # Step 2: Release FinBERT from memory before topic modeling
        del tokenizer, model
        _release_memory()

        # Step 3: Run topic modeling (if 2+ transcripts)
        if len(selected_labels) >= 2:
            progress_bar.progress(0.85, text="Running topic modeling …")
            embed_model = load_embedding_model()
            prepared_texts = {}
            qa_texts = {}
            for label, res in results.items():
                prepared_texts[res.quarter_label] = res.split.prepared_remarks
                qa_texts[res.quarter_label] = res.split.qa_section

            topic_result = compute_topic_shift(
                prepared_texts, qa_texts, embed_model, max_topics=12
            )
            st.session_state["topic_result"] = topic_result
            del embed_model
            _release_memory()

        progress_bar.progress(1.0, text="Analysis complete ✓")
        st.session_state["analysis_results"] = results
        st.session_state["parsed"] = parsed
        st.session_state["selected_labels"] = selected_labels
        # Clear cached insights so they regenerate with new data
        if "sentiment_insight" in st.session_state:
            del st.session_state["sentiment_insight"]
        if "topic_insight" in st.session_state:
            del st.session_state["topic_insight"]

    if "analysis_results" not in st.session_state:
        return

    results = st.session_state["analysis_results"]
    parsed = st.session_state.get("parsed", parsed)
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1].year_sort_key))
    n_results = len(sorted_results)

    # ==================================================================
    # TOP-LEVEL TABS: Sentiment Analysis | Topic Shift Analysis
    # ==================================================================
    tab_sentiment, tab_topic, tab_transcripts = st.tabs([
        "📈 Sentiment Analysis",
        "🔄 Topic Shift Analysis",
        "📄 Raw Transcripts",
    ])

    # ==================================================================
    # TAB 1: Sentiment Analysis
    # ==================================================================
    with tab_sentiment:
        # --- Section 1: Sentiment Cards ---
        st.subheader("🎯 Sentiment Overview")

        # NSS explanation for non-finance users
        with st.expander("ℹ️ What is NSS? (click to learn)", expanded=False):
            st.markdown("""
**Net Sentiment Score (NSS)** measures how positive or negative the language is in an earnings call.

**How it works — simple version:**
- Every sentence in the transcript is scored by an AI model (FinBERT) trained specifically on financial text
- Each sentence gets a probability of being **Positive**, **Negative**, or **Neutral**
- **NSS = P(Positive) − P(Negative)** for each sentence, then averaged across all sentences

**How to read the numbers:**
| NSS Range | Meaning |
|-----------|---------|
| +0.5 to +1.0 | Very positive language (confident, optimistic) |
| +0.1 to +0.5 | Mildly positive (typical for corporate communications) |
| −0.1 to +0.1 | Neutral (factual, balanced) |
| −0.5 to −0.1 | Mildly negative (cautious, hedging) |
| −1.0 to −0.5 | Very negative (distress, concern) |

**Why we show PR and Q&A separately:**
- **PR (Prepared Remarks):** The scripted part — management controls the narrative
- **Q&A:** Unscripted analyst questions — harder to hide problems
- A large gap between PR and Q&A suggests management may be more optimistic in their script than reality warrants

**⚠️ marker** = statistically significant difference between PR and Q&A tone (p < 0.05)
""")

        cols = st.columns(min(n_results, 4))
        for idx, (label, res) in enumerate(sorted_results.items()):
            sent = res.sentiment
            has_qa = sent.qa.n_sentences > 0
            with cols[idx % 4]:
                st.markdown(
                    _sentiment_card_html(
                        res.quarter_label,
                        pr_nss=sent.prepared.mean_nss,
                        qa_nss=sent.qa.mean_nss if has_qa else 0.0,
                        has_qa=has_qa,
                        is_significant=sent.is_significant,
                    ),
                    unsafe_allow_html=True,
                )

        # --- Section 2: Consolidated Chart ---
        if n_results >= 2:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.subheader("📈 Sentiment Shift Across Quarters")
            st.caption(
                "Delta NSS = μ_PR − μ_QA | Red markers = statistically significant (p < 0.05) | "
                "* = significant shift"
            )
            fig_consolidated = _build_consolidated_chart(sorted_results)
            st.plotly_chart(fig_consolidated, use_container_width=True)

        # --- Section 3: LLM-Generated Sentiment Conclusion ---
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("💡 Analysis & Forward-Looking Recommendations")

        if "sentiment_insight" in st.session_state:
            _render_insight_box(
                st.session_state["sentiment_insight"],
                title="Sentiment Analysis — Conclusion & Recommendations",
            )
        else:
            st.info("Click the button below to generate AI-powered analysis and recommendations.")
            if st.button("🤖 Get AI Advice — Sentiment", key="btn_sentiment_llm", type="primary"):
                with st.spinner("Generating sentiment insight …"):
                    st.session_state["sentiment_insight"] = generate_sentiment_insight(sorted_results)
                    st.rerun()

        # --- Section 5: Detailed Metrics (expandable) ---
        with st.expander("📊 Detailed Sentiment Metrics & Raw Data", expanded=False):
            metrics_data = []
            for label, res in sorted_results.items():
                sent = res.sentiment
                metrics_data.append({
                    "Quarter": res.quarter_label,
                    "PR NSS": f"{sent.prepared.mean_nss:+.4f}",
                    "QA NSS": f"{sent.qa.mean_nss:+.4f}" if sent.qa.n_sentences > 0 else "N/A",
                    "Delta": f"{sent.delta:+.4f}",
                    "t-stat": f"{sent.t_statistic:.3f}",
                    "p-value": f"{sent.p_value:.4f}",
                    "Significant": "⚠️ Yes" if sent.is_significant else "✅ No",
                    "PR Hedge/1000w": f"{sent.prepared.hedge_density:.1f}",
                    "QA Hedge/1000w": f"{sent.qa.hedge_density:.1f}" if sent.qa.n_sentences > 0 else "N/A",
                    "PR Avg Len": f"{sent.prepared.avg_sentence_length:.0f}",
                    "QA Avg Len": f"{sent.qa.avg_sentence_length:.0f}" if sent.qa.n_sentences > 0 else "N/A",
                })
            st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)

    # ==================================================================
    # TAB 2: Topic Shift Analysis
    # ==================================================================
    with tab_topic:
        if n_results < 2:
            st.info("Topic analysis requires at least **2 transcripts**. Upload more to enable.")
        else:
            if "topic_result" not in st.session_state:
                st.warning("Topic analysis not yet available. Please re-run analysis.")
            else:
                topic_result = st.session_state["topic_result"]

                if topic_result.n_topics == 0:
                    st.warning("Not enough text data to discover meaningful topics.")
                else:
                    st.success(
                        f"Discovered **{topic_result.n_topics} macro-themes** from "
                        f"{topic_result.n_documents} text segments "
                        f"(POS-pruned keywords: PROPN/NOUN only)."
                    )

                    # --- Factor Risk Quadrant ---
                    anomalies = _build_anomaly_matrix(topic_result, sorted_results)
                    if anomalies:
                        st.markdown("#### 🎯 Factor Risk Quadrant")

                        fig_quadrant = _build_risk_quadrant(topic_result, sorted_results)
                        st.plotly_chart(fig_quadrant, use_container_width=True)

                    # --- Sankey Network ---
                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                    st.markdown("#### 🔀 Attention Flow — Multi-Quarter Sankey Network")
                    st.caption(
                        "Top 5 macro-themes by discussion volume. "
                        "Ribbon width ∝ sentence count. "
                        "**Teal** = expansion (>50% QoQ growth) | "
                        "**Red** = contraction (>40% decline) | "
                        "**Gray** = stable."
                    )

                    # User-adjustable chart height for readability
                    sankey_height = st.slider(
                        "Chart height (drag to zoom in/out)",
                        min_value=400, max_value=1000, value=600, step=50,
                        key="sankey_height_slider",
                    )

                    fig_sankey = _build_sankey_network(topic_result)
                    fig_sankey.update_layout(height=sankey_height)
                    st.plotly_chart(fig_sankey, use_container_width=True)

                    # --- LLM-Generated Topic Conclusion ---
                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                    st.subheader("💡 Analysis & Forward-Looking Recommendations")

                    if "topic_insight" in st.session_state:
                        _render_insight_box(
                            st.session_state["topic_insight"],
                            title="Topic Shift Analysis — Conclusion & Recommendations",
                        )
                    else:
                        st.info("Click the button below to generate AI-powered analysis and recommendations.")
                        if st.button("🤖 Get AI Advice — Topic Shift", key="btn_topic_llm", type="primary"):
                            with st.spinner("Generating topic shift insight …"):
                                st.session_state["topic_insight"] = generate_topic_insight(
                                    topic_result, sorted_results, anomalies
                                )
                                st.rerun()

                    # --- Topic Details (collapsed) ---
                    with st.expander("📋 All Discovered Topics (Details)", expanded=False):
                        topic_table = []
                        for ti in topic_result.topics:
                            topic_table.append({
                                "Topic ID": ti.topic_id,
                                "Keywords (PROPN/NOUN only)": ", ".join(ti.keywords[:6]),
                                "Documents": ti.count,
                            })
                        st.dataframe(pd.DataFrame(topic_table), hide_index=True, use_container_width=True)

    # ==================================================================
    # TAB 3: Raw Transcripts
    # ==================================================================
    with tab_transcripts:
        st.subheader("📄 Raw Transcript Preview")
        st.caption("Expand each quarter to view the full transcript text split into PR, Q&A, and Full sections.")

        for label, res in sorted_results.items():
            with st.expander(f"{res.company_name} ({res.ticker}) — {res.quarter_label}"):
                sub_prep, sub_qa, sub_full = st.tabs(["Prepared Remarks", "Q&A Session", "Full Transcript"])
                with sub_prep:
                    st.text_area("PR", value=res.split.prepared_remarks[:5000],
                                height=250, disabled=True, key=f"prep_{label}")
                with sub_qa:
                    if res.split.qa_section:
                        st.text_area("QA", value=res.split.qa_section[:5000],
                                    height=250, disabled=True, key=f"qa_{label}")
                    else:
                        st.info("No Q&A section detected.")
                with sub_full:
                    meta_obj = parsed[label]
                    st.text_area("Full", value=meta_obj.raw_text[:10000],
                                height=250, disabled=True, key=f"full_{label}")

    # ==================================================================
    # Footer
    # ==================================================================
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.caption(
        f"Earnings Call Lens v{_APP_VERSION}  •  "
        "Sentiment: FinBERT + NSS + t-Test  •  "
        "Topics: BERTopic + POS-Pruned c-TF-IDF  •  "
        "Insights: DeepSeek V4-Flash / Local Narrative Engine  •  "
        "Built with Streamlit"
    )


# ===========================================================================
# Script Entry Point
# ===========================================================================
if __name__ == "__main__":
    main()
