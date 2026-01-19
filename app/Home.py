#!/usr/bin/env python3
# app/Home.py
# Comments in English as requested.

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Bogle & AI Bubble",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp { background: #f6f7fb; }
.block-container { padding-top: 2.8rem !important; }

h1, h2, h3, h4 { color: #111827 !important; letter-spacing: -0.02em; }

.page-title { font-size: 2.9rem; font-weight: 950; color: #0f172a; margin: 0 0 0.25rem 0; }
.page-subtitle { color: #6b7280; font-size: 1.05rem; font-weight: 650; margin: 0 0 1.2rem 0; line-height: 1.55; }

.card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 18px;
  box-shadow: 0 6px 18px rgba(17, 24, 39, 0.06);
  padding: 18px 18px 16px 18px;
}
.card-title { font-size: 1.08rem; font-weight: 900; color: #0f172a; margin: 0 0 0.35rem 0; }
.card-subtle { color: #6b7280; font-size: 0.96rem; line-height: 1.6; margin: 0; }

.hr-soft { border: none; border-top: 1px solid #e5e7eb; margin: 16px 0; }

.kv {
  display: grid;
  grid-template-columns: 180px 1fr;
  gap: 8px 14px;
  align-items: start;
  margin-top: 10px;
}
.k { color:#111827; font-weight: 900; }
.v { color:#6b7280; font-weight: 650; line-height: 1.55; }

.button-row { margin-top: 12px; }

/* Intro bullets */
.bullets { margin: 10px 0 0 0; padding-left: 18px; line-height: 1.7; }
.bullets li { margin-bottom: 6px; }

/* Disclaimer block */
.disclaimer {
  margin-top: 0.6rem;
  padding: 12px 14px;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  background: #ffffff;
  box-shadow: 0 6px 18px rgba(17, 24, 39, 0.06);
  color: #6b7280;
  font-size: 0.96rem;
  line-height: 1.6;
}

/* Make page_link text readable on light background */
a[data-testid="stPageLink-NavLink"], a[data-testid="stPageLink-NavLink"] * {
  color: #0f172a !important;
  font-weight: 800 !important;
  text-decoration: none !important;
}
/* Optional: give the link a subtle pill background */
a[data-testid="stPageLink-NavLink"] {
  display: inline-flex !important;
  align-items: center !important;
  gap: 8px !important;
  padding: 10px 12px !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 12px !important;
  background: #ffffff !important;
  box-shadow: 0 6px 18px rgba(17, 24, 39, 0.06) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Header + official intro
# -----------------------------
st.markdown('<div class="page-title">Bogle &amp; AI Bubble</div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="page-subtitle">
This dashboard consolidates a broad set of frequently-updated indicators and runs deterministic calculations to produce
<b>banded allocation recommendations</b> for a <b>Bogle-style ETF portfolio</b> under elevated uncertainty, including the risk that an
AI-driven cycle is inflating parts of the market.
</div>

<div class="page-subtitle" style="margin-top:0.2rem;">
<ul class="bullets">
  <li><b>Not market timing:</b> the goal is not to “predict tops”, but to translate risk/regime diagnostics into small, auditable
      allocation moves inside predefined bands.</li>
  <li><b>Not investment advice:</b> outputs are best treated as additional context for someone already familiar with allocations
      and who would rebalance anyway.</li>
  <li><b>Backtest reality check:</b> the backtest section shows that these reallocations do not reliably beat a market-cap-weighted
      benchmark over sufficiently long horizons once volatility and the self-correcting nature of DCA are considered.</li>
  <li><b>Why study it anyway:</b> in less bullish sub-periods, banded risk-management scenarios can sometimes reduce the gap vs the benchmark
      and look comparatively better—informative when thinking about adverse-regime risk.</li>
</ul>
</div>

<div class="page-subtitle" style="margin-top:0.4rem;">
<b>Resources</b> is where the “why” questions are answered: why these actionables can still be framed as Bogle-compatible (instead of
market timing), what kinds of market conditions can justify closer monitoring despite a Bogle philosophy, and short primers to help
readers get familiar with the core concepts referenced across the dashboard.
</div>

<div class="disclaimer">
<b>Disclaimer (project framing):</b> this framework was designed in <b>January 2026</b>. It can keep updating daily indefinitely, but if a
structural change occurs—especially related to an AI-bubble unwind—interpret results with that in mind: what updates over time are the
input indicators, while the deterministic rules and overall framing remain those established in January 2026.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<hr class='hr-soft'/>", unsafe_allow_html=True)

# -----------------------------
# Page cards
# -----------------------------
PAGES = [
    {
        "title": "Daily Snapshot",
        "icon": "🧭",
        "path": "pages/01_Daily_Snapshot.py",
        "summary": (
            "Quick read of the current regime and persisted signals. "
            "The signals shown here are the system’s allocation recommendations (subject to persistence and banding)."
        ),
        "details": [
            ("Best for", "Checking the regime and whether recommendations are stable vs noisy."),
            ("What it outputs", "Regime summary + stress drivers + actionable tilts (banded)."),
        ],
        "button_label": "Open Daily Snapshot",
    },
    {
        "title": "AI Bubble Diagnostics",
        "icon": "⚡",
        "path": "pages/02_AI_Bubble_Diagnostics.py",
        "summary": (
            "Diagnostics for AI-cycle froth and fragility: leadership, concentration and run-up proxies combined into "
            "an AI Bubble Score and an amplified Crash Risk. These are additional insights to reinforce (or challenge) "
            "the guardrails implied by the Daily Snapshot."
        ),
        "details": [
            ("Best for", "Understanding whether the AI-cycle looks crowded and fragile."),
            ("What it outputs", "Two gauges + key drivers + interpretability appendix."),
        ],
        "button_label": "Open AI Bubble Diagnostics",
    },
    {
        "title": "Portfolio Actions & Backtest",
        "icon": "📊",
        "path": "pages/03_Portfolio_Actions_Backtest.py",
        "summary": (
            "Materializes the reallocation recommendations using a hypothetical VTI/VXUS/BND portfolio and cashflow schedule, "
            "and shows how those actions would have performed vs the benchmark under DCA and compounding."
        ),
        "details": [
            ("Best for", "Comparing “do nothing” vs “banded actions” across different market regimes."),
            ("What it outputs", "Scenario growth curves + implied allocation moves + appendix on band rules."),
        ],
        "button_label": "Open Portfolio Actions",
    },
    {
        "title": "Resources",
        "icon": "📚",
        "path": "pages/04_Learn_Glossary.py",
        "summary": (
            "Informational texts that add context and definitions: Bogle basics, why the project’s actionables are framed as "
            "Bogle-compatible (not market timing), current-situation discussion, and deeper technical framework + known limitations."
        ),
        "details": [
            ("Best for", "Answering “why” questions and building intuition without cluttering the dashboards."),
            ("What it outputs", "A clean library view with left navigation + content pane."),
        ],
        "button_label": "Open Resources",
    },
]

# Layout: 2x2 grid
row1 = st.columns(2, gap="large")
row2 = st.columns(2, gap="large")
slots = [row1[0], row1[1], row2[0], row2[1]]

for slot, page in zip(slots, PAGES):
    with slot:
        st.markdown(
            f"""
<div class="card">
  <div class="card-title">{page["icon"]} {page["title"]}</div>
  <div class="card-subtle">{page["summary"]}</div>
  <div class="kv">
    <div class="k">{page["details"][0][0]}</div><div class="v">{page["details"][0][1]}</div>
    <div class="k">{page["details"][1][0]}</div><div class="v">{page["details"][1][1]}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='button-row'></div>", unsafe_allow_html=True)

        # IMPORTANT: path must be relative to the entrypoint directory (app/),
        # and target files must be in app/pages/.
        st.page_link(page["path"], label=page["button_label"], icon=page["icon"])

st.markdown("<hr class='hr-soft'/>", unsafe_allow_html=True)
st.caption("Tip: the left sidebar also provides direct navigation between pages.")
