#!/usr/bin/env python3
# app/pages/03_Portfolio_Actions_Backtest.py
# Comments in English as requested.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Portfolio Actions (Backtest)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Styling (match Tab 01/02 aesthetic)
# -----------------------------
st.markdown(
    """
<style>
.stApp { background: #f6f7fb; }

/* Prevent first row from being clipped */
.block-container { padding-top: 3.2rem !important; }

h1, h2, h3, h4 { color: #111827 !important; letter-spacing: -0.02em; }

.page-title { font-size: 2.7rem; font-weight: 900; color: #0f172a; margin: 0 0 0.2rem 0; }
.page-subtitle { color: #6b7280; font-size: 1.05rem; font-weight: 600; margin: 0 0 1.1rem 0; }

.section-title { font-size: 2.2rem; font-weight: 900; color: #0f172a; margin: 1.3rem 0 0.9rem 0; }

.card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 18px;
  box-shadow: 0 6px 18px rgba(17, 24, 39, 0.06);
  padding: 18px 18px 16px 18px;
}
.card-title { font-size: 1.05rem; font-weight: 800; color: #111827; margin-bottom: 8px; }
.card-subtle { color: #6b7280; font-size: 0.92rem; margin-top: 2px; line-height: 1.55; }

.pills { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; margin-bottom: 2px; }
.pill {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 8px 12px; border-radius: 999px;
  border: 1px solid #e5e7eb; background: #f9fafb;
  font-size: 0.92rem; color: #111827; font-weight: 800;
}
.dot { width: 10px; height: 10px; border-radius: 999px; display: inline-block; }
.dot-green { background: #22c55e; }
.dot-amber { background: #f59e0b; }
.dot-gray { background: #9ca3af; }

.mini-grid { display:grid; grid-template-columns:1fr 60px 1fr 60px 1fr; gap:12px; margin-top:12px; }
@media (max-width: 1050px) {
  .mini-grid { grid-template-columns:1fr; }
  .arrow-row { display:none; }
}

.alloc {
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  background: #ffffff;
  box-shadow: 0 6px 18px rgba(17, 24, 39, 0.06);
  padding: 14px 14px 12px 14px;
}
.alloc-title { font-weight:900; color:#111827; font-size:1.02rem; margin-bottom:8px; }
.alloc-sub { color:#6b7280; font-weight:800; font-size:0.92rem; margin-bottom:10px; line-height:1.45; }
.alloc-row { display:flex; gap:10px; flex-wrap:wrap; }
.alloc-pill {
  display:inline-flex; align-items:center; gap:8px;
  padding: 8px 10px; border-radius: 999px;
  border:1px solid #e5e7eb; background:#f9fafb;
  font-weight:900; color:#111827; font-size:0.95rem;
}
.alloc-pill small { color:#6b7280; font-weight:900; font-size:0.85rem; }

.arrow-row { display:flex; align-items:center; justify-content:center; color:#94a3b8; font-weight:900; }

.move-grid { display:grid; grid-template-columns:1fr; gap:8px; margin-top:10px; }
.move-row {
  display:flex; justify-content:space-between; align-items:center;
  padding: 10px 12px;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  background: #f9fafb;
}
.move-left { display:flex; align-items:center; gap:10px; }
.move-asset { font-weight: 900; color:#111827; }
.move-right { font-weight: 900; }
.arrow { font-size: 1.25rem; font-weight: 900; line-height: 1; }
.arrow-up { color: #22c55e; }
.arrow-down { color: #ef4444; }
.arrow-flat { color: #9ca3af; }
.val-up { color: #16a34a; }
.val-down { color: #dc2626; }
.val-flat { color: #111827; opacity: 0.65; }

hr.soft { border: none; border-top: 1px solid #e5e7eb; margin: 18px 0; }

.appendix h4 {
  margin: 12px 0 6px 0;
  font-size: 1.02rem;
  color: #111827;
  font-weight: 900;
}
.appendix ul { margin: 6px 0 0 18px; line-height: 1.65; color:#6b7280; }
.appendix li { margin: 3px 0; }
.appendix p { margin: 10px 0; color:#6b7280; line-height: 1.65; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Repo root + paths
# -----------------------------


def _repo_root() -> Path:
    # app/pages -> parents[2] is repo root
    return Path(__file__).resolve().parents[2]


ROOT = _repo_root()

PRICES_PANEL_PATH = ROOT / "data" / "state" / "backtests" / "prices_panel.parquet"
TARGETS_HIST_PATH = ROOT / "data" / "state" / "portfolio_targets_history" / "portfolio_targets_history.parquet"

# -----------------------------
# Example portfolio + cashflows (fixed for the narrative)
# -----------------------------
BASELINE_WEIGHTS = {"VTI": 0.56, "VXUS": 0.24, "BND": 0.20}

INITIAL_INVEST_USD = 1000.0
CONTRIBUTION_USD = 250.0
CONTRIBUTION_DAYS = (1, 15)

# Policy bands (theoretical)
BAU_BAND_PP = 5.0
EXCEPTIONAL_BAND_PP = 20.0

START_DATE = pd.Timestamp("2012-01-01")


@dataclass
class SimConfig:
    start_date: pd.Timestamp = START_DATE
    initial_invest_usd: float = INITIAL_INVEST_USD
    contrib_usd: float = CONTRIBUTION_USD
    contrib_days: Tuple[int, int] = CONTRIBUTION_DAYS


# -----------------------------
# Loaders
# -----------------------------


def _read_parquet_df(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    return df


@st.cache_data(show_spinner=False)
def load_prices_panel() -> pd.DataFrame:
    df = _read_parquet_df(PRICES_PANEL_PATH)
    if df is None:
        raise FileNotFoundError(
            f"Missing prices_panel.parquet: {PRICES_PANEL_PATH}. Run: python scripts/06.1_build_backtest_panels.py"
        )
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    out = out.dropna(how="any")
    return out


@st.cache_data(show_spinner=False)
def load_targets_history() -> pd.DataFrame:
    df = _read_parquet_df(TARGETS_HIST_PATH)
    if df is None:
        raise FileNotFoundError(f"Missing portfolio_targets_history.parquet: {TARGETS_HIST_PATH}.")
    out = df.copy()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
    out = out.dropna(subset=["as_of_date"]).sort_values("as_of_date").reset_index(drop=True)
    return out


# -----------------------------
# Helpers
# -----------------------------


def _sanitize_weights_3(w: Dict[str, float]) -> Dict[str, float]:
    keys = ["VTI", "VXUS", "BND"]
    ww = {k: float(w.get(k, 0.0)) for k in keys}
    for k in keys:
        if not np.isfinite(ww[k]) or ww[k] < 0:
            ww[k] = 0.0
    s = sum(ww.values())
    if s <= 0:
        return dict(BASELINE_WEIGHTS)
    return {k: ww[k] / s for k in keys}


def _targets_row_asof(targets_df: pd.DataFrame, dt: pd.Timestamp) -> Optional[pd.Series]:
    if targets_df.empty:
        return None
    pos = targets_df["as_of_date"].searchsorted(dt, side="right") - 1
    if pos < 0:
        return None
    return targets_df.iloc[int(pos)]


def _targets_asof(targets_df: pd.DataFrame, dt: pd.Timestamp) -> Optional[Dict[str, float]]:
    r = _targets_row_asof(targets_df, dt)
    if r is None:
        return None
    return {
        "VTI": float(r.get("target_vti", np.nan)),
        "VXUS": float(r.get("target_vxus", np.nan)),
        "BND": float(r.get("target_bnd", np.nan)),
    }


def _build_contribution_dates(start: pd.Timestamp, end: pd.Timestamp, days: Tuple[int, int]) -> List[pd.Timestamp]:
    if start > end:
        return []
    out: List[pd.Timestamp] = []
    cur = pd.Timestamp(year=start.year, month=start.month, day=1)
    while cur <= end:
        for d in days:
            try:
                dt = pd.Timestamp(year=cur.year, month=cur.month, day=int(d))
            except Exception:
                continue
            if start <= dt <= end:
                out.append(dt)
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()
    return sorted(list(dict.fromkeys(out)))


def _shift_to_next_trading_day(dt: pd.Timestamp, trading_index: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    if dt in trading_index:
        return dt
    pos = trading_index.searchsorted(dt, side="left")
    if pos >= len(trading_index):
        return None
    return pd.Timestamp(trading_index[pos])


def _portfolio_value(shares: Dict[str, float], px: Dict[str, float]) -> float:
    return float(sum(float(shares.get(k, 0.0)) * float(px[k]) for k in px.keys()))


def _weights_from_shares(shares: Dict[str, float], px: Dict[str, float]) -> Dict[str, float]:
    v = _portfolio_value(shares, px)
    if v <= 0:
        return {k: 0.0 for k in px.keys()}
    return {k: float(shares.get(k, 0.0)) * float(px[k]) / v for k in px.keys()}


def _allocate_contribution_no_sells(
    shares: Dict[str, float],
    px: Dict[str, float],
    target_w: Dict[str, float],
    cash: float,
) -> Dict[str, float]:
    cur_w = _weights_from_shares(shares, px)
    gaps = {k: max(0.0, float(target_w.get(k, 0.0)) - float(cur_w.get(k, 0.0))) for k in target_w.keys()}
    s = float(sum(gaps.values()))
    if s <= 1e-12:
        alloc_w = target_w
    else:
        alloc_w = {k: gaps[k] / s for k in gaps.keys()}

    for k in alloc_w.keys():
        buy_dollars = float(cash) * float(alloc_w[k])
        if buy_dollars <= 0:
            continue
        shares[k] = float(shares.get(k, 0.0)) + buy_dollars / float(px[k])
    return shares


def _rebalance_full(shares: Dict[str, float], px: Dict[str, float], target_w: Dict[str, float]) -> Dict[str, float]:
    total = _portfolio_value(shares, px)
    if total <= 0:
        return shares
    for k in target_w.keys():
        target_dollars = float(target_w[k]) * total
        shares[k] = target_dollars / float(px[k])
    return shares


def _pill(text: str, dot: str = "gray") -> str:
    dot_class = {
        "green": "dot-green",
        "amber": "dot-amber",
        "gray": "dot-gray",
    }.get(dot, "dot-gray")
    return f'<span class="pill"><span class="dot {dot_class}"></span>{text}</span>'


def _alloc_box(title: str, subtitle: str, w: Dict[str, float], extra_note: str = "") -> str:
    w = _sanitize_weights_3(w)
    eq = float(w["VTI"] + w["VXUS"])
    us_eq_share = float(w["VTI"] / eq) if eq > 0 else 0.0
    rows = (
        f'<span class="alloc-pill">VTI <small>{w["VTI"]*100:,.1f}%</small></span>'
        f'<span class="alloc-pill">VXUS <small>{w["VXUS"]*100:,.1f}%</small></span>'
        f'<span class="alloc-pill">BND <small>{w["BND"]*100:,.1f}%</small></span>'
    )
    eq_line = f"Equity/Bonds: {eq*100:,.0f}% / {w['BND']*100:,.0f}% · US share within equity: {us_eq_share*100:,.0f}%"
    note = f"<div class='alloc-sub' style='margin-top:10px;'>{extra_note}</div>" if extra_note else ""
    return f"""
<div class="alloc">
  <div class="alloc-title">{title}</div>
  <div class="alloc-sub">{subtitle}</div>
  <div class="alloc-row">{rows}</div>
  <div class="alloc-sub" style="margin-top:10px;">{eq_line}</div>
  {note}
</div>
""".strip()


def _move_arrow(delta_pp: float) -> Tuple[str, str, str]:
    if delta_pp > 0.05:
        return "▲", "arrow-up", "val-up"
    if delta_pp < -0.05:
        return "▼", "arrow-down", "val-down"
    return "▶", "arrow-flat", "val-flat"


def _moves_box(delta_pp: Dict[str, float]) -> str:
    rows_html = ""
    for k in ["VTI", "VXUS", "BND"]:
        d = float(delta_pp.get(k, 0.0))
        arrow, arrow_class, val_class = _move_arrow(d)
        rows_html += f"""
<div class="move-row">
  <div class="move-left">
    <div class="arrow {arrow_class}">{arrow}</div>
    <div class="move-asset">{k}</div>
  </div>
  <div class="move-right {val_class}">{d:+.1f}pp</div>
</div>
""".strip()

    return f"""
<div class="alloc">
  <div class="alloc-title">Implied move vs baseline</div>
  <div class="alloc-sub">
    Actions are expressed in percentage points. Typical recommendations are designed to remain within <b>±{BAU_BAND_PP:.0f}pp</b>;
    moves beyond that range are treated as exceptional and may extend up to <b>±{EXCEPTIONAL_BAND_PP:.0f}pp</b> in this research view.
  </div>
  <div class="move-grid">{rows_html}</div>
</div>
""".strip()


def _plot_nav(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for strat in df["strategy"].unique():
        d = df[df["strategy"] == strat].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["nav"],
                mode="lines",
                name=strat,
                hovertemplate="Scenario: %{fullData.name}<br>Date: %{x|%Y-%m-%d}<br>Value: $%{y:,.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=560,
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio value (USD, nominal)",
        legend_title="Scenario",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_yaxes(tickformat=",.0f")
    return fig


# -----------------------------
# Simulations
# -----------------------------


def simulate_vt_buy_hold(prices_vt: pd.Series, cfg: SimConfig) -> pd.DataFrame:
    idx = prices_vt.index
    start_td = _shift_to_next_trading_day(cfg.start_date, idx)
    if start_td is None:
        raise RuntimeError("Start date is after the last available trading day.")

    px0 = float(prices_vt.loc[start_td])
    shares = cfg.initial_invest_usd / px0

    contrib_dates = _build_contribution_dates(start_td, pd.Timestamp(idx[-1]), cfg.contrib_days)
    contrib_td = []
    for dt in contrib_dates:
        tdt = _shift_to_next_trading_day(dt, idx)
        if tdt is not None:
            contrib_td.append(tdt)
    contrib_td = sorted(list(dict.fromkeys(contrib_td)))
    contrib_set = set(contrib_td)

    rows = []
    for dt in idx[idx >= start_td]:
        px = float(prices_vt.loc[dt])
        if dt in contrib_set:
            shares += float(cfg.contrib_usd) / px
        nav = float(shares * px)
        rows.append({"date": dt, "nav": nav})
    out = pd.DataFrame(rows)
    out["strategy"] = "VT (buy & hold)"
    return out


def simulate_3asset(
    prices_3: pd.DataFrame,
    cfg: SimConfig,
    name: str,
    mode: str,
    targets_df: Optional[pd.DataFrame] = None,
    fixed_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    idx = prices_3.index
    start_td = _shift_to_next_trading_day(cfg.start_date, idx)
    if start_td is None:
        raise RuntimeError("Start date is after the last available trading day.")

    shares: Dict[str, float] = {c: 0.0 for c in prices_3.columns}
    px0 = {c: float(prices_3.loc[start_td, c]) for c in prices_3.columns}

    if mode == "fixed_split":
        assert fixed_weights is not None
        w0 = _sanitize_weights_3(fixed_weights)
        for k in shares.keys():
            shares[k] = (cfg.initial_invest_usd * w0[k]) / px0[k]
    elif mode in {"signal_contrib_only", "signal_full_rebalance"}:
        base = _sanitize_weights_3(BASELINE_WEIGHTS)
        for k in shares.keys():
            shares[k] = (cfg.initial_invest_usd * base[k]) / px0[k]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    contrib_dates = _build_contribution_dates(start_td, pd.Timestamp(idx[-1]), cfg.contrib_days)
    contrib_td = []
    for dt in contrib_dates:
        tdt = _shift_to_next_trading_day(dt, idx)
        if tdt is not None:
            contrib_td.append(tdt)
    contrib_td = sorted(list(dict.fromkeys(contrib_td)))
    contrib_set = set(contrib_td)

    rows = []
    for dt in idx[idx >= start_td]:
        px = {c: float(prices_3.loc[dt, c]) for c in prices_3.columns}

        if dt in contrib_set:
            cash = float(cfg.contrib_usd)

            if mode == "fixed_split":
                assert fixed_weights is not None
                w = _sanitize_weights_3(fixed_weights)
                for k in shares.keys():
                    shares[k] = float(shares.get(k, 0.0)) + (cash * w[k]) / px[k]

            elif mode == "signal_contrib_only":
                assert targets_df is not None
                tgt = _targets_asof(targets_df, dt)
                tgt = _sanitize_weights_3(tgt) if tgt is not None else _sanitize_weights_3(BASELINE_WEIGHTS)
                shares = _allocate_contribution_no_sells(shares, px, tgt, cash)

            elif mode == "signal_full_rebalance":
                assert targets_df is not None
                tgt = _targets_asof(targets_df, dt)
                tgt = _sanitize_weights_3(tgt) if tgt is not None else _sanitize_weights_3(BASELINE_WEIGHTS)
                for k in shares.keys():
                    shares[k] = float(shares.get(k, 0.0)) + (cash * tgt[k]) / px[k]
                shares = _rebalance_full(shares, px, tgt)

        nav = _portfolio_value(shares, px)
        rows.append({"date": dt, "nav": float(nav)})

    out = pd.DataFrame(rows)
    out["strategy"] = name
    return out


# -----------------------------
# Render
# -----------------------------

st.markdown('<div class="page-title">Portfolio Actions — Backtest</div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="page-subtitle">
  Historical simulation of the same recommendation logic used today, translated into specific reweighting actions
  within percentage-point bands for an example <b>VTI + VXUS + BND</b> portfolio.
</div>
""",
    unsafe_allow_html=True,
)

# Load inputs
try:
    prices_panel = load_prices_panel()
except Exception as e:
    st.error(f"Could not load prices panel: {e}")
    st.stop()

try:
    targets_df = load_targets_history()
except Exception as e:
    st.error(f"Could not load targets history: {e}")
    st.stop()

required_cols = ["VT", "VTI", "VXUS", "BND"]
missing_cols = [c for c in required_cols if c not in prices_panel.columns]
if missing_cols:
    st.error("prices_panel.parquet is missing required tickers: " + ", ".join(missing_cols))
    st.stop()

cfg = SimConfig()
prices_panel = prices_panel.copy()
prices_panel.index = pd.to_datetime(prices_panel.index, errors="coerce")
prices_panel = prices_panel[~prices_panel.index.isna()].sort_index()
prices_panel = prices_panel[prices_panel.index >= cfg.start_date].dropna(how="any")

if prices_panel.empty or len(prices_panel) < 200:
    st.error("Not enough price history after the configured start date to run the backtest.")
    st.stop()

as_of = pd.Timestamp(prices_panel.index.max())

baseline = _sanitize_weights_3(BASELINE_WEIGHTS)

latest_target = _targets_asof(targets_df, as_of)
latest_target = _sanitize_weights_3(latest_target) if latest_target is not None else _sanitize_weights_3(BASELINE_WEIGHTS)

delta_pp = {k: 100.0 * (latest_target[k] - baseline[k]) for k in baseline.keys()}

# Intro card: baseline, cashflows, theoretical bands (no "today caps" and no extra band tiers)
st.markdown(
    f"""
<div class="card">
  <div class="card-title">Example portfolio + policy bands</div>
  <div class="card-subtle">
    Baseline allocation is <b>VTI 56% / VXUS 24% / BND 20%</b> (i.e., <b>80/20</b> equity/bonds), with <b>70%</b> US within equity. Actions are expressed as <b>percentage-point</b> moves relative to this baseline.
  </div>

  <div class="pills">
    {_pill(f"As-of: {as_of.date().isoformat()}", "gray")}
    {_pill(f"Cashflows: ${INITIAL_INVEST_USD:,.0f} initial + ${CONTRIBUTION_USD:,.0f} on day 1 & 15", "gray")}
    {_pill(f"BAU band: ±{BAU_BAND_PP:.0f}pp", "green")}
    {_pill(f"Exceptional allowance: up to ±{EXCEPTIONAL_BAND_PP:.0f}pp (research)", "amber")}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

# Visual: baseline -> move -> final (kept as-is)
st.markdown(
    """
<div class="mini-grid">
  {box1}
  <div class="arrow-row">→</div>
  {box2}
  <div class="arrow-row">→</div>
  {box3}
</div>
""".format(
        box1=_alloc_box(
            "Original portfolio (baseline)",
            "Reference allocation used as the neutral anchor.",
            baseline,
        ),
        box2=_moves_box(delta_pp=delta_pp),
        box3=_alloc_box(
            "Final portfolio (after actions)",
            "Resulting target allocation used for the simulations below.",
            latest_target,
        ),
    ),
    unsafe_allow_html=True,
)

st.markdown('<hr class="soft"/>', unsafe_allow_html=True)

# Section title before plot
st.markdown('<div class="section-title">Historical simulations vs benchmark</div>', unsafe_allow_html=True)

# Explanation card: wording updates + formal English paragraph
st.markdown(
    f"""
<div class="card">
  <div class="card-title">What you are looking at</div>
  <div class="card-subtle">
    The chart compares wealth paths under identical cashflows, starting from <b>{cfg.start_date.date().isoformat()}</b>:
    ${INITIAL_INVEST_USD:,.0f} initial + ${CONTRIBUTION_USD:,.0f} on day 1 and day 15 each month (invested on the next trading day close).
    <br/><br/>
    <b>Scenarios</b>:
    <ol style="margin:8px 0 0 18px; line-height:1.65;">
      <li><b>VT (buy &amp; hold)</b>: 100% VT with recurring contributions.</li>
      <li><b>Baseline 56/24/20 (no rebalancing)</b>: contributions are always split 56/24/20; no selling.</li>
      <li><b>Signals (contrib-only, no sells)</b>: targets are applied only by routing new contributions toward underweights; no selling.</li>
      <li><b>Signals (full rebalance; tax-advantaged)</b>: at each contribution date, the entire portfolio is rebalanced to target weights (frictionless assumption).
          In other settings, consider additional costs (e.g., taxes, spreads, slippage, and execution constraints).</li>
    </ol>
    <br/>
    <b>Finance details</b>:
    <ul style="margin:8px 0 0 18px; line-height:1.65;">
      <li>Values are <b>nominal USD</b> (not inflation-adjusted).</li>
      <li>Compounding is implicit in the price series.</li>
      <li>No taxes, fees, spreads, or tracking error are modeled.</li>
      <li>Contributions on non-trading days are invested on the <b>next</b> trading day close.</li>
    </ul>
    <br/>
    <b>Important perspective</b>:
    Boglehead-style guidance emphasizes that consistently beating the market is difficult; over long horizons, the simplest approach can be the most robust,
    and frequent allocation changes can be counterproductive. In this backtest, that intuition is broadly supported.
    However, without any intent to cherry-pick, it is also apparent that this window has been dominated by a strong bullish regime, where differences between scenarios
    tend to widen most. In less bullish stretches, managed scenarios have historically reduced the gap and, at times, compared more favorably—most notably during
    outlier drawdowns such as the early-2020 COVID shock. As a result, under a catastrophic risk scenario such as an AI-bubble unwind (diagnosed in Tab 2),
    banded adjustments could plausibly offer some downside protection. This remains an observation rather than a claim: the market-cap-weighted structure of the benchmarks
    and the stabilizing effect of DCA often offset differences over time, and the incremental benefit may not justify the operational burden of active management.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

# Run simulations
prices_vt = prices_panel["VT"].copy()
prices_3 = prices_panel[["VTI", "VXUS", "BND"]].copy()

try:
    vt_df = simulate_vt_buy_hold(prices_vt, cfg)
    baseline_df = simulate_3asset(
        prices_3=prices_3,
        cfg=cfg,
        name="Baseline 56/24/20 (no rebalancing)",
        mode="fixed_split",
        fixed_weights=BASELINE_WEIGHTS,
    )
    contrib_only_df = simulate_3asset(
        prices_3=prices_3,
        cfg=cfg,
        name="Signals (contrib-only, no sells)",
        mode="signal_contrib_only",
        targets_df=targets_df,
    )
    full_reb_df = simulate_3asset(
        prices_3=prices_3,
        cfg=cfg,
        name="Signals (full rebalance; tax-advantaged)",
        mode="signal_full_rebalance",
        targets_df=targets_df,
    )
except Exception as e:
    st.error(f"Backtest failed: {e}")
    st.stop()

plot_df = pd.concat([vt_df, baseline_df, contrib_only_df, full_reb_df], ignore_index=True)

fig = _plot_nav(plot_df, title="Wealth path under different allocation/rebalancing rules")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

st.markdown('<hr class="soft"/>', unsafe_allow_html=True)

# Appendix (expanded; formal English; no internal design references)
st.markdown('<div class="section-title">Appendix</div>', unsafe_allow_html=True)

st.markdown(
    f"""
<div class="card appendix">
  <div class="card-title">Band logic (why moves are usually small, and when they become larger)</div>

  <p>
    The policy is intentionally conservative by default: most recommendations are designed to remain within <b>±{BAU_BAND_PP:.0f} percentage points</b>
    relative to the baseline (56/24/20). Moves beyond that range are reserved for rarer, higher-conviction states and should be interpreted as exceptional.
    Importantly, this is <b>non-linear</b>: each additional point beyond {BAU_BAND_PP:.0f}pp is progressively harder to justify than the previous one.
  </p>

  <h4>Typical moves (≤ ±{BAU_BAND_PP:.0f}pp) tend to occur when</h4>
  <ul>
    <li>The signal environment is mixed or only mildly tilted, without strong alignment across independent diagnostics.</li>
    <li>Market stress and volatility conditions remain broadly normal, suggesting limited downside asymmetry.</li>
    <li>Leadership and concentration measures are elevated but not historically extreme relative to their own distributions.</li>
    <li>The recommendation is best interpreted as a robustness nudge (incremental diversification) rather than a strong tactical stance.</li>
  </ul>

  <h4>Larger moves (&gt; ±{BAU_BAND_PP:.0f}pp) require stronger evidence, for example</h4>
  <ul>
    <li><b>Severity</b>: multiple diagnostics move into extreme territory simultaneously (e.g., narrow leadership plus crowding signals).</li>
    <li><b>Amplifiers</b>: stress/volatility conditions materially increase downside asymmetry for concentrated exposures.</li>
    <li><b>Crash spillover</b>: an AI-bubble unwind can trigger broad de-risking, justifying larger reallocations to protect overall portfolio risk.</li>
    <li><b>Alignment</b>: several independent inputs point in the same direction (not a single proxy).</li>
    <li><b>Persistence</b>: conditions remain elevated over multiple updates, reducing the likelihood of transient noise.</li>
    <li><b>Portfolio relevance</b>: the action directly addresses the dominant risk implied by the state (e.g., raising bond allocation when fragility risk dominates).</li>
  </ul>

  <h4>Why allow up to ±{EXCEPTIONAL_BAND_PP:.0f}pp in this research view</h4>
  <ul>
    <li>To visualize how the policy would behave under tail scenarios and compare historical scenario paths under consistent cashflows.</li>
    <li>It is not intended as routine portfolio management; moves beyond ±{BAU_BAND_PP:.0f}pp should be treated as exceptional.</li>
    <li>Real-world frictions (tax impact, spreads, execution constraints, tracking preferences, and operational complexity) can dominate the net benefit of aggressive rebalancing.</li>
  </ul>

  <p>
    Detailed drivers and diagnostics are covered in other tabs. This page focuses on translating the recommendation stream into fund-level actions
    and comparing historical wealth paths under consistent contributions.
  </p>
</div>
""",
    unsafe_allow_html=True,
)
