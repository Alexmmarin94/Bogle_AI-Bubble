#!/usr/bin/env python3
# app/pages/02_AI_Bubble_Diagnostics.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import plotly.graph_objects as go


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="AI Bubble Diagnostics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Styling (match 01_Daily_Snapshot look & font)
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
.card-subtle { color: #6b7280; font-size: 0.92rem; margin-top: 2px; line-height: 1.35; }

.pills { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; margin-bottom: 12px; }
.pill {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 8px 12px; border-radius: 999px;
  border: 1px solid #e5e7eb; background: #f9fafb;
  font-size: 0.92rem; color: #111827; font-weight: 700;
}
.dot { width: 10px; height: 10px; border-radius: 999px; display: inline-block; }
.dot-green { background: #22c55e; }
.dot-red { background: #ef4444; }
.dot-amber { background: #f59e0b; }
.dot-gray { background: #9ca3af; }

.driver-item { margin: 14px 0 16px 0; }
.driver-k { color: #111827; font-weight: 900; font-size: 1.02rem; }
.driver-v { color: #6b7280; font-weight: 800; }
.driver-explain { color: #6b7280; font-size: 0.95rem; line-height: 1.55; margin-top: 6px; }

.chart-block { margin-top: 6px; margin-bottom: 18px; }
.chart-title { font-size: 1.35rem; font-weight: 900; color: #0f172a; margin: 0 0 6px 0; }
.chart-desc { color: #6b7280; font-size: 0.98rem; line-height: 1.6; margin: 0 0 10px 0; }

hr.soft { border: none; border-top: 1px solid #e5e7eb; margin: 18px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Paths / loaders (same approach as 01)
# -----------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_state_dirs(root: Path) -> List[Path]:
    return [
        root / "data" / "state" / "daily_state",
        root / "app" / "data" / "state" / "daily_state",
    ]


def _latest_daily_state_json(root: Path) -> Path:
    dirs = _candidate_state_dirs(root)
    existing = [d for d in dirs if d.exists() and d.is_dir()]
    if not existing:
        raise FileNotFoundError(f"Missing folder: {dirs[0]}")
    candidates: List[Path] = []
    for d in existing:
        candidates.extend(sorted(d.glob("daily_state_*.json")))
    if not candidates:
        raise FileNotFoundError(f"No daily_state_*.json found in: {', '.join(str(d) for d in existing)}")
    candidates = sorted(candidates, key=lambda p: (p.stem, p.stat().st_mtime))
    return candidates[-1]


@st.cache_data(show_spinner=False)
def load_daily_state(root: Path) -> Dict[str, Any]:
    p = _latest_daily_state_json(root)
    return json.loads(p.read_text(encoding="utf-8"))


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


def _load_fred_series(root: Path, sid: str) -> Optional[pd.Series]:
    p = root / "data" / "raw" / "fred" / f"{sid}.parquet"
    df = _read_parquet_df(p)
    if df is None or "date" not in df.columns or "value" not in df.columns:
        return None
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    if out.empty:
        return None
    s = pd.Series(out["value"].values, index=pd.to_datetime(out["date"]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_tiingo_prices(root: Path, ticker: str) -> Optional[pd.Series]:
    p = root / "data" / "raw" / "tiingo" / f"{ticker}.parquet"
    df = _read_parquet_df(p)
    if df is None or "date" not in df.columns:
        return None
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    if out.empty:
        return None
    price_col = (
        "adjClose"
        if ("adjClose" in out.columns and out["adjClose"].notna().any())
        else ("close" if "close" in out.columns else None)
    )
    if price_col is None:
        return None
    out["price"] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.dropna(subset=["price"])
    if out.empty:
        return None
    s = pd.Series(out["price"].values, index=pd.to_datetime(out["date"]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _value_asof(s: Optional[pd.Series], asof: pd.Timestamp) -> Optional[float]:
    if s is None or s.empty:
        return None
    s2 = s[s.index <= asof].dropna()
    if s2.empty:
        return None
    return float(s2.iloc[-1])


# -----------------------------
# Small utils
# -----------------------------
TRADING_DAYS = 252
LOOKBACK_DAYS = 3650  # trailing window for percentiles (tries ~10y if available)
DEFAULT_CHART_LOOKBACK_DAYS = 1825  # charts: ~5y window if data exists


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _fmt_num(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "—"
    av = abs(v)
    if av >= 100:
        return f"{v:,.0f}"
    if av >= 10:
        return f"{v:,.2f}"
    return f"{v:,.{digits}f}"


def realized_vol_21d_ann(px: pd.Series) -> pd.Series:
    ret = px.pct_change(fill_method=None)
    vol = ret.rolling(21).std(ddof=0) * np.sqrt(TRADING_DAYS)
    return vol


def _percentile_of_last(
    s: pd.Series, asof: pd.Timestamp, lookback_days: int = LOOKBACK_DAYS
) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (last_value, percentile_0_100) within the trailing window up to asof.
    """
    if s is None or s.empty:
        return None, None
    s2 = s.dropna().sort_index()
    s2 = s2[s2.index <= asof]
    if s2.empty:
        return None, None
    start = asof - pd.Timedelta(days=lookback_days)
    w = s2[s2.index >= start]
    if w.empty:
        w = s2
    last = float(w.iloc[-1])
    rank = float((w <= last).mean())
    return last, 100.0 * rank


def _dot_class(level: str) -> str:
    m = {"green": "dot-green", "red": "dot-red", "amber": "dot-amber", "gray": "dot-gray"}
    return m.get(level, "dot-gray")


def pill_html(text: str, dot: str = "gray") -> str:
    return (
        f'<span class="pill">'
        f'<span class="dot {_dot_class(dot)}"></span>'
        f"{text}"
        f"</span>"
    )


# -----------------------------
# Gauges (full-width iframes)
# -----------------------------
def bar_gauge_fragment(
    value_0_100: float,
    marker_top_px: int = 42,
    bar_height_px: int = 18,
) -> str:
    v = float(np.clip(value_0_100, 0.0, 100.0))
    v_int = int(round(v))

    return f"""
<div class="gauge-bar-wrap" style="margin-top:16px;">
  <div class="gauge-bar" style="height:{bar_height_px}px;">
    <div class="gauge-marker" style="left:{v:.2f}%; top:-{marker_top_px}px;">
      <div class="gauge-marker-value">{v_int}</div>
      <div class="gauge-marker-line"></div>
      <div class="gauge-marker-dot"></div>
    </div>
  </div>
  <div class="gauge-labels"><span>0</span><span>100</span></div>
</div>
""".strip()


def render_gauge_card_iframe(
    title: str,
    subtitle: str,
    value: float,
) -> None:
    iframe_css = """
:root { color-scheme: light; }
html, body {
  margin: 0; padding: 0; background: transparent;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}
.card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 18px;
  box-shadow: 0 6px 18px rgba(17, 24, 39, 0.06);
  padding: 18px 18px 16px 18px;
}
.card-title { font-size: 1.05rem; font-weight: 800; color: #111827; margin-bottom: 8px; }
.card-subtle { color: #6b7280; font-size: 0.92rem; margin-top: 2px; line-height: 1.35; }

.gauge-bar {
  position: relative;
  border-radius: 999px;
  background: linear-gradient(90deg, #22c55e 0%, #f59e0b 50%, #ef4444 100%);
  box-shadow: inset 0 0 0 1px rgba(17, 24, 39, 0.08);
}
.gauge-marker {
  position: absolute;
  transform: translateX(-50%);
  text-align: center;
}
.gauge-marker-value {
  font-size: 0.95rem;
  font-weight: 900;
  color: #0f172a;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 999px;
  padding: 4px 10px;
  box-shadow: 0 6px 18px rgba(17, 24, 39, 0.08);
  display: inline-block;
  margin-bottom: 6px;
}
.gauge-marker-line {
  width: 2px;
  height: 22px;
  margin: 0 auto;
  background: #0f172a;
  border-radius: 2px;
}
.gauge-marker-dot {
  width: 10px;
  height: 10px;
  margin: -6px auto 0 auto;
  background: #0f172a;
  border-radius: 999px;
}
.gauge-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 6px;
  color: #6b7280;
  font-weight: 800;
}
.gauge-labels span { font-size: 0.95rem; }
"""
    body = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>{iframe_css}</style>
</head>
<body>
  <div class="card">
    <div class="card-title">{title}</div>
    <div class="card-subtle">{subtitle}</div>
    {bar_gauge_fragment(value, marker_top_px=44, bar_height_px=18)}
  </div>
</body>
</html>
""".strip()

    components.html(body, height=180)


# -----------------------------
# Scoring logic (deterministic)
# -----------------------------
def _label_for_bubble(score_0_100: float) -> Tuple[str, str]:
    s = float(score_0_100)
    if s >= 80:
        return "Bubble: Euphoric", "red"
    if s >= 65:
        return "Bubble: Frothy", "red"
    if s >= 50:
        return "Bubble: Warm", "amber"
    return "Bubble: Normal", "green"


def _label_for_crash_risk(score_0_100: float) -> Tuple[str, str]:
    s = float(score_0_100)
    if s >= 75:
        return "Crash risk: Severe", "red"
    if s >= 55:
        return "Crash risk: High", "red"
    if s >= 35:
        return "Crash risk: Elevated", "amber"
    return "Crash risk: Low", "green"


def _composite_score(items: List[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in items if x is not None and np.isfinite(float(x))]
    if not vals:
        return None
    return float(np.clip(np.mean(vals), 0.0, 100.0))


# -----------------------------
# Plotly helpers
# -----------------------------
def _clip_window(s: pd.Series, asof: pd.Timestamp, lookback_days: int) -> pd.Series:
    s2 = s.dropna().sort_index()
    s2 = s2[s2.index <= asof]
    if s2.empty:
        return s2
    start = asof - pd.Timedelta(days=lookback_days)
    w = s2[s2.index >= start]
    return w if not w.empty else s2


def plotly_line_chart(
    series_list: List[Tuple[str, pd.Series]],
    asof: pd.Timestamp,
    lookback_days: int,
    yaxis_title: str = "",
) -> Optional[go.Figure]:
    valid = [(name, s) for name, s in series_list if s is not None and not s.dropna().empty]
    if not valid:
        return None

    fig = go.Figure()
    for name, s in valid:
        w = _clip_window(s, asof, lookback_days)
        if w.empty:
            continue
        fig.add_trace(go.Scatter(x=w.index, y=w.values, mode="lines", name=name))

    if len(fig.data) == 0:
        return None

    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        xaxis=dict(showgrid=True, gridcolor="rgba(17,24,39,0.08)"),
        yaxis=dict(title=yaxis_title, showgrid=True, gridcolor="rgba(17,24,39,0.08)"),
    )
    return fig


def plotly_bars(
    labels: List[str],
    values: List[float],
    yaxis_title: str = "",
) -> Optional[go.Figure]:
    if not labels or not values:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values))
    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(title=yaxis_title, showgrid=True, gridcolor="rgba(17,24,39,0.08)"),
    )
    return fig


def _normalize_to_z(w: pd.Series) -> pd.Series:
    """
    Z-score normalization computed on the visible window.
    This keeps the plotted lines comparable while preserving the raw values for hover.
    """
    x = w.dropna().astype(float)
    if x.empty:
        return w.astype(float)
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd == 0.0:
        return (w.astype(float) - mu)
    return (w.astype(float) - mu) / sd


def plotly_normalized_with_raw_hover(
    series_raw_list: List[Tuple[str, pd.Series]],
    asof: pd.Timestamp,
    lookback_days: int,
    yaxis_title: str = "Normalized (z-score)",
) -> Optional[go.Figure]:
    """
    Plots normalized series (z-score on the visible window), but hover shows both:
    - Normalized value (y)
    - Raw value (customdata)
    """
    valid = [(name, s) for name, s in series_raw_list if s is not None and not s.dropna().empty]
    if not valid:
        return None

    fig = go.Figure()

    for name, s_raw in valid:
        w_raw = _clip_window(s_raw, asof, lookback_days)
        if w_raw.empty:
            continue

        w_norm = _normalize_to_z(w_raw)

        custom = np.array(w_raw.values, dtype=float).reshape(-1, 1)
        fig.add_trace(
            go.Scatter(
                x=w_norm.index,
                y=w_norm.values,
                mode="lines",
                name=name,
                customdata=custom,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    "%{fullData.name}<br>"
                    "Normalized: %{y:.2f}<br>"
                    "Raw: %{customdata[0]:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    if len(fig.data) == 0:
        return None

    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        xaxis=dict(showgrid=True, gridcolor="rgba(17,24,39,0.08)"),
        yaxis=dict(title=yaxis_title, showgrid=True, gridcolor="rgba(17,24,39,0.08)"),
    )
    return fig


# -----------------------------
# Load state + inputs
# -----------------------------
root = _repo_root()
try:
    state = load_daily_state(root)
except Exception as e:
    st.error(f"Could not load daily_state JSON: {e}")
    st.stop()

as_of = pd.to_datetime(state.get("as_of_date"), errors="coerce")
if pd.isna(as_of):
    as_of = pd.Timestamp.today().normalize()

tilt_signals = state.get("tilt_signals", {}) if isinstance(state.get("tilt_signals", {}), dict) else {}
primary = tilt_signals.get("primary", {}) if isinstance(tilt_signals.get("primary", {}), dict) else {}
if not isinstance(primary, dict):
    primary = {}

mr = state.get("market_regime", {}) if isinstance(state.get("market_regime", {}), dict) else {}
stress_score = _safe_float(mr.get("stress_score"))

us_node = primary.get("US_WEIGHT_WITHIN_EQUITY_BAND", {}) if isinstance(primary.get("US_WEIGHT_WITHIN_EQUITY_BAND", {}), dict) else {}
us_inputs = us_node.get("inputs", {}) if isinstance(us_node.get("inputs", {}), dict) else {}

concentration_composite_state = _safe_float(us_inputs.get("concentration_composite"))
ai_fund_heat_state = _safe_float(us_inputs.get("ai_fundamentals_heat"))
usd_strength_z_state = _safe_float(us_inputs.get("usd_strength_z"))

# -----------------------------
# Load prices for key proxies
# -----------------------------
spy_px = _load_tiingo_prices(root, "SPY")
soxx_px = _load_tiingo_prices(root, "SOXX")
qqq_px = _load_tiingo_prices(root, "QQQ")
rsp_px = _load_tiingo_prices(root, "RSP")

# Ratios
soxx_over_spy = (soxx_px / spy_px).dropna() if soxx_px is not None and spy_px is not None else None
qqq_over_spy = (qqq_px / spy_px).dropna() if qqq_px is not None and spy_px is not None else None
spy_over_rsp = (spy_px / rsp_px).dropna() if spy_px is not None and rsp_px is not None else None

# Returns / run-up
def _return_over(px: Optional[pd.Series], days: int, asof_ts: pd.Timestamp) -> Optional[float]:
    if px is None or px.empty:
        return None
    s = px.dropna().sort_index()
    s = s[s.index <= asof_ts]
    if len(s) < (days + 5):
        return None
    a = float(s.iloc[-1])
    b = float(s.iloc[-(days + 1)])
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return None
    return float(a / b - 1.0)

soxx_ret_3m = _return_over(soxx_px, 63, as_of)
soxx_ret_6m = _return_over(soxx_px, 126, as_of)

# Realized vol
spy_vol_21d = realized_vol_21d_ann(spy_px).dropna() if spy_px is not None and not spy_px.empty else None
spy_vol_last, spy_vol_pct = _percentile_of_last(spy_vol_21d, as_of) if spy_vol_21d is not None else (None, None)

# Percentiles of leadership/concentration proxies
soxx_over_spy_last, soxx_over_spy_pct = _percentile_of_last(soxx_over_spy, as_of) if soxx_over_spy is not None else (None, None)
qqq_over_spy_last, qqq_over_spy_pct = _percentile_of_last(qqq_over_spy, as_of) if qqq_over_spy is not None else (None, None)
spy_over_rsp_last, spy_over_rsp_pct = _percentile_of_last(spy_over_rsp, as_of) if spy_over_rsp is not None else (None, None)

# Run-up percentile proxy (rolling 6m return)
def _rolling_return_series(px: Optional[pd.Series], window_days: int) -> Optional[pd.Series]:
    if px is None or px.empty:
        return None
    s = px.dropna().sort_index()
    return s / s.shift(window_days) - 1.0

soxx_roll_6m = _rolling_return_series(soxx_px, 126)
soxx_roll_6m_last, soxx_roll_6m_pct = _percentile_of_last(soxx_roll_6m, as_of) if soxx_roll_6m is not None else (None, None)

# -----------------------------
# Bubble score + crash risk score
# -----------------------------
bubble_components = [
    soxx_over_spy_pct,
    spy_over_rsp_pct,
    qqq_over_spy_pct,
    soxx_roll_6m_pct,
]
bubble_score = _composite_score(bubble_components)

stress_component = float(np.clip(stress_score, 0.0, 100.0)) if stress_score is not None else None
crash_components = [
    bubble_score,
    stress_component,
    spy_vol_pct,
]
crash_risk = _composite_score(crash_components)

bubble_score = float(bubble_score) if bubble_score is not None else 0.0
crash_risk = float(crash_risk) if crash_risk is not None else 0.0

bubble_label, bubble_dot = _label_for_bubble(bubble_score)
crash_label, crash_dot = _label_for_crash_risk(crash_risk)

# -----------------------------
# Page header
# -----------------------------
st.markdown('<div class="page-title">AI Bubble Diagnostics</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="page-subtitle">Leadership, concentration, and fragility diagnostics (as of {as_of.date().isoformat()}).</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Layout: gauges first (full width)
# -----------------------------
render_gauge_card_iframe(
    title="AI Bubble Score",
    subtitle="Composite of leadership, concentration, and run-up proxies (0–100).",
    value=bubble_score,
)
st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
render_gauge_card_iframe(
    title="Crash Risk (Amplified)",
    subtitle="Bubble score amplified by stress and realized volatility (0–100).",
    value=crash_risk,
)

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

# -----------------------------
# Key drivers (full width)
# -----------------------------
driver_blocks: List[str] = []

if soxx_over_spy_last is not None and soxx_over_spy_pct is not None:
    driver_blocks.append(
        f"""
<div class="driver-item">
  <div class="driver-k">Semis leadership (SOXX/SPY): <span class="driver-v">Percentile: {_fmt_num(soxx_over_spy_pct, 1)}%</span></div>
  <div class="driver-explain">
    Compares semiconductor performance to the broad market. Unusually high percentiles suggest narrow thematic leadership,
    which is a common ingredient in momentum-driven “bubble-like” regimes.
  </div>
</div>
"""
    )

if spy_over_rsp_last is not None and spy_over_rsp_pct is not None:
    driver_blocks.append(
        f"""
<div class="driver-item">
  <div class="driver-k">Concentration proxy (SPY/RSP): <span class="driver-v">Percentile: {_fmt_num(spy_over_rsp_pct, 1)}%</span></div>
  <div class="driver-explain">
    SPY (cap-weighted) outperforming RSP (equal-weighted) indicates breadth is narrowing. Higher percentiles imply the index
    is being driven by fewer large names, increasing fragility to reversals.
  </div>
</div>
"""
    )

if qqq_over_spy_last is not None and qqq_over_spy_pct is not None:
    driver_blocks.append(
        f"""
<div class="driver-item">
  <div class="driver-k">Growth/tech tilt (QQQ/SPY): <span class="driver-v">Percentile: {_fmt_num(qqq_over_spy_pct, 1)}%</span></div>
  <div class="driver-explain">
    Captures growth/tech leadership versus the broad market. Elevated percentiles are consistent with a strong factor tilt
    and can signal crowding when combined with high concentration.
  </div>
</div>
"""
    )

if soxx_ret_3m is not None or soxx_ret_6m is not None:
    driver_blocks.append(
        f"""
<div class="driver-item">
  <div class="driver-k">SOXX run-up (3m / 6m): <span class="driver-v">{_fmt_num(soxx_ret_3m*100 if soxx_ret_3m is not None else None, 2)}% / {_fmt_num(soxx_ret_6m*100 if soxx_ret_6m is not None else None, 2)}%</span></div>
  <div class="driver-explain">
    Measures recent acceleration in the AI-sensitive semiconductor basket. Fast run-ups can mean positioning is one-sided,
    making drawdowns sharper if expectations or liquidity conditions change.
  </div>
</div>
"""
    )

if stress_score is not None or spy_vol_last is not None:
    driver_blocks.append(
        f"""
<div class="driver-item">
  <div class="driver-k">Stress score / SPY realized vol: <span class="driver-v">{_fmt_num(stress_score, 2)} / {_fmt_num((spy_vol_last*100) if spy_vol_last is not None else None, 2)}%</span></div>
  <div class="driver-explain">
    These are “amplifiers.” Higher systemic stress and higher realized volatility increase the probability that crowded
    leadership unwinds abruptly. Low readings reduce near-term amplification, but do not remove valuation/positioning risk.
  </div>
</div>
"""
    )

if concentration_composite_state is not None:
    driver_blocks.append(
        f"""
<div class="driver-item">
  <div class="driver-k">Concentration composite (daily_state): <span class="driver-v">{_fmt_num(concentration_composite_state, 3)}</span></div>
  <div class="driver-explain">
    Internal concentration proxy used elsewhere in the system. Higher values generally indicate narrower leadership and
    more index dependence on a small set of names.
  </div>
</div>
"""
    )

if ai_fund_heat_state is not None:
    driver_blocks.append(
        f"""
<div class="driver-item">
  <div class="driver-k">AI fundamentals heat (SEC context): <span class="driver-v">{_fmt_num(ai_fund_heat_state, 3)}</span></div>
  <div class="driver-explain">
    Slow-moving context signal from fundamentals for an AI-related basket. It is not a timing indicator, but it helps
    distinguish price-only froth from regimes where fundamentals are also improving.
  </div>
</div>
"""
    )

vix = _load_fred_series(root, "VIXCLS")
hy = _load_fred_series(root, "BAMLH0A0HYM2")
ig = _load_fred_series(root, "BAMLC0A0CM")
nfci = _load_fred_series(root, "NFCI")

vix_last = _value_asof(vix, as_of)
hy_last = _value_asof(hy, as_of)
ig_last = _value_asof(ig, as_of)
nfci_last = _value_asof(nfci, as_of)

if any(x is not None for x in [vix_last, hy_last, ig_last, nfci_last]):
    driver_blocks.append(
        f"""
<div class="driver-item">
  <div class="driver-k">VIX / HY OAS / IG OAS / NFCI: <span class="driver-v">{_fmt_num(vix_last, 2)} / {_fmt_num(hy_last, 2)} / {_fmt_num(ig_last, 2)} / {_fmt_num(nfci_last, 3)}</span></div>
  <div class="driver-explain">
    These indicators summarize whether froth is paired with tightening financial conditions. Rising spreads or tighter
    conditions can turn “frothy but stable” into “fragile and unstable” more quickly.
  </div>
</div>
"""
    )

drivers_html = "\n".join(driver_blocks) if driver_blocks else "<div class='card-subtle'>Not enough data to render driver diagnostics.</div>"

st.markdown(
    f"""
<div class="card">
  <div class="card-title">Key drivers</div>
  <div class="pills">
    {pill_html(bubble_label, dot=bubble_dot)}
    {pill_html(crash_label, dot=crash_dot)}
  </div>

  <div class="card-subtle" style="font-weight:900; text-transform:uppercase; letter-spacing:0.08em; margin-top:10px;">
    What is pushing the diagnosis
  </div>

  {drivers_html}
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<hr class="soft"/>', unsafe_allow_html=True)

# -----------------------------
# Diagnostics charts
# -----------------------------
st.markdown('<div class="section-title">Diagnostics charts</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="card-subtle" style="margin-bottom:10px;">Charts use the full locally available history. If your parquet starts in 2024, the chart will reflect that; otherwise it will automatically show a longer window.</div>',
    unsafe_allow_html=True,
)

chart_lookback_days = DEFAULT_CHART_LOOKBACK_DAYS


def chart_block(title: str, desc: str, fig: Optional[go.Figure]) -> None:
    st.markdown(
        f"<div class='chart-block'><div class='chart-title'>{title}</div><div class='chart-desc'>{desc}</div></div>",
        unsafe_allow_html=True,
    )
    if fig is None:
        st.info(f"{title}: not enough data to plot.")
        return
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


# 1) Leadership & concentration ratios
fig = plotly_line_chart(
    [("SOXX/SPY", soxx_over_spy)] if soxx_over_spy is not None else [],
    asof=as_of,
    lookback_days=chart_lookback_days,
)
chart_block(
    title="Semis leadership ratio (SOXX / SPY)",
    desc=(
        "This line tracks semiconductor performance relative to the broad market. A rising or persistently elevated ratio "
        "suggests leadership is concentrated in semis—often the core of the “AI trade.” If it is extreme versus its own history, "
        "it can indicate crowding and late-cycle froth."
    ),
    fig=fig,
)

fig = plotly_line_chart(
    [("QQQ/SPY", qqq_over_spy)] if qqq_over_spy is not None else [],
    asof=as_of,
    lookback_days=chart_lookback_days,
)
chart_block(
    title="Growth/tech tilt ratio (QQQ / SPY)",
    desc=(
        "This ratio compares growth/tech-heavy Nasdaq exposure to the broad market. Higher levels indicate a strong growth/tech tilt. "
        "In bubble-like regimes, it often moves up together with concentration and semis leadership."
    ),
    fig=fig,
)

fig = plotly_line_chart(
    [("SPY/RSP", spy_over_rsp)] if spy_over_rsp is not None else [],
    asof=as_of,
    lookback_days=chart_lookback_days,
)
chart_block(
    title="Concentration proxy (SPY / RSP)",
    desc=(
        "This compares cap-weighted performance (SPY) to equal-weighted performance (RSP). When it rises, breadth is narrowing: "
        "fewer large names drive returns. Persistent elevation is a classic fragility ingredient because reversals can be abrupt."
    ),
    fig=fig,
)

st.markdown('<hr class="soft"/>', unsafe_allow_html=True)

# 2) Run-up & path comparison
if soxx_px is not None and spy_px is not None and (not soxx_px.empty) and (not spy_px.empty):
    start = as_of - pd.Timedelta(days=chart_lookback_days)
    soxx_w = soxx_px[(soxx_px.index >= start) & (soxx_px.index <= as_of)].dropna()
    spy_w = spy_px[(spy_px.index >= start) & (spy_px.index <= as_of)].dropna()
    if not soxx_w.empty and not spy_w.empty:
        soxx_n = 100.0 * (soxx_w / float(soxx_w.iloc[0]))
        spy_n = 100.0 * (spy_w / float(spy_w.iloc[0]))
        fig = plotly_line_chart(
            [("SOXX (normalized)", soxx_n), ("SPY (normalized)", spy_n)],
            asof=as_of,
            lookback_days=chart_lookback_days,
            yaxis_title="Indexed (start = 100)",
        )
    else:
        fig = None
else:
    fig = None

chart_block(
    title="Run-up comparison (normalized paths)",
    desc=(
        "Both series are rebased to 100 at the start of the window. This helps compare how much the AI-sensitive semis basket "
        "has run ahead of the broad market. A widening gap indicates leadership acceleration, which can increase sensitivity to "
        "disappointment or tightening financial conditions."
    ),
    fig=fig,
)

labels: List[str] = []
values: List[float] = []
if soxx_ret_3m is not None:
    labels.append("SOXX 3m")
    values.append(100.0 * soxx_ret_3m)
if soxx_ret_6m is not None:
    labels.append("SOXX 6m")
    values.append(100.0 * soxx_ret_6m)

fig = plotly_bars(labels, values, yaxis_title="Return (%)") if labels else None
chart_block(
    title="SOXX trailing returns (run-up)",
    desc=(
        "This shows recent trailing returns for SOXX over 3 and 6 months. Large positive run-ups are not a crash signal by themselves, "
        "but they increase fragility because positioning can become one-sided and downside moves can be faster."
    ),
    fig=fig,
)

st.markdown('<hr class="soft"/>', unsafe_allow_html=True)

# 3) Stress & volatility context
if spy_vol_21d is not None and not spy_vol_21d.empty:
    fig = plotly_line_chart(
        [("SPY vol (21d, ann.)", spy_vol_21d * 100.0)],
        asof=as_of,
        lookback_days=chart_lookback_days,
        yaxis_title="Percent (%)",
    )
else:
    fig = None

chart_block(
    title="SPY realized volatility (21d, annualized)",
    desc=(
        "Realized volatility is a backward-looking turbulence measure. When volatility rises, drawdowns can become more abrupt—"
        "especially if leadership is crowded. Low volatility reduces near-term amplification but can also allow froth to persist longer."
    ),
    fig=fig,
)

# --- CHANGE #1: normalized chart hover shows both normalized + raw ---
stress_raw_list: List[Tuple[str, pd.Series]] = []
if vix is not None and not vix.empty:
    stress_raw_list.append(("VIX", vix))
if hy is not None and not hy.empty:
    stress_raw_list.append(("HY OAS", hy))
if ig is not None and not ig.empty:
    stress_raw_list.append(("IG OAS", ig))

fig = (
    plotly_normalized_with_raw_hover(stress_raw_list, asof=as_of, lookback_days=chart_lookback_days)
    if stress_raw_list
    else None
)

chart_block(
    title="Market stress amplifiers (normalized)",
    desc=(
        "Lines are normalized (z-score on the visible window) to make shapes comparable, but hover shows both the normalized value "
        "and the raw level for that date.\n\n"
        "If VIX runs above or below credit spreads for a sustained period, it can indicate a divergence between equity-option fear "
        "and credit risk pricing. For example, a VIX spike without spread widening can be a short-lived equity shock (or positioning unwind) "
        "that credit has not repriced yet. Conversely, spreads widening with a muted VIX can reflect slower-moving credit deterioration "
        "that equities have not fully reflected."
    ),
    fig=fig,
)

st.markdown('<hr class="soft"/>', unsafe_allow_html=True)


# -----------------------------
# Appendix (kept; improved formatting in white cards)
# -----------------------------
st.markdown('<hr class="soft"/>', unsafe_allow_html=True)
st.markdown("## Appendix")

def _card(title: str, body_html: str) -> None:
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">{title}</div>
  <div class="card-subtle" style="margin-top:6px;">
    {body_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# --- Interpretation bands (AI Bubble Score + Crash Risk) ---
_card(
    "Interpretation bands (AI Bubble Score and Crash Risk)",
    """
<div style="font-weight:800; color:#111827; margin-bottom:10px; line-height:1.7;">
  The ranges below apply to both <span style="font-weight:900;">AI Bubble Score</span> and
  <span style="font-weight:900;">Crash Risk (Amplified)</span>.
  Scores are on a <span style="font-weight:900;">0–100 percentile scale</span> (higher = more extreme vs the proxy’s trailing history up to the as-of date).
</div>

<div style="display:flex; gap:14px; flex-wrap:wrap; margin-bottom:14px;">
  <div style="flex:1; min-width:260px; padding:12px 14px; border:1px solid #e5e7eb; border-radius:14px; background:#f9fafb;">
    <div style="font-weight:900; color:#111827; margin-bottom:6px;">AI Bubble Score labels</div>
    <ul style="margin:0; padding-left:18px; line-height:1.7;">
      <li><span style="font-weight:900;">0–49</span>: <span style="font-weight:900;">Bubble: Normal</span></li>
      <li><span style="font-weight:900;">50–64</span>: <span style="font-weight:900;">Bubble: Warm</span></li>
      <li><span style="font-weight:900;">65–79</span>: <span style="font-weight:900;">Bubble: Frothy</span></li>
      <li><span style="font-weight:900;">80–100</span>: <span style="font-weight:900;">Bubble: Euphoric</span></li>
    </ul>
    <div style="margin-top:8px; color:#6b7280; font-weight:800; line-height:1.6;">
    </div>
  </div>

  <div style="flex:1; min-width:260px; padding:12px 14px; border:1px solid #e5e7eb; border-radius:14px; background:#f9fafb;">
    <div style="font-weight:900; color:#111827; margin-bottom:6px;">Crash Risk labels</div>
    <ul style="margin:0; padding-left:18px; line-height:1.7;">
      <li><span style="font-weight:900;">0–34</span>: <span style="font-weight:900;">Crash risk: Low</span></li>
      <li><span style="font-weight:900;">35–54</span>: <span style="font-weight:900;">Crash risk: Elevated</span></li>
      <li><span style="font-weight:900;">55–74</span>: <span style="font-weight:900;">Crash risk: High</span></li>
      <li><span style="font-weight:900;">75–100</span>: <span style="font-weight:900;">Crash risk: Severe</span></li>
    </ul>
    <div style="margin-top:8px; color:#6b7280; font-weight:800; line-height:1.6;">
    </div>
  </div>
</div>
""",
)

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

# --- How the scores are computed ---
_card(
    "How the scores are computed",
    """
<div style="font-weight:900; color:#111827; margin-bottom:8px;">Core idea</div>
<div style="line-height:1.7; margin-bottom:12px;">
  Each proxy is converted into a <span style="font-weight:900;">percentile score (0–100)</span> vs its own trailing history up to the as-of date.
  Those percentiles are then averaged into two composites:
  <span style="font-weight:900;">AI Bubble Score</span> and <span style="font-weight:900;">Crash Risk (Amplified)</span>.
</div>

<div style="font-weight:900; color:#111827; margin-bottom:6px;">Step 1 — Build proxy series</div>
<ul style="margin:0 0 12px 0; padding-left:18px; line-height:1.7;">
  <li><span style="font-weight:900;">SOXX/SPY</span>: semis leadership vs broad market</li>
  <li><span style="font-weight:900;">QQQ/SPY</span>: growth/tech tilt vs broad market</li>
  <li><span style="font-weight:900;">SPY/RSP</span>: breadth / concentration proxy (cap-weighted vs equal-weighted)</li>
  <li><span style="font-weight:900;">SOXX 6m rolling return</span>: run-up proxy, computed as <span style="font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">SOXX / SOXX.shift(126) - 1</span></li>
  <li><span style="font-weight:900;">Stress score</span>: from <span style="font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">daily_state.market_regime.stress_score</span> (already 0–100)</li>
  <li><span style="font-weight:900;">SPY realized vol</span>: 21d annualized, <span style="font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">std(ret, 21) * sqrt(252)</span></li>
</ul>

<div style="font-weight:900; color:#111827; margin-bottom:6px;">Step 2 — Convert each proxy into a percentile (0–100)</div>
<div style="line-height:1.7; margin-bottom:12px;">
  For each proxy series <span style="font-weight:900;">x(t)</span>, using only data up to the as-of date, compute:
  <div style="margin-top:8px; padding:10px 12px; background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; display:inline-block;">
    <span style="font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">
      percentile = 100 * mean(x ≤ x_asof)
    </span>
  </div>
  <div style="margin-top:10px; color:#6b7280; font-weight:800;">
    Higher percentiles mean the proxy is more extreme relative to its own historical distribution.
  </div>
</div>

<div style="font-weight:900; color:#111827; margin-bottom:6px;">Step 3 — Aggregate percentiles into composites</div>

<div style="margin-bottom:10px; line-height:1.7;">
  <span style="font-weight:900; color:#111827;">AI Bubble Score (0–100)</span> = mean of available proxy percentiles:
</div>
<ul style="margin:0 0 12px 0; padding-left:18px; line-height:1.7;">
  <li>SOXX/SPY percentile (semis leadership)</li>
  <li>SPY/RSP percentile (breadth / concentration)</li>
  <li>QQQ/SPY percentile (growth/tech tilt)</li>
  <li>SOXX 6m rolling return percentile (run-up)</li>
</ul>

<div style="margin-bottom:10px; line-height:1.7;">
  <span style="font-weight:900; color:#111827;">Crash Risk (Amplified) (0–100)</span> = mean of:
</div>
<ul style="margin:0; padding-left:18px; line-height:1.7;">
  <li>Bubble score</li>
  <li>Stress score (from <span style="font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">daily_state.market_regime.stress_score</span>)</li>
  <li>SPY realized vol percentile (21d annualized)</li>
</ul>

<div style="margin-top:12px; color:#6b7280; font-weight:800; line-height:1.7;">
  Interpretation: crash risk rises when bubble-like leadership coincides with higher stress and/or higher realized volatility.
</div>
""",
)

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

# --- Notes ---
_card(
    "Notes",
    """
<ul style="margin:0; padding-left:18px; line-height:1.7;">
  <li>
    Percentiles are computed over the trailing window up to the as-of date. If local history is short (e.g., data starts in 2024),
    percentiles remain mathematically valid but provide less full-cycle context.
  </li>
  <li>
    The normalized stress chart is for shape comparison only; interpret raw levels on hover.
  </li>
</ul>
""",
)
