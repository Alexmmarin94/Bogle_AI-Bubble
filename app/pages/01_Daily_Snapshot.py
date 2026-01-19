#!/usr/bin/env python3
# app/pages/01_Daily_Snapshot.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Daily Snapshot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Styling (Streamlit page scope)
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
.card-subtle { color: #6b7280; font-size: 0.92rem; margin-top: 2px; }

.mini-row { margin-top: 0.35rem; }
.mini-card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 18px;
  box-shadow: 0 6px 18px rgba(17, 24, 39, 0.06);
  padding: 14px 16px;
}
.mini-label { color: #6b7280; font-size: 0.9rem; font-weight: 700; margin-bottom: 6px; }
.mini-value { font-size: 1.35rem; font-weight: 900; color: #0f172a; }

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

.band-card {
  min-height: 360px;
  height: 100%;
  display: flex;
  flex-direction: column;
}
.band-title-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.band-title { font-size: 1.45rem; font-weight: 900; margin: 0; color: #0f172a; }
.arrow { font-size: 1.22rem; font-weight: 900; line-height: 1; }
.arrow-up { color: #22c55e; }
.arrow-down { color: #ef4444; }
.arrow-flat { color: #f59e0b; }

.drivers-title {
  color: #6b7280;
  font-size: 0.86rem;
  font-weight: 900;
  margin-top: 2px;
  margin-bottom: 6px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.drivers-list { color: #6b7280; font-size: 0.95rem; line-height: 1.55; }
.driver-item { margin: 2px 0; }
.driver-k { color: #111827; font-weight: 800; }
.driver-v { color: #6b7280; }

hr.soft { border: none; border-top: 1px solid #e5e7eb; margin: 18px 0; }

/* ---- Glossary table (wrap + explicit black text) ---- */
.glossary-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  overflow: hidden;
  background: #ffffff;
}
.glossary-table th, .glossary-table td {
  padding: 10px 12px;
  border-bottom: 1px solid #e5e7eb;
  vertical-align: top;
  color: #111827;           /* force black text */
}
.glossary-table th {
  background: #f9fafb;
  font-weight: 900;
  white-space: nowrap;
}
.glossary-table td {
  white-space: normal;
  word-break: break-word;
}
.glossary-table tr:last-child td { border-bottom: none; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Paths / loaders
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
# Helpers
# -----------------------------
TRADING_DAYS = 252
LOOKBACK_DAYS = 3650


def realized_vol_21d_ann(px: pd.Series) -> pd.Series:
    ret = px.pct_change(fill_method=None)
    vol = ret.rolling(21).std(ddof=0) * np.sqrt(TRADING_DAYS)
    return vol


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


def _bucket_human(bucket: Optional[str]) -> str:
    if not bucket:
        return "Unknown"
    m = {"LOW_STRESS": "Low Stress", "MID_STRESS": "Mid Stress", "HIGH_STRESS": "High Stress"}
    return m.get(bucket, bucket.replace("_", " ").title())


def _dir_human(direction: Optional[str]) -> str:
    if not direction:
        return "Hold"
    m = {"TILT_TOWARD": "Tilt Toward", "TILT_AWAY": "Tilt Away", "HOLD": "Hold"}
    return m.get(direction, direction.replace("_", " ").title())


def _arrow_for(direction: Optional[str]) -> Tuple[str, str]:
    d = (direction or "").upper()
    if d == "TILT_TOWARD":
        return "▲", "arrow-up"
    if d == "TILT_AWAY":
        return "▼", "arrow-down"
    return "▶", "arrow-flat"


def _persisted_yesno(persisted_direction: Optional[str]) -> str:
    return "Yes" if persisted_direction else "No"


def mini_card_html(label: str, value: str) -> str:
    return (
        f'<div class="mini-card">'
        f'<div class="mini-label">{label}</div>'
        f'<div class="mini-value">{value}</div>'
        f"</div>"
    )


def pill_html(text: str, dot: str = "gray") -> str:
    dot_class = {"green": "dot-green", "red": "dot-red", "amber": "dot-amber", "gray": "dot-gray"}.get(dot, "dot-gray")
    return (
        f'<span class="pill">'
        f'<span class="dot {dot_class}"></span>'
        f"{text}"
        f"</span>"
    )


# -----------------------------
# Stress bar gauge (HTML fragment)
# -----------------------------
def stress_gauge_bar_fragment(value_0_100: float) -> str:
    v = float(np.clip(value_0_100, 0.0, 100.0))
    v_int = int(round(v))
    return (
        '<div class="stress-bar-wrap">'
        '<div class="stress-bar">'
        f'<div class="stress-marker" style="left:{v:.2f}%;">'
        f'<div class="stress-marker-value">{v_int}</div>'
        '<div class="stress-marker-line"></div>'
        '<div class="stress-marker-dot"></div>'
        "</div>"
        "</div>"
        '<div class="stress-bar-labels"><span>0</span><span>100</span></div>'
        "</div>"
    )


# -----------------------------
# Stress drivers + ranges (only the drivers you want in Stress)
# -----------------------------
def compute_stress_driver_values_and_ranges(
    root: Path,
    asof: pd.Timestamp,
) -> Tuple[Dict[str, float], Dict[str, Tuple[Optional[float], Optional[float]]]]:
    drivers: Dict[str, float] = {}
    ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    fred_labels = {
        "VIXCLS": "VIX (implied volatility)",
        "BAMLH0A0HYM2": "High-yield spread (HY OAS)",
        "BAMLC0A0CM": "Investment-grade spread (IG OAS)",
        "NFCI": "Chicago Fed NFCI (financial conditions)",
    }

    series_map: Dict[str, pd.Series] = {}
    for sid in fred_labels:
        s = _load_fred_series(root, sid)
        if s is not None and not s.empty:
            series_map[sid] = s

    for sid, label in fred_labels.items():
        s = series_map.get(sid)
        v = _value_asof(s, asof)
        if v is not None:
            drivers[label] = float(v)

        if s is not None and not s.empty:
            s2 = s.dropna().sort_index()
            start = asof - pd.Timedelta(days=LOOKBACK_DAYS)
            s2 = s2[(s2.index <= asof) & (s2.index >= start)]
            ranges[label] = (float(s2.min()), float(s2.max())) if not s2.empty else (None, None)
        else:
            ranges[label] = (None, None)

    spy_px = _load_tiingo_prices(root, "SPY")
    if spy_px is not None and not spy_px.empty:
        vol = realized_vol_21d_ann(spy_px)
        v = _value_asof(vol, asof)
        if v is not None:
            drivers["SPY realized volatility (21d, annualized)"] = float(v)

        vol2 = vol.dropna().sort_index()
        start = asof - pd.Timedelta(days=LOOKBACK_DAYS)
        vol2 = vol2[(vol2.index <= asof) & (vol2.index >= start)]
        ranges["SPY realized volatility (21d, annualized)"] = (float(vol2.min()), float(vol2.max())) if not vol2.empty else (None, None)
    else:
        ranges["SPY realized volatility (21d, annualized)"] = (None, None)

    return drivers, ranges


def stress_top_drivers_from_state(
    state: Dict[str, Any],
    driver_values: Dict[str, float],
) -> List[Tuple[str, float]]:
    mr = state.get("market_regime", {}) if isinstance(state.get("market_regime", {}), dict) else {}
    td = mr.get("top_drivers")
    if isinstance(td, list) and td and all(isinstance(x, dict) for x in td):
        out: List[Tuple[str, float]] = []
        for x in td:
            label = x.get("label") or x.get("name")
            val = x.get("value")
            if label and val is not None:
                fv = _safe_float(val)
                if fv is not None:
                    out.append((str(label), float(fv)))
        if out:
            return out[:5]

    fallback_order = [
        "High-yield spread (HY OAS)",
        "Chicago Fed NFCI (financial conditions)",
        "SPY realized volatility (21d, annualized)",
        "Investment-grade spread (IG OAS)",
        "VIX (implied volatility)",
    ]
    out2: List[Tuple[str, float]] = []
    for k in fallback_order:
        if k in driver_values:
            out2.append((k, float(driver_values[k])))
    return out2[:5]


# -----------------------------
# Snapshot text
# -----------------------------
def overall_posture_from_equity(direction: Optional[str]) -> str:
    d = (direction or "").upper()
    if d == "TILT_TOWARD":
        return "Risk-on within band"
    if d == "TILT_AWAY":
        return "Risk-off within band"
    return "Neutral"


def trades_suggested(primary: Dict[str, Any]) -> str:
    for _, node in primary.items():
        if not isinstance(node, dict):
            continue
        d = (node.get("direction") or "").upper()
        persisted = node.get("persisted_direction") is not None
        if persisted and d in {"TILT_TOWARD", "TILT_AWAY"}:
            return "Optional rebalance"
    return "None"


def human_friendly_band_names() -> Dict[str, str]:
    return {
        "EQUITY_WEIGHT_WITHIN_BAND": "Equity within band",
        "BOND_DURATION_WITHIN_BAND": "Duration within bond band",
        "TIPS_SLICE_WITHIN_BAND": "TIPS slice",
        "US_WEIGHT_WITHIN_EQUITY_BAND": "US vs ex-US",
    }


def drivers_for_band(key: str, node: Dict[str, Any]) -> List[Tuple[str, str]]:
    inputs = node.get("inputs", {}) if isinstance(node.get("inputs", {}), dict) else {}
    out: List[Tuple[str, str]] = []

    if key == "EQUITY_WEIGHT_WITHIN_BAND":
        out = [
            ("SPY vs 200-day trend", _fmt_num(_safe_float(inputs.get("spy_200d_trend")), digits=3)),
            ("SPY momentum (12 months)", _fmt_num(_safe_float(inputs.get("spy_momentum_12m")), digits=3)),
            ("SPY momentum (3 months)", _fmt_num(_safe_float(inputs.get("spy_momentum_3m")), digits=3)),
            ("SPY realized volatility (21d, annualized)", _fmt_num(_safe_float(inputs.get("spy_realized_vol_21d_ann")), digits=3)),
            ("3-month cash yield (DGS3MO)", _fmt_num(_safe_float(inputs.get("cash_yield_3m")), digits=3)),
            ("10Y–3M slope (T10Y3M)", _fmt_num(_safe_float(inputs.get("t10y3m")), digits=3)),
            ("Stress bucket", _bucket_human(inputs.get("stress_bucket"))),
        ]
    elif key == "BOND_DURATION_WITHIN_BAND":
        out = [
            ("10Y yield change (30 days)", _fmt_num(_safe_float(inputs.get("dgs10_change_30d")), digits=3)),
            ("Rate volatility (10Y, 63d)", _fmt_num(_safe_float(inputs.get("rate_vol_63d")), digits=3)),
            ("5Y Treasury yield", _fmt_num(_safe_float(inputs.get("dgs5")), digits=3)),
            ("10Y Treasury yield", _fmt_num(_safe_float(inputs.get("dgs10")), digits=3)),
            ("30Y Treasury yield", _fmt_num(_safe_float(inputs.get("dgs30")), digits=3)),
            ("10Y–3M slope (T10Y3M)", _fmt_num(_safe_float(inputs.get("t10y3m")), digits=3)),
            ("Stress bucket", _bucket_human(inputs.get("stress_bucket"))),
        ]
    elif key == "TIPS_SLICE_WITHIN_BAND":
        out = [
            ("10Y breakeven inflation (T10YIE)", _fmt_num(_safe_float(inputs.get("t10y_breakeven")), digits=3)),
            ("10Y breakeven change (30d)", _fmt_num(_safe_float(inputs.get("t10y_breakeven_change_30d")), digits=3)),
            ("10Y real yield (DFII10)", _fmt_num(_safe_float(inputs.get("dfii10_real_yield")), digits=3)),
            ("10Y real yield change (30d)", _fmt_num(_safe_float(inputs.get("dfii10_real_yield_change_30d")), digits=3)),
            ("5Y5Y forward inflation (T5YIFR)", _fmt_num(_safe_float(inputs.get("t5y5y_forward_infl")), digits=3)),
        ]
    elif key == "US_WEIGHT_WITHIN_EQUITY_BAND":
        out = [
            ("Equity concentration composite", _fmt_num(_safe_float(inputs.get("concentration_composite")), digits=3)),
            ("USD strength (broad index z-score)", _fmt_num(_safe_float(inputs.get("usd_strength_z")), digits=3)),
            ("AI fundamentals heat (SEC, context only)", _fmt_num(_safe_float(inputs.get("ai_fundamentals_heat")), digits=3)),
            ("Stress bucket", _bucket_human(inputs.get("stress_bucket"))),
        ]
    return out


def band_card_html(
    title: str,
    direction: Optional[str],
    magnitude: Optional[str],
    persisted_direction: Optional[str],
    drivers: List[Tuple[str, Any]],
) -> str:
    arrow, arrow_class = _arrow_for(direction)
    dir_label = _dir_human(direction)
    mag_label = (magnitude or "—").title()
    persisted = _persisted_yesno(persisted_direction)

    if (direction or "").upper() == "TILT_TOWARD":
        dot = "green"
    elif (direction or "").upper() == "TILT_AWAY":
        dot = "red"
    elif (direction or "").upper() == "HOLD":
        dot = "amber"
    else:
        dot = "gray"

    pills = (
        pill_html(dir_label, dot=dot)
        + pill_html(f"Magnitude: {mag_label}", dot="gray")
        + pill_html("Persisted" if persisted == "Yes" else "Not persisted", dot="green" if persisted == "Yes" else "amber")
    )

    drivers_html = ""
    for k, v in drivers:
        drivers_html += (
            '<div class="driver-item">'
            f'<span class="driver-k">{k}:</span> '
            f'<span class="driver-v">{v}</span>'
            "</div>"
        )

    return (
        '<div class="card band-card">'
        '<div class="band-title-row">'
        f'<div class="arrow {arrow_class}">{arrow}</div>'
        f'<div class="band-title">{title}</div>'
        "</div>"
        f'<div class="pills">{pills}</div>'
        '<div class="drivers-title">Top drivers</div>'
        f'<div class="drivers-list">{drivers_html}</div>'
        "</div>"
    )


# -----------------------------
# Glossary helpers
# -----------------------------
def _range_str_from_history(label: str, ranges: Dict[str, Tuple[Optional[float], Optional[float]]], fallback: str = "Unbounded") -> str:
    mn, mx = ranges.get(label, (None, None))
    if mn is None or mx is None or (not np.isfinite(mn)) or (not np.isfinite(mx)):
        return fallback
    return f"{_fmt_num(mn, digits=3)} .. {_fmt_num(mx, digits=3)}"


def _render_glossary_section(title: str, rows: List[Dict[str, str]]) -> None:
    import html

    st.markdown(f"### {title}")
    df = pd.DataFrame(rows)

    col_widths = {
        "Driver": "20%",
        "What it is": "58%",
        "Units": "9%",
        "Scale (min .. max)": "13%",
    }

    cols = list(df.columns)
    colgroup = "<colgroup>" + "".join(
        f'<col style="width:{html.escape(col_widths.get(c, "auto"))};"/>' for c in cols
    ) + "</colgroup>"

    thead = "<thead><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols) + "</tr></thead>"

    body_rows: List[str] = []
    for _, r in df.iterrows():
        tds: List[str] = []
        for c in cols:
            val = "" if pd.isna(r[c]) else str(r[c])
            tds.append(f"<td>{html.escape(val)}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")
    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"

    st.markdown(
        f"""
<table class="glossary-table">
  {colgroup}
  {thead}
  {tbody}
</table>
""",
        unsafe_allow_html=True,
    )


# -----------------------------
# Load state
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

primary = state.get("tilt_signals", {}).get("primary", {}) if isinstance(state.get("tilt_signals", {}), dict) else {}
if not isinstance(primary, dict):
    primary = {}

equity_node = primary.get("EQUITY_WEIGHT_WITHIN_BAND", {}) if isinstance(primary.get("EQUITY_WEIGHT_WITHIN_BAND", {}), dict) else {}
overall_posture = overall_posture_from_equity(equity_node.get("direction"))
trade_text = trades_suggested(primary)

mr = state.get("market_regime", {}) if isinstance(state.get("market_regime", {}), dict) else {}
stress_score = _safe_float(mr.get("stress_score"))
stress_score = float(stress_score) if stress_score is not None else float("nan")

stress_bucket = mr.get("stress_bucket")
stress_bucket_h = _bucket_human(stress_bucket)

driver_values, driver_ranges = compute_stress_driver_values_and_ranges(root, as_of)
stress_top = stress_top_drivers_from_state(state, driver_values)

# -----------------------------
# Page header
# -----------------------------
st.markdown('<div class="page-title">Daily Snapshot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Banded tilts, stress regime, and top drivers (as of the latest daily state).</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Top summary cards
# -----------------------------
st.markdown('<div class="mini-row"></div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown(mini_card_html("As of", as_of.date().isoformat()), unsafe_allow_html=True)
with c2:
    st.markdown(mini_card_html("Overall posture", overall_posture), unsafe_allow_html=True)
with c3:
    st.markdown(mini_card_html("Trades suggested", trade_text), unsafe_allow_html=True)

# -----------------------------
# Stress section (iframe card)
# -----------------------------
st.markdown('<div class="section-title">Stress</div>', unsafe_allow_html=True)

bucket_dot = "gray"
if (stress_bucket or "") == "LOW_STRESS":
    bucket_dot = "green"
elif (stress_bucket or "") == "MID_STRESS":
    bucket_dot = "amber"
elif (stress_bucket or "") == "HIGH_STRESS":
    bucket_dot = "red"

stress_right_items = ""
for label, val in stress_top:
    stress_right_items += (
        '<div class="driver-item">'
        f'<span class="driver-k">{label}:</span> '
        f'<span class="driver-v">{_fmt_num(val, digits=3)}</span>'
        "</div>"
    )

gauge_fragment = (
    stress_gauge_bar_fragment(stress_score)
    if np.isfinite(stress_score)
    else '<div class="no-score">No stress score available</div>'
)

stress_iframe_css = """
:root { color-scheme: light; }
html, body { margin: 0; padding: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }

/* Outer shell to keep one shadow around both blocks */
.shell {
  border: 1px solid #e5e7eb;
  border-radius: 18px;
  box-shadow: 0 6px 18px rgba(17, 24, 39, 0.06);
  background: transparent;
  overflow: hidden; /* ensures the two panels clip nicely */
}

/* Two panels, glued together (no gap) */
.panels {
  display: grid;
  grid-template-columns: 0.60fr 0.40fr;
  gap: 0;
}

@media (max-width: 900px) {
  .panels { grid-template-columns: 1fr; }
  .panel-left, .panel-right { border-radius: 0 !important; }
  .panel-right { border-top: 1px solid #e5e7eb; border-left: none !important; }
}

.panel {
  background: #ffffff;
  padding: 18px 18px 16px 18px;
}

.panel-left {
  border-right: 1px solid #e5e7eb; /* divider line */
}

.panel-right {
  display: flex;
  flex-direction: column;
  justify-content: center;  /* vertical auto-centering */
}

/* Typography */
.card-title { font-size: 1.05rem; font-weight: 800; color: #111827; margin: 0 0 8px 0; }
.card-subtle { color: #6b7280; font-size: 0.92rem; margin-top: 2px; }

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

.drivers-title {
  color: #6b7280;
  font-size: 0.86rem;
  font-weight: 900;
  margin: 8px 0 6px 0;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.drivers-list { color: #6b7280; font-size: 0.95rem; line-height: 1.55; }
.driver-item { margin: 2px 0; }
.driver-k { color: #111827; font-weight: 800; }
.driver-v { color: #6b7280; }

/* Gauge */
.stress-bar-wrap { margin-top: 10px; }
.stress-bar {
  position: relative;
  height: 18px;
  border-radius: 999px;
  background: linear-gradient(90deg, #22c55e 0%, #f59e0b 50%, #ef4444 100%);
  box-shadow: inset 0 0 0 1px rgba(17, 24, 39, 0.08);
}
.stress-marker {
  position: absolute;
  top: -34px;
  transform: translateX(-50%);
  text-align: center;
}
.stress-marker-value {
  font-size: 0.95rem;
  font-weight: 900;
  color: #0f172a;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 999px;
  padding: 4px 10px;
  box-shadow: 0 6px 18px rgba(17, 24, 39, 0.08);
  display: inline-block;
}
.stress-marker-line {
  width: 2px;
  height: 22px;
  margin: 6px auto 0 auto;
  background: #0f172a;
  border-radius: 2px;
}
.stress-marker-dot {
  width: 10px;
  height: 10px;
  margin: -6px auto 0 auto;
  background: #0f172a;
  border-radius: 999px;
}
.stress-bar-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 6px;
  color: #6b7280;
  font-weight: 800;
}
.stress-bar-labels span { font-size: 0.95rem; }

.no-score {
  height: 70px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
  font-weight: 800;
  border: 1px dashed #e5e7eb;
  border-radius: 12px;
  margin-top: 10px;
}
"""

stress_iframe_html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>{stress_iframe_css}</style>
</head>
<body>
  <div class="shell">
    <div class="panels">
      <div class="panel panel-left">
        <div class="card-title">Market regime</div>
        <div class="pills">
          {pill_html(stress_bucket_h, dot=bucket_dot)}
        </div>
        <div class="card-subtle" style="margin-top:6px;">Stress gauge</div>
        {gauge_fragment}
      </div>

      <div class="panel panel-right">
        <div>
          <div class="card-title" style="margin:0 0 6px 0;">Stress drivers</div>
          <div class="drivers-title">Top drivers</div>
          <div class="drivers-list">
            {stress_right_items}
          </div>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
""".strip()

# Height: allow for both panels; add extra room if many drivers
stress_height = int(max(260, 240 + 26 * max(0, len(stress_top) - 4)))
components.html(stress_iframe_html, height=stress_height)

st.markdown('<hr class="soft"/>', unsafe_allow_html=True)

# -----------------------------
# Band guidance
# -----------------------------
st.markdown('<div class="section-title">Band guidance</div>', unsafe_allow_html=True)

names = human_friendly_band_names()
ordered = [
    "EQUITY_WEIGHT_WITHIN_BAND",
    "BOND_DURATION_WITHIN_BAND",
    "TIPS_SLICE_WITHIN_BAND",
    "US_WEIGHT_WITHIN_EQUITY_BAND",
]

row1 = st.columns(2, gap="medium")
row2 = st.columns(2, gap="medium")


def render_band_card(key: str) -> None:
    node = primary.get(key, {})
    if not isinstance(node, dict) or not node:
        st.markdown(
            (
                '<div class="card band-card">'
                '<div class="band-title-row">'
                '<div class="arrow arrow-flat">▶</div>'
                f'<div class="band-title">{names.get(key, key)}</div>'
                "</div>"
                '<div class="card-subtle">No data available for this tilt.</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    drivers = drivers_for_band(key, node)
    st.markdown(
        band_card_html(
            title=names.get(key, key),
            direction=node.get("direction"),
            magnitude=node.get("magnitude"),
            persisted_direction=node.get("persisted_direction"),
            drivers=drivers,
        ),
        unsafe_allow_html=True,
    )


with row1[0]:
    render_band_card(ordered[0])
with row1[1]:
    render_band_card(ordered[1])
with row2[0]:
    render_band_card(ordered[2])
with row2[1]:
    render_band_card(ordered[3])

st.markdown('<hr class="soft"/>', unsafe_allow_html=True)

# -----------------------------
# Top Drivers Glossary (grouped)
# -----------------------------
st.markdown('<div class="section-title">Top Drivers Glossary</div>', unsafe_allow_html=True)

_render_glossary_section(
    "Stress drivers",
    [
        {
            "Driver": "Stress score (0–100)",
            "What it is": (
                "A composite market-stress indicator scaled to 0–100. It blends fast-moving market signals "
                "(volatility and credit spreads) with slower systemic indicators (financial conditions), "
                "then applies a sigmoid transform so extreme regimes compress toward the ends of the scale."
            ),
            "Units": "Index",
            "Scale (min .. max)": "0 .. 100",
        },
        {
            "Driver": "VIX (implied volatility) [VIXCLS]",
            "What it is": (
                "Derived from S&P 500 option prices and reflects implied volatility over the next ~30 days. "
                "Higher values usually coincide with risk-off regimes and more turbulent equity markets."
            ),
            "Units": "Index points",
            "Scale (min .. max)": _range_str_from_history("VIX (implied volatility)", driver_ranges),
        },
        {
            "Driver": "High-yield spread (HY OAS) [BAMLH0A0HYM2]",
            "What it is": (
                "High-yield credit option-adjusted spread (OAS). It measures how much extra yield sub-investment-grade borrowers "
                "must pay vs Treasuries. Wider spreads typically signal rising credit risk and tighter financial conditions."
            ),
            "Units": "Percentage points",
            "Scale (min .. max)": _range_str_from_history("High-yield spread (HY OAS)", driver_ranges),
        },
        {
            "Driver": "Investment-grade spread (IG OAS) [BAMLC0A0CM]",
            "What it is": (
                "Investment-grade credit option-adjusted spread (OAS). It captures stress in higher-quality corporate credit. "
                "A sustained widening often indicates broader funding stress or worsening risk appetite."
            ),
            "Units": "Percentage points",
            "Scale (min .. max)": _range_str_from_history("Investment-grade spread (IG OAS)", driver_ranges),
        },
        {
            "Driver": "Chicago Fed NFCI (financial conditions) [NFCI]",
            "What it is": (
                "Chicago Fed National Financial Conditions Index (weekly). A broad measure of how easy or tight financing conditions are "
                "based on money-market, debt, and equity indicators. Higher (less negative) generally means tighter conditions."
            ),
            "Units": "Index",
            "Scale (min .. max)": _range_str_from_history("Chicago Fed NFCI (financial conditions)", driver_ranges),
        },
        {
            "Driver": "SPY realized volatility (21d, annualized) [SPY_vol_21d]",
            "What it is": (
                "Realized volatility of SPY over the last 21 trading days, annualized. This is a backward-looking measure of recent equity turbulence "
                "and tends to rise during drawdowns and fast sell-offs."
            ),
            "Units": "Fraction",
            "Scale (min .. max)": _range_str_from_history("SPY realized volatility (21d, annualized)", driver_ranges),
        },
    ],
)

_render_glossary_section(
    "Equity within band (primary tilt inputs)",
    [
        {
            "Driver": "SPY vs 200-day trend",
            "What it is": (
                "A long-horizon trend proxy: SPY / MA(200) − 1. Positive values mean SPY is above its 200-day moving average. "
                "This is commonly used as a slow trend filter rather than a short-term timing signal."
            ),
            "Units": "Fraction",
            "Scale (min .. max)": "Unbounded",
        },
        {
            "Driver": "SPY momentum (12 months)",
            "What it is": (
                "12-month price momentum: SPY / SPY(252 trading days ago) − 1. It summarizes the trailing 1-year return. "
                "Positive values mean SPY is up over the last year; negative means it is down."
            ),
            "Units": "Fraction",
            "Scale (min .. max)": "Unbounded",
        },
        {
            "Driver": "SPY momentum (3 months)",
            "What it is": (
                "3-month price momentum: SPY / SPY(63 trading days ago) − 1. A shorter momentum window that is more responsive, "
                "but also noisier than the 12-month measure."
            ),
            "Units": "Fraction",
            "Scale (min .. max)": "Unbounded",
        },
        {
            "Driver": "SPY realized volatility (21d, annualized)",
            "What it is": (
                "Annualized realized volatility over the last 21 trading days. It penalizes equity risk-taking when recent market moves "
                "have been unusually large, even if the longer trend is still positive."
            ),
            "Units": "Fraction",
            "Scale (min .. max)": ">= 0 (no strict cap)",
        },
        {
            "Driver": "3-month cash yield (DGS3MO)",
            "What it is": (
                "3-month Treasury yield used as a cash-yield anchor. Higher cash yields increase the opportunity cost of holding equities, "
                "so the equity tilt logic can treat it as a headwind."
            ),
            "Units": "Percent",
            "Scale (min .. max)": ">= 0 in most modern history (no strict cap)",
        },
        {
            "Driver": "10Y–3M slope (T10Y3M)",
            "What it is": (
                "Yield-curve slope: 10-year Treasury minus 3-month Treasury. Negative values indicate inversion. "
                "Inversions often coincide with restrictive policy regimes and elevated late-cycle risk, and can act as a macro headwind."
            ),
            "Units": "Percentage points",
            "Scale (min .. max)": "Unbounded",
        },
        {
            "Driver": "Stress bucket",
            "What it is": (
                "A categorical regime label derived from the stress score (Low / Mid / High). "
                "It acts as a high-level adjustment that can nudge the equity tilt within its band."
            ),
            "Units": "Category",
            "Scale (min .. max)": "Low / Mid / High",
        },
    ],
)

_render_glossary_section(
    "Duration within bond band (primary tilt inputs)",
    [
        {
            "Driver": "10Y yield change (30 days)",
            "What it is": (
                "30-day change in the 10-year Treasury yield (percentage points). Positive means yields rose over the last month "
                "(which is typically a headwind for longer-duration bonds)."
            ),
            "Units": "Percentage points",
            "Scale (min .. max)": "Unbounded",
        },
        {
            "Driver": "Rate volatility (10Y, 63d)",
            "What it is": (
                "Rolling 63-day standard deviation of daily changes in the 10-year Treasury yield. Higher values indicate a more unstable rate environment, "
                "where duration risk tends to be less attractive."
            ),
            "Units": "Percentage points",
            "Scale (min .. max)": ">= 0 (no strict cap)",
        },
        {
            "Driver": "5Y / 10Y / 30Y Treasury yields",
            "What it is": (
                "Treasury yields used as level inputs for the duration tilt logic (shorter/mid/long). "
                "They help characterize the rate regime (e.g., unusually high vs unusually low levels historically)."
            ),
            "Units": "Percent",
            "Scale (min .. max)": ">= 0 in most modern history (no strict cap)",
        },
        {
            "Driver": "10Y–3M slope (T10Y3M)",
            "What it is": (
                "Curve slope (10Y minus 3M). More inversion can be interpreted as tighter policy / recession risk, which can make duration "
                "more attractive as a hedge depending on the rest of the regime."
            ),
            "Units": "Percentage points",
            "Scale (min .. max)": "Unbounded",
        },
        {
            "Driver": "Stress bucket",
            "What it is": (
                "Stress regime label derived from the stress score. In higher-stress regimes, the logic can allow a stronger duration-hedge bias, "
                "subject to the configured band."
            ),
            "Units": "Category",
            "Scale (min .. max)": "Low / Mid / High",
        },
    ],
)

_render_glossary_section(
    "TIPS slice (primary tilt inputs)",
    [
        {
            "Driver": "10Y breakeven inflation (T10YIE)",
            "What it is": (
                "10-year breakeven inflation rate (market-implied inflation expectation). "
                "It is the spread between nominal and inflation-linked Treasury yields at the 10-year horizon."
            ),
            "Units": "Percent",
            "Scale (min .. max)": "Unbounded (practically regime-bounded)",
        },
        {
            "Driver": "5Y5Y forward inflation (T5YIFR)",
            "What it is": (
                "5-year, 5-year forward inflation expectation. A longer-horizon inflation expectations proxy that is less sensitive to very short-term noise "
                "than spot breakevens."
            ),
            "Units": "Percent",
            "Scale (min .. max)": "Unbounded (practically regime-bounded)",
        },
        {
            "Driver": "10Y real yield (DFII10)",
            "What it is": (
                "10-year real yield proxy (inflation-indexed Treasury yield). Higher real yields can reduce the relative attractiveness of inflation protection, "
                "while falling real yields can support a larger TIPS allocation within the sleeve."
            ),
            "Units": "Percent",
            "Scale (min .. max)": "Unbounded (practically regime-bounded)",
        },
        {
            "Driver": "30d changes (breakeven / real yield)",
            "What it is": (
                "Short-horizon changes used to capture directionality: rising breakevens suggest increasing inflation pricing; rising real yields can be a headwind "
                "for inflation-linked assets depending on the configuration."
            ),
            "Units": "Percentage points",
            "Scale (min .. max)": "Unbounded",
        },
    ],
)

_render_glossary_section(
    "US vs ex-US (primary tilt inputs)",
    [
        {
            "Driver": "Equity concentration composite",
            "What it is": (
                "A heuristic concentration proxy that combines breadth and leadership measures (e.g., equal-weight vs cap-weight and tech/semis leadership ratios) "
                "into a single composite. Higher values indicate narrower leadership and greater concentration."
            ),
            "Units": "Composite",
            "Scale (min .. max)": "Unbounded",
        },
        {
            "Driver": "USD strength (broad index z-score) [DTWEXBGS_z]",
            "What it is": (
                "Z-score of the broad trade-weighted US dollar index. Higher values mean the USD is strong relative to its own history, "
                "which can be a headwind for unhedged ex-US returns in USD terms and can also reflect global risk dynamics."
            ),
            "Units": "Z-score",
            "Scale (min .. max)": "Unbounded (often near -3..+3 in stable regimes)",
        },
        {
            "Driver": "AI fundamentals heat (SEC, context only)",
            "What it is": (
                "A slow, annual context signal derived from SEC Company Facts for an AI-related basket. "
                "It is explicitly gated by coverage and is not intended to be a high-frequency driver."
            ),
            "Units": "0..1",
            "Scale (min .. max)": "0 .. 1",
        },
        {
            "Driver": "Stress bucket",
            "What it is": (
                "Stress regime label derived from the stress score. It can bias the US vs ex-US tilt modestly, while the main driver is concentration/leadership."
            ),
            "Units": "Category",
            "Scale (min .. max)": "Low / Mid / High",
        },
    ],
)
