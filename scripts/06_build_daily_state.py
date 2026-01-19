#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("pyyaml is required. Install with: pip install pyyaml") from e


# -----------------------------
# Tunables / guardrails
# -----------------------------

TRADING_DAYS = 252

# Persistence / anti-whipsaw
PERSIST_WINDOW_DAYS = 7
PERSIST_REQUIRED_DAYS = 5

# Stress bucket thresholds (0..100)
STRESS_HIGH_THRESHOLD = 70.0
STRESS_MID_THRESHOLD = 50.0

# Conservative defaults for action caps
DEFAULT_MAX_STEP_PP = 2.0

# Lookbacks
MACRO_LOOKBACK_DAYS = TRADING_DAYS * 10  # ~10y
MIN_ZSCORE_POINTS = 120

# SEC fundamentals weak-context gating
SEC_HEAT_MIN_COVERAGE_RATIO = 0.60


@dataclass
class Policy:
    baseline_equity_weight: float = 0.80
    min_equity_weight: float = 0.70
    max_equity_weight: float = 0.85
    max_step_pp: float = DEFAULT_MAX_STEP_PP

    allow_bond_quality_down: bool = False

    gold_enabled: bool = False
    commodities_enabled: bool = False
    bitcoin_enabled: bool = False

    gold_band: Tuple[float, float] = (0.0, 0.05)
    commodities_band: Tuple[float, float] = (0.0, 0.05)
    bitcoin_band: Tuple[float, float] = (0.0, 0.05)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}


def load_policy(repo_root: Path) -> Policy:
    cfg = read_yaml(repo_root / "config" / "policy.yml")
    p = Policy()

    equity = cfg.get("equity", {}) if isinstance(cfg.get("equity", {}), dict) else {}
    bonds = cfg.get("bonds", {}) if isinstance(cfg.get("bonds", {}), dict) else {}
    sleeves = cfg.get("sleeves", {}) if isinstance(cfg.get("sleeves", {}), dict) else {}

    def _f(x: Any, default: float) -> float:
        try:
            return float(x)
        except Exception:
            return default

    p.baseline_equity_weight = _f(equity.get("baseline", p.baseline_equity_weight), p.baseline_equity_weight)
    p.min_equity_weight = _f(equity.get("min", p.min_equity_weight), p.min_equity_weight)
    p.max_equity_weight = _f(equity.get("max", p.max_equity_weight), p.max_equity_weight)
    p.max_step_pp = _f(equity.get("max_step_pp", p.max_step_pp), p.max_step_pp)

    if isinstance(bonds.get("allow_quality_down", None), bool):
        p.allow_bond_quality_down = bool(bonds.get("allow_quality_down"))

    def _load_sleeve(name: str) -> Tuple[bool, Tuple[float, float]]:
        node = sleeves.get(name, {})
        if not isinstance(node, dict):
            return False, (0.0, 0.0)
        enabled = bool(node.get("enabled", False))
        band = node.get("band", None)
        if isinstance(band, (list, tuple)) and len(band) == 2:
            return enabled, (_f(band[0], 0.0), _f(band[1], 0.0))
        return enabled, (0.0, 0.0)

    p.gold_enabled, p.gold_band = _load_sleeve("gold")
    p.commodities_enabled, p.commodities_band = _load_sleeve("commodities")
    p.bitcoin_enabled, p.bitcoin_band = _load_sleeve("bitcoin")

    # Sanity
    p.min_equity_weight = max(0.0, min(p.min_equity_weight, 1.0))
    p.max_equity_weight = max(0.0, min(p.max_equity_weight, 1.0))
    p.baseline_equity_weight = max(p.min_equity_weight, min(p.baseline_equity_weight, p.max_equity_weight))
    p.max_step_pp = max(0.0, p.max_step_pp)

    return p


# -----------------------------
# JSON sanitization (strict JSON)
# -----------------------------

def sanitize_for_json(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if not np.isfinite(x):
            return None
        return x

    if isinstance(obj, (pd.Timestamp,)):
        if pd.isna(obj):
            return None
        return obj.isoformat()

    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj

    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


# -----------------------------
# Loaders
# -----------------------------

def load_tiingo_prices(repo_root: Path, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    tiingo_dir = repo_root / "data" / "raw" / "tiingo"
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        p = tiingo_dir / f"{t}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        price_col = "adjClose" if ("adjClose" in df.columns and df["adjClose"].notna().any()) else "close"
        df["price"] = pd.to_numeric(df.get(price_col), errors="coerce")
        df = df.dropna(subset=["price"])
        if df.empty:
            continue
        out[t] = df[["date", "price"]]
    return out


def load_fred_series(repo_root: Path, series_ids: List[str]) -> Dict[str, pd.Series]:
    fred_dir = repo_root / "data" / "raw" / "fred"
    out: Dict[str, pd.Series] = {}
    for sid in series_ids:
        p = fred_dir / f"{sid}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if df.empty:
            continue
        s = pd.Series(df["value"].values, index=pd.to_datetime(df["date"])).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        out[sid] = s
    return out


def load_ecb_series(repo_root: Path) -> Dict[str, pd.Series]:
    """
    Loads ECB SDMX parquet files produced by scripts/04_backfill_ecb_sdmx.py.

    IMPORTANT:
    The backfill script writes to: data/raw/ecb/
    """
    ecb_dir = repo_root / "data" / "raw" / "ecb"
    out: Dict[str, pd.Series] = {}
    if not ecb_dir.exists():
        return out
    for p in ecb_dir.glob("*.parquet"):
        df = pd.read_parquet(p)
        if df.empty or "date" not in df.columns or "value" not in df.columns:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if df.empty:
            continue
        s = pd.Series(df["value"].values, index=pd.to_datetime(df["date"])).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        out[p.stem] = s
    return out


# -----------------------------
# Math helpers
# -----------------------------

def zscore_series(values: pd.Series, lookback: int = MACRO_LOOKBACK_DAYS) -> pd.Series:
    """
    Computes rolling z-score on the series' native index (do NOT pre-expand to a daily trading calendar).
    """
    s = values.dropna().astype(float)
    if s.empty:
        return pd.Series(dtype=float)
    mu = s.rolling(lookback, min_periods=MIN_ZSCORE_POINTS).mean()
    sd = s.rolling(lookback, min_periods=MIN_ZSCORE_POINTS).std(ddof=0)
    z = (s - mu) / sd
    return z


def bucket_direction_magnitude(score: float) -> Dict[str, str]:
    if np.isnan(score):
        return {"direction": "HOLD", "magnitude": "SMALL"}
    if score > 0.20:
        direction = "TILT_TOWARD"
    elif score < -0.20:
        direction = "TILT_AWAY"
    else:
        direction = "HOLD"

    a = abs(score)
    if a < 0.35:
        mag = "SMALL"
    elif a < 0.70:
        mag = "MEDIUM"
    else:
        mag = "LARGE"

    return {"direction": direction, "magnitude": mag}


def _persisted_bucket(labels: pd.Series, window: int, required: int) -> Optional[str]:
    s = labels.dropna().tail(window)
    if s.empty:
        return None
    vc = s.value_counts()
    if vc.empty:
        return None
    top_label = str(vc.index[0])
    top_count = int(vc.iloc[0])
    return top_label if top_count >= required else None


def _sigmoid(x: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-x))


def _combine_weighted(parts: List[Tuple[float, Optional[pd.Series]]], cal: pd.DatetimeIndex) -> pd.Series:
    parts2 = [(w, s) for (w, s) in parts if s is not None and len(s) > 0]
    if not parts2:
        return pd.Series(index=cal, dtype=float)
    wsum = sum(w for w, _ in parts2)
    out = pd.Series(0.0, index=cal)
    for w, s in parts2:
        out = out.add((w / wsum) * s.reindex(cal), fill_value=0.0)
    return out


# -----------------------------
# SEC fundamentals (optional)
# -----------------------------

def load_sec_fundamentals(repo_root: Path, as_of_dt: pd.Timestamp) -> Dict[str, Any]:
    """
    Offline SEC fundamentals context.
    Expects Company Facts JSON files under: data/raw/sec_companyfacts/<CIK>.json
    and config at: config/sec_ai_basket.yml
    """
    cfg_path = repo_root / "config" / "sec_ai_basket.yml"
    if not cfg_path.exists():
        return {
            "as_of": str(as_of_dt.date()),
            "basket_summary": {
                "ai_fundamentals_heat": None,
                "companies_with_heat": 0,
                "companies_total": 0,
                "coverage_ratio": 0.0,
                "heat_used_in_signals": False,
            },
            "companies": [],
            "notes": ["Missing config/sec_ai_basket.yml"],
        }

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    companies_cfg = cfg.get("companies", [])
    if not isinstance(companies_cfg, list):
        companies_cfg = []

    def _annual_series(company_facts: Dict[str, Any], tag: str) -> pd.Series:
        node = company_facts.get("facts", {}).get("us-gaap", {}).get(tag, {})
        units = node.get("units", {})
        unit_key = "USD" if "USD" in units else (sorted(units.keys())[0] if units else None)
        if unit_key is None:
            return pd.Series(dtype=float)
        rows = units.get(unit_key, [])
        out = []
        for r in rows:
            form = str(r.get("form", "")).upper()
            if form not in {"10-K", "20-F"}:
                continue
            end = r.get("end")
            val = r.get("val")
            if end is None or val is None:
                continue
            dt = pd.to_datetime(end, errors="coerce")
            if pd.isna(dt) or dt > as_of_dt:
                continue
            try:
                out.append((dt, float(val)))
            except Exception:
                continue
        if not out:
            return pd.Series(dtype=float)
        s = pd.Series({d: v for d, v in out}).sort_index()
        return s[~s.index.duplicated(keep="last")]

    def _percentile(s: pd.Series, x: float) -> Optional[float]:
        s2 = s.dropna().astype(float)
        if len(s2) < 5:
            return None
        return float((s2 <= x).mean())

    companies_out: List[Dict[str, Any]] = []
    for c in companies_cfg:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        cik = str(c.get("cik", "")).zfill(10)
        if not cik.strip():
            continue
        capex_tag = c.get("capex_tag", "PaymentsToAcquirePropertyPlantAndEquipment")
        revenue_tag = c.get("revenue_tag", "Revenues")

        facts_path = repo_root / "data" / "raw" / "sec_companyfacts" / f"{cik}.json"
        if not facts_path.exists():
            companies_out.append({"name": name, "cik": cik, "error": f"Missing Company Facts file: {facts_path}"})
            continue

        company_facts = json.loads(facts_path.read_text(encoding="utf-8"))
        capex = _annual_series(company_facts, capex_tag)
        rev = _annual_series(company_facts, revenue_tag)

        ratio = pd.Series(dtype=float)
        ratio_dt = None
        if len(capex) and len(rev):
            # Align by closest fiscal year-end within 60 days
            vals = {}
            for dt_cap, cap in capex.items():
                diffs = (rev.index - dt_cap).days.astype(int)
                j = int(np.abs(diffs).argmin()) if len(diffs) else None
                if j is None:
                    continue
                dt_rev = rev.index[j]
                if abs((dt_rev - dt_cap).days) <= 60:
                    rv = float(rev.loc[dt_rev])
                    if rv != 0 and np.isfinite(rv):
                        vals[dt_cap] = float(cap) / rv
            if vals:
                ratio = pd.Series(vals).sort_index()
                ratio_dt = ratio.index[-1]

        capex_latest = float(capex.iloc[-1]) if len(capex) else None
        ratio_latest = float(ratio.iloc[-1]) if len(ratio) else None

        capex_yoy = None
        if len(capex) >= 2 and capex_latest is not None:
            prev = float(capex.iloc[-2])
            if prev != 0 and np.isfinite(prev):
                capex_yoy = float(capex_latest / prev - 1.0)

        ratio_pctl = _percentile(ratio, ratio_latest) if ratio_latest is not None else None
        capex_yoy_pctl = None
        if capex_yoy is not None and len(capex) >= 6:
            hist = (capex / capex.shift(1) - 1.0).dropna()
            capex_yoy_pctl = _percentile(hist, capex_yoy)

        heat = None
        if ratio_pctl is not None and capex_yoy_pctl is not None:
            heat = float(0.5 * ratio_pctl + 0.5 * capex_yoy_pctl)
        elif ratio_pctl is not None:
            heat = float(ratio_pctl)

        companies_out.append(
            {
                "name": name,
                "cik": cik,
                "capex_to_revenue_latest": ratio_latest,
                "capex_to_revenue_asof": str(ratio_dt.date()) if ratio_dt is not None else None,
                "capex_yoy_latest": capex_yoy,
                "capex_to_revenue_percentile": ratio_pctl,
                "capex_yoy_percentile": capex_yoy_pctl,
                "ai_fundamentals_heat": heat,
                # Cross-sectional ranks are filled later (if coverage is sufficient).
                "cross_sectional": None,
            }
        )

    heats = [c.get("ai_fundamentals_heat") for c in companies_out if c.get("ai_fundamentals_heat") is not None]
    basket_heat = float(np.median(heats)) if heats else None

    companies_total = int(len(companies_out))
    companies_with_heat = int(len(heats))
    coverage_ratio = float(companies_with_heat / companies_total) if companies_total > 0 else 0.0

    notes: List[str] = []
    heat_used_in_signals = coverage_ratio >= SEC_HEAT_MIN_COVERAGE_RATIO and basket_heat is not None
    if companies_total > 0 and coverage_ratio < SEC_HEAT_MIN_COVERAGE_RATIO:
        notes.append(f"SEC heat coverage too low: {companies_with_heat}/{companies_total}")

    # Cross-sectional snapshot (only if coverage is sufficient).
    # This is not a percentile vs own history; it's a same-period ranking across the basket.
    if heat_used_in_signals:
        rows = []
        for i, c in enumerate(companies_out):
            r = c.get("capex_to_revenue_latest")
            y = c.get("capex_yoy_latest")
            if r is None and y is None:
                continue
            rows.append((i, r, y))

        if rows:
            df = pd.DataFrame(rows, columns=["idx", "capex_to_revenue_latest", "capex_yoy_latest"])
            # Higher values are more "aggressive" -> rank 1 is highest.
            if df["capex_to_revenue_latest"].notna().any():
                df["capex_to_revenue_rank"] = df["capex_to_revenue_latest"].rank(method="min", ascending=False)
            else:
                df["capex_to_revenue_rank"] = np.nan

            if df["capex_yoy_latest"].notna().any():
                df["capex_yoy_rank"] = df["capex_yoy_latest"].rank(method="min", ascending=False)
            else:
                df["capex_yoy_rank"] = np.nan

            n_r = int(df["capex_to_revenue_latest"].notna().sum())
            n_y = int(df["capex_yoy_latest"].notna().sum())

            for _, row in df.iterrows():
                idx = int(row["idx"])
                companies_out[idx]["cross_sectional"] = {
                    "capex_to_revenue_rank": int(row["capex_to_revenue_rank"]) if np.isfinite(row["capex_to_revenue_rank"]) else None,
                    "capex_to_revenue_n": n_r,
                    "capex_yoy_rank": int(row["capex_yoy_rank"]) if np.isfinite(row["capex_yoy_rank"]) else None,
                    "capex_yoy_n": n_y,
                    "ranking_convention": "rank 1 = highest (most aggressive)",
                }

    return {
        "as_of": str(as_of_dt.date()),
        "basket_summary": {
            "ai_fundamentals_heat": basket_heat,
            "companies_with_heat": companies_with_heat,
            "companies_total": companies_total,
            "coverage_ratio": coverage_ratio,
            "heat_used_in_signals": bool(heat_used_in_signals),
        },
        "companies": companies_out,
        "notes": notes if notes else None,
    }


# -----------------------------
# Main builder
# -----------------------------

def build_daily_state() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    policy = load_policy(repo_root)

    out_dir = repo_root / "data" / "state" / "daily_state"
    ensure_dir(out_dir)

    # ---- Tiingo universe ----
    tiingo_cfg = read_yaml(repo_root / "config" / "tiingo_universe.yml")
    tickers_raw = tiingo_cfg.get("tickers", [])
    tickers: List[str] = []
    if isinstance(tickers_raw, list):
        if all(isinstance(x, str) for x in tickers_raw):
            tickers = [str(x).strip().upper() for x in tickers_raw if str(x).strip()]
        elif all(isinstance(x, dict) for x in tickers_raw):
            tickers = [str(x.get("ticker", "")).strip().upper() for x in tickers_raw if str(x.get("ticker", "")).strip()]

    if not tickers:
        raise RuntimeError("No Tiingo tickers configured in config/tiingo_universe.yml")

    prices_map = load_tiingo_prices(repo_root, tickers)
    if not prices_map:
        raise RuntimeError("No Tiingo price data found. Run scripts/02_backfill_tiingo.py first.")

    # Determine as_of_dt using core coverage (kept as-is per request).
    core = ["SPY", "IEF", "TLT"]
    core_dates = [prices_map[c]["date"].max() for c in core if c in prices_map and not prices_map[c].empty]
    as_of_dt = min(core_dates) if core_dates else max(df["date"].max() for df in prices_map.values())
    as_of_date = as_of_dt.date().isoformat()

    # Build aligned price panel up to as_of_dt
    panel = []
    for t, df in prices_map.items():
        tmp = df[df["date"] <= as_of_dt].copy()
        tmp = tmp.set_index("date")[["price"]].rename(columns={"price": t})
        panel.append(tmp)
    price_df = pd.concat(panel, axis=1).sort_index()

    # Calendar index (trading-day calendar)
    cal = pd.DatetimeIndex(price_df.index).sort_values()
    cal_asof = cal[cal <= as_of_dt]
    last_days = cal_asof[-max(30, PERSIST_WINDOW_DAYS + 5):]

    ret_df = price_df.pct_change(fill_method=None)

    # ---- FRED series ----
    fred_cfg = read_yaml(repo_root / "config" / "fred_series.yml")
    series_list = fred_cfg.get("series", []) if isinstance(fred_cfg.get("series", []), list) else []
    fred_ids = [str(s.get("id", "")).strip() for s in series_list if isinstance(s, dict) and str(s.get("id", "")).strip()]
    fred_map = load_fred_series(repo_root, fred_ids)

    # --- Frequency-safe alignment helpers ---
    def _asof_native(s: Optional[pd.Series]) -> Optional[pd.Series]:
        if s is None or len(s) == 0:
            return None
        s2 = s.copy()
        s2 = s2[s2.index <= as_of_dt]
        s2 = s2.dropna().sort_index()
        return None if s2.empty else s2

    def _to_cal_levels_ffill(s: Optional[pd.Series]) -> Optional[pd.Series]:
        s2 = _asof_native(s)
        if s2 is None:
            return None
        return s2.reindex(cal).ffill()

    def _z_to_cal_from_native(s: Optional[pd.Series]) -> Optional[pd.Series]:
        """
        Compute z-score on the series' native frequency first, then expand to trading calendar via ffill.
        This avoids artificial compression/duplication when series are weekly/monthly (or have gaps).
        """
        s2 = _asof_native(s)
        if s2 is None:
            return None
        z = zscore_series(s2, MACRO_LOOKBACK_DAYS)
        if z.empty:
            return None
        return z.reindex(cal).ffill()

    # ---- ECB series (context only) ----
    ecb_map = load_ecb_series(repo_root)

    def _ecb_latest(key: str) -> Optional[float]:
        s = ecb_map.get(key)
        if s is None or len(s) == 0:
            return None
        s2 = s[s.index <= as_of_dt].dropna()
        if s2.empty:
            return None
        return float(s2.iloc[-1])

    eur_yc_2y = _ecb_latest("EUR_YC_2Y")
    eur_yc_10y = _ecb_latest("EUR_YC_10Y")

    ecb_drivers = {
        "EUR_YC_2Y": eur_yc_2y,
        "EUR_YC_10Y": eur_yc_10y,
        "EUR_YC_30Y": _ecb_latest("EUR_YC_30Y"),
        "EUR_YC_10Y_2Y_SPREAD": (eur_yc_10y - eur_yc_2y) if (eur_yc_10y is not None and eur_yc_2y is not None) else None,
        "EUR_HICP_YOY": _ecb_latest("EUR_HICP_YOY"),
    }

    # -----------------------------
    # Stress score (0-100): FAST + SLOW + curve inversion
    # -----------------------------
    # Levels (for penalties / deltas)
    t10y3m_s = _to_cal_levels_ffill(fred_map.get("T10Y3M"))
    if t10y3m_s is None:
        dgs10_lvl = _to_cal_levels_ffill(fred_map.get("DGS10"))
        dgs3m_lvl = _to_cal_levels_ffill(fred_map.get("DGS3MO"))
        if dgs10_lvl is not None and dgs3m_lvl is not None:
            t10y3m_s = dgs10_lvl - dgs3m_lvl

    # Frequency-safe z-scores (compute z on native, then ffill to cal)
    vix_z = _z_to_cal_from_native(fred_map.get("VIXCLS"))
    hy_z = _z_to_cal_from_native(fred_map.get("BAMLH0A0HYM2"))
    ig_z = _z_to_cal_from_native(fred_map.get("BAMLC0A0CM"))
    nfci_z = _z_to_cal_from_native(fred_map.get("NFCI"))
    stlfsi_z = _z_to_cal_from_native(fred_map.get("STLFSI4"))

    spy_px = price_df["SPY"].copy() if "SPY" in price_df.columns else None
    if spy_px is None or spy_px.dropna().empty:
        raise RuntimeError("SPY is required in Tiingo universe to build daily_state.")

    spy_ret = spy_px.pct_change(fill_method=None)
    spy_vol_21d = spy_ret.rolling(21).std(ddof=0) * np.sqrt(TRADING_DAYS)
    vol_z = zscore_series(spy_vol_21d, MACRO_LOOKBACK_DAYS).reindex(cal)

    curve_penalty = pd.Series(0.0, index=cal)
    if t10y3m_s is not None:
        curve_penalty = (-t10y3m_s).clip(lower=0.0, upper=2.0)

    fast_raw = _combine_weighted(
        [
            (0.25, vix_z),
            (0.25, hy_z),
            (0.15, ig_z),
            (0.35, vol_z),
        ],
        cal,
    )

    slow_raw = _combine_weighted(
        [
            (0.60, nfci_z),
            (0.40, stlfsi_z),
        ],
        cal,
    )

    stress_raw = 0.55 * fast_raw + 0.30 * slow_raw + 0.15 * curve_penalty
    stress_score_s = (100.0 * _sigmoid(stress_raw)).clip(lower=0.0, upper=100.0)

    stress_score = float(stress_score_s.loc[as_of_dt])

    def _stress_bucket(v: float) -> str:
        if np.isnan(v):
            return "MID_STRESS"
        if v >= STRESS_HIGH_THRESHOLD:
            return "HIGH_STRESS"
        if v >= STRESS_MID_THRESHOLD:
            return "MID_STRESS"
        return "LOW_STRESS"

    stress_bucket_s = stress_score_s.apply(_stress_bucket)
    stress_bucket = _stress_bucket(stress_score)

    stress_bucket_recent = stress_bucket_s.loc[last_days]
    stress_bucket_persisted = _persisted_bucket(
        stress_bucket_recent, window=PERSIST_WINDOW_DAYS, required=PERSIST_REQUIRED_DAYS
    )
    if stress_bucket_persisted is not None:
        stress_bucket = stress_bucket_persisted

    # -----------------------------
    # Primary tilts
    # -----------------------------
    primary: Dict[str, Any] = {}

    # ---- Equity weight within band ----
    dgs3m_lvl = _to_cal_levels_ffill(fred_map.get("DGS3MO"))
    cash_yield_z = _z_to_cal_from_native(fred_map.get("DGS3MO"))

    spy_ma200 = spy_px.rolling(200).mean()
    spy_trend = (spy_px / spy_ma200) - 1.0
    spy_trend_z = zscore_series(spy_trend, MACRO_LOOKBACK_DAYS).reindex(cal)

    mom_3m = (spy_px / spy_px.shift(63)) - 1.0
    mom_12m = (spy_px / spy_px.shift(252)) - 1.0
    mom_3m_z = zscore_series(mom_3m, MACRO_LOOKBACK_DAYS).reindex(cal)
    mom_12m_z = zscore_series(mom_12m, MACRO_LOOKBACK_DAYS).reindex(cal)

    invert_flag = pd.Series(0.0, index=cal)
    if t10y3m_s is not None:
        invert_flag = (t10y3m_s < 0).astype(float)

    equity_score_s = (
        0.20 * spy_trend_z.reindex(cal)
        + 0.20 * mom_12m_z.reindex(cal)
        + 0.10 * mom_3m_z.reindex(cal)
        - 0.15 * vol_z.reindex(cal)
        - (0.10 * cash_yield_z.reindex(cal) if cash_yield_z is not None else 0.0)
        - 0.20 * invert_flag.reindex(cal)
        - 0.65 * (stress_bucket_s == "HIGH_STRESS").astype(float).reindex(cal)
        + 0.20 * (stress_bucket_s == "LOW_STRESS").astype(float).reindex(cal)
    )

    equity_score = float(equity_score_s.loc[as_of_dt])
    equity_raw = bucket_direction_magnitude(equity_score)
    equity_dir_recent = equity_score_s.loc[last_days].apply(lambda x: bucket_direction_magnitude(float(x))["direction"])
    equity_dir_persisted = _persisted_bucket(
        equity_dir_recent, window=PERSIST_WINDOW_DAYS, required=PERSIST_REQUIRED_DAYS
    )
    equity_dir_final = equity_dir_persisted if equity_dir_persisted is not None else equity_raw["direction"]

    primary["EQUITY_WEIGHT_WITHIN_BAND"] = {
        "direction": equity_dir_final,
        "raw_direction": equity_raw["direction"],
        "persisted_direction": equity_dir_persisted,
        "magnitude": equity_raw["magnitude"],
        "score": equity_score,
        "inputs": {
            "stress_bucket": stress_bucket,
            "spy_200d_trend": float(spy_trend.loc[as_of_dt]),
            "spy_momentum_3m": float(mom_3m.loc[as_of_dt]),
            "spy_momentum_12m": float(mom_12m.loc[as_of_dt]),
            "spy_realized_vol_21d_ann": float(spy_vol_21d.loc[as_of_dt]),
            "cash_yield_3m": float(dgs3m_lvl.loc[as_of_dt]) if dgs3m_lvl is not None else None,
            "t10y3m": float(t10y3m_s.loc[as_of_dt]) if t10y3m_s is not None else None,
        },
        "rules": {
            "direction_thresholds": {"TILT_TOWARD": "> +0.20", "HOLD": "[-0.20, +0.20]", "TILT_AWAY": "< -0.20"},
            "persistence": {"window_days": PERSIST_WINDOW_DAYS, "required_days": PERSIST_REQUIRED_DAYS},
        },
    }

    # ---- Bond duration within band ----
    dgs5_lvl = _to_cal_levels_ffill(fred_map.get("DGS5"))
    dgs10_lvl = _to_cal_levels_ffill(fred_map.get("DGS10"))
    dgs30_lvl = _to_cal_levels_ffill(fred_map.get("DGS30"))

    def _avg_level_z_from_native(series_list: List[Optional[pd.Series]]) -> pd.Series:
        zs = []
        for s in series_list:
            z = _z_to_cal_from_native(s)
            if z is not None and len(z) > 0:
                zs.append(z)
        if not zs:
            return pd.Series(0.0, index=cal)
        out = pd.Series(0.0, index=cal)
        for z in zs:
            out = out.add(z.reindex(cal), fill_value=0.0)
        return out / float(len(zs))

    rate_level_z = _avg_level_z_from_native([fred_map.get("DGS5"), fred_map.get("DGS10"), fred_map.get("DGS30")])

    rate_vol_63d = dgs10_lvl.diff().rolling(63).std(ddof=0) if dgs10_lvl is not None else None
    rate_vol_z = zscore_series(rate_vol_63d.dropna(), MACRO_LOOKBACK_DAYS).reindex(cal) if rate_vol_63d is not None else None

    dgs10_change_30d = dgs10_lvl.diff(30) if dgs10_lvl is not None else pd.Series(np.nan, index=cal)

    # Curve z-score: compute on native T10Y3M if present, then ffill.
    curve_z = _z_to_cal_from_native(fred_map.get("T10Y3M"))
    if curve_z is None:
        curve_z = pd.Series(0.0, index=cal)

    dur_score_s = (
        -0.40 * (dgs10_change_30d / 0.50).clip(lower=-2.0, upper=2.0).reindex(cal)
        - (0.35 * rate_vol_z.reindex(cal) if rate_vol_z is not None else 0.0)
        + 0.20 * rate_level_z.reindex(cal)
        - 0.20 * curve_z.reindex(cal)
        + 0.25 * (stress_bucket_s == "HIGH_STRESS").astype(float).reindex(cal)
    )

    dur_score = float(dur_score_s.loc[as_of_dt])
    dur_raw = bucket_direction_magnitude(dur_score)
    dur_dir_recent = dur_score_s.loc[last_days].apply(lambda x: bucket_direction_magnitude(float(x))["direction"])
    dur_dir_persisted = _persisted_bucket(dur_dir_recent, window=PERSIST_WINDOW_DAYS, required=PERSIST_REQUIRED_DAYS)
    dur_dir_final = dur_dir_persisted if dur_dir_persisted is not None else dur_raw["direction"]

    primary["BOND_DURATION_WITHIN_BAND"] = {
        "direction": dur_dir_final,
        "raw_direction": dur_raw["direction"],
        "persisted_direction": dur_dir_persisted,
        "magnitude": dur_raw["magnitude"],
        "score": dur_score,
        "inputs": {
            "dgs5": float(dgs5_lvl.loc[as_of_dt]) if dgs5_lvl is not None else None,
            "dgs10": float(dgs10_lvl.loc[as_of_dt]) if dgs10_lvl is not None else None,
            "dgs30": float(dgs30_lvl.loc[as_of_dt]) if dgs30_lvl is not None else None,
            "dgs10_change_30d": float(dgs10_change_30d.loc[as_of_dt]) if dgs10_lvl is not None else None,
            "rate_vol_63d": float(rate_vol_63d.loc[as_of_dt]) if rate_vol_63d is not None else None,
            "t10y3m": float(t10y3m_s.loc[as_of_dt]) if t10y3m_s is not None else None,
            "stress_bucket": stress_bucket,
        },
        "rules": {
            "intent": {"TILT_TOWARD": "longer duration", "TILT_AWAY": "shorter duration"},
            "direction_thresholds": {"TILT_TOWARD": "> +0.20", "HOLD": "[-0.20, +0.20]", "TILT_AWAY": "< -0.20"},
            "persistence": {"window_days": PERSIST_WINDOW_DAYS, "required_days": PERSIST_REQUIRED_DAYS},
        },
    }

    # ---- TIPS slice within band ----
    dfii10_lvl = _to_cal_levels_ffill(fred_map.get("DFII10"))
    t10yie_lvl = _to_cal_levels_ffill(fred_map.get("T10YIE"))
    t5yifr_lvl = _to_cal_levels_ffill(fred_map.get("T5YIFR"))

    breakeven_chg30 = t10yie_lvl.diff(30) if t10yie_lvl is not None else None
    real_yield_chg30 = dfii10_lvl.diff(30) if dfii10_lvl is not None else None

    breakeven_z = _z_to_cal_from_native(fred_map.get("T10YIE"))
    fwd_infl_z = _z_to_cal_from_native(fred_map.get("T5YIFR"))
    real_yield_z = _z_to_cal_from_native(fred_map.get("DFII10"))

    tip_score_s = pd.Series(0.0, index=cal)
    if breakeven_z is not None:
        tip_score_s = tip_score_s + 0.35 * breakeven_z.reindex(cal)
    if fwd_infl_z is not None:
        tip_score_s = tip_score_s + 0.15 * fwd_infl_z.reindex(cal)
    if real_yield_z is not None:
        tip_score_s = tip_score_s + 0.25 * real_yield_z.reindex(cal)
    if breakeven_chg30 is not None:
        tip_score_s = tip_score_s + 0.15 * (breakeven_chg30 / 0.25).clip(lower=-2.0, upper=2.0).reindex(cal)
    if real_yield_chg30 is not None:
        tip_score_s = tip_score_s - 0.10 * (real_yield_chg30 / 0.25).clip(lower=-2.0, upper=2.0).reindex(cal)

    tip_score = float(tip_score_s.loc[as_of_dt])
    tip_raw = bucket_direction_magnitude(tip_score)
    tip_dir_recent = tip_score_s.loc[last_days].apply(lambda x: bucket_direction_magnitude(float(x))["direction"])
    tip_dir_persisted = _persisted_bucket(tip_dir_recent, window=PERSIST_WINDOW_DAYS, required=PERSIST_REQUIRED_DAYS)
    tip_dir_final = tip_dir_persisted if tip_dir_persisted is not None else tip_raw["direction"]

    primary["TIPS_SLICE_WITHIN_BAND"] = {
        "direction": tip_dir_final,
        "raw_direction": tip_raw["direction"],
        "persisted_direction": tip_dir_persisted,
        "magnitude": tip_raw["magnitude"],
        "score": tip_score,
        "inputs": {
            "t10y_breakeven": float(t10yie_lvl.loc[as_of_dt]) if t10yie_lvl is not None else None,
            "t10y_breakeven_change_30d": float(breakeven_chg30.loc[as_of_dt]) if breakeven_chg30 is not None else None,
            "dfii10_real_yield": float(dfii10_lvl.loc[as_of_dt]) if dfii10_lvl is not None else None,
            "dfii10_real_yield_change_30d": float(real_yield_chg30.loc[as_of_dt]) if real_yield_chg30 is not None else None,
            "t5y5y_forward_infl": float(t5yifr_lvl.loc[as_of_dt]) if t5yifr_lvl is not None else None,
        },
        "rules": {
            "intent": {"TILT_TOWARD": "larger TIPS slice", "TILT_AWAY": "smaller TIPS slice"},
            "direction_thresholds": {"TILT_TOWARD": "> +0.20", "HOLD": "[-0.20, +0.20]", "TILT_AWAY": "< -0.20"},
            "persistence": {"window_days": PERSIST_WINDOW_DAYS, "required_days": PERSIST_REQUIRED_DAYS},
        },
    }

    # ---- US vs ex-US within equity band ----
    def _ratio_log_z(num: str, den: str) -> Optional[pd.Series]:
        if num not in price_df.columns or den not in price_df.columns:
            return None
        r = (price_df[num] / price_df[den]).replace([np.inf, -np.inf], np.nan).dropna()
        if r.empty:
            return None
        return zscore_series(np.log(r), MACRO_LOOKBACK_DAYS)

    rsp_spy_z = _ratio_log_z("RSP", "SPY")
    qqq_spy_z = _ratio_log_z("QQQ", "SPY")
    soxx_spy_z = _ratio_log_z("SOXX", "SPY")
    xlk_spy_z = _ratio_log_z("XLK", "SPY")

    conc_parts: List[Tuple[float, Optional[pd.Series]]] = []
    if rsp_spy_z is not None:
        conc_parts.append((0.50, -rsp_spy_z))
    if qqq_spy_z is not None:
        conc_parts.append((0.25, qqq_spy_z))
    if xlk_spy_z is not None:
        conc_parts.append((0.15, xlk_spy_z))
    if soxx_spy_z is not None:
        conc_parts.append((0.25, soxx_spy_z))
    concentration_s = _combine_weighted(conc_parts, cal)

    dtwex_s = _z_to_cal_from_native(fred_map.get("DTWEXBGS"))
    usd_z = dtwex_s.reindex(cal) if dtwex_s is not None else pd.Series(0.0, index=cal)

    us_exus_score_s = (
        +0.10 * (stress_bucket_s == "LOW_STRESS").astype(float).reindex(cal)
        - 0.20 * (stress_bucket_s == "HIGH_STRESS").astype(float).reindex(cal)
        - 0.25 * concentration_s.reindex(cal)
        - 0.15 * usd_z.reindex(cal)
    )

    # SEC fundamentals heat (weak context; gated by coverage)
    sec_fund = load_sec_fundamentals(repo_root, as_of_dt)
    basket_summary = sec_fund.get("basket_summary", {}) if isinstance(sec_fund.get("basket_summary", {}), dict) else {}
    ai_heat_raw = basket_summary.get("ai_fundamentals_heat")
    heat_used_in_signals = bool(basket_summary.get("heat_used_in_signals", False))
    ai_heat_used = ai_heat_raw if heat_used_in_signals else None

    conc_today = float(concentration_s.loc[as_of_dt]) if as_of_dt in concentration_s.index else float("nan")
    usd_today = float(usd_z.loc[as_of_dt]) if as_of_dt in usd_z.index else float("nan")

    us_exus_score = float(us_exus_score_s.loc[as_of_dt])

    # Soft penalties (only when extremes align) AND only if SEC heat is allowed for signals.
    if ai_heat_used is not None and np.isfinite(conc_today) and float(ai_heat_used) >= 0.90 and conc_today >= 1.0:
        us_exus_score -= 0.15
    if np.isfinite(conc_today) and np.isfinite(usd_today) and conc_today >= 1.0 and usd_today >= 1.0:
        us_exus_score -= 0.10

    us_raw = bucket_direction_magnitude(us_exus_score)
    us_dir_recent = us_exus_score_s.loc[last_days].apply(lambda x: bucket_direction_magnitude(float(x))["direction"])
    us_dir_persisted = _persisted_bucket(us_dir_recent, window=PERSIST_WINDOW_DAYS, required=PERSIST_REQUIRED_DAYS)
    us_dir_final = us_dir_persisted if us_dir_persisted is not None else us_raw["direction"]

    primary["US_WEIGHT_WITHIN_EQUITY_BAND"] = {
        "direction": us_dir_final,
        "raw_direction": us_raw["direction"],
        "persisted_direction": us_dir_persisted,
        "magnitude": us_raw["magnitude"],
        "score": us_exus_score,
        "inputs": {
            "stress_bucket": stress_bucket,
            "concentration_composite": conc_today if np.isfinite(conc_today) else None,
            "usd_strength_z": usd_today if np.isfinite(usd_today) else None,
            "ai_fundamentals_heat": ai_heat_raw,
            "ai_fundamentals_heat_used": ai_heat_used,
            "ai_fundamentals_coverage_ratio": basket_summary.get("coverage_ratio"),
            "ai_fundamentals_used_in_signals": heat_used_in_signals,
        },
        "rules": {
            "intent": {"TILT_TOWARD": "more US equity", "TILT_AWAY": "less US equity (toward ex-US)"},
            "direction_thresholds": {"TILT_TOWARD": "> +0.20", "HOLD": "[-0.20, +0.20]", "TILT_AWAY": "< -0.20"},
            "persistence": {"window_days": PERSIST_WINDOW_DAYS, "required_days": PERSIST_REQUIRED_DAYS},
        },
    }

    # -----------------------------
    # AI cycle (context)
    # -----------------------------
    leadership_today = np.nanmean(
        [
            float(qqq_spy_z.loc[as_of_dt]) if (qqq_spy_z is not None and as_of_dt in qqq_spy_z.index) else np.nan,
            float(soxx_spy_z.loc[as_of_dt]) if (soxx_spy_z is not None and as_of_dt in soxx_spy_z.index) else np.nan,
        ]
    )
    conc_today2 = conc_today

    ai_cycle_score = (
        0.60 * leadership_today + 0.40 * conc_today2
        if (np.isfinite(leadership_today) and np.isfinite(conc_today2))
        else float("nan")
    )

    # Only let SEC heat influence AI-cycle score when coverage is sufficient.
    if ai_heat_used is not None and np.isfinite(ai_cycle_score):
        ai_cycle_score += 0.25 * ((float(ai_heat_used) - 0.5) * 2.0)

    # Phase labeling: SEC-driven "OVERHEATED" requires usable heat.
    if (
        (ai_heat_used is not None)
        and (float(ai_heat_used) >= 0.90)
        and np.isfinite(conc_today2)
        and np.isfinite(leadership_today)
        and conc_today2 >= 1.0
        and leadership_today >= 0.75
    ):
        ai_phase = "OVERHEATED"
    elif np.isfinite(leadership_today) and np.isfinite(conc_today2) and leadership_today >= 1.0 and conc_today2 >= 0.75:
        ai_phase = "HOT"
    elif np.isfinite(leadership_today) and np.isfinite(conc_today2) and leadership_today <= -0.75 and conc_today2 <= 0.0:
        ai_phase = "COOL"
    else:
        ai_phase = "NEUTRAL"

    if np.isnan(ai_cycle_score):
        ai_cycle = "NEUTRAL"
    elif ai_cycle_score >= 1.25:
        ai_cycle = "HOT"
    elif ai_cycle_score >= 0.40:
        ai_cycle = "WARM"
    elif ai_cycle_score <= -0.80:
        ai_cycle = "COLD"
    elif ai_cycle_score <= -0.20:
        ai_cycle = "COOL"
    else:
        ai_cycle = "NEUTRAL"

    def _ratio_features(num: str, den: str, z_s: Optional[pd.Series]) -> Optional[Dict[str, Any]]:
        if num not in price_df.columns or den not in price_df.columns:
            return None
        r = (price_df[num] / price_df[den]).replace([np.inf, -np.inf], np.nan)
        if as_of_dt not in r.index or pd.isna(r.loc[as_of_dt]):
            return None
        ret90 = (r / r.shift(90) - 1.0).loc[as_of_dt] if as_of_dt in r.index else np.nan
        z_today = None
        if z_s is not None:
            if as_of_dt in z_s.index and np.isfinite(z_s.loc[as_of_dt]):
                z_today = float(z_s.loc[as_of_dt])
            else:
                z2 = z_s.dropna()
                if len(z2) > 0:
                    z_today = float(z2.iloc[-1])
        return {
            "ratio": float(r.loc[as_of_dt]),
            "ratio_90d_return": float(ret90) if np.isfinite(ret90) else None,
            "z": z_today,
        }

    ai_ratios: Dict[str, Any] = {}
    for num, z_s in [("RSP", rsp_spy_z), ("QQQ", qqq_spy_z), ("SOXX", soxx_spy_z), ("XLK", xlk_spy_z)]:
        feat = _ratio_features(num, "SPY", z_s)
        if feat is not None:
            ai_ratios[f"{num}_over_SPY"] = feat

    # -----------------------------
    # Optional sleeves (eligibility-only)
    # -----------------------------
    def _trend_200d(ticker: str) -> Optional[float]:
        if ticker not in price_df.columns:
            return None
        px = price_df[ticker].dropna()
        if len(px) < 220 or as_of_dt not in px.index:
            return None
        ma200 = px.rolling(200).mean()
        if pd.isna(ma200.loc[as_of_dt]):
            return None
        return float(px.loc[as_of_dt] / ma200.loc[as_of_dt] - 1.0)

    def _eligibility(enabled: bool, ticker: str) -> Optional[Dict[str, Any]]:
        if not enabled:
            return None
        tr = _trend_200d(ticker)
        if tr is None:
            return {"eligible": None, "reason": f"Missing {ticker} or insufficient history"}
        eligible = (stress_bucket != "HIGH_STRESS") and (tr > 0.0)
        reason = "Trend>0 and stress not HIGH" if eligible else "Trend<=0 or HIGH_STRESS"
        return {"eligible": bool(eligible), "reason": reason, "trend_200d": float(tr)}

    sleeves = {
        "GOLD_SLEEVE_ELIGIBILITY": _eligibility(policy.gold_enabled, "GLD"),
        "COMMODITIES_SLEEVE_ELIGIBILITY": _eligibility(policy.commodities_enabled, "DBC"),
        "BITCOIN_SLEEVE_ELIGIBILITY": _eligibility(policy.bitcoin_enabled, "BITO"),
    }

    # -----------------------------
    # Final assembly
    # -----------------------------
    # Promote SEC warning notes to diagnostics for visibility (still kept under sec_fundamentals).
    diag_notes = []
    if isinstance(sec_fund.get("notes", None), list):
        diag_notes.extend([str(x) for x in sec_fund["notes"] if str(x).strip()])

    state: Dict[str, Any] = {
        "as_of_date": as_of_date,
        "policy": {
            "baseline_equity_weight": policy.baseline_equity_weight,
            "min_equity_weight": policy.min_equity_weight,
            "max_equity_weight": policy.max_equity_weight,
            "max_step_pp": policy.max_step_pp,
            "allow_bond_quality_down": policy.allow_bond_quality_down,
            "optional_sleeves": {
                "gold": {"enabled": policy.gold_enabled, "band": {"min": policy.gold_band[0], "max": policy.gold_band[1]}},
                "commodities": {"enabled": policy.commodities_enabled, "band": {"min": policy.commodities_band[0], "max": policy.commodities_band[1]}},
                "bitcoin": {"enabled": policy.bitcoin_enabled, "band": {"min": policy.bitcoin_band[0], "max": policy.bitcoin_band[1]}},
            },
        },
        "market_regime": {
            "stress_score": stress_score,
            "stress_bucket": stress_bucket,
            "persistence": {
                "window_days": PERSIST_WINDOW_DAYS,
                "required_days": PERSIST_REQUIRED_DAYS,
                "stress_bucket_persisted": stress_bucket_persisted,
            },
            "ecb_context": ecb_drivers,
        },
        "ai_cycle": {
            "phase": ai_phase,
            "cycle": ai_cycle,
            "leadership_ratios": ai_ratios,
            "ai_fundamentals_heat": ai_heat_raw,
            "ai_fundamentals_heat_used": ai_heat_used,
            "ai_fundamentals_coverage_ratio": basket_summary.get("coverage_ratio"),
            "ai_fundamentals_used_in_signals": heat_used_in_signals,
        },
        "tilt_signals": {
            "primary": primary,
            "optional_sleeves": sleeves,
        },
        "diagnostics": {
            "notes": diag_notes if diag_notes else None,
            "stress_score_recent": [
                {"date": d.date().isoformat(), "stress_score": float(stress_score_s.loc[d])}
                for d in stress_score_s.dropna().index[-PERSIST_WINDOW_DAYS:]
            ],
        },
        "data_coverage": {
            "tiingo_tickers_loaded": sorted(list(prices_map.keys())),
            "fred_series_present": sorted(list(fred_map.keys())),
            "ecb_series_present": sorted(list(ecb_map.keys())),
            "last_price_date": as_of_date,
        },
        "sec_fundamentals": sec_fund,
    }

    return sanitize_for_json(state)


def main() -> None:
    state = build_daily_state()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "data" / "state" / "daily_state"
    ensure_dir(out_dir)

    out_path = out_dir / f"daily_state_{state['as_of_date']}.json"
    out_path.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[OK] daily_state written: {out_path}")


if __name__ == "__main__":
    main()
