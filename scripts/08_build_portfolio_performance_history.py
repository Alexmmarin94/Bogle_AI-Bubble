#!/usr/bin/env python3
# scripts/08_build_portfolio_performance_history.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


START_DATE = "2012-01-01"
INITIAL_USD = 1000.0
CONTRIB_USD = 250.0

STATIC_WEIGHTS = {"VTI": 0.56, "VXUS": 0.24, "BND": 0.20}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_price(root: Path, ticker: str) -> pd.Series:
    p = root / "data" / "raw" / "tiingo" / f"{ticker}.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    if df is None or df.empty or "date" not in df.columns:
        return pd.Series(dtype=float)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    price_col = "adjClose" if ("adjClose" in df.columns and df["adjClose"].notna().any()) else "close"
    if price_col not in df.columns:
        return pd.Series(dtype=float)

    df["price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["price"])

    s = pd.Series(df["price"].values, index=pd.to_datetime(df["date"])).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def align_prices(root: Path, tickers: list[str], start: pd.Timestamp) -> pd.DataFrame:
    prices = {t: load_price(root, t) for t in tickers}
    missing = [t for t, s in prices.items() if s.empty]
    if missing:
        raise RuntimeError(f"Missing Tiingo data for {missing}. Run Tiingo backfill first.")

    idx = None
    for s in prices.values():
        s2 = s[s.index >= start].dropna()
        idx = s2.index if idx is None else idx.union(s2.index)

    if idx is None or len(idx) == 0:
        raise RuntimeError("Not enough price history after start date.")

    idx = pd.DatetimeIndex(sorted(idx))
    df = pd.DataFrame(index=idx)
    for t, s in prices.items():
        s2 = s[s.index >= start].dropna().sort_index()
        df[t] = s2.reindex(idx).ffill()

    df = df.dropna(how="any")
    return df


def next_trading_day(index: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = index.searchsorted(dt)
    if pos >= len(index):
        return None
    return pd.Timestamp(index[pos])


def build_contrib_series(index: pd.DatetimeIndex) -> pd.Series:
    idx = pd.DatetimeIndex(index).sort_values()
    start = idx.min().normalize()
    end = idx.max().normalize()
    months = pd.period_range(start=start, end=end, freq="M")

    contrib = pd.Series(0.0, index=idx)
    for m in months:
        d1 = pd.Timestamp(m.start_time.date())
        d15 = d1 + pd.Timedelta(days=14)

        t1 = next_trading_day(idx, d1)
        t15 = next_trading_day(idx, d15)

        if t1 is not None:
            contrib.loc[t1] += CONTRIB_USD
        if t15 is not None:
            contrib.loc[t15] += CONTRIB_USD

    return contrib


def simulate(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    initial_usd: float,
    contrib: pd.Series,
    rebalance_full: bool,
    rebalance_dates: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers], dtype=float)
    w = w / w.sum()

    px = prices[tickers].copy()
    idx = px.index

    shares = np.zeros(len(tickers), dtype=float)

    # Initial buy
    p0 = px.iloc[0].values
    shares += (initial_usd * w) / p0

    reb_set = set(pd.DatetimeIndex(rebalance_dates).normalize()) if (rebalance_full and rebalance_dates is not None) else set()

    nav = pd.Series(index=idx, dtype=float)
    for i, dt in enumerate(idx):
        p = px.iloc[i].values

        c = float(contrib.loc[dt]) if dt in contrib.index else 0.0
        if c > 0:
            shares += (c * w) / p

        if rebalance_full and (dt.normalize() in reb_set):
            total = float(np.sum(shares * p))
            shares = (total * w) / p

        nav.loc[dt] = float(np.sum(shares * p))

    return nav


def main() -> None:
    root = repo_root()
    out_dir = root / "data" / "state" / "portfolio_history"
    ensure_dir(out_dir)

    start = pd.Timestamp(START_DATE).normalize()

    # Build aligned prices
    prices_all = align_prices(root, ["VT", "VTI", "VXUS", "BND"], start)

    contrib = build_contrib_series(prices_all.index)

    # Rebalance dates: first trading day of each month in the panel index
    cal = prices_all.index
    month_starts = pd.date_range(cal.min().normalize(), cal.max().normalize(), freq="BMS")
    reb = []
    for d in month_starts:
        t = next_trading_day(cal, d)
        if t is not None:
            reb.append(t)
    reb_dates = pd.DatetimeIndex(sorted(set(reb)))

    # 1) VT hold
    nav_vt = simulate(
        prices=prices_all[["VT"]],
        weights={"VT": 1.0},
        initial_usd=INITIAL_USD,
        contrib=contrib,
        rebalance_full=False,
    )

    # 2) Static 56/24/20 (contrib-only)
    nav_static_contrib = simulate(
        prices=prices_all,
        weights=STATIC_WEIGHTS,
        initial_usd=INITIAL_USD,
        contrib=contrib,
        rebalance_full=False,
    )

    # 3) Static 56/24/20 (full monthly rebalance)
    nav_static_full = simulate(
        prices=prices_all,
        weights=STATIC_WEIGHTS,
        initial_usd=INITIAL_USD,
        contrib=contrib,
        rebalance_full=True,
        rebalance_dates=reb_dates,
    )

    out = pd.DataFrame(
        {
            "nav_vt_hold": nav_vt,
            "nav_static_contrib_only": nav_static_contrib,
            "nav_static_full_rebalance": nav_static_full,
            "contrib_usd": contrib.reindex(prices_all.index).fillna(0.0),
        }
    ).dropna()

    out_path = out_dir / "portfolio_nav_history.parquet"
    out.to_parquet(out_path, index=True)

    print(f"[OK] written: {out_path}")
    print(f"[INFO] start={out.index.min().date().isoformat()} end={out.index.max().date().isoformat()} rows={len(out)}")


if __name__ == "__main__":
    main()