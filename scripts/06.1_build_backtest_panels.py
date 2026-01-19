#!/usr/bin/env python3
# scripts/06_build_backtest_panels.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


START_DATE = "2012-01-01"
TICKERS = ["VT", "VTI", "VXUS", "BND"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_price_series(root: Path, ticker: str) -> pd.Series:
    p = root / "data" / "raw" / "tiingo" / f"{ticker}.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    if df is None or df.empty or "date" not in df.columns:
        return pd.Series(dtype=float)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    price_col = "adjClose" if ("adjClose" in df.columns and df["adjClose"].notna().any()) else "close"
    if price_col not in df.columns:
        return pd.Series(dtype=float)

    df["price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["price"])
    s = pd.Series(df["price"].values, index=pd.to_datetime(df["date"])).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def align_panel(prices: Dict[str, pd.Series], start: pd.Timestamp) -> pd.DataFrame:
    idx = None
    for s in prices.values():
        s2 = s[s.index >= start].dropna()
        if s2.empty:
            continue
        idx = s2.index if idx is None else idx.union(s2.index)
    if idx is None or len(idx) == 0:
        return pd.DataFrame()

    idx = pd.DatetimeIndex(sorted(idx))
    df = pd.DataFrame(index=idx)
    for k, s in prices.items():
        s2 = s[s.index >= start].dropna().sort_index()
        df[k] = s2.reindex(idx).ffill()
    df = df.dropna(how="any")
    return df


def main() -> None:
    root = repo_root()
    out_dir = root / "data" / "state" / "backtests"
    ensure_dir(out_dir)

    start = pd.Timestamp(START_DATE)

    prices_map = {t: load_price_series(root, t) for t in TICKERS}
    missing = [t for t, s in prices_map.items() if s.empty]
    if missing:
        raise RuntimeError(f"Missing Tiingo data for {missing}. Run Tiingo backfill first.")

    prices_panel = align_panel(prices_map, start)
    if prices_panel.empty:
        raise RuntimeError("Unable to build aligned prices panel.")

    returns_panel = prices_panel.pct_change(fill_method=None).dropna()

    prices_path = out_dir / "prices_panel.parquet"
    returns_path = out_dir / "returns_panel.parquet"
    meta_path = out_dir / "metadata.json"

    prices_panel.to_parquet(prices_path, index=True)
    returns_panel.to_parquet(returns_path, index=True)

    meta = {
        "start_date": START_DATE,
        "tickers": TICKERS,
        "panel_start": str(prices_panel.index.min().date()),
        "panel_end": str(prices_panel.index.max().date()),
        "rows_prices": int(len(prices_panel)),
        "rows_returns": int(len(returns_panel)),
        "notes": "Prices use adjClose if present, otherwise close.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] prices_panel: {prices_path}")
    print(f"[OK] returns_panel: {returns_path}")
    print(f"[OK] metadata: {meta_path}")


if __name__ == "__main__":
    main()
