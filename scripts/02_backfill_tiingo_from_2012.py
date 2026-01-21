#!/usr/bin/env python3

from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    import tomli  # type: ignore
except Exception as e:
    raise RuntimeError("tomli is required. Install with: pip install tomli") from e

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("pyyaml is required. Install with: pip install pyyaml") from e


TIINGO_EOD_URL_TMPL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
DEFAULT_START = "2012-01-01"
SLEEP_BETWEEN_REQUESTS = 0.6


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_secrets_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing secrets file: {path}")
    with path.open("rb") as f:
        return tomli.load(f)


def require_secret(secrets: Dict[str, Any], key: str) -> str:
    v = secrets.get(key)
    if not v or not str(v).strip():
        raise ValueError(f"Missing required secret: {key}")
    return str(v).strip()


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def last_date_in_parquet(path: Path) -> Optional[pd.Timestamp]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty or "date" not in df.columns:
        return None
    return pd.to_datetime(df["date"], errors="coerce").max()


def parse_tickers(cfg: Dict[str, Any]) -> List[Tuple[str, Optional[str]]]:
    """
    Supports either:
      tickers:
        - SPY
        - QQQ
    or:
      tickers:
        - ticker: SPY
          backfill_start: "2010-01-01"
    Returns list of (ticker, optional_backfill_start).
    """
    raw = cfg.get("tickers", []) if isinstance(cfg, dict) else []
    out: List[Tuple[str, Optional[str]]] = []

    if not raw:
        return out

    if all(isinstance(x, str) for x in raw):
        for x in raw:
            t = str(x).strip().upper()
            if t:
                out.append((t, None))
        return out

    if all(isinstance(x, dict) for x in raw):
        for x in raw:
            t = str(x.get("ticker", "")).strip().upper()
            if not t:
                continue
            bf = str(x.get("backfill_start", "")).strip() or None
            out.append((t, bf))
        return out

    raise RuntimeError("Invalid format for config/tiingo_universe.yml tickers.")


def fetch_tiingo_eod(
    session: requests.Session,
    token: str,
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeout: int = 60,
) -> pd.DataFrame:
    url = TIINGO_EOD_URL_TMPL.format(ticker=ticker)
    params: Dict[str, str] = {"token": token}
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date

    r = session.get(url, params=params, timeout=timeout)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        head = (r.text or "")[:500]
        raise requests.HTTPError(f"Tiingo HTTP error for {ticker}: {e} | response_head={head}") from e

    data = r.json()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "date" not in df.columns:
        raise RuntimeError(f"Tiingo response missing 'date' for {ticker}. Columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    keep_cols = [c for c in ["date", "open", "high", "low", "close", "adjClose", "volume"] if c in df.columns]
    out = df[keep_cols].copy()

    for c in ["open", "high", "low", "close", "adjClose", "volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["date"])
    return out


def upsert_parquet(existing_path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing_path.exists():
        old = pd.read_parquet(existing_path)
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df.copy()

    out = out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def main() -> None:
    root = repo_root()

    # Match your existing "working" approach
    secrets = load_secrets_toml(root / ".streamlit" / "secrets.toml")
    token = require_secret(secrets, "TIINGO_API_KEY")

    cfg = read_yaml(root / "config" / "tiingo_universe.yml")
    tickers = parse_tickers(cfg)

    if not tickers:
        raise RuntimeError("No tickers found in config/tiingo_universe.yml under key 'tickers'.")

    # For a historical backfill, we want full refresh behavior unless a parquet already exists.
    full_refresh = os.getenv("TIINGO_FULL_REFRESH", "0").strip() == "1"

    out_dir = root / "data" / "raw" / "tiingo"
    ensure_dir(out_dir)

    session = requests.Session()

    # End date optional: by default let Tiingo go to latest available.
    end_date = os.getenv("TIINGO_END_DATE", "").strip() or None

    for ticker, cfg_backfill_start in tickers:
        out_path = out_dir / f"{ticker}.parquet"
        last_dt = None if full_refresh else last_date_in_parquet(out_path)

        if last_dt is None:
            start_date = DEFAULT_START
        else:
            start_date = (last_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        df_new = fetch_tiingo_eod(session, token, ticker, start_date=start_date, end_date=end_date)

        if df_new.empty:
            print(f"[SKIP] Tiingo {ticker} - no new data")
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        df_all = upsert_parquet(out_path, df_new)
        df_all.to_parquet(out_path, index=False)

        last_date = pd.to_datetime(df_all["date"], errors="coerce").max()
        first_date = pd.to_datetime(df_all["date"], errors="coerce").min()
        print(f"[OK] Tiingo {ticker} - rows={len(df_all)} range={first_date.date()}..{last_date.date()}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("Done.")


if __name__ == "__main__":
    main()
