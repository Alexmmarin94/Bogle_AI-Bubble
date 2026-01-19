#!/usr/bin/env python3

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


FRED_SERIES_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"
SLEEP_BETWEEN_REQUESTS = 0.2
PAGE_LIMIT = 100000


def load_secrets_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomli.load(f)


def require_secret(secrets: Dict[str, Any], key: str) -> str:
    v = secrets.get(key)
    if not v or not str(v).strip():
        raise ValueError(f"Missing required secret: {key}")
    return str(v).strip()


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_existing_last_date(path: Path) -> Optional[pd.Timestamp]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).max()


def fetch_fred_series_observations(
    session: requests.Session,
    api_key: str,
    series_id: str,
    observation_start: Optional[str],
    timeout: int = 45,
) -> pd.DataFrame:
    """
    Fetch observations for a FRED series incrementally.
    Uses pagination via offset/limit.
    """
    rows: List[Dict[str, Any]] = []
    offset = 0

    while True:
        params: Dict[str, Any] = {
            "api_key": api_key,
            "file_type": "json",
            "series_id": series_id,
            "sort_order": "asc",
            "limit": PAGE_LIMIT,
            "offset": offset,
        }
        if observation_start:
            params["observation_start"] = observation_start

        r = session.get(FRED_SERIES_OBS_URL, params=params, timeout=timeout)
        r.raise_for_status()
        payload = r.json()

        obs = payload.get("observations", [])
        if not obs:
            break

        rows.extend(obs)

        # If we got fewer than PAGE_LIMIT, we are done.
        if len(obs) < PAGE_LIMIT:
            break

        offset += PAGE_LIMIT

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # FRED returns 'date' and 'value' as strings. Missing values can be '.'
    if "date" not in df.columns or "value" not in df.columns:
        raise RuntimeError(f"Unexpected FRED response schema for {series_id}. Columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df[["date", "value"]]


def upsert_parquet(existing_path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing_path.exists():
        old = pd.read_parquet(existing_path)
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df.copy()

    out = out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    secrets = load_secrets_toml(repo_root / ".streamlit" / "secrets.toml")
    fred_api_key = require_secret(secrets, "FRED_API_KEY")

    cfg = read_yaml(repo_root / "config" / "fred_series.yml")
    series_list: List[Dict[str, Any]] = cfg.get("series", []) if isinstance(cfg, dict) else []
    if not series_list:
        raise RuntimeError("No FRED series found in config/fred_series.yml under key 'series'.")

    out_dir = repo_root / "data" / "raw" / "fred"
    ensure_dir(out_dir)

    session = requests.Session()

    for s in series_list:
        series_id = str(s.get("id", "")).strip()
        if not series_id:
            continue

        out_path = out_dir / f"{series_id}.parquet"
        last_dt = load_existing_last_date(out_path)

        # If series parquet doesn't exist -> backfill from configured backfill_start (or FRED default earliest)
        # If it exists -> incremental from last_dt + 1 day.
        if last_dt is None:
            observation_start = str(s.get("backfill_start", "")).strip() or None
        else:
            observation_start = (last_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        df_new = fetch_fred_series_observations(
            session=session,
            api_key=fred_api_key,
            series_id=series_id,
            observation_start=observation_start,
        )

        if df_new.empty:
            print(f"[SKIP] FRED {series_id} - no new data")
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        df_all = upsert_parquet(out_path, df_new)
        df_all.to_parquet(out_path, index=False)

        last_date = pd.to_datetime(df_all["date"]).max().date()
        print(f"[OK] FRED {series_id} - rows={len(df_all)} last_date={last_date}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("Done.")


if __name__ == "__main__":
    main()
