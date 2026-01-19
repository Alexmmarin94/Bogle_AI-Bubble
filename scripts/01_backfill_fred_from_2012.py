#!/usr/bin/env python3
# Comments in English as requested.

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

try:
    import tomli  # type: ignore
except Exception as e:
    raise RuntimeError("tomli is required. Install with: pip install tomli") from e


FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_START = "2012-01-01"
SLEEP_SECONDS = 0.4


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_secrets_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomli.load(f)


def fred_key(root: Path) -> str:
    # 1) Env var (keeps compatibility with your current script)
    k = os.getenv("FRED_API_KEY", "").strip()
    if k:
        return k

    # 2) Streamlit secrets (same pattern as Tiingo)
    secrets = load_secrets_toml(root / ".streamlit" / "secrets.toml")
    k2 = str(secrets.get("FRED_API_KEY", "")).strip()
    if k2:
        return k2

    raise RuntimeError("Missing FRED_API_KEY. Set env var or add it to .streamlit/secrets.toml")


def read_series_from_config(root: Path) -> List[str]:
    """
    Minimal YAML parsing without requiring pyyaml.
    Expects lines with: id: <SERIES_ID>
    """
    cfg = root / "config" / "fred_series.yml"
    if not cfg.exists():
        return []

    ids: List[str] = []
    for line in cfg.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("id:"):
            v = s.split("id:", 1)[1].strip().strip('"').strip("'")
            if v:
                ids.append(v)
    return sorted(list(dict.fromkeys(ids)))


def default_series() -> List[str]:
    return sorted(
        list(
            dict.fromkeys(
                [
                    "VIXCLS",
                    "BAMLH0A0HYM2",
                    "BAMLC0A0CM",
                    "NFCI",
                    "STLFSI4",
                    "T10Y3M",
                    "DGS3MO",
                    "DGS5",
                    "DGS10",
                    "DGS30",
                    "DFII10",
                    "T10YIE",
                    "T5YIFR",
                    "DTWEXBGS",
                ]
            )
        )
    )


def fetch_series(api_key: str, sid: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    params = {
        "series_id": sid,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }
    if end:
        params["observation_end"] = end

    r = requests.get(FRED_BASE, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"FRED error {sid}: {r.status_code} {r.text[:200]}")
    obj = r.json()
    obs = obj.get("observations", [])
    if not isinstance(obs, list) or not obs:
        return pd.DataFrame()

    df = pd.DataFrame(obs)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def load_existing(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df is None or df.empty or "date" not in df.columns:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df if not df.empty else None


def merge_dedup(existing: Optional[pd.DataFrame], incoming: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        out = incoming.copy()
    else:
        out = pd.concat([existing, incoming], axis=0, ignore_index=True)

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out


def main() -> None:
    root = repo_root()
    out_dir = root / "data" / "raw" / "fred"
    ensure_dir(out_dir)

    api_key = fred_key(root)

    series_ids = read_series_from_config(root)
    if not series_ids:
        series_ids = default_series()

    end = os.getenv("FRED_END_DATE", "").strip() or None

    print(f"[INFO] Series: {len(series_ids)}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Range: {DEFAULT_START}..{end or 'latest'}")

    for i, sid in enumerate(series_ids, start=1):
        path = out_dir / f"{sid}.parquet"
        existing = load_existing(path)

        try:
            print(f"[{i}/{len(series_ids)}] Fetch {sid} from {DEFAULT_START}")
            incoming = fetch_series(api_key, sid, DEFAULT_START, end=end)
            if incoming.empty:
                print(f"[WARN] No data returned for {sid}")
                time.sleep(SLEEP_SECONDS)
                continue

            merged = merge_dedup(existing, incoming)
            merged.to_parquet(path, index=False)

            last_dt = pd.to_datetime(merged["date"]).max()
            first_dt = pd.to_datetime(merged["date"]).min()
            print(f"[OK] {sid}: rows={len(merged)} range={first_dt.date()}..{last_dt.date()}")

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"[ERROR] {sid}: {e}")
            time.sleep(2.0)
            continue


if __name__ == "__main__":
    main()
