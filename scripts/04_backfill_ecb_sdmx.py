#!/usr/bin/env python3

from __future__ import annotations

import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

try:
    import tomli  # type: ignore
except Exception as e:
    raise RuntimeError("tomli is required. pip install tomli") from e

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("pyyaml is required. pip install pyyaml") from e


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
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def last_date_in_parquet(path: Path) -> Optional[pd.Timestamp]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    return pd.to_datetime(df["date"], errors="coerce").max()


def fetch_ecb_csv(
    session: requests.Session,
    base_url: str,
    dataflow: str,
    key: str,
    start_period: Optional[str],
    end_period: Optional[str],
    timeout: int = 45,
) -> pd.DataFrame:
    """
    Fetch ECB SDMX series as CSV (format=csvdata) and return a normalized DataFrame with:
      - date (datetime64[ns])
      - value (float)

    Robustness:
      - Handles empty response bodies (returns empty DF)
      - Handles pandas EmptyDataError (returns empty DF)
      - Handles non-standard schemas (returns empty DF)
    """
    url = f"{base_url.rstrip('/')}/data/{dataflow}/{key}"

    params: Dict[str, str] = {"format": "csvdata"}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period

    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()

    text = (r.text or "").strip()
    if not text:
        return pd.DataFrame(columns=["date", "value"])

    try:
        df = pd.read_csv(StringIO(text))
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["date", "value"])
    except Exception:
        # Any unexpected parsing failure -> treat as "no data" instead of crashing.
        return pd.DataFrame(columns=["date", "value"])

    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        # Sometimes ECB may return an error payload or unexpected schema.
        return pd.DataFrame(columns=["date", "value"])

    out = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
    out = out.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "value"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
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
    repo_root = Path(__file__).resolve().parents[1]
    secrets = load_secrets_toml(repo_root / ".streamlit" / "secrets.toml")
    ecb_base = require_secret(secrets, "ECB_SDMX_BASE_URL")  # e.g., https://data-api.ecb.europa.eu/service

    cfg = read_yaml(repo_root / "config" / "ecb_series.yml")
    series_list: List[Dict[str, Any]] = cfg.get("series", []) if isinstance(cfg.get("series", []), list) else []

    out_dir = repo_root / "data" / "raw" / "ecb"
    ensure_dir(out_dir)

    session = requests.Session()

    for s in series_list:
        sid = str(s.get("id", "")).strip()
        dataflow = str(s.get("dataflow", "")).strip()
        key = str(s.get("key", "")).strip()

        if not sid or not dataflow or not key:
            print(f"[SKIP] Invalid ECB series config entry: {s}")
            continue

        out_path = out_dir / f"{sid}.parquet"
        last_dt = last_date_in_parquet(out_path)

        # Support optional backfill_start in config when file does not exist yet
        if last_dt is None:
            bf = str(s.get("backfill_start", "")).strip()
            start_period = bf or None
        else:
            start_period = (last_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        df_new = fetch_ecb_csv(
            session=session,
            base_url=ecb_base,
            dataflow=dataflow,
            key=key,
            start_period=start_period,
            end_period=None,
        )

        if df_new.empty:
            print(f"[SKIP] ECB {sid} - no new data (flow={dataflow}, key={key})")
            continue

        df_all = upsert_parquet(out_path, df_new)
        df_all.to_parquet(out_path, index=False)

        print(
            f"[OK] ECB {sid} - rows={len(df_all)} last_date={df_all['date'].max().date()} "
            f"(flow={dataflow}, key={key})"
        )

        time.sleep(0.3)

    print("Done.")


if __name__ == "__main__":
    main()
