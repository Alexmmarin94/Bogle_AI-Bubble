#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List

import requests

try:
    import tomli  # type: ignore
except Exception as e:
    raise RuntimeError("tomli is required. pip install tomli") from e

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("pyyaml is required. pip install pyyaml") from e


SEC_COMPANYFACTS_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


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


def normalize_cik(cik: str) -> str:
    # SEC endpoints expect 10-digit zero-padded CIK in the URL path.
    cik_digits = "".join([c for c in cik if c.isdigit()])
    return cik_digits.zfill(10)


def fetch_companyfacts(session: requests.Session, cik10: str, user_agent: str, timeout: int = 30) -> Dict[str, Any]:
    url = SEC_COMPANYFACTS_TMPL.format(cik=cik10)
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }
    r = session.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    secrets = load_secrets_toml(repo_root / ".streamlit" / "secrets.toml")
    user_agent = require_secret(secrets, "SEC_USER_AGENT")

    cfg = read_yaml(repo_root / "config" / "sec_ai_basket.yml")
    companies: List[Dict[str, Any]] = cfg.get("companies", [])

    out_dir = repo_root / "data" / "raw" / "sec"
    ensure_dir(out_dir)

    session = requests.Session()

    for c in companies:
        cik10 = normalize_cik(c["cik"])
        out_path = out_dir / f"companyfacts_{cik10}.json"

        # Always re-download for now (small basket). Later: conditional fetch by last filing date.
        data = fetch_companyfacts(session, cik10, user_agent=user_agent)
        out_path.write_text(json.dumps(data), encoding="utf-8")

        name = data.get("entityName", "Unknown")
        print(f"[OK] SEC companyfacts CIK={cik10} name={name}")

        # SEC fair access: max 10 req/sec => sleep >=0.11s; we go slower.
        time.sleep(0.3)

    print("Done.")


if __name__ == "__main__":
    main()
