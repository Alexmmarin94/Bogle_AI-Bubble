#!/usr/bin/env python3
# Comments in English as requested.

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

try:
    import tomli  # type: ignore
except Exception as e:
    raise RuntimeError("tomli is required. Install with: pip install tomli") from e


SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SLEEP_BETWEEN_REQUESTS = 0.25


def load_secrets_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomli.load(f)


def require_secret(secrets: Dict[str, Any], key: str) -> str:
    v = secrets.get(key)
    if not v or not str(v).strip():
        raise ValueError(f"Missing required secret: {key}")
    return str(v).strip()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_cik(cik: str) -> str:
    # SEC expects 10-digit, zero-padded CIK.
    digits = "".join(ch for ch in cik if ch.isdigit())
    if not digits:
        raise ValueError(f"Invalid CIK: {cik}")
    return digits.zfill(10)


def download_companyfacts(session: requests.Session, cik10: str, user_agent: str, timeout: int = 45) -> Dict[str, Any]:
    url = SEC_COMPANYFACTS_URL.format(cik=cik10)
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "application/json",
    }
    r = session.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    secrets = load_secrets_toml(repo_root / ".streamlit" / "secrets.toml")

    # SEC requires a proper User-Agent identifying the requester.
    sec_user_agent = require_secret(secrets, "SEC_USER_AGENT")

    out_dir = repo_root / "data" / "raw" / "sec_companyfacts"
    ensure_dir(out_dir)

    # You can move this list to a config file later if you want.
    companies: List[Dict[str, str]] = [
        {"name": "Microsoft", "cik": "0000789019"},
        {"name": "Alphabet", "cik": "0001652044"},
        {"name": "Amazon", "cik": "0001018724"},
        {"name": "Meta", "cik": "0001326801"},
        {"name": "NVIDIA", "cik": "0001045810"},
    ]

    session = requests.Session()

    for c in companies:
        name = c["name"]
        cik10 = normalize_cik(c["cik"])
        out_path = out_dir / f"{cik10}.json"

        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[SKIP] SEC {name} ({cik10}) - already exists: {out_path}")
            continue

        try:
            payload = download_companyfacts(session=session, cik10=cik10, user_agent=sec_user_agent)
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            body_preview = (getattr(e.response, "text", "") or "")[:200].replace("\n", " ")
            print(f"[ERR] SEC {name} ({cik10}) - HTTP {status} - {body_preview}")
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue
        except Exception as e:
            print(f"[ERR] SEC {name} ({cik10}) - {type(e).__name__}: {e}")
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[OK] SEC {name} ({cik10}) - saved: {out_path}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("Done.")


if __name__ == "__main__":
    main()
