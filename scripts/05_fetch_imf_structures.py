#!/usr/bin/env python3

from __future__ import annotations

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


def http_get_text(session: requests.Session, url: str, params: Dict[str, str], timeout: int = 45) -> str:
    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.text


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    secrets = load_secrets_toml(repo_root / ".streamlit" / "secrets.toml")
    imf_base = require_secret(secrets, "IMF_SDMX_BASE_URL")  # https://sdmxcentral.imf.org/ws/public/sdmxapi/rest

    cfg = read_yaml(repo_root / "config" / "imf_structures.yml")
    structures: List[Dict[str, Any]] = cfg.get("structures", [])

    out_dir = repo_root / "data" / "raw" / "imf" / "structures"
    ensure_dir(out_dir)

    session = requests.Session()

    for s in structures:
        sid = s["id"]
        resource = s["resource"]
        agency = s.get("agency", "IMF")
        flow = s.get("flow", "all")
        version = s.get("version", "latest")
        detail = s.get("detail", "full")
        references = s.get("references", "all")
        fmt = s.get("format", "sdmx-2.1")

        # For this repo, we keep it simple: fetch SDMX-ML structures and save them.
        url = f"{imf_base.rstrip('/')}/{resource}/{agency}/{flow}/{version}/"
        params = {"detail": detail, "references": references, "format": fmt}

        text = http_get_text(session, url, params=params)

        if "Structure" not in text and "message:Structure" not in text:
            raise RuntimeError(f"IMF response did not look like an SDMX Structure message for {sid}.")

        out_path = out_dir / f"{sid}.xml"
        out_path.write_text(text, encoding="utf-8")

        size_kb = out_path.stat().st_size / 1024.0
        print(f"[OK] IMF {sid} - saved {out_path} ({size_kb:.1f} KB) url={url}")

        time.sleep(0.3)

    print("Done.")


if __name__ == "__main__":
    main()
