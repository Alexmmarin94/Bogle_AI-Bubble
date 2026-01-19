#!/usr/bin/env python3
# Comments in English as requested.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {' '.join(cmd)}")


def main() -> None:
    root = repo_root()

    # 1) Raw data refresh (incremental)
    run([sys.executable, "scripts/02_backfill_tiingo_from_2012.py"], cwd=root)
    run([sys.executable, "scripts/01_backfill_fred_from_2012.py"], cwd=root)

    # 2) Panels / derived time series used by Tab 1-2 (your renamed script)
    run([sys.executable, "scripts/06.1_build_backtest_panels.py"], cwd=root)

    # 3) Daily state history (incremental action dates)
    run([sys.executable, "scripts/07_build_daily_state_history.py"], cwd=root)

    # 4) Portfolio targets history (incremental)
    run([sys.executable, "scripts/07.1_build_portfolio_targets_history.py"], cwd=root)

    # 5) Portfolio performance history
    run([sys.executable, "scripts/08_build_portfolio_performance_history.py"], cwd=root)

    # 6) Daily State (current)
    run([sys.executable, "scripts/06_build_daily_state.py"], cwd=root)

    print("\n[OK] All updates completed.")


if __name__ == "__main__":
    main()
