#!/usr/bin/env python3
# Comments in English as requested.

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config (edit here only if needed)
# -----------------------------

DEFAULT_BASELINE = {"VTI": 0.56, "VXUS": 0.24, "BND": 0.20}

# Normal vs exceptional max moves (per rebalance decision date)
NORMAL_MAX_MOVE_PP = 5.0     # typical
EXCEPTIONAL_MAX_MOVE_PP = 20.0  # absolute cap

# These define how "strong" a set of signals must be to justify >5pp.
# We map a severity score into an allowed move in {0,2,5,8,12,16,20}.
MOVE_STEPS_PP = [0.0, 2.0, 5.0, 8.0, 12.0, 16.0, 20.0]

# Output paths
HISTORY_IN = "data/state/daily_state_history/daily_state_history.parquet"
TARGETS_OUT = "data/state/portfolio_targets_history/portfolio_targets_history.parquet"


@dataclass
class TargetPolicy:
    # Baseline weights
    vti: float = DEFAULT_BASELINE["VTI"]
    vxus: float = DEFAULT_BASELINE["VXUS"]
    bnd: float = DEFAULT_BASELINE["BND"]

    # Hard limits
    min_equity: float = 0.60
    max_equity: float = 0.90
    min_bnd: float = 0.10
    max_bnd: float = 0.40

    # Movement caps
    normal_max_move_pp: float = NORMAL_MAX_MOVE_PP
    exceptional_max_move_pp: float = EXCEPTIONAL_MAX_MOVE_PP


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pp(x: float) -> float:
    return float(x) * 100.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _normalize_equity_split(vti: float, vxus: float, equity_total: float) -> Tuple[float, float]:
    s = vti + vxus
    if s <= 0:
        # fallback to baseline equity split if something weird happens
        vti_share = DEFAULT_BASELINE["VTI"] / (DEFAULT_BASELINE["VTI"] + DEFAULT_BASELINE["VXUS"])
        return equity_total * vti_share, equity_total * (1.0 - vti_share)
    return equity_total * (vti / s), equity_total * (vxus / s)


def _severity_from_signals(
    stress_bucket: str,
    equity_dir: Optional[str],
    us_weight_dir: Optional[str],
    ai_phase: Optional[str],
) -> int:
    """
    Convert qualitative signals into an integer severity 0..6,
    which maps to MOVE_STEPS_PP.
    This is intentionally conservative.
    """
    sev = 0

    # Stress dominates. HIGH_STRESS is a strong push to reduce equity.
    if str(stress_bucket).upper() == "HIGH_STRESS":
        sev += 3
    elif str(stress_bucket).upper() == "MID_STRESS":
        sev += 1

    # Equity direction indicates whether to tilt within band.
    if equity_dir == "TILT_AWAY":
        sev += 2
    elif equity_dir == "TILT_TOWARD":
        sev += 1

    # US weight direction acts as a secondary signal (smaller impact).
    if us_weight_dir in {"TILT_AWAY", "TILT_TOWARD"}:
        sev += 1

    # AI phase can justify "exceptional" risk control only when OVERHEATED.
    if ai_phase == "OVERHEATED":
        sev += 2
    elif ai_phase == "HOT":
        sev += 1

    return int(_clamp(sev, 0, 6))


def _move_pp_from_severity(sev: int) -> float:
    sev = int(_clamp(sev, 0, len(MOVE_STEPS_PP) - 1))
    return float(MOVE_STEPS_PP[sev])


def _equity_target_from_dirs(
    policy_equity_baseline: float,
    min_equity: float,
    max_equity: float,
    stress_bucket: str,
    equity_dir: Optional[str],
    sev_move_pp: float,
) -> float:
    """
    Decide equity total target using the direction + a capped move magnitude.
    Rules:
      - If stress HIGH -> prefer equity down (if any move allowed).
      - If LOW and equity_dir TILT_TOWARD -> equity up.
      - Otherwise -> hold baseline.
    """
    baseline = float(policy_equity_baseline)

    stress = str(stress_bucket).upper()

    move = sev_move_pp / 100.0

    if stress == "HIGH_STRESS":
        # reduce equity
        return _clamp(baseline - move, min_equity, max_equity)

    if equity_dir == "TILT_AWAY":
        return _clamp(baseline - move, min_equity, max_equity)

    if stress == "LOW_STRESS" and equity_dir == "TILT_TOWARD":
        return _clamp(baseline + move, min_equity, max_equity)

    # default: baseline
    return _clamp(baseline, min_equity, max_equity)


def _us_share_target_from_dir(
    baseline_us_share_within_equity: float,
    us_weight_dir: Optional[str],
    move_pp: float,
) -> float:
    """
    Adjust US share inside equities (VTI vs VXUS) while keeping equity total fixed.
    Baseline US share is computed from baseline VTI/(VTI+VXUS).
    Move is capped similarly but we apply smaller effect by scaling 0.5.
    """
    base = baseline_us_share_within_equity
    step = 0.5 * (move_pp / 100.0)  # smaller than equity move
    if us_weight_dir == "TILT_TOWARD":
        return _clamp(base + step, 0.50, 0.95)
    if us_weight_dir == "TILT_AWAY":
        return _clamp(base - step, 0.35, 0.80)
    return _clamp(base, 0.35, 0.95)


def compute_targets_row(row: pd.Series, pol: TargetPolicy) -> Dict[str, object]:
    # Prefer policy values from daily_state_history if present
    policy_baseline_equity = row.get("policy_baseline_equity_weight", None)
    if policy_baseline_equity is None or (isinstance(policy_baseline_equity, float) and not np.isfinite(policy_baseline_equity)):
        policy_baseline_equity = pol.vti + pol.vxus  # baseline equity from weights

    # Baseline US share inside equity
    base_us_share = pol.vti / (pol.vti + pol.vxus)

    stress_bucket = str(row.get("stress_bucket", "MID_STRESS"))
    equity_dir = row.get("equity_dir", None)
    us_weight_dir = row.get("us_weight_dir", None)
    ai_phase = row.get("ai_phase", None)

    sev = _severity_from_signals(stress_bucket, equity_dir, us_weight_dir, ai_phase)
    move_pp = _move_pp_from_severity(sev)

    # Enforce normal vs exceptional cap: allow >5pp only when sev>=3.
    if move_pp > pol.normal_max_move_pp and sev < 3:
        move_pp = pol.normal_max_move_pp

    # Absolute cap
    move_pp = min(move_pp, pol.exceptional_max_move_pp)

    equity_target = _equity_target_from_dirs(
        policy_equity_baseline=float(policy_baseline_equity),
        min_equity=pol.min_equity,
        max_equity=pol.max_equity,
        stress_bucket=stress_bucket,
        equity_dir=equity_dir,
        sev_move_pp=move_pp,
    )

    us_share = _us_share_target_from_dir(base_us_share, us_weight_dir, move_pp)

    vti_t, vxus_t = _normalize_equity_split(
        vti=us_share,
        vxus=1.0 - us_share,
        equity_total=equity_target,
    )
    bnd_t = 1.0 - (vti_t + vxus_t)

    # Clamp bonds
    bnd_t = _clamp(bnd_t, pol.min_bnd, pol.max_bnd)
    # Re-normalize equity split to fill remainder
    equity_total = 1.0 - bnd_t
    vti_t, vxus_t = _normalize_equity_split(vti_t, vxus_t, equity_total)

    # Output + explanations
    explanation = []
    explanation.append(f"Severity={sev} => move_cap={move_pp:.1f}pp")
    explanation.append(f"stress_bucket={stress_bucket}, equity_dir={equity_dir}, us_weight_dir={us_weight_dir}, ai_phase={ai_phase}")
    explanation.append(f"equity_target={_pp(equity_target):.1f}%, us_share_in_equity={_pp(us_share):.1f}%")

    return {
        "as_of_date": str(row["as_of_date"]),
        "target_vti": float(vti_t),
        "target_vxus": float(vxus_t),
        "target_bnd": float(bnd_t),
        "target_equity_total": float(vti_t + vxus_t),
        "severity": int(sev),
        "move_cap_pp": float(move_pp),
        "explanation": " | ".join(explanation),
    }


def main() -> None:
    root = repo_root()
    in_path = root / HISTORY_IN
    out_path = root / TARGETS_OUT
    ensure_dir(out_path.parent)

    if not in_path.exists():
        raise RuntimeError(f"Missing input: {in_path}. Run scripts/07_build_daily_state_history.py first.")

    full_refresh = os.getenv("TARGETS_FULL_REFRESH", "0").strip() == "1"

    hist = pd.read_parquet(in_path)
    if hist.empty or "as_of_date" not in hist.columns:
        raise RuntimeError("daily_state_history is empty or missing as_of_date.")

    hist = hist.copy()
    hist["as_of_date"] = pd.to_datetime(hist["as_of_date"], errors="coerce").dt.date.astype(str)
    hist = hist.dropna(subset=["as_of_date"]).sort_values("as_of_date").reset_index(drop=True)

    existing = None
    existing_dates: set[str] = set()
    if out_path.exists() and not full_refresh:
        try:
            existing = pd.read_parquet(out_path)
            if existing is not None and not existing.empty:
                existing_dates = set(existing["as_of_date"].astype(str).tolist())
        except Exception:
            existing = None
            existing_dates = set()

    missing = hist[~hist["as_of_date"].astype(str).isin(existing_dates)].copy()
    if missing.empty:
        print("[OK] portfolio_targets_history is up to date.")
        return

    pol = TargetPolicy()

    rows: List[Dict[str, object]] = []
    for _, r in missing.iterrows():
        rows.append(compute_targets_row(r, pol))

    new_df = pd.DataFrame(rows)
    new_df = new_df.sort_values("as_of_date").reset_index(drop=True)

    if existing is not None and not existing.empty and not full_refresh:
        out_df = pd.concat([existing, new_df], ignore_index=True)
    else:
        out_df = new_df.copy()

    out_df["as_of_date"] = pd.to_datetime(out_df["as_of_date"], errors="coerce").dt.date.astype(str)
    out_df = out_df.drop_duplicates(subset=["as_of_date"], keep="last").sort_values("as_of_date").reset_index(drop=True)
    out_df.to_parquet(out_path, index=False)

    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] rows={len(out_df)} range={out_df['as_of_date'].iloc[0]}..{out_df['as_of_date'].iloc[-1]}")


if __name__ == "__main__":
    main()
