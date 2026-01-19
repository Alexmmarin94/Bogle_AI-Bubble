# Technical Framework, limitations and extended glossary

This document defines:
- the **daily-state JSON schema** used by the pipeline as the basis for recommendations,
- the **rules** behind each actionable tilt,
- the **inputs** (connectors + proxies),
- the **quantitative rules** behind some outputs, like the diagnosis of the bubble state and recommended reallocation bands,
- known current limitations,
- and a **glossary** for every indicator referenced by the logic.

---

## 1) Output conventions

### 1.1 Direction labels (actionables)
All **primary tilts** map a continuous score into:

- `TILT_TOWARD`: move **toward the preferred end** of the allowed band
- `HOLD`: stay near baseline; rebalance-as-usual
- `TILT_AWAY`: move **away from the preferred end** (toward the opposite end)

Primary tilts are *banded* by design (small moves, not regime flips).

### 1.2 Persistence (anti-flip)
To reduce day-to-day noise, primary tilts and stress buckets apply:

- `PERSIST_WINDOW_DAYS = 7`
- `PERSIST_REQUIRED_DAYS = 5`

A signal is applied only if it appears at least `required_days` times within the most recent `window_days`.

### 1.3 Direction fields: `direction` vs `raw_direction` vs `persisted_direction`
Each primary tilt emits three direction-related fields:

- `raw_direction`  
  The direction implied by **today’s score only**, using the direction thresholds (e.g., `> +0.20` → `TILT_TOWARD`).  
  This is the “un-smoothed” daily signal.

- `persisted_direction`  
  The direction produced by the **persistence rule** applied to the most recent window:
  the label (`TILT_TOWARD` / `HOLD` / `TILT_AWAY`) must appear at least
  `PERSIST_REQUIRED_DAYS` times within the last `PERSIST_WINDOW_DAYS`.  
  If the rule is not met, this is `null`.

- `direction`  
  The **final direction used by the system**:
  - if `persisted_direction` is not `null`, then `direction = persisted_direction`
  - otherwise, `direction = raw_direction`

This preserves transparency:
- you can see what the model would do **today** (`raw_direction`),
- whether it achieved **persistence** (`persisted_direction`),
- and what will actually be **acted upon** (`direction`).

---

## 2) Data sources and proxies

### 2.1 FRED (macro, rates, stress)
FRED series are treated as the authoritative source for:
- volatility, credit spreads, financial conditions,
- yield curve slopes and real yields,
- inflation expectations and USD strength.

### 2.2 ECB SDMX (Europe context only)
ECB SDMX series are included as **context** for Europe (not as primary actionables by default).

**Important correction (inflation):**
- For HICP (inflation), use the **annual rate of change (YoY)** series (commonly coded as `ANR` in the ICP dataset).
- Do not use `MOR` for a YoY inflation regime: in the ICP dataset, `MOR` typically denotes **monthly rate of change** (MoM), which is much noisier and not interchangeable with YoY.

The ECB block is intended for narrative context in the report (e.g., “Euro-area yields are rising while HICP YoY is cooling”), not as an allocation trigger.

### 2.3 Tiingo (ETF price proxy)
Tiingo is used as an internal **price proxy** for liquid ETFs (SPY, QQQ, SOXX, TIP, etc.).  
It is not treated as a “macro truth source”; it is used to compute **relative leadership**, **trend**, and **realized volatility** consistently.

**Implementability note (especially for EU investors):**
Some tickers (e.g., BIL/SHV/VGSH/VGIT/VGLT) are used as **internal liquid proxies**.  
They may require **UCITS equivalents** (or a different brokerage setup) outside the US.  
The system reports signals based on these proxies; mapping to tradable instruments is a separate execution layer.

### 2.4 Frequency-safe macro normalization (critical)
Many macro series are **not daily** (weekly NFCI/STLFSI; monthly HICP; some ECB series have gaps).
To prevent artificial distortion when aligning to the trading calendar:

- **Compute z-scores on the series’ native frequency first** (weekly/monthly/original sampling).
- **Only then** expand to the trading calendar using `reindex(cal).ffill()`.

This avoids repeated-value “stair-steps” inflating or deflating rolling volatility and producing misleading z-scores.

---

## 3) Stress score (0–100) and stress buckets

The daily stress score blends:

### 3.1 FAST component (daily, market-speed)
- VIX z-score (`VIXCLS`)
- HY OAS z-score (`BAMLH0A0HYM2`)
- IG OAS z-score (`BAMLC0A0CM`)
- 21d realized equity vol z-score (derived from SPY)

**Z-score convention:** for FRED series, the z-score is computed on the raw series first, then expanded to the trading calendar.

### 3.2 SLOW component (weekly, system-wide)
- Chicago Fed NFCI z-score (`NFCI`)
- St. Louis Fed Financial Stress Index z-score (`STLFSI4`)

**Z-score convention:** computed on the weekly series first, then expanded to the trading calendar.

### 3.3 Curve inversion penalty (non-linear)
- 10Y–3M (`T10Y3M`) contributes **only when inverted** (penalty = `max(0, -T10Y3M)`).

**Note on the curve term:**
The curve contribution in the stress score is a **non-linear level penalty**, not a z-score:
`curve_penalty = max(0, -T10Y3M)` (clipped).  
This is intentional: only inversions increase stress; positive slopes do not “reward” risk-taking.

### 3.4 Aggregation and scaling
- `stress_raw = 0.55 * FAST + 0.30 * SLOW + 0.15 * curve_penalty`
- `stress_score = 100 * sigmoid(stress_raw)`

### 3.5 Buckets
- `LOW`: below the mid threshold
- `MID`: between mid and high
- `HIGH`: above the high threshold

A persisted bucket may override the raw daily bucket using the persistence rule.

---

## 4) Primary tilts (actionables)

Each primary tilt produces:
- a **score** (continuous),
- a **direction** (`TILT_TOWARD` / `HOLD` / `TILT_AWAY`) with persistence,
- key **inputs** used for transparency.

### 4.1 `EQUITY_WEIGHT_WITHIN_BAND`
**Intent**
- `TILT_TOWARD`: keep equity closer to the **top** of its allowed band
- `TILT_AWAY`: keep equity closer to the **bottom** of its allowed band

**Core drivers**
- SPY vs 200d MA (trend) z-score
- 12m momentum z-score
- 3m momentum z-score
- 21d realized vol z-score (penalty)
- 3M cash yield z-score (`DGS3MO`, penalty; computed on native sampling then expanded)
- curve inversion flag (`T10Y3M < 0`, penalty)
- stress bucket adjustment:
  - `HIGH`: negative adjustment
  - `LOW`: positive adjustment
  - `MID`: neutral

**Direction thresholds**
- `TILT_TOWARD`: score `> +0.20`
- `HOLD`: score in `[-0.20, +0.20]`
- `TILT_AWAY`: score `< -0.20`

### 4.2 `BOND_DURATION_WITHIN_BAND`
**Intent**
- `TILT_TOWARD`: **longer duration**
- `TILT_AWAY`: **shorter duration**

**Core drivers**
- 10Y yield change (30d) scaled (penalty if rising fast)
- rate volatility z-score (63d, penalty)
- rate level z-score (avg of 5Y/10Y/30Y; z computed on native sampling then expanded)
- curve z-score (more inversion → favors duration; z computed on native sampling then expanded)
- stress bucket `HIGH` bonus

### 4.3 `TIPS_SLICE_WITHIN_BAND`
**Intent**
- `TILT_TOWARD`: larger TIPS share within the inflation-hedge sleeve
- `TILT_AWAY`: smaller TIPS share

**Core drivers**
- 10Y breakeven inflation z-score (`T10YIE`; native then expanded)
- 5Y5Y forward inflation z-score (`T5YIFR`; native then expanded)
- 10Y real yield z-score (`DFII10`; native then expanded)
- 30d changes in breakeven (+) and real yield (-)

### 4.4 `US_WEIGHT_WITHIN_EQUITY_BAND`
**Intent**
- `TILT_TOWARD`: more US equity
- `TILT_AWAY`: less US equity (toward ex-US)

**Core drivers**
- Concentration composite (penalty):
  - breadth: `-(z(log(RSP/SPY)))`
  - leadership: `z(log(QQQ/SPY))`, `z(log(SOXX/SPY))`
- USD strength z-score (`DTWEXBGS`, penalty; native then expanded)
- stress bucket adjustment:
  - `LOW`: small positive adjustment
  - `HIGH`: negative adjustment

**Soft penalties (only when extremes align)**
- If **SEC heat is usable** AND `ai_fundamentals_heat >= 0.90` AND concentration composite `>= 1.0`: subtract 0.15
- If concentration composite `>= 1.0` AND USD z-score `>= 1.0`: subtract 0.10

This tilt is expected to be **rare**.

---

## 5) Optional sleeve eligibility (risk budget gate)

If a sleeve exists in the portfolio policy, the daily state can emit:

- `GOLD_SLEEVE_ELIGIBILITY`
- `COMMODITIES_SLEEVE_ELIGIBILITY`
- `BITCOIN_SLEEVE_ELIGIBILITY`

`ON` means holding mid-band is acceptable.  
`OFF` means keep at minimum (do not increase sleeve risk budget).

---

## 6) SEC fundamentals (AI basket) — weak context only

SEC Company Facts are used to compute an annual, slow-moving “heat” indicator.

### 6.1 Inputs (annual)
For each company in `config/sec_ai_basket.yml`:
- CAPEX annual
- Revenue annual
- CAPEX / Revenue
- CAPEX YoY

### 6.2 Heat definition (per-company)
Per-company heat is a slow-moving proxy computed **vs the company’s own history**:
- `heat = avg(percentile(CAPEX/Revenue), percentile(CAPEX YoY))`
- If YoY history is too short, heat falls back to the CAPEX/Revenue percentile.

Basket heat:
- `ai_fundamentals_heat = median(company_heats)`

### 6.3 Coverage gating (do not let weak data drive signals)
SEC fundamentals can be incomplete (missing tags, missing filings, unit issues).
To avoid accidental decision-making based on partial coverage:

- Compute `coverage_ratio = companies_with_heat / companies_total`.
- **Do not use `ai_fundamentals_heat` for any scoring/penalties/phase logic unless `coverage_ratio >= 0.60`.**
- If coverage is below the threshold, include a warning:
  - `notes: ["SEC heat coverage too low: 2/5"]`

In the JSON:
- `ai_fundamentals_heat` is still reported (informational).
- `ai_fundamentals_heat_used` becomes `null` when coverage is insufficient.
- `ai_fundamentals_used_in_signals` explicitly states whether it was allowed to influence decisions.

### 6.4 Cross-sectional snapshot (only when coverage is sufficient)
Percentiles are **within-company**. To add interpretability about “who is more aggressive today” across the basket,
the script adds cross-sectional ranks when coverage is sufficient:

- Rank of `CAPEX/Revenue` **across the basket** for the latest available year (`rank 1 = highest`).
- Rank of `CAPEX YoY` **across the basket** for the latest available year (`rank 1 = highest`).

These ranks live under each company’s `cross_sectional` block and do not replace the historical percentiles.

---

## 6.5 AI bubble diagnostics verdict logic (Tab 02)

This block describes the **deterministic** rule-set used to produce the Tab 02 “AI Bubble” verdicts.
It is intentionally **price + stress based** (not valuation-based) and uses only information available **up to the as-of date**.

### 6.5.1 Inputs (local, daily)
**Tiingo EOD price proxies** (uses `adjClose` when available, else `close`):
- `SPY`, `SOXX`, `QQQ`, `RSP`

**Daily-state stress**:
- `stress_score = daily_state.market_regime.stress_score` (already scaled to 0–100, see Section 3)

### 6.5.2 Derived proxy series (daily)
Ratios (leadership / concentration proxies):
- `SOXX/SPY` (semis leadership vs broad market)
- `QQQ/SPY` (growth/tech tilt vs broad market)
- `SPY/RSP` (cap-weighted vs equal-weighted; higher = narrower breadth)

Run-up proxy:
- `SOXX_6m_roll = SOXX / SOXX.shift(126) - 1`  (≈ 6-month rolling return)

Volatility proxy:
- `SPY_vol_21d_ann = std( SPY.pct_change(), 21 ) * sqrt(252)`  (21-trading-day realized vol, annualized)

### 6.5.3 Percentile transform (0–100)
Each proxy is converted into a **trailing empirical percentile** vs its own history up to the as-of date.

For a proxy series `x(t)`:
1. Restrict to history up to `as_of` and a trailing window of up to `LOOKBACK_DAYS = 3650` (~10y when available):
   - `W = { x(u) : u <= as_of and u >= as_of - LOOKBACK_DAYS }`
2. Let `x_asof` be the last value in `W`.
3. Compute:
   - `percentile(x) = 100 * mean( W <= x_asof )`

This is an empirical CDF (rank-based) transform:
- **0** means “extremely low vs its own trailing history”
- **100** means “extremely high vs its own trailing history”

### 6.5.4 Composite scores
**AI Bubble Score (0–100)** is the mean of available proxy percentiles:
- `p(SOXX/SPY)`  (semis leadership)
- `p(SPY/RSP)`   (breadth / concentration)
- `p(QQQ/SPY)`   (growth/tech tilt)
- `p(SOXX_6m_roll)` (run-up)

**Crash Risk (Amplified) (0–100)** is the mean of:
- `AI Bubble Score`
- `stress_score`  (from daily_state)
- `p(SPY_vol_21d_ann)`  (realized vol percentile)

Interpretation:
- Bubble score rises when **leadership/concentration/run-up** are extreme vs their own history.
- Crash risk rises when that **froth** coincides with **higher stress and/or higher realized volatility**.

### 6.5.5 Verdict thresholds (labels)
Bubble labels:
- `>= 80`: **Bubble: Euphoric**
- `>= 65`: **Bubble: Frothy**
- `>= 50`: **Bubble: Warm**
- `< 50`: **Bubble: Normal**

Crash risk labels:
- `>= 75`: **Crash risk: Severe**
- `>= 55`: **Crash risk: High**
- `>= 35`: **Crash risk: Elevated**
- `< 35`: **Crash risk: Low**

### 6.5.6 Chart conventions (Tab 02)
- Ratio charts (`SOXX/SPY`, `QQQ/SPY`, `SPY/RSP`) are shown on the **raw scale**.
- The “Market stress amplifiers” chart normalizes each raw series on the **visible window** using a z-score:
  - `z = (x - mean(x)) / std(x)`
  Hover always includes both **normalized** and **raw** values.

## 6.6 Portfolio band recommendations (VTI / VXUS / BND) (Tab 03)

This block explains how the system turns **directions** into **banded portfolio actions** for the example
three-fund setup used elsewhere (VTI / VXUS / BND).

### 6.6.1 Baseline anchor and what is being adjusted
A baseline allocation is treated as the neutral anchor. The action layer does not attempt to “optimize” a portfolio;
it only recommends **small, constrained moves** around the baseline.

The mapping is conceptually split into two knobs:

1) **Equity vs bonds (BND share)**  
   Primarily driven by `EQUITY_WEIGHT_WITHIN_BAND`.

2) **US vs ex-US within equity (VTI vs VXUS split)**  
   Primarily driven by `US_WEIGHT_WITHIN_EQUITY_BAND`.

Other primary tilts (e.g., duration, TIPS) are relevant only if those sleeves exist in the portfolio policy.
For the three-fund example, the execution focuses on the two knobs above.

### 6.6.2 Direction → action sign
Directions are interpreted consistently:

- `TILT_TOWARD` moves toward the preferred end of the allowed band.
- `TILT_AWAY` moves toward the opposite end of the band.
- `HOLD` stays near the baseline (rebalance-as-usual).

Persistence applies first (Section 1.2): a direction that does not achieve persistence is treated as lower-conviction,
and the action layer remains conservative.

### 6.6.3 Magnitude and the non-linear “difficulty” beyond ±5pp

Tab 03 turns **the same daily_state directions** into numeric target moves for a simple ETF portfolio:
- **Total equity**: `VTI + VXUS`
- **Bonds**: `BND`
- **Within equities**: split between `VTI` (US) and `VXUS` (ex-US)

The system is deliberately banded:
- **Typical (BAU) moves** are capped at **±5 percentage points**.
- **Larger moves** are allowed **up to ±20pp**, but only when the underlying signal is both **strong** and **persistent**, and the broader environment is flagged as unusually stressed/fragile.

#### Mapping directions → target deltas
For each relevant primary tilt, the execution layer uses:
- `score` (continuous)
- `direction` (`TILT_TOWARD` / `HOLD` / `TILT_AWAY`)
- `persisted_direction` (from the persistence rule)
- `market_regime` (stress score/bucket) and the Tab 02 fragility composites

The numeric move is computed as a **two-stage** function of `|score|`:

1) **BAU component (0–5pp)**  
Let `s = score` and `a = |s|`. The direction thresholds already define `HOLD` around zero (Section 4), with a boundary at `|s| = 0.20`.
Define a BAU intensity:
- `bau_intensity = clip( (a - 0.20) / 0.40, 0, 1 )`

Then:
- `bau_move_pp = 5 * bau_intensity`

This means:
- Scores just beyond the threshold lead to **small** moves.
- Scores around `|s| ≈ 0.60` saturate the **full ±5pp** BAU move.

2) **Extreme add-on (0–15pp, convex penalty)**  
Beyond the BAU range, the system introduces a **convex** extra term:
- `ext_intensity = clip( (a - 0.60) / 0.60, 0, 1 )`
- `extra_move_pp = 15 * (ext_intensity^2)`

This is the “non-linear difficulty”:
- the first few points beyond 5pp require a meaningful increase in `|score|`
- each additional point becomes progressively harder (because the add-on is squared)

3) **Gates for allowing moves beyond ±5pp**
The extra term is only applied when **all** of the following hold:
- **Persistence gate**: `persisted_direction` is not null and matches `direction`
- **Fragility/stress gate**: the environment is flagged as unusually fragile (e.g., `market_regime` is in a high-stress bucket and/or Tab 02 indicates elevated crash-risk conditions)

If the gates do not pass:
- `extra_move_pp = 0` (i.e., the system stays within the **±5pp** BAU band)

4) **Final cap (±20pp)**
- `total_move_pp = sign(direction) * (bau_move_pp + extra_move_pp)`
- `total_move_pp` is clipped to **±20pp**

The consequence is intentional:
- **±5pp is easy to reach** with moderately strong signals.
- **>±5pp requires “multiple things to be true at once”** (strong score + persistence + stressed/fragile regime).


### 6.6.4 What tends to trigger larger actions (mechanics)

Moves larger than ±5pp are not driven by “new” indicators; they occur when the **same score drivers** (defined in Section 4)
align strongly enough that `|score|` moves well beyond the direction threshold *and* the persistence + stress/fragility gates pass.

Below is a technical way to read “what has to line up” for larger adjustments.

#### A) Total equity vs bonds (VTI+VXUS vs BND)
This is controlled by `EQUITY_WEIGHT_WITHIN_BAND` (Section 4.1). The score is a composite of trend, momentum, volatility, rates, curve, and stress adjustments.

**Larger risk-off moves (reduce equity / increase BND)** tend to require a combination such as:
- **Market stress bucket = HIGH** (or stress score elevated)
- **Equity volatility elevated** (21d realized vol term penalizing)
- **Curve inversion** (`T10Y3M < 0`) adding penalty
- **Trend/momentum weakening** (SPY vs MA200 down; 3m/12m momentum down)
- **Cash yield punitive** (high `DGS3MO` relative to its history)

When several of these are simultaneously negative, `|score|` moves deeper into the extreme range, making the >5pp band reachable.

**Larger risk-on moves (increase equity / reduce BND)** tend to require the opposite alignment:
- stress bucket LOW (or stress score low)
- low realized vol penalty
- positive or improving trend/momentum
- curve not inverted (or improving)
- cash yield not strongly punitive

#### B) US vs ex-US within equities (VTI vs VXUS)
This is controlled by `US_WEIGHT_WITHIN_EQUITY_BAND` (Section 4.4). The score is driven by a concentration composite and USD strength, with optional SEC heat penalties when usable.

**Larger “tilt away from US” moves (reduce VTI / increase VXUS)** are most likely when:
- **Concentration composite is high** (narrow leadership):
  - `SPY/RSP` elevated (cap-weighted outperformance → narrowing breadth)
  - `QQQ/SPY` elevated and/or `SOXX/SPY` elevated (growth/tech + semis leadership)
- **USD strength z-score is high** (`DTWEXBGS` elevated vs its history), which is a headwind for ex-US and also a “crowding + macro” warning when it coincides with narrow US leadership
- Optional additional penalty applies only when **SEC heat is usable** and aligns with extremes (Section 4.4 “soft penalties”)

These conditions push the tilt score negative and, if persistent, can justify moving beyond the BAU band.

**Larger “tilt toward US” moves (increase VTI / reduce VXUS)** are comparatively rarer and typically require:
- breadth improving (less concentration)
- leadership ratios not extreme
- USD not unusually strong (or weakening)
- stress not elevated

#### C) Why the add-on is gated (avoid overreaction)
Even if the raw score is extreme on a single day, the system avoids jumping to large moves unless:
- the direction is **persistent** over the recent window (anti-flip rule), and
- the broader context is flagged as **stressed/fragile**, so that a “bigger response” is more defensible than normal BAU rebalancing.

This is the intended guardrail: larger moves are reserved for regimes where multiple independent risk signals agree.

## 7) Key limitations (known and accepted)

This section is intentionally explicit about what the system **does not** do.  
The goal is to make it easy to audit the output and to avoid over-interpreting the daily state as a “macro trading engine”.

### 7.1 The model is regime-aware, not an expected-return engine
**What this means:** the system is designed to detect “conditions” (stress, rate regime, leadership/concentration), not to estimate forward returns.

**Practical implication:** even if the model says `TILT_TOWARD` equities, it is not claiming that equities have positive alpha tomorrow. It is only saying that, *given the configured heuristics*, conditions are consistent with staying near the top of the pre-defined equity band.

**What is missing (by design):**
- equity valuation (earnings yield, forward P/E, CAPE, equity risk premium),
- earnings/revision cycles,
- positioning/flows (COT, fund flows, dealer gamma),
- macro growth indicators (PMIs, unemployment, GDP surprises).

### 7.2 ETFs are internal proxies; execution may differ by country/broker
**What this means:** Tiingo tickers are used to compute consistent price-derived features (trend, vol, relative performance).

**Practical implication:** a user in Spain might not be able to buy BIL/VGSH/VGLT directly. The signal is still valid as a *proxy-driven* direction, but translating it into trades requires:
- UCITS equivalents (if available), or
- a broker that offers the original instrument, or
- an execution mapping layer.

**Risk if ignored:** users may implement a “duration tilt” with a mismatched instrument (e.g., mixing corporate credit when the intent was government duration), breaking the meaning of the signal.

### 7.3 Mixed data frequencies limit “responsiveness” and can create stale context
**What this means:** some inputs update daily (VIX, spreads), others weekly/monthly (NFCI/STLFSI4, ECB HICP). Weekly/monthly inputs are forward-filled.

**Practical implication:** the daily JSON may change because market prices changed even if the slow macro series did not update. This is acceptable: slow series are intended to anchor the regime, not to react daily.

**Risk if misunderstood:** users might expect the system to reflect new macro prints immediately; it will reflect them only when the source series updates.

### 7.4 Stress score is a heuristic composite (weights are not “learned”)
**What this means:** `FAST`, `SLOW`, and `curve_penalty` are combined with fixed weights and then passed through a sigmoid.

**Practical implication:** the numerical value `stress_score = 35` should be read as “low stress relative to this model’s scaling”, not as “35% probability of a crash”.

**Risk if over-interpreted:** treating stress_score as a calibrated risk probability will lead to false precision.

### 7.5 Long-history z-scores still suffer from regime breaks
**What this means:** 10-year windows reduce overfitting to the last cycle, but they do not eliminate structural breaks (e.g., a post-2022 higher-rate regime vs post-2008 ZIRP).

**Practical implication:** “+1 z-score” today may not represent the same economic meaning as “+1 z-score” a decade ago, especially for rates and inflation-related series.

**Risk if ignored:** expecting stable thresholds across decades can be misleading. Persistence and banding mitigate this, but cannot remove it.

### 7.6 Feature overlap can produce ambiguous narratives
**What this means:** trend, momentum, leadership ratios, and concentration measures are naturally correlated in many regimes.

**Practical implication:** when equities rally led by megacaps, you may simultaneously see:
- strong trend/momentum (pushes equity tilt toward),
- high concentration (pushes US weight tilt away),
- “HOT” AI-cycle context.

This is not a contradiction: it reflects that the system separates “overall risk appetite” from “where within equities to allocate”.

**Risk if misunderstood:** expecting all indicators to align in the same direction will cause confusion. The system is multi-objective and outputs separate knobs.

### 7.7 Signals are not allocations, and the system does not include trading constraints
**What this means:** daily state outputs directions/magnitudes inside bands, but does not compute final weights or trades.

**Practical implication:** the user still needs an execution policy:
- rebalance cadence (monthly/quarterly),
- minimum trade size,
- transaction cost assumptions,
- tax constraints,
- cashflows (contributions vs rebalancing).

**Risk if ignored:** acting on every daily update would defeat the Bogle-compatible intent. The persistence rule helps, but execution discipline is still required.

### 7.8 SEC fundamentals are slow, incomplete, and can change historically (restatements)
**What this means:** Company Facts data is annual and arrives with reporting lags; tags and units can vary; restatements exist.

**Practical implication:** SEC-derived “heat” must be treated as slow context. The system gates usage by coverage ratio and sets `ai_fundamentals_heat_used = null` when coverage is insufficient.

**Risk if ignored:** using incomplete SEC coverage to drive decisions can introduce look-ahead-like issues or unstable signals. The coverage gate is a safety valve, not a guarantee of perfect fundamentals comparability.

### 7.9 The “Bogle-compatibility” boundary is enforced by guardrails, not by prediction quality
**What this means:** the system is compatible with Bogle principles because it:
- restricts changes to small bands,
- uses persistence to reduce flip-flopping,
- emphasizes risk management over precision.

**Practical implication:** the success criterion is not “outperform the market”, but “avoid behavioral mistakes and reduce avoidable regime risk” while staying diversified.

**Risk if misunderstood:** evaluating it like a tactical trading strategy (daily hit rate, short-horizon performance) will push the project toward overfitting and away from its intended use.

---

# Glossary

This glossary covers:
- every field in the JSON,
- and any indicator required to reproduce the scores.

## A) FRED series

### `VIXCLS`
CBOE Volatility Index (VIX).

### `BAMLH0A0HYM2`
ICE BofA US High Yield Option-Adjusted Spread (OAS).

### `BAMLC0A0CM`
ICE BofA US Corporate (Investment Grade) OAS.

### `NFCI`
Chicago Fed National Financial Conditions Index (weekly).

### `STLFSI4`
St. Louis Fed Financial Stress Index (weekly).

### `T10Y3M`
10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity (yield-curve slope).

### `DGS3MO`
3-Month Treasury Constant Maturity Rate (cash yield anchor).

### `DGS5`, `DGS10`, `DGS30`
Treasury yields used for duration level and rate-vol computations.

### `DFII10`
10-Year Treasury Inflation-Indexed Security, Constant Maturity (real yield proxy).

### `T10YIE`
10-Year Breakeven Inflation Rate.

### `T5YIFR`
5-Year, 5-Year Forward Inflation Expectation Rate.

### `DTWEXBGS`
Trade Weighted U.S. Dollar Index: Broad (USD strength proxy).

## B) Derived market measures

### `spy_realized_vol_21d_ann`
21-trading-day realized volatility of SPY, annualized.

### `spy_200d_trend`
`SPY / MA200 - 1`.

### `spy_momentum_3m`, `spy_momentum_12m`
3-month and 12-month price momentum for SPY.

### `concentration_composite`
Composite concentration proxy built from:
- `RSP/SPY` (breadth)
- `QQQ/SPY`, `SOXX/SPY` (tech/semis leadership)
- `XLK/SPY` (sector tech)

## C) Output blocks

### `tilt_signals.primary`
Dictionary containing:
- `score`
- `direction`, `raw_direction`, `persisted_direction`
- `magnitude`
- `inputs` and `rules` (for explainability)

### `sleeve_eligibility`
Optional risk-budget gate flags for sleeves.

### `sec_fundamentals`
Company-level and basket summary metrics, including:
- `ai_fundamentals_heat`
- `coverage_ratio`
- `heat_used_in_signals`
- optional `notes` warnings
- optional `cross_sectional` ranks per company

### `market_regime`
Contains the stress score, bucket, and the driver series used to compute it.

### `ecb_context`
Context-only ECB series for EUR narrative (not direct allocation triggers)."
