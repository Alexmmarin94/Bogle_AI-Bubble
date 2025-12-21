## Limitations, Mitigations, and Rationale (Free Data Sources)

This project intentionally relies on **free / public** data sources and a daily (or weekly) batch pipeline. The goal is educational: demonstrate a rigorous, data-science-driven workflow for portfolio tilts and scenario validation **without** depending on expensive institutional data feeds.

### Data sources used (current connectors)

* **FRED** (macro time series, rates, financial conditions proxies) ([FRED][1])
* **Tiingo (free tier)** (market prices for selected tickers/ETFs) ([tiingo.com][2])
* **SEC/EDGAR** (US issuer fundamentals via XBRL `companyfacts` and filings metadata) + fair-access constraints ([SEC][3])
* **ECB SDMX** (official ECB datasets such as FX / rates; SDMX API) ([data.ecb.europa.eu][4])
* **IMF SDMX Central** (SDMX **structures**: dataflows/DSDs/metadata for discovery) ([dsbb.imf.org][5])

---

# 1) Coverage limitations

### 1.1 “AI overvaluation” is computed only for **SEC filers**

We compute AI-basket fundamentals (e.g., revenue, earnings, shares) using **SEC XBRL facts**, which works well for **US-listed / SEC-reporting** issuers, but does not guarantee coverage for key non-US companies that do not file XBRL in EDGAR. ([SEC][3])

**Impact:** our “AI valuation” view can be incomplete if major AI-adjacent players are non-SEC filers.

**Mitigation (within the free-source philosophy):**

* Be explicit: all “valuation” outputs are **“US SEC-filer basket only”**.
* Add a **coverage flag**: show what % of the intended AI basket is actually covered by SEC fundamentals, and reduce confidence when coverage drops.

---

### 1.2 RAM/HBM “crowding-out” is **proxy-based**, not a direct memory price index

We do **not** ingest a direct HBM/DRAM supply–demand or price index from industry vendors. Instead, we infer “memory supply constraint pressure” using:

* SEC-filer fundamentals and language signals (where possible),
* plus market price behavior of relevant public tickers (from Tiingo).

**Impact:** this is an inference, not a measurement. It can miss supply dynamics dominated by non-SEC companies and can sometimes confuse “pricing power” with other effects.

**Mitigation:**

* Publish a “Supply Constraint Confidence” score that degrades when:

  * key coverage is missing,
  * signals disagree (e.g., margins down but prices up),
  * or data is stale.
* Keep the model explainable: every “RAM/HBM constraint” output must list the top drivers.

---

### 1.3 Tiingo free tier limits constrain breadth and update cadence

Tiingo’s free plan imposes hard caps: **50 requests/hour, 1,000 requests/day, 500 unique symbols/month, 1GB/month**. ([tiingo.com][2])

**Impact:**

* We cannot naively query hundreds of tickers daily.
* Building/refreshing long daily histories for many assets requires careful caching and batching.

**Mitigation:**

* Use a **small, stable universe** (core ETFs + a limited set of proxies).
* Aggressively cache and only pull missing data since the last stored date.
* Prefer **ETF proxies** over many single stocks to reduce symbol count.
* Downsample for analytics when appropriate (e.g., compute regime features on weekly/monthly resamples once daily data is stored).

---

### 1.4 SEC/EDGAR is rate-limited and requires fair-access behavior

The SEC enforces a **10 requests/second** maximum and expects responsible automated access. ([SEC][3])

**Impact:** any fundamentals ingestion must be throttled, cached, and incremental.

**Mitigation:**

* Strict rate limiting + retry/backoff.
* Cache SEC responses and only refresh new filings.
* Always send an identifying **User-Agent** (configured in secrets) and avoid unnecessary downloads.

---

### 1.5 IMF SDMX Central is used primarily for **structures**, not guaranteed data delivery

IMF SDMX Central provides a standards-based REST interface for SDMX structures; in practice it is best treated as **metadata/discovery** rather than a guaranteed source of time-series data for all IMF datasets. ([dsbb.imf.org][5])

**Impact:** IMF integration may be limited to:

* listing dataflows and metadata,
* validating dataset structure,
* and enabling future extensions.

**Mitigation:**

* Keep IMF as “structure layer” in v1.
* Rely on FRED/ECB for most operational macro series used in the daily state.
* Document which IMF datasets (if any) are operationally supported.

---

# 2) Modeling limitations (by design)

### 2.1 This is not a “real-time trading” system

The project is designed for **periodic ETF contributions** (weekly/monthly) and regime-aware tilts, not intraday execution. Tiingo free limits also make real-time breadth unrealistic. ([tiingo.com][2])

**Mitigation:**

* Provide 1D / 7D / 30D consistency views so users can distinguish noise vs persistent signals.

### 2.2 The model prioritizes explainability over black-box optimization

All outputs are meant to be auditable: regime state, AI-cycle state, and tilt recommendations must be explainable via drivers.

**Mitigation:**

* Each recommendation includes:

  * direction and magnitude,
  * confidence,
  * “why” drivers,
  * “what would change my mind” thresholds.

---

# 3) Paid options that could cover these gaps “100%”

Even with strong data science, free sources have intrinsic constraints (coverage, latency, and missing industrial indicators). A paid stack can cover them more completely:

### 3.1 Global fundamentals + clean point-in-time coverage

* **S&P Capital IQ / Compustat** (global standardized fundamentals; broad coverage via API offerings). ([S&P Global][6])
* **Bloomberg Terminal** (multi-asset data + analytics; premium). ([Bloomberg][7])
* Similar institutional providers (FactSet / Refinitiv) are widely used for full-coverage workflows (not integrated here). ([Wall Street Prep][8])

### 3.2 Direct memory market intelligence (HBM/DRAM/NAND prices, tightness, forecasts)

Industry vendors such as TrendForce/DRAMeXchange offer subscription products with historical downloads and research reports; membership tiers are listed in the thousands to tens of thousands USD per year. ([giiresearch.com][9])

### 3.3 Market data APIs without restrictive symbol/request caps

Upgrading Tiingo tiers or using other commercial market data APIs would remove most of the free-tier constraints. (This repo intentionally stays on the free tier to demonstrate engineering + modeling under constraints.) ([tiingo.com][2])

---

# 4) Why this approach is still valuable (educational purpose)

Rather than “perfect data,” the project demonstrates:

* how to build a **reproducible daily state** from heterogeneous public sources,
* how to create **regime + AI-cycle** signals with uncertainty and consistency checks,
* how to produce **portfolio tilts** that are transparent and testable,
* how to use scenario analysis to validate robustness,
* and how to explicitly surface **coverage/quality limitations** so users can interpret recommendations responsibly.

In short: we use data science to **compensate** for the limits of free data (through careful feature design, caching, incremental ingestion, confidence scoring, and explainability), while being honest about what cannot be measured directly without paid feeds.
