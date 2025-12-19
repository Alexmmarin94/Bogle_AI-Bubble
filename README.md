# AI-Bubble```markdown
# FIRE-Tactical Portfolio Rebalancer (Streamlit)

A Streamlit app that generates **explainable, regime-aware portfolio recommendations** across **asset classes and regions** (US, Europe, Japan, Emerging Markets, etc.), with a dedicated **AI-cycle risk overlay** (concentration, supply-chain stress, “expectations risk”) and a **Scenario Engine** for stress-testing.

> ⚠️ **Disclaimer**: This project is for educational and research purposes. It does **not** provide financial advice or personalized investment recommendations. Always do your own research and/or consult a licensed professional.

---

## Why this exists

Most portfolio tools either:
- stay purely “strategic” (set-and-forget) and ignore the current cycle, or
- go fully “tactical” with opaque black-box predictions.

This app aims for a pragmatic middle-ground:
- **Strategic Allocation (SAA)** as the anchor,
- **Tactical Overlay (TAA)** as small, bounded tilts based on transparent signals,
- plus an **AI-cycle overlay** that explicitly addresses concentration and semiconductor/memory supply-chain dynamics that can spill over into broader equity performance and macro conditions. (Recent reporting highlights memory supply constraints and production shifts toward AI-related HBM as a real macro/industry factor.) :contentReference[oaicite:0]{index=0}

---

## Product definition

### Advanced mode (finance-savvy users)
**Minimal inputs**
- Base currency (e.g., EUR, USD)
- Risk profile (Conservative / Moderate / Aggressive)
- Horizon (years)
- Current portfolio (tickers + weights) **or** “use model portfolio”
- Constraints (optional): max tech %, no EM, UCITS-only, min bond quality, etc.

**Minimal outputs**
- Target weights by **asset class** and **region**
- Suggested trades/rebalance actions (what to buy/sell)
- A **clear explanation** referencing the exact signals used (macro regime + AI-cycle dashboard)
- Scenario Engine results (portfolio sensitivity / drawdown expectations under shocks)

### Light mode (FIRE democratization)
A user-friendly flow that avoids detailed portfolio inputs.

**Light-mode questions (example)**
- Intention: FIRE / wealth accumulation / capital preservation
- Age
- Monthly budget / contribution
- “How much drawdown can you tolerate?” (plain-language choices)
- Time horizon
- Current savings/investments (rough bands)
- Region preference (home bias vs global)
- Optional: “Do you want simplicity above all?” (yes/no)

**Light-mode outputs**
- A recommended “starter” allocation template (regions + asset classes)
- Simple explanations and action steps
- A simplified scenario view (“what if markets drop 30%?”)

---

## Core features

- **Macro/Market Regime Dashboard**: growth, inflation, rates/curve, financial stress.
- **Regional Allocation Engine**: US vs Europe vs Japan vs EM tilts with guardrails.
- **AI-Cycle Overlay**:
  - **Market concentration risk** (e.g., top holdings dominance; optional equal-weight/alt exposure prompts). :contentReference[oaicite:1]{index=1}
  - **Supply-chain stress proxies** (memory/HBM constraints, capex intensity, spillovers). :contentReference[oaicite:2]{index=2}
  - **Expectations risk** via scenarios (AI drawdown, multiple compression, etc.).
- **Market Notes Tab** (plain-English bullets by bucket):
  - S&P 500 / US equities
  - Europe / Japan / EM equities
  - Bonds (gov/IG/HY), duration
  - Gold / cash
  - “Based on current signals, consider increasing/reducing weight” with rationale
- **Scenario Engine**: coherent shock sets and portfolio response (drawdown, volatility, contribution to risk).
- **Auditability**: every recommendation is reproducible from stored signals + parameters.

---

## Data sources and ingestion

We prioritize APIs with stable documentation and programmatic access.

### Macro & rates (official / institutional)
- **FRED (St. Louis Fed)** for US macro series and market proxies. :contentReference[oaicite:3]{index=3}
- **ECB Data Portal (SDMX REST)** for Euro Area / ECB-related series. Note: SDW endpoints are being repointed to the ECB Data Portal API (data-api). :contentReference[oaicite:4]{index=4}

### Volatility / stress indicators
- **VIX**: either directly from Cboe’s downloadable historical dataset or via FRED’s VIX series. :contentReference[oaicite:5]{index=5}
- **Credit spreads**: e.g., ICE BofA High Yield OAS series. :contentReference[oaicite:6]{index=6}

### Market prices (ETFs, indices proxies)
- **Tiingo** End-of-Day prices + fundamentals (clean REST endpoints). :contentReference[oaicite:7]{index=7}
- **Polygon** (optional) for broader market/reference endpoints (tickers, metadata, aggregates). :contentReference[oaicite:8]{index=8}
- **OpenBB (optional)** as a unifying connector layer (ODP/Workspace integration) if we want to swap providers with minimal code changes. :contentReference[oaicite:9]{index=9}

---

## External API endpoints (planned)

Below is the **project-level list** of endpoints we may call (not an exhaustive list of each provider’s full API surface).

### FRED (US macro) — base
- Base: `https://api.stlouisfed.org/fred/` :contentReference[oaicite:10]{index=10}

**Endpoints**
- Series metadata:  
  `https://api.stlouisfed.org/fred/series?series_id={SERIES_ID}&api_key={KEY}&file_type=json` :contentReference[oaicite:11]{index=11}
- Observations (time series):  
  `https://api.stlouisfed.org/fred/series/observations?series_id={SERIES_ID}&api_key={KEY}&file_type=json` :contentReference[oaicite:12]{index=12}
- (Optional) Real-time vintage parameters for revision-aware studies:  
  same endpoint with `realtime_start` / `realtime_end` :contentReference[oaicite:13]{index=13}

**Common series we’ll likely use**
- S&P 500 price index: `SP500` :contentReference[oaicite:14]{index=14}
- VIX: `VIXCLS` :contentReference[oaicite:15]{index=15}
- High yield OAS: `BAMLH0A0HYM2` :contentReference[oaicite:16]{index=16}

### ECB Data Portal (Euro area / EU macro) — SDMX REST
- Overview: `https://data.ecb.europa.eu/help/api/overview` :contentReference[oaicite:17]{index=17}
- Data API entry point (noted by ECB): `https://data-api.ecb.europa.eu` :contentReference[oaicite:18]{index=18}

**Endpoints (SDMX REST pattern)**
- Data retrieval pattern (ECB docs):  
  `protocol://wsEntryPoint/resource/flowRef/key?parameters` :contentReference[oaicite:19]{index=19}
- SDMX data mode documentation:  
  `https://data.ecb.europa.eu/help/getting-data-web-services-sdmx` :contentReference[oaicite:20]{index=20}

> Implementation note: ECB queries are SDMX-structured; we’ll provide a small “data discovery helper” module to find flowRef and key definitions when adding new series.

### Cboe VIX (direct download)
- VIX historical data download page:  
  `https://www.cboe.com/tradable_products/vix/vix_historical_data` :contentReference[oaicite:21]{index=21}

### Tiingo (market prices / fundamentals)
- Latest/historical daily prices:  
  `https://api.tiingo.com/tiingo/daily/{TICKER}/prices` :contentReference[oaicite:22]{index=22}
- Fundamentals (definitions / metrics endpoints):  
  `https://api.tiingo.com/tiingo/fundamentals/` :contentReference[oaicite:23]{index=23}
- Tiingo documentation root:  
  `https://www.tiingo.com/documentation/` :contentReference[oaicite:24]{index=24}

### Polygon (optional, for reference/metadata/aggregates)
- Reference APIs docs (Python wrapper docs; used as guidance for endpoints):  
  `https://polygon.readthedocs.io/en/latest/References.html` :contentReference[oaicite:25]{index=25}

> If Polygon is enabled, we’ll add endpoints for ticker discovery/metadata and aggregates as needed.

### OpenBB (optional integration layer)
- OpenBB docs root: `https://docs.openbb.co/` :contentReference[oaicite:26]{index=26}
- ODP Python docs: `https://docs.openbb.co/python` :contentReference[oaicite:27]{index=27}
- Data integration docs (custom backend idea):  
  `https://docs.openbb.co/workspace/developers/data-integration` :contentReference[oaicite:28]{index=28}

---

## Analysis logic (high level)

### 1) Build a “Regime Score” (macro + market stress)
We compute a composite score from:
- **Growth** proxies
- **Inflation** trend/surprises
- **Rates/Curve** proxies
- **Financial stress** proxies (VIX + credit spreads)

Then classify into a small set of regimes:
- Risk-on
- Neutral
- Risk-off
- Inflationary
- Recessionary

The app always shows:
- latest values,
- rolling context (recent history),
- how each signal influenced the regime classification.

### 2) Strategic allocation (SAA) as the anchor
Each risk profile maps to a baseline global allocation:
- equities split by region (US / Europe / Japan / EM),
- bonds split by type/duration (gov / IG / HY),
- diversifiers (gold/cash).

### 3) Tactical overlay (TAA) with guardrails
Regime-dependent tilts are applied **within tight bounds** (e.g., ±5–15 percentage points by bucket), so the system remains “portfolio management”, not trading.

### 4) AI-cycle overlay (dedicated logic)
This overlay does not “predict AI”; it manages **portfolio risk created by AI-cycle dynamics**.

**Signals and mechanics**
- **Concentration risk**
  - Detect heavy dependence on top holdings / US large-cap dominance.
  - Optional mitigations: more ex-US allocation, equal-weight style, factor diversification. :contentReference[oaicite:29]{index=29}
- **Supply-chain stress**
  - Track proxies related to memory/HBM scarcity and production shifts that can propagate into device markets and capex cycles. :contentReference[oaicite:30]{index=30}
- **Expectations risk**
  - Push the user to evaluate “what if AI multiples compress?” via scenarios rather than narratives.

---

## Scenario Engine

The Scenario Engine runs a small set of coherent “macro + market” shocks and reports:
- expected drawdown range (based on historical analogs / stress factors),
- volatility uplift,
- contribution to risk by bucket,
- recovery sensitivity (optional).

**Default scenarios (v1)**
- AI drawdown (tech/semis shock + spread widening + VIX spike)
- Soft landing
- Stagflation
- Recession
- Rates up / duration shock

Users can:
- tweak probabilities (optional),
- toggle scenario severity (light/standard/severe),
- export scenario results.

---

## Streamlit app structure (tabs)

Proposed Streamlit navigation:

1. **Welcome / Mode Selector**
   - Light vs Advanced
   - “What this tool is / isn’t”

2. **Data Status**
   - API keys configured?
   - last refresh timestamps
   - dataset coverage warnings

3. **Market Regime Dashboard**
   - regime classification
   - signal decomposition (growth/inflation/rates/stress)

4. **AI-Cycle Dashboard**
   - concentration indicators
   - supply-chain stress indicators (memory/HBM proxies)
   - “risk map” for AI-sensitive exposure

5. **Market Notes (Actionable Bullets)**
   - US / Europe / Japan / EM
   - Bonds (gov/IG/HY, duration posture)
   - Gold / cash
   - Each bullet ties back to current regime + AI overlay

6. **Portfolio Analyzer**
   - current exposures by asset class + region
   - risk metrics (rolling vol, drawdowns, correlations)
   - concentration metrics

7. **Recommendations**
   - target weights
   - rebalance steps (what to change)
   - rationale (signal-by-signal)

8. **Scenario Studio**
   - run default scenarios
   - adjust severity
   - compare current vs recommended portfolio

9. **About / Methodology**
   - assumptions
   - limitations
   - citations and data sources

---

## Tentative folder structure

This is a first proposal and will evolve as the app matures.

```
.
├── app/
│   ├── Home.py
│   ├── pages/
│   │   ├── 01_Data_Status.py
│   │   ├── 02_Market_Regime.py
│   │   ├── 03_AI_Cycle.py
│   │   ├── 04_Market_Notes.py
│   │   ├── 05_Portfolio_Analyzer.py
│   │   ├── 06_Recommendations.py
│   │   ├── 07_Scenario_Studio.py
│   │   └── 08_Methodology.py
│   └── ui/
│       ├── components.py
│       └── styles.py
├── src/
│   ├── config/
│   │   ├── settings.py
│   │   └── tickers_universe.yaml
│   ├── data/
│   │   ├── clients/
│   │   │   ├── fred_client.py
│   │   │   ├── ecb_sdmx_client.py
│   │   │   ├── tiingo_client.py
│   │   │   └── cboe_client.py
│   │   ├── cache.py
│   │   └── pipelines.py
│   ├── features/
│   │   ├── macro_signals.py
│   │   ├── stress_signals.py
│   │   ├── concentration_signals.py
│   │   └── ai_supplychain_signals.py
│   ├── allocation/
│   │   ├── strategic_allocations.py
│   │   ├── tactical_overlay.py
│   │   ├── regional_tilts.py
│   │   └── recommendation_engine.py
│   ├── scenarios/
│   │   ├── scenario_definitions.py
│   │   ├── scenario_engine.py
│   │   └── reporting.py
│   ├── portfolio/
│   │   ├── analytics.py
│   │   ├── risk.py
│   │   └── rebalancing.py
│   └── utils/
│       ├── dates.py
│       ├── logging.py
│       └── validation.py
├── tests/
│   ├── unit/
│   └── integration/
├── data_samples/
│   └── example_portfolios/
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── README.md
└── LICENSE

```

---

## Next steps (implementation order)

1) Implement data clients + caching (FRED + ECB + Tiingo + VIX source). :contentReference[oaicite:31]{index=31}  
2) Build the Market Regime computation and visualization.  
3) Add Portfolio Analyzer (exposures, drawdowns, correlations).  
4) Add Recommendations engine (SAA + bounded TAA).  
5) Implement AI-cycle overlay (concentration + supply-chain + expectations scenarios). :contentReference[oaicite:32]{index=32}  
6) Implement Scenario Studio and exportable reports.

---

## License
Choose a permissive license (MIT/Apache-2.0) unless you plan to commercialize.
```

