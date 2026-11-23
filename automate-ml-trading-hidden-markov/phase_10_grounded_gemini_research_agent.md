# PHASE 10: Grounded Gemini Market Research Agent

**Purpose:** Extend the existing 9-phase HMM Regime Trader with a bounded, read-only AI research layer that combines the project's quantitative HMM context with current web research through Gemini + Google Search grounding.

**Important:** This phase does **not** replace or rewrite Phases 1–9. It must preserve all existing HMM, strategy, backtesting, risk-management, brokerage, orchestration, monitoring, and Streamlit behavior.

The research agent is an **information and evidence layer**, not a trading authority.

---

## 0. PHASE GOAL

Build one end-to-end research workflow:

```text
User selects ticker / asks research question
        ↓
Load current HMM + quantitative context from existing system
        ↓
Gemini Research Orchestrator
        ↓
Read-only internal tool calls
        +
Google Search grounding for current external information
        ↓
Source-backed evidence extraction
        ↓
Structured Pydantic ResearchBrief
        ↓
Quant-vs-Research comparison
        ↓
Streamlit Research page
```

The key engineering principle is separation of responsibilities:

```text
HMM / quant engine       = statistical market context
Gemini research agent    = current external evidence
Risk Manager             = deterministic safety authority
Broker layer             = execution only
Human                    = final interpretation / decision
```

The Gemini agent must never:

- submit, modify, or cancel orders
- call the broker directly
- change portfolio allocations
- modify strategy parameters
- modify HMM parameters
- modify risk thresholds
- modify circuit breakers
- promote a strategy version
- write production configuration
- claim that research evidence is financial advice

It may only read approved system state and generate structured research.

---

# 1. FIRST: INSPECT THE EXISTING REPOSITORY

Before changing code:

1. Read the repository structure and existing documentation.
2. Find the current:
   - HMM engine
   - `RegimeState` / equivalent prediction model
   - regime metadata model
   - market-data layer
   - backtest/performance layer
   - portfolio state model
   - settings/config loader
   - structured logging implementation
   - Streamlit entrypoint/page structure
   - test conventions
3. Run the current test suite and record the baseline result.
4. Do not rename or move existing modules unless absolutely required.
5. Preserve backward compatibility.
6. Do not duplicate an existing abstraction if the repository already has one that can be safely reused.
7. Do not touch unrelated behavior.

At the end of Phase 10, the entire pre-existing test suite must still pass.

---

# 2. ADD THE RESEARCH PACKAGE

Add a bounded package similar to the following. Adapt paths to the existing repository rather than forcing this exact tree if equivalent modules already exist.

```text
research/
├── __init__.py
├── agent.py                 # Gemini client / interaction wrapper
├── orchestrator.py          # end-to-end research workflow
├── schemas.py               # Pydantic request/result models
├── prompts.py               # research-system instructions
├── grounding.py             # Google Search grounding + source extraction
├── comparison.py            # quant-vs-research comparison
├── cache.py                 # result cache + TTL
├── exceptions.py
└── tools/
    ├── __init__.py
    ├── hmm_tools.py         # read-only HMM state/history
    ├── backtest_tools.py    # read-only historical metrics
    ├── market_tools.py      # optional local market-data context
    └── portfolio_tools.py   # optional read-only portfolio context

evaluation/
├── research_questions.json
├── evaluator.py
├── README.md
└── google_finance_reference/
    └── .gitkeep

tests/
├── test_research_schemas.py
├── test_research_tools.py
├── test_research_orchestrator.py
├── test_research_grounding.py
├── test_research_comparison.py
├── test_research_cache.py
├── test_research_failures.py
└── test_research_permissions.py
```

If the existing Streamlit app uses a pages directory, add the new page there. Otherwise add a Research tab/section to the current app without creating a second dashboard application.

---

# 3. DEPENDENCIES AND CONFIGURATION

Use the official Google GenAI Python SDK.

Add only required dependencies, for example:

```text
google-genai
pydantic
```

Reuse existing dependencies for HTTP, pandas, logging, caching, testing, and Streamlit where possible.

Do not introduce LangChain, LangGraph, CrewAI, AutoGen, a vector database, or another orchestration framework for this phase. They are not required for the bounded workflow.

Add configuration through the existing settings system.

Example logical configuration:

```yaml
research:
  enabled: true
  provider: gemini
  model: gemini-3.8-flash
  max_internal_tool_calls: 5
  max_search_rounds: 2
  max_sources: 12
  min_sources: 3
  cache_ttl_minutes: 360
  max_retries: 2
  request_timeout_seconds: 90
  default_lookback_days: 14
  enable_url_context: true
  enable_google_search: true
  allow_broker_tools: false
  allow_strategy_mutation: false
  allow_risk_mutation: false
```

The exact model name must exist in one central config location. Do not scatter model names through the codebase.

Add to `.env.example`:

```bash
# Gemini API authentication
GEMINI_API_KEY=

# Optional model override
GEMINI_MODEL=gemini-3.8-flash
```

Requirements:

- never hardcode the API key
- never log the API key
- never send broker credentials to Gemini
- never include secrets in tool results
- keep `.env` ignored by Git
- fail gracefully if research is enabled but no Gemini key is configured

The existing HMM/trading system must remain usable when Gemini is disabled or unavailable.

---

# 4. DO NOT AUTOMATE THE GOOGLE FINANCE WEBSITE

Do **not** scrape, browser-control, reverse-engineer, or automate the user's logged-in Google Finance / Google One interface.

For automated research use:

```text
Gemini API
+
Google Search grounding
```

The user's Google Finance Deep Search / Google AI Pro access may be used manually as an optional comparison/reference workflow in the evaluation section later in this phase.

Do not assume a Google One / Google AI Pro consumer subscription automatically provides Gemini API quota or billing. The programmatic integration must use normal Gemini API authentication and its configured quota.

---

# 5. DEFINE STRICT PYDANTIC SCHEMAS

All agent boundaries must use validated structured objects.

Implement equivalent schemas to the following.

## ResearchRequest

```python
class ResearchRequest(BaseModel):
    symbol: str
    question: str | None = None
    lookback_days: int = 14
    force_refresh: bool = False
```

Validation:

- normalize symbol to uppercase
- reject empty symbols
- cap lookback to a reasonable configured range
- sanitize/limit question length

## QuantContext

This is a snapshot of the existing project. Do not recompute HMM logic inside the research package.

```python
class QuantContext(BaseModel):
    symbol: str
    timestamp: datetime

    regime_id: int | None
    regime_label: str | None
    regime_probability: float | None
    state_probabilities: dict[str, float]

    expected_volatility: float | None
    expected_return: float | None
    volatility_rank: float | None

    is_confirmed: bool
    consecutive_bars: int
    flicker_rate: float | None
    is_flickering: bool

    model_version: str | None
    feature_version: str | None

    recent_return: float | None
    recent_volatility: float | None
```

Important:

- Preserve the distinction already established in Phase 2/3:
  - regime labels may be assigned by historical return for readability
  - strategy behavior is driven independently by volatility characteristics
- The research agent must not infer investment meaning from labels like `BULL` or `BEAR` alone.
- Provide actual confidence, volatility, stability, and regime metadata.

## BacktestContext

```python
class BacktestContext(BaseModel):
    symbol: str
    available: bool
    period_start: date | None
    period_end: date | None
    total_return: float | None
    sharpe: float | None
    max_drawdown: float | None
    win_rate: float | None
    regime_metrics: dict[str, Any]
```

## Evidence

```python
class Evidence(BaseModel):
    claim: str
    category: Literal[
        "earnings",
        "guidance",
        "product",
        "ai",
        "cloud",
        "competition",
        "regulation",
        "management",
        "macro",
        "valuation",
        "other",
    ]
    impact: Literal["positive", "negative", "mixed", "unclear"]
    source_title: str
    source_url: str
    source_domain: str | None
    published_at: datetime | None
    retrieved_at: datetime
    confidence: float
```

## ResearchBrief

```python
class ResearchBrief(BaseModel):
    symbol: str
    generated_at: datetime
    question: str

    summary: str
    catalysts: list[Evidence]
    risks: list[Evidence]
    recent_events: list[Evidence]

    conflicting_evidence: list[str]
    unknowns: list[str]

    evidence_stance: Literal[
        "positive",
        "negative",
        "mixed",
        "insufficient_evidence",
    ]

    overall_confidence: float
    source_count: int
    search_queries: list[str]
```

## QuantResearchComparison

```python
class QuantResearchComparison(BaseModel):
    symbol: str
    generated_at: datetime

    quant_summary: str
    research_summary: str

    relationship: Literal[
        "supportive",
        "partially_supportive",
        "conflicting",
        "orthogonal",
        "insufficient_evidence",
    ]

    agreements: list[str]
    contradictions: list[str]
    information_quant_model_cannot_see: list[str]
    information_research_cannot_establish: list[str]

    human_review_recommended: bool
    explanation: str
```

## ResearchResult

```python
class ResearchResult(BaseModel):
    request: ResearchRequest
    quant_context: QuantContext
    backtest_context: BacktestContext | None
    brief: ResearchBrief
    comparison: QuantResearchComparison
    cache_hit: bool
    latency_ms: int
```

Do not store hidden chain-of-thought. Store only final structured fields, tool traces, search metadata, and concise generated explanations intended for the user.

---

# 6. BUILD READ-ONLY INTERNAL TOOLS

Expose small, typed, read-only tools backed by the existing project.

Minimum tools:

## `get_hmm_snapshot(symbol)`

Returns the latest `QuantContext`.

It should read from the existing HMM/model/runtime state. It must not train a new model or mutate state merely because research was requested.

## `get_regime_history(symbol, bars=60)`

Returns recent regime history using existing recorded state where available.

Return only fields useful for research:

- timestamp
- regime/state ID
- label
- probability
- volatility information
- stability/flicker metadata

## `get_backtest_summary(symbol)`

Returns existing walk-forward/performance metrics if available.

If no backtest exists, return a structured `available=False` result instead of throwing an avoidable error.

## OPTIONAL: `get_portfolio_context(symbol)`

Only if the existing project already has a clean read-only portfolio model.

Return non-secret, high-level state such as:

- currently held: yes/no
- portfolio weight
- unrealized P&L

Do NOT expose:

- credentials
- account IDs
- buying-power secrets
- order-submission functions
- methods capable of changing positions

### Tool security

The `research.tools` package must have **no import dependency on order submission methods** unless needed solely for an immutable type, and preferably none at all.

Add an explicit allowlist:

```python
ALLOWED_RESEARCH_TOOLS = {
    "get_hmm_snapshot",
    "get_regime_history",
    "get_backtest_summary",
}
```

Tests must prove that broker mutation functions are not registered as research tools.

---

# 7. RESEARCH AGENT ARCHITECTURE

Implement a bounded three-stage workflow rather than one giant uncontrolled prompt.

```text
Stage A — Quant Context / Tool Planning
Stage B — Grounded External Research
Stage C — Structured Synthesis + Comparison
```

This separation makes tool use, grounding, testing, retries, and structured output easier to reason about.

---

# 8. STAGE A — INTERNAL TOOL-CALLING / CONTEXT COLLECTION

The orchestrator receives `ResearchRequest`.

Collect the required quantitative context using the approved read-only tools.

Preferred initial implementation:

1. Always call `get_hmm_snapshot(symbol)`.
2. Call `get_regime_history(symbol)` when the question concerns trend/regime changes or when the current regime is unstable.
3. Call `get_backtest_summary(symbol)` when evaluation/history context is useful.
4. Enforce `max_internal_tool_calls`.

To demonstrate Gemini function calling, optionally let Gemini choose among the approved internal tools, but the application must execute the functions itself and enforce the allowlist.

If Gemini requests an unknown or disallowed function:

- do not execute it
- log the rejected tool name
- return a controlled tool error
- continue or fail safely depending on context

The research package must never dynamically execute arbitrary function names.

---

# 9. STAGE B — GOOGLE SEARCH GROUNDED RESEARCH

Use Gemini with the built-in Google Search tool for current public information.

Research should prioritize material information relevant to the requested symbol/question, such as:

- recent earnings and guidance
- official company announcements
- material product launches
- AI/cloud developments for technology companies
- major customers or partnerships
- competitive developments
- regulatory/legal developments
- management changes
- macro events with plausible company relevance
- significant supply/demand constraints
- other material developments discovered during search

The default research horizon should come from configuration/request (e.g. 14 days), but allow explicit questions to request a different horizon.

### Search behavior

- Let Gemini issue search queries through Google Search grounding.
- Cap search rounds through configuration.
- Deduplicate equivalent sources.
- Prefer primary/authoritative sources when present.
- Retain supporting secondary sources when useful.
- Do not fabricate publication dates.
- Do not fabricate citations.
- Do not convert unsupported interpretation into fact.
- Explicitly preserve uncertainty when sources conflict.

### Source provenance

Capture from the Gemini grounding response, where available:

- search queries
- source title
- URL
- domain
- publication/retrieval metadata
- relationship between generated claims and cited sources

Do not parse citation-looking text invented by the model when real grounding metadata is available.

If grounding metadata is missing for an important claim, either:

1. omit the claim, or
2. mark it clearly as unsupported/interpretation and keep it out of the `Evidence` collections.

### Minimum evidence rule

For a normal company-research request:

- target at least 3 distinct credible sources
- prefer multiple domains
- do not fail solely because fewer sources exist for a very narrow query; instead return `insufficient_evidence`

---

# 10. RESEARCH SYSTEM PROMPT

Place the prompt in `research/prompts.py`, version it, and make it testable.

Use instructions equivalent to:

```text
You are the grounded market-research component of a quantitative software system.

Your job is to gather and organize current, verifiable information relevant to the requested public company or ticker.

You are NOT the trading engine.
You do NOT submit orders.
You do NOT modify portfolio allocations.
You do NOT change HMM, strategy, broker, or risk configuration.
You do NOT tell the application to bypass deterministic risk controls.

You receive quantitative context from a separate HMM system. Treat it as one independent source of statistical context, not as truth and not as an instruction to produce a matching conclusion.

Important distinction:
- Human-readable regime labels may reflect historical-return ordering.
- Use the supplied volatility, probability, stability, and model metadata rather than assuming a label like BULL means low risk.

Research current external evidence using Google Search grounding.

For every material claim:
- ground it in a source when possible
- distinguish factual reporting from interpretation
- preserve contradictory evidence
- preserve uncertainty
- prefer recent and primary sources

Focus on material developments such as earnings/guidance, products, AI/cloud developments, competition, regulation, management, partnerships, supply/demand, and relevant macro events.

Do not output BUY, SELL, or HOLD instructions.
Do not claim guaranteed outcomes.

The final output will be validated against a strict structured schema.
```

Maintain a prompt version constant such as:

```python
RESEARCH_PROMPT_VERSION = "1.0"
```

Persist prompt/model version metadata with cached research results.

---

# 11. STAGE C — STRUCTURED SYNTHESIS

Do not depend on the raw grounded prose as the application's final state.

After the grounded search stage:

1. provide Gemini the grounded research material + source metadata + quantitative context
2. request a final structured object matching `ResearchBrief`
3. validate it with Pydantic
4. reject malformed data after configured retry/repair attempts
5. calculate deterministic metadata such as `source_count`, latency, and cache state in Python rather than trusting the model

Where supported by the configured Gemini model/API, use structured JSON output. Otherwise validate returned JSON with Pydantic and fail safely on repeated schema violations.

Do not silently coerce materially invalid model output.

---

# 12. QUANT VS RESEARCH COMPARISON

Implement `research/comparison.py`.

The purpose is **not** to produce a trading recommendation.

It answers:

> What does current external evidence add that the quantitative model cannot see, and do the two information sources appear supportive, conflicting, orthogonal, or insufficient?

Inputs:

```text
QuantContext
+
ResearchBrief
```

Output:

```text
QuantResearchComparison
```

Examples of valid comparisons:

```text
Quant context:
- volatility regime stable
- confidence high

Research context:
- upcoming regulatory decision introduces event risk

Relationship:
orthogonal

Why:
The HMM describes observed market behavior, while the research layer identifies a discrete future event not represented directly in historical price/volume features.
```

or:

```text
Quant context:
- volatility rising
- regime unstable / flickering

Research context:
- conflicting earnings guidance and recent product news

Relationship:
supportive

Why:
Both systems independently indicate elevated uncertainty, although they reach that conclusion from different evidence.
```

Do not force a positive/negative agreement when the information is about different dimensions.

---

# 13. CACHE AND COST CONTROL

Implement `research/cache.py`.

The research UI should not spend Gemini/Search quota every time Streamlit reruns.

Cache completed validated `ResearchResult`s using a deterministic key containing at least:

```text
symbol
normalized question
lookback_days
research prompt version
Gemini model
important quant-context fingerprint / timestamp bucket
```

Recommended behavior:

- default TTL: 6 hours
- user can explicitly request refresh
- UI clearly shows `generated_at` and whether result came from cache
- never cache failed or schema-invalid responses as successful results
- do not serve obviously stale results as "current" without labeling their age

Use the simplest reliable persistence compatible with the repository, such as SQLite or the project's existing local persistence layer.

Do not introduce an external database solely for this phase.

Track lightweight usage metrics when available:

- request count
- cache hits/misses
- latency
- search query count
- source count
- model name

Do not log prompts containing secrets.

---

# 14. FAILURE HANDLING

Research is an optional sidecar capability. Failure must never stop the HMM/trading system.

Handle at minimum:

## Gemini unavailable / quota exceeded

Return a typed error state:

```text
Research temporarily unavailable.
Quantitative system continues normally.
```

Do not modify trading state.

## Search grounding unavailable

Do not silently substitute ungrounded current-market claims.

Either:

- retry according to configured retry policy, or
- return a research-unavailable/insufficient-evidence result

## Internal tool failure

If `get_backtest_summary()` fails but HMM snapshot is available:

- continue without backtest context
- mark it unavailable
- log the failure

If the required HMM snapshot cannot be loaded:

- allow a standalone external-research result only if the UI/request explicitly supports it
- otherwise return a clear controlled error

## Structured-output validation failure

- retry/repair at most the configured number of times
- log schema validation failures without secrets
- never render invalid data as a successful report

## Timeouts

- enforce request timeout
- show retry/refresh action in UI
- do not block the main Streamlit app indefinitely

---

# 15. STREAMLIT INTEGRATION

Add an `AI Research` page/tab to the **existing** Streamlit dashboard.

Do not create a separate dashboard service.

## Controls

At minimum:

- ticker selector/input
- optional research question
- lookback period
- `Research` button
- `Force refresh` option

Research must run on explicit user action, not every Streamlit rerender.

## Section A — Quantitative Context

Display compact cards/table for:

- symbol
- current HMM regime/state
- confidence
- expected volatility / volatility rank if available
- stability bars
- flicker/uncertainty status
- model version
- snapshot timestamp

## Section B — Current Research

Display:

- concise summary
- catalysts
- risks
- recent material events
- conflicting evidence
- unknowns
- overall confidence
- generated timestamp

Each evidence item must show a clickable source link and source title.

Do not hide source provenance in a generic "Sources" footer only; connect sources visibly to evidence.

## Section C — Quant vs Research

Display:

```text
Relationship
supportive / partially supportive / conflicting / orthogonal / insufficient
```

Then:

- agreements
- contradictions
- information the HMM cannot directly see
- information research cannot establish
- human-review flag

Do not render this as a BUY/SELL signal.

## Section D — Research Trace / Diagnostics

Expandable developer section:

- search queries used
- source domains
- cache hit/miss
- latency
- model
- prompt version
- tool calls executed

Do not expose hidden model chain-of-thought.

---

# 16. OPTIONAL: GOOGLE FINANCE DEEP SEARCH REFERENCE WORKFLOW

The user has Google AI Pro / Google Finance Deep Search access. Use it as an **optional manual evaluation reference**, not as an automated dependency.

Add documentation to:

```text
evaluation/README.md
```

Workflow:

1. Run one question in this project's Gemini Research Agent.
2. Ask the same question manually in Google Finance Deep Search.
3. Save/paste the Google Finance report into:

```text
evaluation/google_finance_reference/
```

4. Compare the two on engineering/research quality metrics.

Do not score which system gives the better investment recommendation.

Evaluate instead:

- source coverage
- source freshness
- material-event recall
- citation completeness
- unsupported-claim rate
- contradiction detection
- schema completeness
- latency/cost where measurable

Create a starter `evaluation/research_questions.json` with approximately 10–20 questions across a small technology-stock universe, for example:

```json
[
  {
    "symbol": "GOOGL",
    "question": "What material developments in the last 14 days could affect the company over the next quarter?"
  },
  {
    "symbol": "NVDA",
    "question": "What recent developments affect data-center demand, supply constraints, competition, or export risk?"
  },
  {
    "symbol": "MSFT",
    "question": "What recent AI and cloud developments are material to the company's near-term outlook?"
  }
]
```

Keep this benchmark small and understandable.

---

# 17. CLI SUPPORT

Integrate with the existing CLI style if present.

Add equivalent commands such as:

```bash
python main.py research --symbol GOOGL
python main.py research --symbol NVDA --lookback-days 7
python main.py research --symbol MSFT --question "What changed after the latest earnings report?"
python main.py research --symbol GOOGL --force-refresh
```

CLI output should include:

- quant snapshot
- research summary
- quant/research relationship
- source list
- cache status

The CLI must not place trades.

---

# 18. LOGGING AND OBSERVABILITY

Integrate with the existing structured logging system.

Emit structured events such as:

```text
research.request.started
research.tool.called
research.tool.rejected
research.search.started
research.search.completed
research.schema.validation_failed
research.cache.hit
research.cache.miss
research.completed
research.failed
```

Include safe metadata:

- symbol
- request ID
- model
- prompt version
- latency
- source count
- cache hit
- tool name
- error type

Never log:

- Gemini API keys
- brokerage credentials
- `.env` contents
- hidden chain-of-thought

Add research health information to the existing monitoring UI where appropriate:

```text
Gemini Research: healthy / unavailable / disabled
Last successful research: timestamp
Cache hit rate: ...
```

---

# 19. TESTS — REQUIRED

Use mocks/fixtures for Gemini and Google Search responses in automated tests. Unit tests must not consume real API quota.

## A. Schema tests

Test:

- valid `ResearchRequest`
- invalid/empty symbols
- invalid confidence values
- invalid evidence categories
- malformed URLs
- malformed ResearchBrief
- malformed comparison

## B. Tool tests

Verify:

- `get_hmm_snapshot()` maps the existing runtime state correctly
- state probabilities are preserved
- volatility metadata is preserved
- regime history ordering is correct
- missing backtest results return `available=False`
- no tool mutates HMM/strategy/risk/broker state

## C. Permission/safety tests

Mandatory:

```python
def test_research_agent_has_no_order_tools():
    ...


def test_research_agent_cannot_modify_strategy():
    ...


def test_research_agent_cannot_modify_risk_limits():
    ...
```

Also verify the research tool allowlist contains no broker mutation method.

## D. Grounding tests

Using recorded fixtures, verify:

- source URLs are extracted from grounding metadata
- duplicate URLs are deduplicated
- source counts are deterministic
- unsupported claims are not silently converted into Evidence
- conflicting sources can be represented
- query metadata is retained

## E. Orchestrator tests

Verify:

```text
request
→ internal context
→ grounded research
→ structured brief
→ quant/research comparison
→ ResearchResult
```

Test:

- normal success
- missing optional backtest context
- Gemini timeout
- quota error
- grounding unavailable
- malformed JSON
- Pydantic validation failure
- retry succeeds
- retry exhausted

## F. Cache tests

Verify:

- identical fresh request = cache hit
- force refresh bypasses cache
- expired entry refreshes
- changed research question creates a different key
- changed model/prompt version invalidates the relevant cache entry
- failed responses are not cached as successful research

## G. Main-system isolation test

Mandatory integration test:

```text
Gemini unavailable
→ research page reports unavailable
→ HMM still predicts
→ risk manager still operates
→ existing orchestration remains healthy
```

Phase 10 failure must not take down Phases 1–9.

## H. Existing regression suite

Run the entire repository test suite after implementation.

There must be no regression in:

- HMM inference
- look-ahead protection
- strategies
- walk-forward testing
- risk management
- broker integration
- main loop
- existing monitoring/dashboard

---

# 20. MANUAL INTEGRATION TEST

After mocked tests pass, allow an explicitly invoked real Gemini integration test.

Example:

```bash
python main.py research --symbol GOOGL --question "What material developments occurred in the last 7 days?" --force-refresh
```

Verify manually:

1. Gemini authentication succeeds.
2. Internal HMM context is loaded.
3. Google Search grounding executes.
4. Real source links are returned.
5. The final output validates against Pydantic.
6. Streamlit renders it correctly.
7. Repeating the same request uses cache.
8. Force refresh performs a new research request.
9. No broker call is made.
10. No strategy/risk configuration changes.

Do not make the real API test part of the default unit-test suite.

---

# 21. README / DOCUMENTATION UPDATE

Document Phase 10 clearly.

Add an architecture section similar to:

```text
                    EXISTING QUANT SYSTEM
                           │
Market Data → Features → HMM → Strategy → Risk → Broker
                           │
                           │ read-only context
                           ▼
                  RESEARCH ORCHESTRATOR
                    /              \
                   /                \
        Internal Quant Tools     Gemini Search
                   \                /
                    \              /
                     ▼            ▼
                    Grounded Evidence
                           │
                           ▼
                   Structured ResearchBrief
                           │
                           ▼
                 Quant-vs-Research Comparison
                           │
                           ▼
                     Streamlit Research UI
```

Document:

- why the research layer exists
- why it is read-only
- how Google Search grounding is used
- how citations/source provenance are preserved
- why Google Finance UI is not automated
- how the optional Google Finance reference evaluation works
- how caching controls API usage
- how to configure Gemini credentials
- how to disable the feature
- failure behavior
- research limitations

Explicitly state:

> The research agent is informational. It cannot execute trades or override deterministic risk controls.

---

# 22. IMPLEMENTATION ORDER

Implement in this order to keep the change bounded:

```text
1. Repository inspection + baseline tests
2. Schemas
3. Read-only HMM/backtest tools
4. Gemini client wrapper
5. Google Search grounding + source extraction
6. Research orchestrator
7. Structured synthesis
8. Quant-vs-research comparison
9. Cache
10. CLI
11. Streamlit page
12. Evaluation harness
13. Failure handling
14. Tests
15. Documentation
16. Full regression suite
```

Do not start UI work before the backend workflow passes tests.

---

# 23. ACCEPTANCE CRITERIA

Phase 10 is complete only when all of the following are true:

### Functional

- [ ] User can request research for a supported ticker.
- [ ] Current HMM context is retrieved from the existing system.
- [ ] Gemini performs grounded current-web research.
- [ ] Real source provenance is retained.
- [ ] Final result validates against Pydantic.
- [ ] Quantitative and research evidence are compared without generating a trade instruction.
- [ ] Existing Streamlit dashboard displays the result.
- [ ] CLI research mode works.
- [ ] Results are cached.

### Safety / isolation

- [ ] Research package has no ability to submit orders.
- [ ] Research package cannot mutate strategy parameters.
- [ ] Research package cannot mutate HMM parameters.
- [ ] Research package cannot mutate risk thresholds.
- [ ] Gemini outage does not stop the trading system.
- [ ] Missing/invalid research never changes positions.
- [ ] No secrets appear in prompts, logs, or cached reports.

### Quality

- [ ] Important claims have source provenance.
- [ ] Conflicting evidence is preserved.
- [ ] Insufficient evidence is represented explicitly.
- [ ] Cached/stale results are labeled with generation time.
- [ ] Agent output is schema validated.
- [ ] Internal tool calls are allowlisted.
- [ ] All new tests pass.
- [ ] All previous Phases 1–9 tests still pass.

---

# 24. FINAL IMPLEMENTATION REPORT

When implementation is complete, report:

1. files created
2. files modified
3. dependencies added
4. existing abstractions reused
5. research architecture/data flow
6. Gemini model/configuration used
7. internal tools exposed to the research workflow
8. safety boundaries implemented
9. cache behavior
10. tests added
11. old test-suite baseline vs new test-suite result
12. one real example output using a technology ticker
13. known limitations
14. recommended Phase 10.1 improvements only if supported by evidence from the finished system

Do not automatically begin another phase.

---

# 25. NON-GOALS FOR PHASE 10

Do **not** add any of the following merely to make the project appear more complex:

- multi-agent debate system
- autonomous strategy rewriting
- autonomous risk-rule changes
- live trading decisions from Gemini
- knowledge graph
- vector database / RAG
- FinBERT
- earnings transcript pipeline
- MCP server ecosystem
- local LLM
- Google Finance scraping
- browser automation of Google Finance
- second broker integration
- new technical indicators unrelated to the research feature

Any future component must solve a demonstrated limitation from Phase 10 evaluation rather than being added for technology count.

---

# FINAL TARGET

After Phase 10, the project should demonstrate this single clean capability:

```text
"The quantitative model tells me what market behavior it sees.
 The grounded research agent tells me what is happening in the real world.
 The system preserves both, highlights where they agree or conflict,
 cites the external evidence, and never lets the LLM control execution."
```

That is the finished scope of Phase 10.
