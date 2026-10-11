# ⚡ Renewable Energy Orchestrator

## Live Demo

- **Web App:** https://renewable-energy-orchestrator.streamlit.app
- **Demo Video:** https://drive.google.com/file/d/1UHWBCifoITIeU2mOyPZjpbem0HnfVcUF/view?usp=drive_link

A bounded-autonomy renewable energy dispatch prototype for the **ET AI Hackathon — Agentic Edition**.

The system monitors changing operating conditions every 15 minutes, selects an appropriate optimization strategy, solves the numerical dispatch problem, validates the proposed action, executes only the next interval, updates battery state, and explains the decision.

The prototype coordinates:

- Solar generation
- Wind generation
- Battery storage
- Grid imports and exports
- Electricity prices
- Demand

> **Core principle:** the LLM decides **what should matter**, the optimizer decides **the optimal MW values**, and deterministic validation decides **whether the action is safe to execute**.

---

## 1. Problem

Renewable energy operations change continuously. Solar and wind output vary, electricity prices move, demand changes, and battery state of charge evolves over time.

A single day-ahead schedule can therefore become stale when actual conditions diverge from forecast conditions.

The orchestration problem is to repeatedly answer:

1. What has changed?
2. Is the current operating strategy still appropriate?
3. What objective should dominate under the new conditions?
4. What is the optimal dispatch under that objective?
5. Is the proposed dispatch physically valid?
6. What should be executed for the next 15-minute interval?

The prototype addresses this as a **rolling-horizon agentic optimization problem**.

---

## 2. What the System Does

At every 15-minute interval, the orchestrator follows this loop:

1. Read forecast and current/actual operating conditions.
2. Compare forecast vs actual values.
3. Detect material events such as a price shock or solar shortfall.
4. If conditions materially change, ask the local LLM planner to select an approved objective profile.
5. Validate the LLM's strategy choice using deterministic guardrails.
6. If the strategy is unsuitable, reject it and use a safe deterministic fallback.
7. Formulate the dispatch optimization problem in Pyomo.
8. Solve the numerical dispatch using HiGHS.
9. Validate energy balance, SOC limits and battery operating limits.
10. Execute only the next 15-minute action.
11. Update battery SOC.
12. Continue through the remaining horizon.
13. Use the local LLM to explain the validated decision in readable language.

The LLM **does not directly choose MW dispatch quantities**.

---

## 3. Architecture

The architecture is shown below using plain text so that it renders consistently in GitHub, VS Code and other Markdown viewers.

```text
               FORECAST DATA              ACTUAL DATA
                     \                         /
                      \                       /
                       +----> OPERATING STATE
                                   |
                                   v
                           EVENT DETECTION
                                   |
                          Material change?
                           /             \
                         No               Yes
                         |                 |
              Retain current       LOCAL QWEN PLANNER
                 strategy           selects objective
                         |                 |
                         |                 v
                         |          STRATEGY GUARDRAIL
                         |            /          \
                         |       Accepted      Rejected
                         |          |              |
                         |          |       SAFE FALLBACK
                         |          |              |
                         +----------+--------------+
                                    |
                                    v
                         SELECTED OBJECTIVE PROFILE
                        Cost / Green / Battery Preserve
                                    |
                                    v
                              PYOMO MODEL
                                    |
                                    v
                              HiGHS SOLVER
                                    |
                                    v
                           OPTIMAL DISPATCH
                                    |
                                    v
                     DETERMINISTIC VALIDATOR
                           /                \
                        Fail                Pass
                         |                   |
                  Safe fallback      Execute next 15 min
                                             |
                                             v
                                       UPDATE SOC
                                             |
                                             v
                                      LLM EXPLANATION
                                             |
                                             v
                                         NEXT STEP
```

### Separation of responsibilities

| Component | Responsibility |
|---|---|
| Local Qwen planner | Interpret context and select an approved objective profile |
| Strategy guardrail | Reject contextually unsuitable LLM strategy choices |
| Pyomo | Formulate the mathematical optimization problem |
| HiGHS | Solve for the optimal numerical dispatch |
| Dispatch validator | Check energy balance and operational constraints |
| Python orchestrator | Maintain state and coordinate the decision loop |
| Qwen explanation layer | Explain the validated decision in readable language |
| Streamlit | Interactive demo interface |

---

## 4. Optimization Model

The dispatch problem is formulated as a linear optimization model in **Pyomo** and solved using the open-source **HiGHS** solver.

### Decision variables

For each 15-minute interval, the optimizer determines:

- Grid electricity purchase
- Grid electricity sale
- Battery charging
- Battery discharging
- Renewable curtailment
- Battery state of charge (SOC)

### Energy balance

At every interval:

```text
Solar + Wind + Grid Buy + Battery Discharge
=
Demand + Battery Charge + Grid Sell + Curtailment
```

### Battery dynamics

Battery SOC changes according to charging/discharging power, interval duration and efficiency.

The optimizer respects:

- Minimum SOC
- Maximum battery capacity
- Maximum charging rate
- Maximum discharging rate
- Charging efficiency
- Discharging efficiency
- Terminal SOC requirement where applicable

### Reliability

Reliability is treated as a **hard constraint**, not merely a soft objective.

The optimizer is not allowed to reduce cost by leaving demand unserved.

---

## 5. Selected Optimality Criteria

The prototype does not use a single fixed objective under all conditions.

The generic objective is:

```text
Minimize:
    Net Grid Cost
  + Curtailment Penalty
  + Battery Throughput / Degradation Penalty
```

Different objective profiles change the relative importance of these terms.

### `COST_FIRST`

**Primary objective:** minimize net electricity cost.

Typical use:

- Electricity purchase-price shocks
- Material changes in buy/sell economics
- No dominant renewable-curtailment or resilience issue

### `GREEN_FIRST`

**Primary objective:** maximize renewable utilization by strongly penalizing curtailment.

Typical use:

- Renewable surplus
- Renewable output likely to be curtailed
- Storage/export constraints create a risk of wasting clean generation

`GREEN_FIRST` is **not** selected merely because renewable generation is present.

### `BATTERY_PRESERVE`

**Primary objective:** preserve battery flexibility and discourage unnecessary battery throughput.

Typical use:

- Low battery SOC
- Renewable forecast uncertainty
- Situations where future flexibility is more valuable than immediate arbitrage

This profile is used in the prototype as the resilience-oriented operating mode.

### Strategy selection is bounded

The LLM may propose an objective profile, but it does not have unchecked authority.

For example, if it proposes `GREEN_FIRST` when there is no renewable surplus or curtailment risk, the deterministic strategy guardrail rejects the proposal and applies a safe fallback.

---

## 6. Rolling Horizon Operation

The base dataset contains a 24-hour horizon at 15-minute granularity: **96 intervals**.

A full-day optimizer can calculate all 96 actions at once. The orchestrator instead uses a rolling/receding-horizon approach:

```text
Observe current state
        ↓
Update current period information
        ↓
Optimize the remaining horizon
        ↓
Execute only the first 15-minute action
        ↓
Update SOC
        ↓
Move to next interval
```

### Consistency test

Under perfect information, the rolling horizon implementation reproduced the original full day optimization cost:

| Test | Cost |
|---|---:|
| Full-day optimization | ₹31,04,503.29 |
| Rolling-horizon optimization | ₹31,04,503.29 |
| Difference | ₹0.00 |

This was used as a consistency check before introducing uncertainty.

---

## 7. Demo Scenario 1 — Unexpected Electricity Price Shock

### Event

Between **15:00 and 16:00**:

- Forecast purchase price: **₹3/kWh**
- Actual purchase price: **₹20/kWh**

The system detects:

```text
PRICE_SHOCK
```

### Agent strategy

The planning agent selects:

```text
COST_FIRST
```

The price shock is the dominant operating risk and there is no competing renewable curtailment condition.

Once selected, the strategy is retained while the same event persists instead of asking the LLM to randomly re-decide the policy every 15 minutes.

### Example optimized response at 15:00

| Metric | Value |
|---|---:|
| Demand | 58 MW |
| Renewable generation | 50 MW |
| Battery discharge | 15 MW |
| Grid purchase | 0 MW |
| Grid export | 7 MW |

Energy balance:

```text
50 MW renewable + 15 MW battery discharge
=
58 MW demand + 7 MW grid export
```

### Economic comparison

| Case | Simulated cost |
|---|---:|
| Blindly follow the pre-shock plan under actual prices | ₹34.28 lakh |
| Adaptive rolling horizon response | ₹31.16 lakh |
| Saving from re-optimization | **₹3.12 lakh** |

> The ₹3.12 lakh result is from the synthetic demonstration scenario. It is not a claimed real-world realized saving.

---

## 8. Demo Scenario 2 — Solar Forecast Miss

### Event

Between **11:00 and 13:00**, actual solar output falls to **25% of forecast**.

At event onset:

- Solar forecast: **35 MW**
- Actual solar generation: **8.75 MW**
- Battery SOC: approximately **5 MWh**

The system detects:

```text
SOLAR_SHORTFALL
```

### Bounded autonomy example

In one test, the local LLM proposed:

```text
GREEN_FIRST
```

The deterministic strategy guardrail rejected that proposal because:

- There was no renewable surplus.
- Renewable curtailment was not the dominant problem.
- Battery SOC was already very low.

The system then used the safe resilience-oriented fallback:

```text
BATTERY_PRESERVE
```

The decision chain was therefore:

```text
LLM strategy proposal
        ↓
Deterministic strategy validation
        ↓
Inappropriate strategy rejected
        ↓
Safe fallback selected
        ↓
Pyomo + HiGHS optimization
        ↓
Dispatch validation
        ↓
Execute next 15-minute action
```

During the shortfall, the optimizer increased grid purchases, avoided unnecessary battery discharge and subsequently rebuilt battery SOC.

### Scenario impact

| Case | Simulated cost |
|---|---:|
| Normal operating case | ~₹31.05 lakh |
| Adaptive operation with solar shortfall | ~₹32.84 lakh |
| Additional operating impact | **~₹1.80 lakh** |

The additional cost primarily reflects the loss of expected solar generation. It should not be interpreted as a pure forecast error penalty.

---

## 9. Why the System Is Agentic

The system is intentionally more than an LLM explanation layer.

The local LLM has a real planning responsibility:

```text
Operating context
      ↓
LLM selects what should be optimized for
      ↓
Approved objective profile
      ↓
Mathematical optimizer determines exact numerical actions
```

The LLM does **not** directly invent values such as "discharge 12.7 MW".

That separation is deliberate:

- **LLM:** contextual reasoning and strategy selection
- **Optimizer:** exact numerical decision-making
- **Validator:** physical and operational safety

This is a bounded autonomy design rather than unrestricted LLM control.

---

## 10. Reliability, Guardrails and Fallbacks

### Strategy guardrails

The system checks whether an LLM-selected objective profile is appropriate for the observed situation.

An unsuitable strategy can be rejected before optimization.

### Dispatch validation

Before an optimizer action is accepted, the validator checks:

- Energy balance
- Battery SOC limits
- Maximum charge rate
- Maximum discharge rate

### LLM failure fallback

If local LLM inference is unavailable, the numerical optimization and deterministic fallback logic can continue operating.

The optimizer is therefore not dependent on a paid LLM API.

---

## 11. Technology Stack

| Layer | Technology |
|---|---|
| Local LLM | Qwen3 4B |
| Local model runtime | Ollama |
| Orchestration | Python |
| Mathematical modelling | Pyomo |
| Optimization solver | HiGHS |
| Data processing | pandas |
| Input/prototyping | Excel |
| Demo application | Streamlit |

---

## 12. Data

No operational dataset was supplied for this prototype.

The demo therefore uses a **synthetic 24-hour dataset at 15-minute granularity** containing:

- `timestamp`
- `demand_mw`
- `solar_available_mw`
- `wind_available_mw`
- `buy_price_rs_kwh`
- `sell_price_rs_kwh`

Synthetic disturbances are deliberately injected to test adaptation.

The input workbook used by the current prototype is:

```text
energy_optimizer_input.xlsx
```

The architecture is designed so that the synthetic inputs can later be replaced with real sources such as:

- SCADA / EMS feeds
- Weather forecasts
- Power market APIs
- Battery management systems
- Demand forecasts

---

## 13. Repository Structure

Recommended repository layout:

```text
renewable_orchestrator/
|
|-- app.py                       # Streamlit demo
|-- agent.py                     # Agent loop, LLM planner, guardrails, explanations
|-- optimizer_engine.py          # Pyomo + HiGHS dispatch optimization
|-- energy_optimizer_input.xlsx  # Synthetic base input data
|-- requirements.txt
|-- README.md
|
|-- outputs/                     # Optional saved scenario outputs
|   |-- agent_price_shock_results.xlsx
|   `-- solar_forecast_miss_results.xlsx
|
`-- experiments/                 # Optional development / stress-test code
    |-- stress_test.py
    |-- optimize.py
    `-- simulation_engine_backup.py
```

Only `app.py`, `agent.py`, `optimizer_engine.py`, the input workbook, `requirements.txt` and this README are required for the main demo path.

---

## 14. Quick Start — Windows

These instructions assume the repository has already been downloaded or cloned.

### Step 1 — Open PowerShell in the project folder

Example:

```powershell
cd D:\renewable_orchestrator
```

### Step 2 — Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run Python using the environment directly or adjust the local execution policy as appropriate for your machine.

### Step 3 — Install Python dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

A minimal `requirements.txt` for the current prototype is:

```text
streamlit
pandas
openpyxl
pyomo
highspy
ollama
pydantic
```

### Step 4 — Install Ollama

Install Ollama separately on the machine if it is not already available.

Verify installation:

```powershell
ollama --version
```

### Step 5 — Download the local Qwen model

```powershell
ollama pull qwen3:4b-instruct
```

Optional manual test:

```powershell
ollama run qwen3:4b-instruct
```

Use `/bye` to exit the interactive Ollama session.

### Step 6 — Make sure Ollama is running

On Windows, the Ollama desktop application normally starts the local service automatically.

If the Python client cannot connect to Ollama, start the service manually in another terminal:

```powershell
ollama serve
```

Leave that terminal running.

### Step 7 — Run the Streamlit application

From the project folder:

```powershell
streamlit run app.py
```

Streamlit normally opens the browser automatically.

If it does not, open:

```text
http://localhost:8501
```

---

## 15. Using the Demo

In the Streamlit sidebar:

1. Select a scenario.
2. Click **Run Orchestrator**.
3. Wait for the rolling-horizon simulation to complete.
4. Review the results from top to bottom.

The main sections show:

### Adaptive Response

Overall cost impact and, where applicable, the value of adaptive re-optimization.

### Event Detected

The first material deviation between forecast and actual operating conditions.

### Selected Optimality Criteria

The operating strategy selected by the planner, such as:

- `COST_FIRST`
- `GREEN_FIRST`
- `BATTERY_PRESERVE`

### Optimized Dispatch

The numerical action returned by Pyomo + HiGHS.

### Validation

Whether the proposed dispatch satisfies the implemented operating constraints.

### Agent Explanation

A local-Qwen explanation of the validated action.

### Charts and Decision Log

Full-day operating behavior and the 15-minute audit trail.

---

## 16. Running Without the Local LLM

The numerical optimization engine does not require the LLM to calculate dispatch values.

The code includes deterministic fallback logic for strategy selection and explanation when LLM inference is unavailable.

Therefore:

```text
LLM unavailable
      ↓
Deterministic fallback
      ↓
Pyomo + HiGHS
      ↓
Validator
      ↓
Dispatch continues
```

For the full agentic demonstration, however, run Ollama with `qwen3:4b-instruct` so that objective-profile selection and natural-language explanation are visible.

---

## 17. Current Limitations

The current prototype is deliberately scoped for a hackathon demonstration.

It does not yet model:

- Multiple individual solar farms
- Multiple individual wind farms
- Multiple independent batteries
- Carbon prices
- Grid frequency
- Transmission-network constraints
- Demand response
- Maintenance scheduling
- Learned solar/wind/demand forecasting
- Detailed electrochemical battery-degradation physics
- Unit commitment or generator start-up/shutdown decisions

The current data is synthetic and the objective-profile penalty values are prototype policy parameters rather than calibrated production coefficients.

---

## 18. Hackathon Scope Self-Assessment

### Feature coverage: **F3**

The prototype:

- Determines a coordinated cluster of dispatch actions.
- Selects optimality criteria according to the observed situation.
- Adjusts behavior under forecast uncertainty and disturbances.
- Simulates a sequence of changing conditions over a 24-hour period.

### Solution depth: **D2**

The prototype uses predominantly structured inputs and includes explicit guardrails, validation and fallback behavior.

It does **not** claim D3 because the current implementation does not ingest highly heterogeneous multimodal operational data.

---

## 19. Important Interpretation Notes

- All demonstrated monetary results are based on **synthetic data**.
- The ₹3.12 lakh price shock saving is a simulated comparison against following a stale pre-shock schedule.
- The solar-shortfall cost increase primarily reflects reduced renewable availability, not a pure forecasting error penalty.
- The LLM selects among approved operating strategies; it does not directly calculate dispatch quantities.
- Reliability is enforced through deterministic constraints and validation.

---

## 20. Future Extensions

Natural next steps include:

- Real weather and renewable-generation feeds
- Real power market price APIs
- Probabilistic forecasting
- Explicit carbon intensity and carbon price objectives
- Transmission constraints
- Demand response
- Multi-battery / multi-site portfolios
- More detailed battery degradation models
- Explicit reserve requirements for resilience mode
- Human approval thresholds for high-impact actions
- Production observability and audit logging

---

## Summary

The prototype demonstrates a complete bounded-autonomy loop:

```text
Observe
  ↓
Detect changing conditions
  ↓
Select operating objective
  ↓
Validate strategy
  ↓
Optimize dispatch
  ↓
Validate numerical action
  ↓
Execute next 15-minute decision
  ↓
Update state
  ↓
Explain
  ↓
Repeat
```

The design deliberately combines **LLM reasoning**, **mathematical optimization**, and **deterministic safety controls** instead of asking one component to do everything.
