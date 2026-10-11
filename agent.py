from ollama import chat
from optimizer_engine import solve_dispatch
import pandas as pd
import json

ALLOWED_OBJECTIVE_PROFILES = [
    "COST_FIRST",
    "GREEN_FIRST",
    "BATTERY_PRESERVE"
]

def get_current_state(
    current_period,
    forecast_df,
    actual_df,
    current_soc,
    battery_available=True
):

    forecast = forecast_df.iloc[current_period]
    actual = actual_df.iloc[current_period]

    state = {
        "timestamp": actual["timestamp"],

        "demand_mw": actual["demand_mw"],

        "solar_forecast_mw": forecast["solar_available_mw"],
        "solar_actual_mw": actual["solar_available_mw"],

        "wind_forecast_mw": forecast["wind_available_mw"],
        "wind_actual_mw": actual["wind_available_mw"],

        "price_forecast": forecast["buy_price_rs_kwh"],
        "price_actual": actual["buy_price_rs_kwh"],

        "battery_soc_mwh": current_soc,

        "battery_available": battery_available
    }

    return state


def detect_event(state):

    events = []

    solar_forecast = state["solar_forecast_mw"]
    solar_actual = state["solar_actual_mw"]

    price_forecast = state["price_forecast"]
    price_actual = state["price_actual"]

    # Solar shortfall
    if solar_forecast > 0:

        solar_error_pct = (
            solar_actual - solar_forecast
        ) / solar_forecast

        if solar_error_pct <= -0.25:
            events.append("SOLAR_SHORTFALL")

    # Price shock
    if price_actual >= price_forecast * 1.5:
        events.append("PRICE_SHOCK")

    # Battery outage
    if not state["battery_available"]:
        events.append("BATTERY_OUTAGE")

    if not events:
        events.append("NORMAL")

    return events

def validate_action(action, assets):

    errors = []

    tolerance = 1e-6

    supply = (
        action["solar_available_mw"]
        + action["wind_available_mw"]
        + action["grid_buy_mw"]
        + action["battery_discharge_mw"]
    )

    use = (
        action["demand_mw"]
        + action["battery_charge_mw"]
        + action["grid_sell_mw"]
        + action["curtailment_mw"]
    )

    if abs(supply - use) > tolerance:
        errors.append("ENERGY_BALANCE_VIOLATION")

    if (
        action["soc_mwh"]
        < assets["minimum_soc_mwh"] - tolerance
    ):
        errors.append("SOC_BELOW_MINIMUM")

    if (
        action["soc_mwh"]
        > assets["battery_capacity_mwh"] + tolerance
    ):
        errors.append("SOC_ABOVE_MAXIMUM")

    if (
        action["battery_charge_mw"]
        > assets["maximum_charge_mw"] + tolerance
    ):
        errors.append("CHARGE_LIMIT_VIOLATION")

    if (
        action["battery_discharge_mw"]
        > assets["maximum_discharge_mw"] + tolerance
    ):
        errors.append("DISCHARGE_LIMIT_VIOLATION")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def explain_action(state, events, action):

    explanations = []

    if "PRICE_SHOCK" in events:

        explanations.append(
            f"Market price increased unexpectedly from "
            f"₹{state['price_forecast']:.2f}/kWh to "
            f"₹{state['price_actual']:.2f}/kWh."
        )

    if "SOLAR_SHORTFALL" in events:

        explanations.append(
            f"Solar generation was below forecast: "
            f"{state['solar_forecast_mw']:.1f} MW expected "
            f"versus {state['solar_actual_mw']:.1f} MW actual."
        )

    if "BATTERY_OUTAGE" in events:

        explanations.append(
            "The battery is unavailable and has been removed "
            "from the current dispatch options."
        )

    explanations.append(
        f"Optimized action: buy "
        f"{action['grid_buy_mw']:.1f} MW from the grid, "
        f"charge battery at "
        f"{action['battery_charge_mw']:.1f} MW, "
        f"and discharge at "
        f"{action['battery_discharge_mw']:.1f} MW."
    )

    return " ".join(explanations)

def optimize_action(
    current_period,
    forecast_df,
    actual_df,
    assets,
    current_soc,
    battery_available=True,
    objective_profile="COST_FIRST"
):

    # Future horizon is what we currently forecast
    remaining_horizon = (
        forecast_df
        .iloc[current_period:]
        .copy()
        .reset_index(drop=True)
    )

    # But current-period information is already observed,
    # so replace the first forecast row with actual values.
    actual_now = actual_df.iloc[current_period]

    current_columns = [
        "demand_mw",
        "solar_available_mw",
        "wind_available_mw",
        "buy_price_rs_kwh",
        "sell_price_rs_kwh"
    ]

    for col in current_columns:
        remaining_horizon.loc[0, col] = actual_now[col]

    current_assets = assets.copy()

    if battery_available:
        terminal_soc = assets["initial_soc_mwh"]
    else:
        current_assets["maximum_charge_mw"] = 0
        current_assets["maximum_discharge_mw"] = 0

        # Failed battery cannot magically return to 25 MWh
        terminal_soc = current_soc

    plan = solve_dispatch(
        remaining_horizon,
        current_assets,
        initial_soc_mwh=current_soc,
        terminal_soc_mwh=terminal_soc,
        objective_profile=objective_profile
    )

    # Agent executes ONLY the immediate 15-minute action
    action = plan.iloc[0].to_dict()

    return action, plan

def explain_action_with_llm(
    state,
    events,
    action,
    model_name="qwen3:4b-instruct"
):

    renewable_supply = (
        state["solar_actual_mw"]
        + state["wind_actual_mw"]
    )
    net_load_before_battery = (
        state["demand_mw"]
        - renewable_supply
    )
    total_supply = (
        renewable_supply
        + action["battery_discharge_mw"]
        + action["grid_buy_mw"]
    )

    total_use = (
        state["demand_mw"]
        + action["battery_charge_mw"]
        + action["grid_sell_mw"]
        + action["curtailment_mw"]
    )
    prompt = f"""
You are an operations assistant for a renewable-energy utility.

A mathematical optimizer has already calculated and validated the
dispatch below.

Your job is ONLY to explain the result.
Do not change the dispatch.
Do not invent numbers.
Do not infer facts that are not provided.
Do not say that all battery discharge serves demand if some
power is exported. Explain the energy balance using the supplied
demand, renewable generation, battery discharge and grid export.
Do not attribute exported electricity specifically to the battery,
solar, or wind. Electricity in the system is fungible.

Explain the system-level energy balance only.

For example, if renewable generation is 50 MW, battery discharge
is 15 MW, demand is 58 MW and export is 7 MW, say that total
available supply is 65 MW, of which 58 MW serves demand and
7 MW is exported.

Do not say that all 15 MW of battery discharge serves demand.

ENERGY BALANCE FACTS
Renewable generation: {renewable_supply:.2f} MW
Battery discharge: {action['battery_discharge_mw']:.2f} MW
Grid purchase: {action['grid_buy_mw']:.2f} MW
Total supply: {total_supply:.2f} MW

Demand: {state['demand_mw']:.2f} MW
Battery charging: {action['battery_charge_mw']:.2f} MW
Grid export: {action['grid_sell_mw']:.2f} MW
Curtailment: {action['curtailment_mw']:.2f} MW
Total use: {total_use:.2f} MW

CURRENT STATE
Renewable generation available now:
{renewable_supply:.2f} MW

Demand remaining after renewable generation:
{max(net_load_before_battery, 0):.2f} MW

Timestamp: {state['timestamp']}
Demand: {state['demand_mw']:.2f} MW

Solar forecast: {state['solar_forecast_mw']:.2f} MW
Solar actual: {state['solar_actual_mw']:.2f} MW

Wind forecast: {state['wind_forecast_mw']:.2f} MW
Wind actual: {state['wind_actual_mw']:.2f} MW

Grid purchase price forecast: ₹{state['price_forecast']:.2f}/kWh
Grid purchase price actual: ₹{state['price_actual']:.2f}/kWh

Battery SOC before action:
{state['battery_soc_mwh']:.2f} MWh

EVENT DETECTED
{', '.join(events)}

OPTIMIZED ACTION
Grid purchase: {action['grid_buy_mw']:.2f} MW
Grid sale: {action['grid_sell_mw']:.2f} MW
Battery charging: {action['battery_charge_mw']:.2f} MW
Battery discharging: {action['battery_discharge_mw']:.2f} MW
Renewable curtailment: {action['curtailment_mw']:.2f} MW
Battery SOC after action: {action['soc_mwh']:.2f} MWh

Explain in 2 concise sentences:
1. what changed,
2. how the optimized dispatch responds.

Distinguish clearly between power used to serve demand
and power exported to the grid.
"""

    response = chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    return response.message.content

def plan_with_llm(
    state,
    detected_events,
    model_name="qwen3:4b-instruct"
):

    renewable_actual = (
        state["solar_actual_mw"]
        + state["wind_actual_mw"]
    )

    net_load_mw = (
        state["demand_mw"]
        - renewable_actual
    )

    renewable_surplus_mw = max(
        renewable_actual - state["demand_mw"],
        0
    )

    price_change_pct = (
        (
            state["price_actual"]
            - state["price_forecast"]
        )
        / state["price_forecast"]
        * 100
        if state["price_forecast"] > 0
        else 0
    )
    prompt = f"""
You are the planning agent for a renewable-energy utility.

You do NOT calculate MW dispatch values.
A mathematical optimization engine will calculate those.

Your job is to choose the most appropriate operating objective
for the current situation.

CURRENT STATE
Timestamp: {state['timestamp']}
Demand: {state['demand_mw']:.2f} MW

Solar forecast: {state['solar_forecast_mw']:.2f} MW
Solar actual: {state['solar_actual_mw']:.2f} MW

Wind forecast: {state['wind_forecast_mw']:.2f} MW
Wind actual: {state['wind_actual_mw']:.2f} MW

Price forecast: ₹{state['price_forecast']:.2f}/kWh
Price actual: ₹{state['price_actual']:.2f}/kWh

Battery SOC: {state['battery_soc_mwh']:.2f} MWh
Battery available: {state['battery_available']}

Detected condition:
{', '.join(detected_events)}

DERIVED OPERATING FACTS

Actual renewable generation:
{renewable_actual:.2f} MW

Demand:
{state['demand_mw']:.2f} MW

Net load before battery/grid:
{net_load_mw:.2f} MW

Renewable surplus:
{renewable_surplus_mw:.2f} MW

Forecast purchase price:
₹{state['price_forecast']:.2f}/kWh

Actual purchase price:
₹{state['price_actual']:.2f}/kWh

Price deviation:
{price_change_pct:.1f}%

AVAILABLE STRATEGIES

COST_FIRST

Select when market economics are the dominant operating issue.

Typical triggers:
- major electricity price spike
- material change in buy/sell economics
- no significant renewable-curtailment risk
- sufficient operational feasibility remains

A PRICE_SHOCK normally implies COST_FIRST unless a more severe
reliability or battery-reserve issue exists.


GREEN_FIRST

Select only when renewable energy would otherwise be wasted
or curtailed.

Typical triggers:
- renewable generation exceeds demand
- storage/export capacity is insufficient
- curtailment is occurring or likely

Do NOT select GREEN_FIRST merely because renewable generation
is being fully utilized.
If there is zero renewable surplus and zero curtailment risk,
GREEN_FIRST is normally inappropriate.


BATTERY_PRESERVE

Select when maintaining battery flexibility is more important
than short-term arbitrage.

Typical triggers:
- very low SOC
- severe forecast uncertainty
- operational contingency
- expected future reliability requirement

Choose the strategy that addresses the DOMINANT operational risk.

For a pure PRICE_SHOCK with no renewable surplus or curtailment,
select COST_FIRST.

Never claim that renewable generation equals demand unless the
provided DERIVED OPERATING FACTS explicitly state that it does.

When multiple criteria could apply, identify the dominant operational
risk and select the profile that addresses that risk.
Do not switch profiles without a material change in conditions.

Return ONLY valid JSON in this exact structure:

{{
    "decision": "REOPTIMIZE",
    "objective_profile": "COST_FIRST",
    "reason": "Brief reason for selecting this strategy."
}}

objective_profile must be exactly one of:
COST_FIRST
GREEN_FIRST
BATTERY_PRESERVE
"""

    response = chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    result = json.loads(
        response.message.content
    )

    if (
        result.get("objective_profile")
        not in ALLOWED_OBJECTIVE_PROFILES
    ):
        raise ValueError(
            "LLM returned invalid objective profile"
        )

    return result

def fallback_objective_profile(
    state,
    events
):

    if "PRICE_SHOCK" in events:

        return {
            "decision": "REOPTIMIZE",
            "objective_profile": "COST_FIRST",
            "reason": (
                "Fallback rule: large market-price deviation "
                "requires cost-focused re-optimization."
            )
        }

    if "SOLAR_SHORTFALL" in events:

        if state["battery_soc_mwh"] <= 10:

            profile = "BATTERY_PRESERVE"

            reason = (
                "Fallback rule: solar shortfall coincides "
                "with low battery SOC, so stored flexibility "
                "should be preserved."
            )

        else:

            profile = "COST_FIRST"

            reason = (
                "Fallback rule: solar shortfall requires "
                "economic rebalancing while sufficient "
                "battery flexibility remains."
            )

        return {
            "decision": "REOPTIMIZE",
            "objective_profile": profile,
            "reason": reason
        }

    if "BATTERY_OUTAGE" in events:

        return {
            "decision": "REOPTIMIZE",
            "objective_profile": "COST_FIRST",
            "reason": (
                "Fallback rule: available resources must "
                "be re-optimized after battery loss."
            )
        }

    return {
        "decision": "REOPTIMIZE",
        "objective_profile": "COST_FIRST",
        "reason": "Normal default operating strategy."
    }

def select_strategy(
    state,
    events
):

    try:

        strategy = plan_with_llm(
            state,
            events
        )

        valid, validation_reason = (
            validate_strategy(
                state,
                events,
                strategy
            )
        )

        if valid:
            return strategy

        print(
            "LLM strategy rejected:",
            validation_reason
        )

        fallback = fallback_objective_profile(
            state,
            events
        )

        fallback["reason"] = (
            f"LLM proposed "
            f"{strategy['objective_profile']}, "
            f"but strategy guardrail rejected it: "
            f"{validation_reason} "
            f"Fallback selected "
            f"{fallback['objective_profile']}."
        )

        return fallback

    except Exception as e:

        print(
            f"Planner unavailable; "
            f"using fallback: {e}"
        )

        return fallback_objective_profile(
            state,
            events
        )

def validate_strategy(
    state,
    events,
    strategy
):

    profile = strategy["objective_profile"]

    renewable_actual = (
        state["solar_actual_mw"]
        + state["wind_actual_mw"]
    )

    renewable_surplus = (
        renewable_actual
        - state["demand_mw"]
    )

    # ------------------------------------
    # GREEN_FIRST guardrail
    # ------------------------------------

    if profile == "GREEN_FIRST":

        # No renewable surplus and no
        # curtailment-related event:
        # GREEN_FIRST has no justification.
        if (
            renewable_surplus <= 0
            and
            "RENEWABLE_SURPLUS" not in events
        ):
            return False, (
                "GREEN_FIRST rejected: "
                "there is no renewable surplus "
                "or curtailment risk."
            )

    # ------------------------------------
    # PRICE_SHOCK guardrail
    # ------------------------------------

    if "PRICE_SHOCK" in events:

        if (
            profile == "GREEN_FIRST"
            and
            renewable_surplus <= 0
        ):
            return False, (
                "GREEN_FIRST rejected during price shock: "
                "market cost is the dominant issue and "
                "renewables are not being curtailed."
            )

    return True, "Strategy accepted."

class RenewableOrchestratorAgent:

    def __init__(self, assets):

        self.assets = assets

        self.logs = []

    def observe(
        self,
        current_period,
        forecast_df,
        actual_df,
        current_soc,
        battery_available=True
    ):

        return get_current_state(
            current_period,
            forecast_df,
            actual_df,
            current_soc,
            battery_available
        )

    def assess(self, state):

        return detect_event(state)

    def validate(self, action):

        return validate_action(
            action,
            self.assets
        )

    def explain(
        self,
        state,
        events,
        action,
        use_llm=True
    ):

        if use_llm:

            try:
                return explain_action_with_llm(
                    state,
                    events,
                    action
                )

            except Exception as e:

                print(
                    f"LLM unavailable. "
                    f"Using deterministic fallback: {e}"
                )

        return explain_action(
            state,
            events,
            action
        )

    def optimize(
        self,
        current_period,
        forecast_df,
        actual_df,
        current_soc,
        battery_available=True,
        objective_profile="COST_FIRST"
    ):

        return optimize_action(
            current_period=current_period,
            forecast_df=forecast_df,
            actual_df=actual_df,
            assets=self.assets,
            current_soc=current_soc,
            battery_available=battery_available,
            objective_profile=objective_profile
        )

    def execute(self, action, validation):

        if not validation["valid"]:
            raise ValueError(
                f"Dispatch rejected: {validation['errors']}"
            )

        new_soc = action["soc_mwh"]

        return new_soc

def run_agent_simulation(
    agent,
    forecast_df,
    actual_df
):

    current_soc = agent.assets["initial_soc_mwh"]

    logs = []
    previous_events = ["NORMAL"]
    current_strategy = {
        "decision": "REOPTIMIZE",
        "objective_profile": "COST_FIRST",
        "reason": "Normal operating strategy."
    }

    for current_period in range(len(forecast_df)):

        # -------------------------
        # OBSERVE
        # -------------------------

        state = agent.observe(
            current_period=current_period,
            forecast_df=forecast_df,
            actual_df=actual_df,
            current_soc=current_soc,
            battery_available=True
        )

        # -------------------------
        # ASSESS
        # -------------------------

        events = agent.assess(state)

        event_changed = events != previous_events

        if events == ["NORMAL"]:

            current_strategy = {
                "decision": "REOPTIMIZE",
                "objective_profile": "COST_FIRST",
                "reason": "Normal operating strategy."
            }

        elif event_changed:

            # Material operating condition changed:
            # let the planning agent choose a strategy.
            current_strategy = select_strategy(
                state,
                events
            )

        else:

            # Existing event persists:
            # retain the previously selected operating strategy.
            current_strategy = {
                "decision": "REOPTIMIZE",
                "objective_profile":
                    current_strategy["objective_profile"],
                "reason": (
                    "Existing operating event persists; "
                    "previously selected strategy retained."
                )
            }

        strategy = current_strategy

        objective_profile = (
            strategy["objective_profile"]
        )

        previous_events = events.copy()

        # -------------------------
        # OPTIMIZE
        # -------------------------

        action, plan = agent.optimize(
            current_period=current_period,
            forecast_df=forecast_df,
            actual_df=actual_df,
            current_soc=current_soc,
            battery_available=True,
            objective_profile=objective_profile
        )
        

        # -------------------------
        # VALIDATE
        # -------------------------

        validation = agent.validate(action)

        # -------------------------
        # EXECUTE
        # -------------------------

        current_soc = agent.execute(
            action,
            validation
        )

        # -------------------------
        # EXPLAIN
        # -------------------------

        if events != ["NORMAL"]:

            explanation = agent.explain(
                state,
                events,
                action,
                use_llm=True
            )

        else:

            explanation = (
                "Normal operation. "
                "No material deviation detected."
            )

        # -------------------------
        # LOG
        # -------------------------

        logs.append({
            "timestamp": state["timestamp"],

            "events": ", ".join(events),

            "demand_mw": state["demand_mw"],

            "solar_forecast_mw":
                state["solar_forecast_mw"],

            "solar_actual_mw":
                state["solar_actual_mw"],

            "price_forecast":
                state["price_forecast"],

            "price_actual":
                state["price_actual"],

            "grid_buy_mw":
                action["grid_buy_mw"],

            "grid_sell_mw":
                action["grid_sell_mw"],

            "battery_charge_mw":
                action["battery_charge_mw"],

            "battery_discharge_mw":
                action["battery_discharge_mw"],

            "soc_mwh":
                action["soc_mwh"],

            "curtailment_mw":
                action["curtailment_mw"],

            "net_cost_rs": action["net_cost_rs"],

            "validation_passed":
                validation["valid"],

            "explanation":
                explanation,

            "objective_profile":
                objective_profile,

            "strategy_reason":
                strategy["reason"]
        })

    return pd.DataFrame(logs)

if __name__ == "__main__":

    assets = {
        "battery_capacity_mwh": 50,
        "initial_soc_mwh": 25,
        "minimum_soc_mwh": 5,
        "maximum_charge_mw": 15,
        "maximum_discharge_mw": 15,
    }

    # What the system expected
    forecast_df = pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2026-10-01 15:00"),
            "demand_mw": 60.0,
            "solar_available_mw": 28.0,
            "wind_available_mw": 22.0,
            "buy_price_rs_kwh": 3.0,
            "sell_price_rs_kwh": 2.25
        }
    ])

    # What actually happened
    actual_df = pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2026-10-01 15:00"),
            "demand_mw": 60.0,
            "solar_available_mw": 28.0,
            "wind_available_mw": 22.0,
            "buy_price_rs_kwh": 20.0,
            "sell_price_rs_kwh": 15.0
        }
    ])

    agent = RenewableOrchestratorAgent(assets)

    state = agent.observe(
        current_period=0,
        forecast_df=forecast_df,
        actual_df=actual_df,
        current_soc=25,
        battery_available=True
    )

    events = agent.assess(state)

    # Temporary sample optimizer output
    # action = {
    #     "demand_mw": 60.0,
    #     "solar_available_mw": 28.0,
    #     "wind_available_mw": 22.0,
    #     "grid_buy_mw": 0.0,
    #     "grid_sell_mw": 0.0,
    #     "battery_charge_mw": 0.0,
    #     "battery_discharge_mw": 10.0,
    #     "curtailment_mw": 0.0,
    #     "soc_mwh": 22.37
    # }

    # validation = agent.validate(action)

    # explanation = agent.explain(
    #     state,
    #     events,
    #     action
    # )

    # print("\nSTATE")
    # print(state)

    # print("\nEVENTS")
    # print(events)

    # print("\nVALIDATION")
    # print(validation)

    # print("\nEXPLANATION")
    # print(explanation)

if __name__ == "__main__":

    INPUT_FILE = "energy_optimizer_input.xlsx"

    # -------------------------
    # LOAD DATA
    # -------------------------

    forecast_df = pd.read_excel(
        INPUT_FILE,
        sheet_name="timeseries"
    )

    assets_df = pd.read_excel(
        INPUT_FILE,
        sheet_name="assets"
    )

    forecast_df.columns = (
        forecast_df.columns
        .str.strip()
        .str.lower()
    )

    forecast_df["timestamp"] = pd.to_datetime(
        forecast_df["timestamp"]
    )

    numeric_columns = [
        "demand_mw",
        "solar_available_mw",
        "wind_available_mw",
        "buy_price_rs_kwh",
        "sell_price_rs_kwh"
    ]

    forecast_df[numeric_columns] = (
        forecast_df[numeric_columns].astype(float)
    )

    assets = dict(
        zip(
            assets_df["parameter"],
            assets_df["value"]
        )
    )

    # -------------------------
    # CREATE ACTUAL WORLD
    # -------------------------

    # actual_df = forecast_df.copy()

    # # Unexpected price shock:
    # # 15:00–16:00 actual price becomes ₹20
    # price_shock_mask = (
    #     (actual_df["timestamp"].dt.hour >= 15)
    #     &
    #     (actual_df["timestamp"].dt.hour < 16)
    # )

    # actual_df.loc[
    #     price_shock_mask,
    #     "buy_price_rs_kwh"
    # ] = 20.0

    # actual_df.loc[
    #     price_shock_mask,
    #     "sell_price_rs_kwh"
    # ] = 15.0

    actual_df = forecast_df.copy()

    # Ensure solar column accepts decimals
    actual_df["solar_available_mw"] = (
        actual_df["solar_available_mw"].astype(float)
    )

    solar_miss_mask = (
        (actual_df["timestamp"].dt.hour >= 11)
        &
        (actual_df["timestamp"].dt.hour < 13)
    )

    actual_df.loc[
        solar_miss_mask,
        "solar_available_mw"
    ] = (
        actual_df.loc[
            solar_miss_mask,
            "solar_available_mw"
        ] * 0.25
    )

    # -------------------------
    # FIND 15:00 PERIOD
    # -------------------------

    current_period = actual_df.index[
        (
            actual_df["timestamp"].dt.hour == 15
        )
        &
        (
            actual_df["timestamp"].dt.minute == 0
        )
    ][0]

    # For smoke test only.
    # Later this comes from previous simulator state.
    current_soc = 39.73438

    # -------------------------
    # AGENT
    # -------------------------

    agent = RenewableOrchestratorAgent(assets)

    state = agent.observe(
        current_period=current_period,
        forecast_df=forecast_df,
        actual_df=actual_df,
        current_soc=current_soc,
        battery_available=True
    )

    events = agent.assess(state)

    action, plan = agent.optimize(
        current_period=current_period,
        forecast_df=forecast_df,
        actual_df=actual_df,
        current_soc=current_soc,
        battery_available=True
    )

    validation = agent.validate(action)

    explanation = agent.explain(
        state,
        events,
        action
    )

    print("\nSTATE")
    print(state)

    print("\nEVENTS")
    print(events)

    print("\nOPTIMIZER ACTION")
    print({
        "grid_buy_mw": action["grid_buy_mw"],
        "grid_sell_mw": action["grid_sell_mw"],
        "battery_charge_mw": action["battery_charge_mw"],
        "battery_discharge_mw": action["battery_discharge_mw"],
        "soc_mwh": action["soc_mwh"]
    })

    print("\nVALIDATION")
    print(validation)

    print("\nEXPLANATION")
    print(explanation)

    agent = RenewableOrchestratorAgent(assets)

    agent_results = run_agent_simulation(
        agent=agent,
        forecast_df=forecast_df,
        actual_df=actual_df
    )

    agent_results.to_excel(
        "agent_price_shock_results.xlsx",
        index=False
    )

    event_rows = agent_results[
        agent_results["events"] != "NORMAL"
    ]

    print("\nABNORMAL EVENTS")
    print(
        event_rows[
            [
                "timestamp",
                "events",
                "objective_profile",
                "strategy_reason",
                "price_forecast",
                "price_actual",
                "grid_buy_mw",
                "grid_sell_mw",
                "battery_discharge_mw",
                "soc_mwh",
                "validation_passed"
            ]
        ]
    )

    print(
        "\nAll actions valid:",
        agent_results["validation_passed"].all()
    )

    print(
        "Final SOC:",
        agent_results["soc_mwh"].iloc[-1]
    )