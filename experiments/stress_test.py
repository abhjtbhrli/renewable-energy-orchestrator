import pandas as pd

from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Var,
    NonNegativeReals,
    Constraint,
    Objective,
    minimize,
    SolverFactory,
    value
)

BASE_FILE = "energy_optimizer_input.xlsx"

base_timeseries = pd.read_excel(
    BASE_FILE,
    sheet_name="timeseries"
)

base_timeseries.columns = (
    base_timeseries.columns
    .str.strip()
    .str.lower()
)

assets_df = pd.read_excel(
    BASE_FILE,
    sheet_name="assets"
)

base_assets = dict(
    zip(
        assets_df["parameter"],
        assets_df["value"]
    )
)


def solve_dispatch(timeseries, assets, initial_soc_mwh=None,
    terminal_soc_mwh=None):

    
    timeseries = timeseries.copy()
    assets = assets.copy()

    n_periods = len(timeseries)

    DT_HOURS = 0.25
    if initial_soc_mwh is None:
        initial_soc_mwh = assets["initial_soc_mwh"]
    if terminal_soc_mwh is None:
        terminal_soc_mwh = assets["initial_soc_mwh"]

    demand = timeseries["demand_mw"].to_numpy()
    solar = timeseries["solar_available_mw"].to_numpy()
    wind = timeseries["wind_available_mw"].to_numpy()
    buy_price = timeseries["buy_price_rs_kwh"].to_numpy()
    sell_price = timeseries["sell_price_rs_kwh"].to_numpy()

    model = ConcreteModel()

    model.T = RangeSet(0, n_periods - 1)

    # Decision variables

    model.grid_buy = Var(
        model.T,
        domain=NonNegativeReals
    )

    model.grid_sell = Var(
        model.T,
        domain=NonNegativeReals
    )

    model.battery_charge = Var(
        model.T,
        domain=NonNegativeReals
    )

    model.battery_discharge = Var(
        model.T,
        domain=NonNegativeReals
    )

    model.curtailment = Var(
        model.T,
        domain=NonNegativeReals
    )

    model.soc = Var(
        model.T,
        domain=NonNegativeReals
    )

    # -----------------------------
    # ENERGY BALANCE
    # -----------------------------

    def energy_balance_rule(model, t):

        return (
            solar[t]
            + wind[t]
            + model.grid_buy[t]
            + model.battery_discharge[t]
            ==
            demand[t]
            + model.battery_charge[t]
            + model.grid_sell[t]
            + model.curtailment[t]
        )

    model.energy_balance = Constraint(
        model.T,
        rule=energy_balance_rule
    )

    # -----------------------------
    # SOC TRANSITION
    # -----------------------------

    def soc_rule(model, t):

        charge_added = (
            model.battery_charge[t]
            * DT_HOURS
            * assets["charge_efficiency"]
        )

        discharge_removed = (
            model.battery_discharge[t]
            * DT_HOURS
            / assets["discharge_efficiency"]
        )

        if t == 0:

            return (
                model.soc[t]
                ==
                initial_soc_mwh
                + charge_added
                - discharge_removed
            )

        return (
            model.soc[t]
            ==
            model.soc[t - 1]
            + charge_added
            - discharge_removed
        )

    model.soc_transition = Constraint(
        model.T,
        rule=soc_rule
    )

    # -----------------------------
    # SOC LIMITS
    # -----------------------------

    def soc_min_rule(model, t):

        return (
            model.soc[t]
            >= assets["minimum_soc_mwh"]
        )

    model.soc_min = Constraint(
        model.T,
        rule=soc_min_rule
    )

    def soc_max_rule(model, t):

        return (
            model.soc[t]
            <= assets["battery_capacity_mwh"]
        )

    model.soc_max = Constraint(
        model.T,
        rule=soc_max_rule
    )

    # -----------------------------
    # BATTERY LIMITS
    # -----------------------------

    def charge_limit_rule(model, t):

        return (
            model.battery_charge[t]
            <= assets["maximum_charge_mw"]
        )

    model.charge_limit = Constraint(
        model.T,
        rule=charge_limit_rule
    )

    def discharge_limit_rule(model, t):

        return (
            model.battery_discharge[t]
            <= assets["maximum_discharge_mw"]
        )

    model.discharge_limit = Constraint(
        model.T,
        rule=discharge_limit_rule
    )

    # -----------------------------
    # GRID LIMITS
    # -----------------------------

    def import_limit_rule(model, t):

        return (
            model.grid_buy[t]
            <= assets["grid_import_limit_mw"]
        )

    model.import_limit = Constraint(
        model.T,
        rule=import_limit_rule
    )

    def export_limit_rule(model, t):

        return (
            model.grid_sell[t]
            <= assets["grid_export_limit_mw"]
        )

    model.export_limit = Constraint(
        model.T,
        rule=export_limit_rule
    )

    # -----------------------------
    # CURTAILMENT
    # -----------------------------

    def curtailment_limit_rule(model, t):

        return (
            model.curtailment[t]
            <= solar[t] + wind[t]
        )

    model.curtailment_limit = Constraint(
        model.T,
        rule=curtailment_limit_rule
    )

    # -----------------------------
    # TERMINAL SOC
    # -----------------------------

    if terminal_soc_mwh is not None:
        model.terminal_soc = Constraint(
            expr=
            model.soc[n_periods - 1]
            ==
            terminal_soc_mwh
        )

    # -----------------------------
    # OBJECTIVE
    # -----------------------------

    def objective_rule(model):

        return sum(
            (
                buy_price[t] * model.grid_buy[t]
                -
                sell_price[t] * model.grid_sell[t]
            )
            * DT_HOURS
            * 1000

            for t in model.T
        )

    model.objective = Objective(
        rule=objective_rule,
        sense=minimize
    )

    # -----------------------------
    # SOLVE
    # -----------------------------

    solver = SolverFactory("appsi_highs")

    result = solver.solve(model)

    # -----------------------------
    # OUTPUT
    # -----------------------------

    output = timeseries.copy()

    output["grid_buy_mw"] = [
        value(model.grid_buy[t])
        for t in model.T
    ]

    output["grid_sell_mw"] = [
        value(model.grid_sell[t])
        for t in model.T
    ]

    output["battery_charge_mw"] = [
        value(model.battery_charge[t])
        for t in model.T
    ]

    output["battery_discharge_mw"] = [
        value(model.battery_discharge[t])
        for t in model.T
    ]

    output["curtailment_mw"] = [
        value(model.curtailment[t])
        for t in model.T
    ]

    output["soc_mwh"] = [
        value(model.soc[t])
        for t in model.T
    ]

    output["net_cost_rs"] = (
        (
            output["grid_buy_mw"]
            * output["buy_price_rs_kwh"]
        )
        -
        (
            output["grid_sell_mw"]
            * output["sell_price_rs_kwh"]
        )
    ) * DT_HOURS * 1000

    return output

def run_rolling_simulation(timeseries, assets):

    timeseries = timeseries.copy().reset_index(drop=True)

    current_soc = assets["initial_soc_mwh"]

    terminal_soc = assets["initial_soc_mwh"]

    executed_actions = []

    for current_period in range(len(timeseries)):

        # Everything from NOW until end of day
        remaining_horizon = (
            timeseries
            .iloc[current_period:]
            .copy()
            .reset_index(drop=True)
        )

        # Re-optimize using the CURRENT battery SOC
        plan = solve_dispatch(
            remaining_horizon,
            assets,
            initial_soc_mwh=current_soc,
            terminal_soc_mwh=terminal_soc
        )

        # Execute ONLY the first decision
        action = plan.iloc[0].copy()

        action["simulation_period"] = current_period

        executed_actions.append(action)

        # The SOC produced after this 15-minute action
        # becomes next period's starting state
        current_soc = action["soc_mwh"]

    rolling_results = pd.DataFrame(executed_actions)

    return rolling_results

def validate_solution(results, assets):

    DT_HOURS = 0.25

    results = results.copy()

    # Energy balance residual

    results["balance_residual"] = (

        results["solar_available_mw"]
        + results["wind_available_mw"]
        + results["grid_buy_mw"]
        + results["battery_discharge_mw"]

        -

        (
            results["demand_mw"]
            + results["battery_charge_mw"]
            + results["grid_sell_mw"]
            + results["curtailment_mw"]
        )
    )

    max_balance_error = (
        results["balance_residual"]
        .abs()
        .max()
    )

    min_soc = results["soc_mwh"].min()

    max_soc = results["soc_mwh"].max()

    final_soc = results["soc_mwh"].iloc[-1]

    total_cost = results["net_cost_rs"].sum()

    total_charge_mwh = (
        results["battery_charge_mw"].sum()
        * DT_HOURS
    )

    total_discharge_mwh = (
        results["battery_discharge_mw"].sum()
        * DT_HOURS
    )

    total_grid_import_mwh = (
        results["grid_buy_mw"].sum()
        * DT_HOURS
    )

    total_grid_export_mwh = (
        results["grid_sell_mw"].sum()
        * DT_HOURS
    )

    total_curtailment_mwh = (
        results["curtailment_mw"].sum()
        * DT_HOURS
    )

    return {
        "total_cost_rs": total_cost,
        "max_balance_error_mw": max_balance_error,
        "min_soc_mwh": min_soc,
        "max_soc_mwh": max_soc,
        "final_soc_mwh": final_soc,
        "battery_charge_mwh": total_charge_mwh,
        "battery_discharge_mwh": total_discharge_mwh,
        "grid_import_mwh": total_grid_import_mwh,
        "grid_export_mwh": total_grid_export_mwh,
        "curtailment_mwh": total_curtailment_mwh
    }

base_result = solve_dispatch(
    base_timeseries,
    base_assets,
    initial_soc_mwh=25,
    terminal_soc_mwh=25
)

base_summary = validate_solution(
    base_result,
    base_assets
)

print("\nBASE CASE")
print(base_summary)

base_result.to_excel(
    "scenario_0_base.xlsx",
    index=False
)

scenario_1 = base_timeseries.copy()
scenario_1["buy_price_rs_kwh"] = 5.0
scenario_1["sell_price_rs_kwh"] = 3.75
result_1 = solve_dispatch(
    scenario_1,
    base_assets
)

summary_1 = validate_solution(
    result_1,
    base_assets
)

print("\nSCENARIO 1 - FLAT PRICE")
print(summary_1)

result_1.to_excel(
    "scenario_1_flat_price.xlsx",
    index=False
)

scenario_2 = base_timeseries.copy()
scenario_2["timestamp"] = pd.to_datetime(
    scenario_2["timestamp"]
)

spike_mask = (
    (scenario_2["timestamp"].dt.hour >= 19)
    &
    (scenario_2["timestamp"].dt.hour < 20)
)

scenario_2.loc[
    spike_mask,
    "buy_price_rs_kwh"
] = 20.0

scenario_2.loc[
    spike_mask,
    "sell_price_rs_kwh"
] = 15.0

result_2 = solve_dispatch(
    scenario_2,
    base_assets
)

summary_2 = validate_solution(
    result_2,
    base_assets
)

print("\nSCENARIO 2 - PRICE SPIKE")
print(summary_2)

result_2.to_excel(
    "scenario_2_price_spike.xlsx",
    index=False
)

scenario_3 = base_timeseries.copy()
scenario_3["timestamp"] = pd.to_datetime(
    scenario_3["timestamp"]
)
solar_mask = (
    (scenario_3["timestamp"].dt.hour >= 11)
    &
    (scenario_3["timestamp"].dt.hour < 14)
)

scenario_3.loc[
    solar_mask,
    "solar_available_mw"
] = 130

result_3 = solve_dispatch(
    scenario_3,
    base_assets
)

summary_3 = validate_solution(
    result_3,
    base_assets
)

print("\nSCENARIO 3 - MASSIVE SOLAR SURPLUS")
print(summary_3)

result_3.to_excel(
    "scenario_3_solar_surplus.xlsx",
    index=False
)

scenario_4 = base_timeseries.copy()

assets_4 = base_assets.copy()
assets_4["maximum_charge_mw"] = 0
assets_4["maximum_discharge_mw"] = 0

result_4 = solve_dispatch(
    scenario_4,
    assets_4
)

summary_4 = validate_solution(
    result_4,
    assets_4
)

print("\nSCENARIO 4 - BATTERY DISABLED")
print(summary_4)

result_4.to_excel(
    "scenario_4_battery_disabled.xlsx",
    index=False
)

comparison = pd.DataFrame(
    [
        {
            "scenario": "Base",
            **base_summary
        },
        {
            "scenario": "Flat Price",
            **summary_1
        },
        {
            "scenario": "₹20 Price Spike",
            **summary_2
        },
        {
            "scenario": "Solar Surplus",
            **summary_3
        },
        {
            "scenario": "Battery Disabled",
            **summary_4
        }
    ]
)

print("\nSCENARIO COMPARISON")
print(comparison)

comparison.to_excel(
    "stress_test_summary.xlsx",
    index=False
)

rolling_result = run_rolling_simulation(
    base_timeseries,
    base_assets
)

rolling_result.to_excel(
    "rolling_horizon_results.xlsx",
    index=False
)

rolling_cost = rolling_result["net_cost_rs"].sum()

print(
    f"Rolling-horizon cost: ₹{rolling_cost:,.2f}"
)

full_day_result = solve_dispatch(
    base_timeseries,
    base_assets,
    initial_soc_mwh=25,
    terminal_soc_mwh=25
)

full_day_cost = full_day_result["net_cost_rs"].sum()

print(
    f"Full-day optimization cost: ₹{full_day_cost:,.2f}"
)

print(
    f"Rolling optimization cost: ₹{rolling_cost:,.2f}"
)

print(
    f"Difference: ₹{rolling_cost - full_day_cost:,.2f}"
)

forecast_df = base_timeseries.copy()
actual_df = base_timeseries.copy()

forecast_df["solar_available_mw"] = (
    forecast_df["solar_available_mw"].astype(float)
)

actual_df["solar_available_mw"] = (
    actual_df["solar_available_mw"].astype(float)
)

forecast_df["timestamp"] = pd.to_datetime(
    forecast_df["timestamp"]
)

actual_df["timestamp"] = pd.to_datetime(
    actual_df["timestamp"]
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

price_forecast_df = base_timeseries.copy()
price_actual_df = base_timeseries.copy()

price_forecast_df["timestamp"] = pd.to_datetime(
    price_forecast_df["timestamp"]
)

price_actual_df["timestamp"] = pd.to_datetime(
    price_actual_df["timestamp"]
)

price_actual_df["buy_price_rs_kwh"] = (
    price_actual_df["buy_price_rs_kwh"].astype(float)
)

price_actual_df["sell_price_rs_kwh"] = (
    price_actual_df["sell_price_rs_kwh"].astype(float)
)

price_shock_mask = (
    (price_actual_df["timestamp"].dt.hour >= 15)
    &
    (price_actual_df["timestamp"].dt.hour < 16)
)

price_actual_df.loc[
    price_shock_mask,
    "buy_price_rs_kwh"
] = 20.0

price_actual_df.loc[
    price_shock_mask,
    "sell_price_rs_kwh"
] = 15.0

def run_uncertain_simulation(
    forecast_df,
    actual_df,
    assets
):

    forecast_df = (
        forecast_df
        .copy()
        .reset_index(drop=True)
    )

    actual_df = (
        actual_df
        .copy()
        .reset_index(drop=True)
    )

    current_soc = assets["initial_soc_mwh"]

    terminal_soc = assets["initial_soc_mwh"]

    executed_actions = []

    for current_period in range(len(forecast_df)):

        # -------------------------
        # FUTURE = FORECAST
        # -------------------------

        remaining_horizon = (
            forecast_df
            .iloc[current_period:]
            .copy()
            .reset_index(drop=True)
        )

        # -------------------------
        # CURRENT PERIOD = OBSERVED ACTUAL
        # -------------------------

        actual_now = actual_df.iloc[current_period]

        current_columns = [
            "demand_mw",
            "solar_available_mw",
            "wind_available_mw",
            "buy_price_rs_kwh",
            "sell_price_rs_kwh"
        ]

        for col in current_columns:

            remaining_horizon.loc[0, col] = (
                actual_now[col]
            )

        # -------------------------
        # RE-OPTIMIZE
        # -------------------------

        plan = solve_dispatch(
            remaining_horizon,
            assets,
            initial_soc_mwh=current_soc,
            terminal_soc_mwh=terminal_soc
        )

        # Execute only first action
        action = plan.iloc[0].copy()

        # -------------------------
        # LOG FORECAST VS ACTUAL
        # -------------------------

        forecast_now = forecast_df.iloc[current_period]

        action["solar_forecast_mw"] = (
            forecast_now["solar_available_mw"]
        )

        action["solar_actual_mw"] = (
            actual_now["solar_available_mw"]
        )

        action["solar_forecast_error_mw"] = (
            actual_now["solar_available_mw"]
            -
            forecast_now["solar_available_mw"]
        )
        action["price_forecast_rs_kwh"] = (
            forecast_now["buy_price_rs_kwh"]
        )

        action["price_actual_rs_kwh"] = (
            actual_now["buy_price_rs_kwh"]
        )

        action["price_forecast_error_rs_kwh"] = (
            actual_now["buy_price_rs_kwh"]
            -
            forecast_now["buy_price_rs_kwh"]
        )

        

        action["simulation_period"] = current_period

        executed_actions.append(action)

        # New battery state
        current_soc = action["soc_mwh"]

    return pd.DataFrame(executed_actions)

uncertain_result = run_uncertain_simulation(
    forecast_df,
    actual_df,
    base_assets
)

uncertain_result.to_excel(
    "solar_forecast_miss_results.xlsx",
    index=False
)

uncertain_cost = (
    uncertain_result["net_cost_rs"].sum()
)

print(
    f"Perfect forecast cost: ₹{full_day_cost:,.2f}"
)

print(
    f"Solar forecast miss cost: ₹{uncertain_cost:,.2f}"
)

print(
    f"Forecast error penalty: ₹{uncertain_cost - full_day_cost:,.2f}"
)

price_shock_result = run_uncertain_simulation(
    price_forecast_df,
    price_actual_df,
    base_assets
)

price_shock_result.to_excel(
    "dynamic_price_shock_results.xlsx",
    index=False
)

price_shock_cost = price_shock_result["net_cost_rs"].sum()

print(
    f"Perfect forecast cost: ₹{full_day_cost:,.2f}"
)

print(
    f"Dynamic price shock cost: ₹{price_shock_cost:,.2f}"
)

print(
    f"Price shock penalty: ₹{price_shock_cost - full_day_cost:,.2f}"
)

def run_battery_outage_simulation(
    forecast_df,
    actual_df,
    assets,
    outage_start_hour=18,
    outage_start_minute=30
):

    forecast_df = (
        forecast_df
        .copy()
        .reset_index(drop=True)
    )

    actual_df = (
        actual_df
        .copy()
        .reset_index(drop=True)
    )

    current_soc = assets["initial_soc_mwh"]

    terminal_soc = assets["initial_soc_mwh"]

    executed_actions = []

    for current_period in range(len(forecast_df)):

        remaining_horizon = (
            forecast_df
            .iloc[current_period:]
            .copy()
            .reset_index(drop=True)
        )

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

        current_time = pd.to_datetime(
            actual_now["timestamp"]
        )

        # Copy normal asset configuration
        current_assets = assets.copy()

        outage_started = (
            (current_time.hour > outage_start_hour)
            or
            (
                current_time.hour == outage_start_hour
                and current_time.minute >= outage_start_minute
            )
        )

        if outage_started:
            current_assets["maximum_charge_mw"] = 0
            current_assets["maximum_discharge_mw"] = 0

        if outage_started:
            required_terminal_soc = current_soc
        else:
            required_terminal_soc = terminal_soc

        plan = solve_dispatch(
            remaining_horizon,
            current_assets,
            initial_soc_mwh=current_soc,
            terminal_soc_mwh=required_terminal_soc
        )
        action = plan.iloc[0].copy()

        action["battery_available"] = (
            0 if outage_started else 1
        )

        action["simulation_period"] = current_period

        executed_actions.append(action)

        current_soc = action["soc_mwh"]

    return pd.DataFrame(executed_actions)

battery_outage_result = run_battery_outage_simulation(
    base_timeseries,
    base_timeseries,
    base_assets
)

battery_outage_result.to_excel(
    "dynamic_battery_outage_results.xlsx",
    index=False
)

battery_outage_cost = (
    battery_outage_result["net_cost_rs"].sum()
)

print(
    f"Normal cost: ₹{full_day_cost:,.2f}"
)

print(
    f"Battery outage cost: ₹{battery_outage_cost:,.2f}"
)

print(
    f"Battery outage penalty: ₹{battery_outage_cost - full_day_cost:,.2f}"
)

static_under_shock = full_day_result.copy()

static_under_shock["actual_buy_price_rs_kwh"] = (
    price_actual_df["buy_price_rs_kwh"].to_numpy()
)

static_under_shock["actual_sell_price_rs_kwh"] = (
    price_actual_df["sell_price_rs_kwh"].to_numpy()
)

static_under_shock["actual_net_cost_rs"] = (
    (
        static_under_shock["grid_buy_mw"]
        * static_under_shock["actual_buy_price_rs_kwh"]
    )
    -
    (
        static_under_shock["grid_sell_mw"]
        * static_under_shock["actual_sell_price_rs_kwh"]
    )
) * 0.25 * 1000

static_shock_cost = (
    static_under_shock["actual_net_cost_rs"].sum()
)

adaptive_saving = (
    static_shock_cost
    - price_shock_cost
)

print(
    f"Static plan under shock: ₹{static_shock_cost:,.2f}"
)

print(
    f"Adaptive rolling response: ₹{price_shock_cost:,.2f}"
)

print(
    f"Saving from re-optimization: ₹{adaptive_saving:,.2f}"
)