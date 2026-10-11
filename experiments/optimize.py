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

timeseries = pd.read_excel(
    "energy_optimizer_input.xlsx",
    sheet_name="timeseries"
)

assets_df = pd.read_excel(
    "energy_optimizer_input.xlsx",
    sheet_name="assets"
)

assets = dict(
    zip(
        assets_df["parameter"],
        assets_df["value"]
    )
)

model = ConcreteModel()

n_periods = len(timeseries)

model.T = RangeSet(0, n_periods - 1)

DT_HOURS = 0.25

demand = timeseries["demand_mw"].to_numpy()

solar = timeseries["solar_available_mw"].to_numpy()

wind = timeseries["wind_available_mw"].to_numpy()

buy_price = timeseries["buy_price_rs_kwh"].to_numpy()

sell_price = timeseries["sell_price_rs_kwh"].to_numpy()

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
            assets["initial_soc_mwh"]
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

def curtailment_limit_rule(model, t):

    return (
        model.curtailment[t]
        <= solar[t] + wind[t]
    )


model.curtailment_limit = Constraint(
    model.T,
    rule=curtailment_limit_rule
)

model.terminal_soc = Constraint(
    expr=
    model.soc[n_periods - 1]
    ==
    assets["initial_soc_mwh"]
)

def objective_rule(model):

    total_cost = sum(

        (
            buy_price[t] * model.grid_buy[t]
            -
            sell_price[t] * model.grid_sell[t]
        )

        * DT_HOURS
        * 1000

        for t in model.T
    )

    return total_cost


model.objective = Objective(
    rule=objective_rule,
    sense=minimize
)

solver = SolverFactory("appsi_highs")

result = solver.solve(model)

print()
print("Solver status:")
print(result.solver.termination_condition)

results = timeseries.copy()

results["grid_buy_mw"] = [
    value(model.grid_buy[t])
    for t in model.T
]

results["grid_sell_mw"] = [
    value(model.grid_sell[t])
    for t in model.T
]

results["battery_charge_mw"] = [
    value(model.battery_charge[t])
    for t in model.T
]

results["battery_discharge_mw"] = [
    value(model.battery_discharge[t])
    for t in model.T
]

results["curtailment_mw"] = [
    value(model.curtailment[t])
    for t in model.T
]

results["soc_mwh"] = [
    value(model.soc[t])
    for t in model.T
]

results["renewable_used_mw"] = (
    results["solar_available_mw"]
    + results["wind_available_mw"]
    - results["curtailment_mw"]
)

results["net_cost_rs"] = (

    results["grid_buy_mw"]
    * results["buy_price_rs_kwh"]

    -

    results["grid_sell_mw"]
    * results["sell_price_rs_kwh"]

) * DT_HOURS * 1000

total_cost = results["net_cost_rs"].sum()

print()
print(f"Optimal total cost: ₹{total_cost:,.2f}")

results.to_excel(
    "optimization_results.xlsx",
    index=False
)

print()
print("Results saved to optimization_results.xlsx")