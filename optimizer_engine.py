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

OBJECTIVE_PROFILES = {

    "COST_FIRST": {
        "curtailment_penalty_rs_kwh": 0.25,
        "battery_degradation_rs_kwh": 0.10,
        "description": (
            "Prioritize operating cost while maintaining "
            "reliability constraints."
        )
    },

    "GREEN_FIRST": {
        "curtailment_penalty_rs_kwh": 5.00,
        "battery_degradation_rs_kwh": 0.10,
        "description": (
            "Prioritize renewable utilization by strongly "
            "penalizing renewable curtailment."
        )
    },

    "BATTERY_PRESERVE": {
        "curtailment_penalty_rs_kwh": 0.50,
        "battery_degradation_rs_kwh": 3.00,
        "description": (
            "Preserve battery life/flexibility by strongly "
            "penalizing battery throughput."
        )
    }
}

def solve_dispatch(
    timeseries,
    assets,
    initial_soc_mwh=None,
    terminal_soc_mwh=None,
    objective_profile="COST_FIRST"
):

    
    if objective_profile not in OBJECTIVE_PROFILES:
        raise ValueError(
            f"Unknown objective profile: {objective_profile}"
        )

    profile = OBJECTIVE_PROFILES[objective_profile]

    curtailment_penalty = (
            profile["curtailment_penalty_rs_kwh"]
        )

    battery_degradation_cost = (
            profile["battery_degradation_rs_kwh"]
        )
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

        grid_cost = sum(
            (
                buy_price[t] * model.grid_buy[t]
                -
                sell_price[t] * model.grid_sell[t]
            )
            * DT_HOURS
            * 1000

            for t in model.T
        )

        curtailment_cost = sum(
            model.curtailment[t]
            * curtailment_penalty
            * DT_HOURS
            * 1000

            for t in model.T
        )

        battery_degradation_cost_total = sum(
            (
                model.battery_charge[t]
                +
                model.battery_discharge[t]
            )
            * battery_degradation_cost
            * DT_HOURS
            * 1000

            for t in model.T
        )

        return (
            grid_cost
            +
            curtailment_cost
            +
            battery_degradation_cost_total
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
    output["objective_profile"] = objective_profile
    output["curtailment_penalty_rs_kwh"] = (
        curtailment_penalty
    )

    output["battery_degradation_rs_kwh"] = (
        battery_degradation_cost
    )

    return output