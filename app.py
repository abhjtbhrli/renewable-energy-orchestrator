import pandas as pd
import streamlit as st

from agent import (
    RenewableOrchestratorAgent,
    run_agent_simulation
)

from optimizer_engine import solve_dispatch


# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Renewable Energy Orchestrator",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Renewable Energy Orchestrator")

st.caption(
    "Agentic renewable dispatch optimization under uncertainty"
)


# --------------------------------------------------
# LOAD INPUTS
# --------------------------------------------------

INPUT_FILE = "energy_optimizer_input.xlsx"

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


# --------------------------------------------------
# BUILD ACTUAL WORLD
# --------------------------------------------------

scenario = st.sidebar.selectbox(
    "Scenario",
    [
        "Price Shock",
        "Solar Forecast Miss"
    ]
)


actual_df = forecast_df.copy()

if scenario == "Price Shock":

    price_shock_mask = (
        (actual_df["timestamp"].dt.hour >= 15)
        &
        (actual_df["timestamp"].dt.hour < 16)
    )

    actual_df.loc[
        price_shock_mask,
        "buy_price_rs_kwh"
    ] = 20.0

    actual_df.loc[
        price_shock_mask,
        "sell_price_rs_kwh"
    ] = 15.0

    scenario_description = (
        "Unexpected market price spike from "
        "₹3/kWh to ₹20/kWh between 15:00–16:00."
    )


elif scenario == "Solar Forecast Miss":

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

    scenario_description = (
        "Actual solar generation falls to 25% of forecast "
        "between 11:00–13:00."
    )

# --------------------------------------------------
# RUN BUTTON
# --------------------------------------------------

st.sidebar.header("Scenario")

st.sidebar.write(scenario_description)

run_button = st.sidebar.button(
    "▶ Run Orchestrator",
    type="primary"
)


if run_button:

    with st.spinner(
        "Running rolling-horizon optimization and agent..."
    ):

        # ------------------------------------------
        # NORMAL DAY-AHEAD PLAN
        # ------------------------------------------

        static_plan = solve_dispatch(
            forecast_df,
            assets,
            initial_soc_mwh=assets["initial_soc_mwh"],
            terminal_soc_mwh=assets["initial_soc_mwh"]
        )
        normal_cost = static_plan["net_cost_rs"].sum()


        # ------------------------------------------
        # COST OF BLINDLY FOLLOWING OLD PLAN
        # ------------------------------------------

        static_under_shock = static_plan.copy()

        static_under_shock["actual_buy_price"] = (
            actual_df["buy_price_rs_kwh"].to_numpy()
        )

        static_under_shock["actual_sell_price"] = (
            actual_df["sell_price_rs_kwh"].to_numpy()
        )

        static_under_shock["actual_net_cost_rs"] = (

            static_under_shock["grid_buy_mw"]
            * static_under_shock["actual_buy_price"]

            -

            static_under_shock["grid_sell_mw"]
            * static_under_shock["actual_sell_price"]

        ) * 0.25 * 1000

        static_shock_cost = (
            static_under_shock["actual_net_cost_rs"].sum()
        )


        # ------------------------------------------
        # AGENTIC ROLLING RESPONSE
        # ------------------------------------------

        agent = RenewableOrchestratorAgent(
            assets
        )

        agent_results = run_agent_simulation(
            agent=agent,
            forecast_df=forecast_df,
            actual_df=actual_df
        )

        adaptive_cost = (
            agent_results["net_cost_rs"].sum()
        )

        adaptive_saving = (
            static_shock_cost
            - adaptive_cost
        )


        # ------------------------------------------
        # STORE RESULT
        # ------------------------------------------

        st.session_state["agent_results"] = agent_results
        st.session_state["static_shock_cost"] = static_shock_cost
        st.session_state["adaptive_cost"] = adaptive_cost
        st.session_state["adaptive_saving"] = adaptive_saving
        st.session_state["normal_cost"] = normal_cost
        st.session_state["scenario_run"] = scenario 


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

if (
    "agent_results" in st.session_state
    and
    st.session_state.get("scenario_run") == scenario
):

    results = st.session_state["agent_results"]
    normal_cost = st.session_state["normal_cost"]

    static_shock_cost = (
        st.session_state["static_shock_cost"]
    )

    adaptive_cost = (
        st.session_state["adaptive_cost"]
    )

    adaptive_saving = (
        st.session_state["adaptive_saving"]
    )
    solar_event_impact = adaptive_cost - normal_cost


    # --------------------------------------------------
    # MAIN KPI
    # --------------------------------------------------

    st.subheader("Adaptive Response")
    if scenario == "Price Shock":

        k1, k2, k3 = st.columns(3)

        k1.metric(
            "Static Plan Under Shock",
            f"₹{static_shock_cost / 100000:.2f} lakh"
        )

        k2.metric(
            "Adaptive Rolling Cost",
            f"₹{adaptive_cost / 100000:.2f} lakh"
        )

        k3.metric(
            "Saving from Re-optimization",
            f"₹{adaptive_saving / 100000:.2f} lakh"
        )
    elif scenario == "Solar Forecast Miss":

        k1, k2, k3 = st.columns(3)

        k1.metric(
            "Normal Operating Cost",
            f"₹{normal_cost / 100000:.2f} lakh"
        )

        k2.metric(
            "Adaptive Cost with Solar Miss",
            f"₹{adaptive_cost / 100000:.2f} lakh"
        )

        k3.metric(
            "Solar Shortfall Impact",
            f"₹{solar_event_impact / 100000:.2f} lakh"
        )

    # k1, k2, k3 = st.columns(3)

    # k1.metric(
    #     "Static Plan Under Shock",
    #     f"₹{static_shock_cost / 100000:.2f} lakh"
    # )

    # k2.metric(
    #     "Adaptive Rolling Cost",
    #     f"₹{adaptive_cost / 100000:.2f} lakh"
    # )

    # k3.metric(
    #     "Saving from Re-optimization",
    #     f"₹{adaptive_saving / 100000:.2f} lakh"
    # )


    st.success(
        f"Adaptive re-optimization saved "
        f"₹{adaptive_saving:,.0f} versus blindly "
        f"following the pre-shock dispatch plan "
        f"in this simulated scenario."
    )


    # --------------------------------------------------
    # EVENT SNAPSHOT
    # --------------------------------------------------

    event_rows = results[
        results["events"] != "NORMAL"
    ]

    first_event = event_rows.iloc[0]

    event_name = first_event["events"].replace("_", " ").title()

    event_time = pd.to_datetime(
        first_event["timestamp"]
    ).strftime("%H:%M")

    st.subheader(
        f"⚠ {event_name} Detected — {event_time}"
    )

    if scenario == "Price Shock":

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Demand",
            f"{first_event['demand_mw']:.1f} MW"
        )

        c2.metric(
            "Battery SOC",
            f"{first_event['soc_mwh']:.1f} MWh"
        )

        c3.metric(
            "Forecast Price",
            f"₹{first_event['price_forecast']:.0f}/kWh"
        )

        c4.metric(
            "Actual Price",
            f"₹{first_event['price_actual']:.0f}/kWh",
            delta=(
                f"+₹{first_event['price_actual'] - first_event['price_forecast']:.0f}/kWh"
            ),
            delta_color="inverse"
        )

    elif scenario == "Solar Forecast Miss":

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Demand",
            f"{first_event['demand_mw']:.1f} MW"
        )

        c2.metric(
            "Battery SOC",
            f"{first_event['soc_mwh']:.1f} MWh"
        )

        c3.metric(
            "Solar Forecast",
            f"{first_event['solar_forecast_mw']:.1f} MW"
        )

        solar_delta = (
            first_event["solar_actual_mw"]
            -
            first_event["solar_forecast_mw"]
        )

        c4.metric(
            "Actual Solar",
            f"{first_event['solar_actual_mw']:.1f} MW",
            delta=f"{solar_delta:.1f} MW",
            delta_color="normal"
        )

    # c1, c2, c3, c4 = st.columns(4)

    # c1.metric(
    #     "Demand",
    #     f"{first_shock['demand_mw']:.1f} MW"
    # )

    # c2.metric(
    #     "Battery SOC",
    #     f"{first_shock['soc_mwh']:.1f} MWh"
    # )

    # c3.metric(
    #     "Forecast Price",
    #     f"₹{first_shock['price_forecast']:.0f}/kWh"
    # )

    # c4.metric(
    #     "Actual Price",
    #     f"₹{first_shock['price_actual']:.0f}/kWh",
    #     delta=f"+₹{first_shock['price_actual'] - first_shock['price_forecast']:.0f}/kWh",
    #     delta_color="inverse"
    # )


    # --------------------------------------------------
    # OPTIMIZED ACTION
    # --------------------------------------------------

    st.subheader("Optimized Dispatch")

    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "Grid Buy",
        f"{first_event['grid_buy_mw']:.1f} MW"
    )

    a2.metric(
        "Grid Sell",
        f"{first_event['grid_sell_mw']:.1f} MW"
    )

    a3.metric(
        "Battery Discharge",
        f"{first_event['battery_discharge_mw']:.1f} MW"
    )

    a4.metric(
        "Battery Charge",
        f"{first_event['battery_charge_mw']:.1f} MW"
    )


    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    if event_rows["validation_passed"].all():

        st.success(
            "✓ Dispatch validated — all operating constraints satisfied."
        )

    else:

        st.error(
            "Dispatch validation failed."
        )


    # --------------------------------------------------
    # QWEN EXPLANATION
    # --------------------------------------------------

    st.subheader("Agent Explanation")
    print(
        "DEBUG EXPLANATION:",
        repr(first_event["explanation"]),
        type(first_event["explanation"])
    )
    st.info(
        first_event["explanation"]
    )


    # --------------------------------------------------
    # PRICE CHART
    # --------------------------------------------------

    st.subheader("Market Price")

    price_chart = (
        results[
            [
                "timestamp",
                "price_forecast",
                "price_actual"
            ]
        ]
        .set_index("timestamp")
    )

    st.line_chart(
        price_chart,
        height=280
    )


    # --------------------------------------------------
    # ENERGY SUPPLY / DEMAND
    # --------------------------------------------------

    st.subheader("Demand and Renewable Generation")

    energy_chart = results.copy()

    energy_chart["renewable_generation_mw"] = (
        energy_chart["solar_actual_mw"]
        +
        forecast_df["wind_available_mw"].to_numpy()
    )

    energy_chart = (
        energy_chart[
            [
                "timestamp",
                "demand_mw",
                "renewable_generation_mw"
            ]
        ]
        .set_index("timestamp")
    )

    st.line_chart(
        energy_chart,
        height=280
    )


    # --------------------------------------------------
    # BATTERY ACTION
    # --------------------------------------------------

    st.subheader("Battery Dispatch")

    battery_chart = (
        results[
            [
                "timestamp",
                "battery_charge_mw",
                "battery_discharge_mw"
            ]
        ]
        .set_index("timestamp")
    )

    st.line_chart(
        battery_chart,
        height=280
    )


    # --------------------------------------------------
    # SOC
    # --------------------------------------------------

    st.subheader("Battery State of Charge")

    soc_chart = (
        results[
            [
                "timestamp",
                "soc_mwh"
            ]
        ]
        .set_index("timestamp")
    )

    st.line_chart(
        soc_chart,
        height=280
    )


    # --------------------------------------------------
    # AGENT AUDIT LOG
    # --------------------------------------------------

    with st.expander("Agent Decision Log"):

        st.dataframe(
            results[
                [
                    "timestamp",
                    "events",
                    "grid_buy_mw",
                    "grid_sell_mw",
                    "battery_charge_mw",
                    "battery_discharge_mw",
                    "soc_mwh",
                    "validation_passed"
                ]
            ],
            use_container_width=True
        )


else:

    st.info(
        "Select the scenario and click "
        "'Run Orchestrator' to start the simulation."
    )