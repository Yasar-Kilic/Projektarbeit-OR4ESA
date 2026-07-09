"""
OR4ESA PV-battery investment model with grid connection limit scenarios.

Main model:
- PV investment and operation
- Battery energy capacity and battery power capacity optimized separately
- Optional grid charging scenario
- Grid connection limit sensitivity: no, weak, light, medium
- Curtailment variable to measure unused PV generation
- Result tables and figures

Optional data analysis:
- Redispatch data
- ID-AEP / intraday proxy
- Activated aFRR
- Curtailment/ABSM data from Netztransparenz
- Annual solar market values

Folder structure on your PC:
OR4ESA_Model/
│
├─ or4esa_model_gridlimit_extended.py
│
├─ data/
│  ├─ Day-ahead_prices_final.csv
│  └─ CapacityValues.csv
│
└─ data_optional/
   ├─ Redispatch_Daten.csv
   ├─ Aktivierte aFRR qual..csv
   ├─ Index Ausgleichsenergiepreis [2026-07-08 17-43-20].csv
   └─ Ausgewiesene Abregelungsstrommenge [2026-07-08 17-58-44].csv
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import linopy


# =============================================================================
# 1. GENERAL SETTINGS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OPTIONAL_DATA_DIR = BASE_DIR / "data_optional"
RESULTS_DIR = BASE_DIR / "results_gridlimit"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Main analysis period and locations
YEARS = list(range(2015, 2025))  # 2015–2024
LOCATIONS = ["Stuttgart", "Duesseldorf", "Schleswig-Holstein"]

# Solver. If None, Linopy chooses a solver. "highs" is recommended if installed.
SOLVER_NAME = "highs"

# Set this to True only if you want a fast test run first.
# It will use only one year, one location and fewer scenarios.
QUICK_TEST = False

# Optional extra analyses from Netztransparenz files.
RUN_OPTIONAL_DATA_ANALYSIS = True

# Optional battery cost sensitivity.
# This adds several model runs and can take longer.
RUN_BATTERY_COST_SENSITIVITY = True
BATTERY_COST_FACTORS = [1.0, 0.75, 0.50, 0.25, 0.10]


# =============================================================================
# 2. TECHNICAL AND ECONOMIC PARAMETERS
# =============================================================================

# Budget
BUDGET_EUR = 10_000_000.0

# PV investment cost [EUR/kW]
PV_INV_COST_EUR_PER_KW = 800.0

# Battery investment costs.
# IMPORTANT: Adjust these if your group has a better source.
# We split battery costs into energy capacity [EUR/kWh] and power capacity [EUR/kW].
BAT_ENERGY_COST_EUR_PER_KWH = 300.0
BAT_POWER_COST_EUR_PER_KW = 100.0

# Annual fixed O&M costs. Set to 0 if you do not want to include O&M.
PV_FIXED_OM_EUR_PER_KW_YEAR = 0.0
BAT_FIXED_OM_EUR_PER_KWH_YEAR = 0.0
BAT_FIXED_OM_EUR_PER_KW_YEAR = 0.0

# Optional cycling/degradation cost [EUR/kWh discharged].
# Set to 0 for the base case. A small value makes battery use more realistic.
BAT_DEGRADATION_COST_EUR_PER_KWH = 0.0

# Efficiencies
ETA_CHARGE = 0.95
ETA_DISCHARGE = 0.95

# Lifetimes and discount rate for annualisation
LIFETIME_PV_YEARS = 25
LIFETIME_BAT_YEARS = 12
DISCOUNT_RATE = 0.05

# Grid limit scenarios.
# The values are export limits in kW. None means no export limit.
# We include up to medium limitation, as discussed.
GRID_LIMIT_SCENARIOS = {
    "no_limit": None,
    "weak_limit_12_5MW": 12_500.0,
    "light_limit_10MW": 10_000.0,
    "medium_limit_7_5MW": 7_500.0,
}

# Main model cases.
# PV_only: no battery
# PV_battery_no_grid_buy: battery can only store PV electricity
# PV_battery_grid_buy: battery can also buy electricity from the day-ahead market
MAIN_SCENARIOS = {
    "PV_only": {"battery_allowed": False, "grid_buy_allowed": False},
    "PV_battery_no_grid_buy": {"battery_allowed": True, "grid_buy_allowed": False},
    "PV_battery_grid_buy": {"battery_allowed": True, "grid_buy_allowed": True},
}

# Manual official annual solar market values copied from Netztransparenz [ct/kWh].
SOLAR_MARKET_VALUE_CT_PER_KWH = {
    2015: 3.171,
    2016: 2.952,
    2017: 3.474,
    2018: 4.515,
    2019: 3.776,
    2020: 2.879,
    2021: 9.562,
    2022: 20.806,
    2023: 8.003,
    2024: 5.858,
}


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def annuity_factor(r: float, lifetime: int) -> float:
    """Return annuity factor for annualising investment costs."""
    if r == 0:
        return 1.0 / lifetime
    return (r * (1 + r) ** lifetime) / ((1 + r) ** lifetime - 1)


def present_value_of_annuity(annual_cashflow: float, r: float, years: int) -> float:
    """Present value of a constant yearly cashflow."""
    if years <= 0:
        return 0.0
    if r == 0:
        return annual_cashflow * years
    return annual_cashflow * ((1 + r) ** years - 1) / (r * (1 + r) ** years)


def discounted_payback_years(capex: float, annual_cashflow: float, r: float, max_years: int = 40) -> float:
    """Approximate discounted payback period. Returns NaN if payback is not reached."""
    if capex <= 0 or annual_cashflow <= 0:
        return np.nan
    cumulative = 0.0
    for year in range(1, max_years + 1):
        previous = cumulative
        cumulative += annual_cashflow / ((1 + r) ** year)
        if cumulative >= capex:
            # Linear interpolation within the year for a smoother estimate
            discounted_cf = cumulative - previous
            if discounted_cf <= 0:
                return float(year)
            fraction = (capex - previous) / discounted_cf
            return (year - 1) + fraction
    return np.nan


def find_existing_file(candidates: list[Path]) -> Path:
    """Return first existing file from candidate list."""
    for path in candidates:
        if path.exists():
            return path
    candidate_text = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"None of these files exists:\n{candidate_text}")


def read_main_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read day-ahead prices and PV capacity factors."""
    price_path = find_existing_file([
        DATA_DIR / "Day-ahead_prices_final.csv",
        BASE_DIR / "Day-ahead_prices_final.csv",
    ])
    pv_path = find_existing_file([
        DATA_DIR / "CapacityValues.csv",
        DATA_DIR / "CapacityValues (1).csv",
        BASE_DIR / "CapacityValues.csv",
        BASE_DIR / "CapacityValues (1).csv",
    ])

    prices = pd.read_csv(price_path, encoding="latin1")
    pv = pd.read_csv(pv_path, encoding="latin1")

    pv["time"] = pd.to_datetime(pv["time"], utc=True)
    pv["year"] = pv["time"].dt.year

    print("Loaded main input data:")
    print(f"  Prices: {price_path.name}, shape={prices.shape}")
    print(f"  PV:     {pv_path.name}, shape={pv.shape}")

    return prices, pv


def prepare_multiyear_data(
    prices_raw: pd.DataFrame,
    pv_raw: pd.DataFrame,
    years: list[int],
    location: str,
) -> pd.DataFrame:
    """Prepare hourly price and PV data for several years.

    Leap years are kept. Non-existing price entries are dropped. PV and price data are
    matched by number of rows per year.
    """
    all_data = []

    for year in years:
        if str(year) not in prices_raw.columns:
            raise ValueError(f"Year {year} is missing in price file columns.")
        if location not in pv_raw.columns:
            raise ValueError(f"Location '{location}' is missing in PV file columns.")

        pv_scenario = pv_raw[pv_raw["year"] == year].copy().reset_index(drop=True)
        price_scenario = pd.to_numeric(prices_raw[str(year)], errors="coerce").dropna().reset_index(drop=True)

        n = min(len(pv_scenario), len(price_scenario))
        if n == 0:
            raise ValueError(f"No data for year {year} and location {location}.")

        if len(pv_scenario) != len(price_scenario):
            warnings.warn(
                f"Length mismatch in {year}: PV={len(pv_scenario)}, price={len(price_scenario)}. "
                f"Using first {n} rows."
            )

        year_data = pd.DataFrame({
            "time": pv_scenario.loc[: n - 1, "time"].values,
            "price_eur_per_kwh": price_scenario.iloc[:n].values / 1000.0,  # EUR/MWh -> EUR/kWh
            "pv_cf": pd.to_numeric(pv_scenario.loc[: n - 1, location], errors="coerce").values,
            "year": year,
        })
        all_data.append(year_data)

    data = pd.concat(all_data, ignore_index=True)
    data = data.dropna(subset=["price_eur_per_kwh", "pv_cf"])
    data = data.set_index("time")
    data.index.name = "time"

    return data


def calculate_market_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple annual market indicators from price data."""
    rows = []
    df = data.copy()
    df["date"] = df.index.date
    df["price_eur_per_mwh"] = df["price_eur_per_kwh"] * 1000.0

    for year, ydf in df.groupby("year"):
        daily = ydf.groupby("date")["price_eur_per_mwh"].agg(lambda x: x.max() - x.min())
        rows.append({
            "year": int(year),
            "avg_price_EUR_per_MWh": ydf["price_eur_per_mwh"].mean(),
            "std_price_EUR_per_MWh": ydf["price_eur_per_mwh"].std(),
            "avg_daily_spread_EUR_per_MWh": daily.mean(),
            "max_daily_spread_EUR_per_MWh": daily.max(),
            "avg_pv_cf": ydf["pv_cf"].mean(),
            "solar_market_value_EUR_per_MWh": SOLAR_MARKET_VALUE_CT_PER_KWH.get(int(year), np.nan) * 10.0,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 4. OPTIMISATION MODEL
# =============================================================================

def run_model(
    prices_raw: pd.DataFrame,
    pv_raw: pd.DataFrame,
    years: list[int],
    location: str,
    scenario_name: str,
    grid_limit_name: str,
    grid_limit_kw: float | None,
    battery_cost_factor: float = 1.0,
) -> dict:
    """Run one PV-battery optimisation scenario."""
    scenario = MAIN_SCENARIOS[scenario_name]
    battery_allowed = scenario["battery_allowed"]
    grid_buy_allowed = scenario["grid_buy_allowed"]

    data = prepare_multiyear_data(prices_raw, pv_raw, years, location)
    time = data.index
    time.name = "time"
    years_count = len(years)
    dt = 1.0  # hourly time step

    price = xr.DataArray(
        data["price_eur_per_kwh"].values,
        coords={"time": time},
        dims=["time"],
    )
    pv_cf = xr.DataArray(
        data["pv_cf"].values,
        coords={"time": time},
        dims=["time"],
    )

    # Annualised investment costs
    pv_annuity = annuity_factor(DISCOUNT_RATE, LIFETIME_PV_YEARS)
    bat_annuity = annuity_factor(DISCOUNT_RATE, LIFETIME_BAT_YEARS)

    bat_energy_cost = BAT_ENERGY_COST_EUR_PER_KWH * battery_cost_factor
    bat_power_cost = BAT_POWER_COST_EUR_PER_KW * battery_cost_factor

    pv_cost_annual = PV_INV_COST_EUR_PER_KW * pv_annuity + PV_FIXED_OM_EUR_PER_KW_YEAR
    bat_energy_cost_annual = bat_energy_cost * bat_annuity + BAT_FIXED_OM_EUR_PER_KWH_YEAR
    bat_power_cost_annual = bat_power_cost * bat_annuity + BAT_FIXED_OM_EUR_PER_KW_YEAR

    # Create optimisation model
    m = linopy.Model()

    # Capacity variables
    C_pv = m.add_variables(lower=0, name="C_pv")  # kW
    E_bat = m.add_variables(lower=0, name="E_bat")  # kWh
    P_bat = m.add_variables(lower=0, name="P_bat")  # kW

    # Operational variables
    pv_to_grid = m.add_variables(lower=0, dims=["time"], coords={"time": time}, name="pv_to_grid")
    charge_pv = m.add_variables(lower=0, dims=["time"], coords={"time": time}, name="charge_pv")
    charge_grid = m.add_variables(lower=0, dims=["time"], coords={"time": time}, name="charge_grid")
    discharge = m.add_variables(lower=0, dims=["time"], coords={"time": time}, name="discharge")
    soc = m.add_variables(lower=0, dims=["time"], coords={"time": time}, name="soc")
    q_sell = m.add_variables(lower=0, dims=["time"], coords={"time": time}, name="q_sell")
    curtailment = m.add_variables(lower=0, dims=["time"], coords={"time": time}, name="curtailment")

    # PV split: available PV is either sold directly, charged into battery, or curtailed
    m.add_constraints(
        pv_to_grid + charge_pv + curtailment == C_pv * pv_cf,
        name="pv_split",
    )

    # Electricity sold to the market consists of direct PV sales plus battery discharge
    m.add_constraints(
        q_sell == pv_to_grid + discharge,
        name="sell_definition",
    )

    # Grid export limit scenario
    if grid_limit_kw is not None:
        m.add_constraints(q_sell <= grid_limit_kw, name="grid_export_limit")

    # Battery capacity and power limits
    m.add_constraints(charge_pv + charge_grid <= P_bat, name="charge_power_limit")
    m.add_constraints(discharge <= P_bat, name="discharge_power_limit")
    m.add_constraints(soc <= E_bat, name="soc_energy_limit")

    # Battery state of charge dynamics. SOC is interpreted as end-of-hour SOC.
    # This vectorised formulation is much faster than adding one constraint in a Python loop.
    # The first hour uses an implicit initial SOC of zero through fill_value=0.
    m.add_constraints(
        soc
        == soc.shift(time=1, fill_value=0)
        + ETA_CHARGE * (charge_pv + charge_grid) * dt
        - discharge * dt / ETA_DISCHARGE,
        name="soc_balance",
    )

    # End condition avoids leaving energy in the battery after the model period.
    m.add_constraints(soc.sel(time=time[-1]) == 0, name="soc_final_zero")

    # If no battery is allowed, force all battery variables to zero
    if not battery_allowed:
        m.add_constraints(E_bat == 0, name="no_battery_energy")
        m.add_constraints(P_bat == 0, name="no_battery_power")
        m.add_constraints(charge_pv == 0, name="no_battery_charge_pv")
        m.add_constraints(charge_grid == 0, name="no_battery_charge_grid")
        m.add_constraints(discharge == 0, name="no_battery_discharge")
        m.add_constraints(soc == 0, name="no_battery_soc")

    # If grid buying is not allowed, battery can only charge from PV
    if not grid_buy_allowed:
        m.add_constraints(charge_grid == 0, name="no_grid_buy")

    # Budget constraint uses full CAPEX, not annualised costs
    m.add_constraints(
        PV_INV_COST_EUR_PER_KW * C_pv
        + bat_energy_cost * E_bat
        + bat_power_cost * P_bat
        <= BUDGET_EUR,
        name="budget_limit",
    )

    # Objective: total profit over all modelled years using annualised investment costs
    revenue = (price * q_sell * dt).sum()
    grid_purchase_cost = (price * charge_grid * dt).sum()
    degradation_cost = BAT_DEGRADATION_COST_EUR_PER_KWH * (discharge * dt).sum()

    annualised_costs_total = (
        pv_cost_annual * C_pv
        + bat_energy_cost_annual * E_bat
        + bat_power_cost_annual * P_bat
    ) * years_count

    objective = revenue - grid_purchase_cost - degradation_cost - annualised_costs_total
    m.add_objective(objective, sense="max")

    # Solve
    if SOLVER_NAME is None:
        solve_result = m.solve()
    else:
        solve_result = m.solve(solver_name=SOLVER_NAME)

    return {
        "model": m,
        "solve_result": solve_result,
        "data": data,
        "price": price,
        "pv_cf": pv_cf,
        "C_pv": C_pv,
        "E_bat": E_bat,
        "P_bat": P_bat,
        "pv_to_grid": pv_to_grid,
        "charge_pv": charge_pv,
        "charge_grid": charge_grid,
        "discharge": discharge,
        "soc": soc,
        "q_sell": q_sell,
        "curtailment": curtailment,
        "years": years,
        "location": location,
        "scenario_name": scenario_name,
        "grid_limit_name": grid_limit_name,
        "grid_limit_kw": grid_limit_kw,
        "battery_cost_factor": battery_cost_factor,
        "bat_energy_cost": bat_energy_cost,
        "bat_power_cost": bat_power_cost,
    }


def scalar_value(x) -> float:
    """Convert Linopy/xarray solution values to a Python float."""
    return float(np.asarray(x).reshape(-1)[0])


def summarise_result(res: dict) -> dict:
    """Create one summary row from a solved model."""
    years_count = len(res["years"])
    m = res["model"]
    data = res["data"]

    C_pv = scalar_value(res["C_pv"].solution.values)
    E_bat = scalar_value(res["E_bat"].solution.values)
    P_bat = scalar_value(res["P_bat"].solution.values)

    q_sell = res["q_sell"].solution
    charge_grid = res["charge_grid"].solution
    charge_pv = res["charge_pv"].solution
    discharge = res["discharge"].solution
    curtailment = res["curtailment"].solution

    price = res["price"]

    revenue = float((price * q_sell).sum())
    grid_purchase_cost = float((price * charge_grid).sum())
    degradation_cost = float(BAT_DEGRADATION_COST_EUR_PER_KWH * discharge.sum())
    total_profit = float(m.objective.value)
    avg_annual_profit = total_profit / years_count

    used_capex = (
        PV_INV_COST_EUR_PER_KW * C_pv
        + res["bat_energy_cost"] * E_bat
        + res["bat_power_cost"] * P_bat
    )

    pv_available_mwh = float((C_pv * res["pv_cf"]).sum()) / 1000.0
    curtailment_mwh = float(curtailment.sum()) / 1000.0
    q_sell_mwh = float(q_sell.sum()) / 1000.0
    grid_buy_mwh = float(charge_grid.sum()) / 1000.0
    battery_discharge_mwh = float(discharge.sum()) / 1000.0
    battery_charge_pv_mwh = float(charge_pv.sum()) / 1000.0

    curtailment_share = curtailment_mwh / pv_available_mwh if pv_available_mwh > 0 else 0.0
    return_on_capex = avg_annual_profit / used_capex if used_capex > 0 else np.nan

    annual_fixed_om = (
        PV_FIXED_OM_EUR_PER_KW_YEAR * C_pv
        + BAT_FIXED_OM_EUR_PER_KWH_YEAR * E_bat
        + BAT_FIXED_OM_EUR_PER_KW_YEAR * P_bat
    )
    operating_cashflow_total = revenue - grid_purchase_cost - degradation_cost - annual_fixed_om * years_count
    average_annual_operating_cashflow = operating_cashflow_total / years_count

    simple_payback_years = (
        used_capex / average_annual_operating_cashflow
        if average_annual_operating_cashflow > 0
        else np.nan
    )
    discounted_payback = discounted_payback_years(
        used_capex,
        average_annual_operating_cashflow,
        DISCOUNT_RATE,
        max_years=40,
    )

    # These NPV metrics are evaluation indicators, not the optimisation objective.
    # They assume that the average yearly operating cashflow repeats over the horizon.
    npv_12y = -used_capex + present_value_of_annuity(
        average_annual_operating_cashflow, DISCOUNT_RATE, LIFETIME_BAT_YEARS
    )
    npv_25y = -used_capex + present_value_of_annuity(
        average_annual_operating_cashflow, DISCOUNT_RATE, LIFETIME_PV_YEARS
    )

    # Approximate equivalent full cycles per year
    if E_bat > 0:
        cycles_per_year = (battery_discharge_mwh * 1000.0) / E_bat / years_count
    else:
        cycles_per_year = 0.0

    return {
        "location": res["location"],
        "years": f"{res['years'][0]}-{res['years'][-1]}",
        "scenario": res["scenario_name"],
        "grid_limit_scenario": res["grid_limit_name"],
        "grid_limit_kW": res["grid_limit_kw"] if res["grid_limit_kw"] is not None else np.nan,
        "battery_cost_factor": res["battery_cost_factor"],
        "solver_status": str(res["solve_result"]),
        "pv_capacity_kW": C_pv,
        "battery_energy_capacity_kWh": E_bat,
        "battery_power_capacity_kW": P_bat,
        "used_capex_EUR": used_capex,
        "budget_EUR": BUDGET_EUR,
        "total_profit_EUR": total_profit,
        "average_annual_profit_EUR": avg_annual_profit,
        "return_on_capex_per_year": return_on_capex,
        "operating_cashflow_total_EUR": operating_cashflow_total,
        "average_annual_operating_cashflow_EUR": average_annual_operating_cashflow,
        "simple_payback_years": simple_payback_years,
        "discounted_payback_years": discounted_payback,
        "npv_12y_EUR": npv_12y,
        "npv_25y_EUR": npv_25y,
        "revenue_EUR": revenue,
        "grid_purchase_cost_EUR": grid_purchase_cost,
        "degradation_cost_EUR": degradation_cost,
        "pv_available_MWh": pv_available_mwh,
        "electricity_sold_MWh": q_sell_mwh,
        "grid_buy_MWh": grid_buy_mwh,
        "battery_charge_from_pv_MWh": battery_charge_pv_mwh,
        "battery_discharge_MWh": battery_discharge_mwh,
        "curtailment_MWh": curtailment_mwh,
        "curtailment_share": curtailment_share,
        "battery_cycles_per_year": cycles_per_year,
    }


def save_example_dispatch(res: dict, filename: str, first_days: int = 14) -> None:
    """Save a short dispatch time series for one scenario."""
    n_hours = first_days * 24
    idx = res["data"].index[:n_hours]
    out = pd.DataFrame({
        "time": idx,
        "price_EUR_per_MWh": res["data"].loc[idx, "price_eur_per_kwh"].values * 1000.0,
        "pv_cf": res["data"].loc[idx, "pv_cf"].values,
        "q_sell_kWh": res["q_sell"].solution.sel(time=idx).values,
        "pv_to_grid_kWh": res["pv_to_grid"].solution.sel(time=idx).values,
        "charge_pv_kWh": res["charge_pv"].solution.sel(time=idx).values,
        "charge_grid_kWh": res["charge_grid"].solution.sel(time=idx).values,
        "discharge_kWh": res["discharge"].solution.sel(time=idx).values,
        "soc_kWh": res["soc"].solution.sel(time=idx).values,
        "curtailment_kWh": res["curtailment"].solution.sel(time=idx).values,
    })
    out.to_csv(RESULTS_DIR / filename, index=False, sep=";", decimal=",")

    plt.figure(figsize=(10, 5))
    plt.plot(out["time"], out["q_sell_kWh"], label="sold")
    plt.plot(out["time"], out["soc_kWh"], label="SOC")
    plt.plot(out["time"], out["curtailment_kWh"], label="curtailment")
    plt.xticks(rotation=45)
    plt.ylabel("kWh / kW-equivalent")
    plt.title("Example dispatch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename.replace(".csv", ".png"), dpi=200)
    plt.close()


# =============================================================================
# 5. OPTIONAL DATA ANALYSIS FUNCTIONS
# =============================================================================

def read_semicolon_decimal_file(path: Path) -> pd.DataFrame:
    """Read German CSV files with semicolon separator and comma decimals."""
    return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")


def analyse_redispatch() -> pd.DataFrame | None:
    path = OPTIONAL_DATA_DIR / "Redispatch_Daten.csv"
    if not path.exists():
        return None

    df = read_semicolon_decimal_file(path)
    if "BEGINN_DATUM" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["BEGINN_DATUM"], dayfirst=True, errors="coerce")
    df["year"] = df["date"].dt.year
    if "GESAMTE_ARBEIT_MWH" in df.columns:
        df["GESAMTE_ARBEIT_MWH"] = pd.to_numeric(df["GESAMTE_ARBEIT_MWH"], errors="coerce")
    else:
        return None

    summary = (
        df.groupby(["year", "RICHTUNG"], dropna=False)["GESAMTE_ARBEIT_MWH"]
        .sum()
        .reset_index()
        .rename(columns={"GESAMTE_ARBEIT_MWH": "redispatch_energy_MWh"})
    )
    summary.to_csv(RESULTS_DIR / "optional_redispatch_summary.csv", index=False, sep=";", decimal=",")

    yearly = df.groupby("year")["GESAMTE_ARBEIT_MWH"].sum().reset_index()
    plt.figure(figsize=(8, 4))
    plt.bar(yearly["year"], yearly["GESAMTE_ARBEIT_MWH"])
    plt.ylabel("Redispatch energy [MWh]")
    plt.title("Redispatch energy in optional data")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "optional_redispatch_energy.png", dpi=200)
    plt.close()

    return summary


def analyse_id_aep() -> pd.DataFrame | None:
    candidates = list(OPTIONAL_DATA_DIR.glob("Index Ausgleichsenergiepreis*.csv"))
    if not candidates:
        return None
    path = candidates[0]
    df = read_semicolon_decimal_file(path)

    date_col = "Datum von" if "Datum von" in df.columns else df.columns[0]
    price_col = None
    for col in df.columns:
        if "ID AEP" in col:
            price_col = col
            break
    if price_col is None:
        return None

    df["date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df["year"] = df["date"].dt.year
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    summary = df.groupby("year")[price_col].agg(["mean", "std", "min", "max"]).reset_index()
    summary = summary.rename(columns={
        "mean": "id_aep_mean_EUR_per_MWh",
        "std": "id_aep_std_EUR_per_MWh",
        "min": "id_aep_min_EUR_per_MWh",
        "max": "id_aep_max_EUR_per_MWh",
    })
    summary.to_csv(RESULTS_DIR / "optional_id_aep_summary.csv", index=False, sep=";", decimal=",")

    plt.figure(figsize=(8, 4))
    plt.plot(summary["year"], summary["id_aep_mean_EUR_per_MWh"], marker="o")
    plt.ylabel("Average ID-AEP [EUR/MWh]")
    plt.title("ID-AEP as intraday-near proxy")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "optional_id_aep_average.png", dpi=200)
    plt.close()

    return summary


def analyse_afrr() -> pd.DataFrame | None:
    path = OPTIONAL_DATA_DIR / "Aktivierte aFRR qual..csv"
    if not path.exists():
        return None
    df = read_semicolon_decimal_file(path)
    if "Datum" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["Datum"], dayfirst=True, errors="coerce")
    df["year"] = df["date"].dt.year

    pos_col = "Deutschland (Positiv)"
    neg_col = "Deutschland (Negativ)"
    if pos_col not in df.columns or neg_col not in df.columns:
        return None

    df[pos_col] = pd.to_numeric(df[pos_col], errors="coerce")
    df[neg_col] = pd.to_numeric(df[neg_col], errors="coerce")

    # 15-minute MW values -> MWh by multiplying by 0.25 h
    df["aFRR_positive_MWh"] = df[pos_col] * 0.25
    df["aFRR_negative_MWh"] = df[neg_col] * 0.25

    summary = df.groupby("year")[["aFRR_positive_MWh", "aFRR_negative_MWh"]].sum().reset_index()
    summary.to_csv(RESULTS_DIR / "optional_afrr_summary.csv", index=False, sep=";", decimal=",")

    plt.figure(figsize=(8, 4))
    plt.plot(summary["year"], summary["aFRR_positive_MWh"], marker="o", label="positive")
    plt.plot(summary["year"], summary["aFRR_negative_MWh"], marker="o", label="negative")
    plt.ylabel("Activated aFRR [MWh]")
    plt.title("Activated aFRR in Germany")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "optional_afrr_energy.png", dpi=200)
    plt.close()

    return summary


def analyse_absm() -> pd.DataFrame | None:
    candidates = list(OPTIONAL_DATA_DIR.glob("Ausgewiesene Abregelungsstrommenge*.csv"))
    if not candidates:
        return None
    path = candidates[0]
    df = read_semicolon_decimal_file(path)
    if "Datum" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["Datum"], dayfirst=True, errors="coerce")
    df["year"] = df["date"].dt.year

    value_cols = [c for c in df.columns if c.startswith("H") or c.startswith("T")]
    if not value_cols:
        return None
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["total_ABSM_MW"] = df[value_cols].sum(axis=1)
    df["total_ABSM_MWh"] = df["total_ABSM_MW"] * 0.25

    summary = df.groupby("year")["total_ABSM_MWh"].sum().reset_index()
    summary.to_csv(RESULTS_DIR / "optional_absm_summary.csv", index=False, sep=";", decimal=",")

    plt.figure(figsize=(6, 4))
    plt.bar(summary["year"].astype(str), summary["total_ABSM_MWh"])
    plt.ylabel("ABSM [MWh]")
    plt.title("Curtailment / ABSM data available in file")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "optional_absm_energy.png", dpi=200)
    plt.close()

    return summary


def analyse_solar_market_values() -> pd.DataFrame:
    df = pd.DataFrame({
        "year": list(SOLAR_MARKET_VALUE_CT_PER_KWH.keys()),
        "solar_market_value_ct_per_kWh": list(SOLAR_MARKET_VALUE_CT_PER_KWH.values()),
    })
    df["solar_market_value_EUR_per_MWh"] = df["solar_market_value_ct_per_kWh"] * 10.0
    df.to_csv(RESULTS_DIR / "solar_market_values.csv", index=False, sep=";", decimal=",")

    plt.figure(figsize=(8, 4))
    plt.plot(df["year"], df["solar_market_value_EUR_per_MWh"], marker="o")
    plt.ylabel("Solar market value [EUR/MWh]")
    plt.title("Annual solar market value")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "solar_market_values.png", dpi=200)
    plt.close()
    return df


# =============================================================================
# 6. COMPARATIVE METRICS AND PLOTTING MAIN RESULTS
# =============================================================================

def add_comparative_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Add comparison against the PV-only case for the same location and grid limit."""
    df = summary_df.copy()
    baseline = df[
        (df["scenario"] == "PV_only")
        & (df["battery_cost_factor"] == 1.0)
    ][[
        "location",
        "years",
        "grid_limit_scenario",
        "total_profit_EUR",
        "average_annual_profit_EUR",
        "curtailment_MWh",
        "electricity_sold_MWh",
    ]].rename(columns={
        "total_profit_EUR": "pv_only_total_profit_EUR",
        "average_annual_profit_EUR": "pv_only_average_annual_profit_EUR",
        "curtailment_MWh": "pv_only_curtailment_MWh",
        "electricity_sold_MWh": "pv_only_electricity_sold_MWh",
    })

    df = df.merge(
        baseline,
        on=["location", "years", "grid_limit_scenario"],
        how="left",
    )
    df["profit_gain_vs_pv_only_EUR"] = df["total_profit_EUR"] - df["pv_only_total_profit_EUR"]
    df["average_annual_profit_gain_vs_pv_only_EUR"] = (
        df["average_annual_profit_EUR"] - df["pv_only_average_annual_profit_EUR"]
    )
    df["curtailment_reduction_vs_pv_only_MWh"] = (
        df["pv_only_curtailment_MWh"] - df["curtailment_MWh"]
    )
    df["additional_sold_energy_vs_pv_only_MWh"] = (
        df["electricity_sold_MWh"] - df["pv_only_electricity_sold_MWh"]
    )
    return df


def create_main_plots(summary_df: pd.DataFrame) -> None:
    """Create plots from the main model result summary."""
    # Average annual profit by grid limit and scenario
    for location in summary_df["location"].unique():
        sdf = summary_df[(summary_df["location"] == location) & (summary_df["battery_cost_factor"] == 1.0)]

        plt.figure(figsize=(11, 5))
        labels = sdf["scenario"] + "\n" + sdf["grid_limit_scenario"]
        plt.bar(labels, sdf["average_annual_profit_EUR"])
        plt.xticks(rotation=90)
        plt.ylabel("Average annual profit [EUR/year]")
        plt.title(f"Average annual profit - {location}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"profit_{location}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(11, 5))
        plt.bar(labels, sdf["curtailment_MWh"])
        plt.xticks(rotation=90)
        plt.ylabel("Curtailment [MWh over model period]")
        plt.title(f"Curtailment - {location}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"curtailment_{location}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(11, 5))
        plt.bar(labels, sdf["battery_energy_capacity_kWh"])
        plt.xticks(rotation=90)
        plt.ylabel("Battery energy capacity [kWh]")
        plt.title(f"Battery energy capacity - {location}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"battery_energy_{location}.png", dpi=200)
        plt.close()

    # Battery cost sensitivity if available
    sensitivity = summary_df[summary_df["battery_cost_factor"] != 1.0]
    if not sensitivity.empty:
        plt.figure(figsize=(8, 4))
        for location in sensitivity["location"].unique():
            sdf = sensitivity[sensitivity["location"] == location]
            plt.plot(
                sdf["battery_cost_factor"],
                sdf["battery_energy_capacity_kWh"],
                marker="o",
                label=location,
            )
        plt.gca().invert_xaxis()
        plt.xlabel("Battery cost factor")
        plt.ylabel("Battery energy capacity [kWh]")
        plt.title("Battery cost sensitivity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "battery_cost_sensitivity.png", dpi=200)
        plt.close()


# =============================================================================
# 7. MAIN SCRIPT
# =============================================================================

def main() -> None:
    prices_raw, pv_raw = read_main_data()

    years = YEARS
    locations = LOCATIONS
    scenarios = MAIN_SCENARIOS
    grid_limits = GRID_LIMIT_SCENARIOS

    if QUICK_TEST:
        years = [2024]
        locations = ["Stuttgart"]
        scenarios = {"PV_battery_no_grid_buy": MAIN_SCENARIOS["PV_battery_no_grid_buy"]}
        grid_limits = {"no_limit": None, "medium_limit_7_5MW": 7_500.0}
        print("QUICK_TEST=True: running only a small test case.")

    # Save market indicators for all locations
    all_market_indicators = []
    for location in locations:
        data = prepare_multiyear_data(prices_raw, pv_raw, years, location)
        indicators = calculate_market_indicators(data)
        indicators.insert(0, "location", location)
        all_market_indicators.append(indicators)
    market_indicators_df = pd.concat(all_market_indicators, ignore_index=True)
    market_indicators_df.to_csv(RESULTS_DIR / "market_indicators.csv", index=False, sep=";", decimal=",")

    # Run main scenarios
    summary_rows = []
    example_saved = False

    for location in locations:
        for scenario_name in scenarios:
            for grid_limit_name, grid_limit_kw in grid_limits.items():
                print(f"Running: {location} | {scenario_name} | {grid_limit_name}")
                res = run_model(
                    prices_raw=prices_raw,
                    pv_raw=pv_raw,
                    years=years,
                    location=location,
                    scenario_name=scenario_name,
                    grid_limit_name=grid_limit_name,
                    grid_limit_kw=grid_limit_kw,
                    battery_cost_factor=1.0,
                )
                summary_rows.append(summarise_result(res))

                if not example_saved and location == "Stuttgart" and scenario_name == "PV_battery_no_grid_buy":
                    save_example_dispatch(res, "example_dispatch_Stuttgart.csv")
                    example_saved = True

    # Optional battery cost sensitivity: only one scenario per location to avoid too many runs.
    if RUN_BATTERY_COST_SENSITIVITY and not QUICK_TEST:
        for location in locations:
            for factor in BATTERY_COST_FACTORS:
                print(f"Running battery cost sensitivity: {location} | factor={factor}")
                res = run_model(
                    prices_raw=prices_raw,
                    pv_raw=pv_raw,
                    years=years,
                    location=location,
                    scenario_name="PV_battery_no_grid_buy",
                    grid_limit_name="medium_limit_7_5MW",
                    grid_limit_kw=7_500.0,
                    battery_cost_factor=factor,
                )
                summary_rows.append(summarise_result(res))

    summary_df = pd.DataFrame(summary_rows)
    summary_df = add_comparative_metrics(summary_df)
    summary_df.to_csv(RESULTS_DIR / "results_summary_gridlimit.csv", index=False, sep=";", decimal=",")
    summary_df.to_excel(RESULTS_DIR / "results_summary_gridlimit.xlsx", index=False)
    create_main_plots(summary_df)

    # Optional external data analysis
    analyse_solar_market_values()
    if RUN_OPTIONAL_DATA_ANALYSIS:
        print("Running optional data analyses...")
        analyse_redispatch()
        analyse_id_aep()
        analyse_afrr()
        analyse_absm()

    print("\nDone.")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"Figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
