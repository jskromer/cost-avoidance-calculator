"""
Cost & Carbon Avoidance Calculator — CMVP Training Tool

Demonstrates why cost avoidance must be calculated by applying the full rate
schedule to BOTH the adjusted baseline consumption AND the actual consumption
independently, then subtracting. The naive approach (energy savings x average
rate) produces errors that this tool quantifies.

The same principle applies to carbon avoidance: you can't multiply saved kWh
by an average grid emissions factor. You need marginal emissions factors (MEF)
that vary by hour — low midday when solar/wind is on the margin, high in the
evening when gas peakers fire up.

Author: Counterfactual Designs
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import calendar

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cost & Carbon Avoidance Calculator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants — 2026 calendar (not a leap year, Jan 1 = Thursday)
# ---------------------------------------------------------------------------
YEAR = 2026
HOURS_IN_YEAR = 8760
DAYS_IN_MONTHS = [calendar.monthrange(YEAR, m)[1] for m in range(1, 13)]
HOURS_IN_MONTHS = [d * 24 for d in DAYS_IN_MONTHS]

# Build arrays mapping each of the 8760 hours to metadata
_hour_of_year_to_month = np.zeros(HOURS_IN_YEAR, dtype=int)      # 1-12
_hour_of_year_to_hour_of_day = np.zeros(HOURS_IN_YEAR, dtype=int) # 0-23
_hour_of_year_to_day_of_week = np.zeros(HOURS_IN_YEAR, dtype=int) # 0=Mon ... 6=Sun
_hour_of_year_to_day_of_year = np.zeros(HOURS_IN_YEAR, dtype=int) # 0-364

_dt_start = datetime(YEAR, 1, 1)
for h in range(HOURS_IN_YEAR):
    dt = _dt_start + timedelta(hours=h)
    _hour_of_year_to_month[h] = dt.month
    _hour_of_year_to_hour_of_day[h] = dt.hour
    _hour_of_year_to_day_of_week[h] = dt.weekday()
    _hour_of_year_to_day_of_year[h] = dt.timetuple().tm_yday - 1

MONTH_ARR = _hour_of_year_to_month
HOUR_ARR = _hour_of_year_to_hour_of_day
DOW_ARR = _hour_of_year_to_day_of_week
DOY_ARR = _hour_of_year_to_day_of_year

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# ---------------------------------------------------------------------------
# Climate profiles — approximate outdoor temperature sinusoids
# ---------------------------------------------------------------------------
CLIMATE_PROFILES = {
    "Mild Coastal CA (SF Bay)": {"avg": 57, "amp": 8, "cool_base": 65, "heat_base": 60},
    "Hot Inland CA (Fresno)":   {"avg": 64, "amp": 20, "cool_base": 75, "heat_base": 55},
    "New York City":            {"avg": 55, "amp": 22, "cool_base": 72, "heat_base": 58},
    "Denver CO":                {"avg": 50, "amp": 25, "cool_base": 72, "heat_base": 55},
    "Honolulu HI":              {"avg": 77, "amp": 5, "cool_base": 78, "heat_base": 65},
    "New Jersey":               {"avg": 54, "amp": 21, "cool_base": 72, "heat_base": 58},
    "Central CA (CAISO)":       {"avg": 64, "amp": 20, "cool_base": 75, "heat_base": 55},
}

UTILITY_CLIMATE_MAP = {
    "PG&E E-1 (Tiered)": "Mild Coastal CA (SF Bay)",
    "SCE Schedule D (Tiered)": "Hot Inland CA (Fresno)",
    "Con Edison Rate I (Seasonal Delivery)": "New York City",
    "Xcel Energy CO RE-TOU (Time-of-Use)": "Denver CO",
    "Hawaiian Electric Sch R (Flat)": "Honolulu HI",
    "PSEG NJ RS (Flat + Seasonal Supply)": "New Jersey",
    "CAISO Wholesale (Neg. Pricing)": "Central CA (CAISO)",
}

UTILITY_NAMES = list(UTILITY_CLIMATE_MAP.keys())


# ---------------------------------------------------------------------------
# Temperature simulation
# ---------------------------------------------------------------------------
def simulate_temperature(climate_key: str) -> np.ndarray:
    """Return 8760 hourly outdoor temperatures (°F) using sinusoidal annual cycle
    with a diurnal swing."""
    prof = CLIMATE_PROFILES[climate_key]
    avg, amp = prof["avg"], prof["amp"]
    # Annual cycle: peak around day 200 (mid-July)
    annual = avg + amp * np.sin(2 * np.pi * (DOY_ARR - 110) / 365)
    # Diurnal cycle: +/- 8°F, peak at hour 15
    diurnal = 8.0 * np.sin(2 * np.pi * (HOUR_ARR - 9) / 24)
    return annual + diurnal


# ---------------------------------------------------------------------------
# CAISO wholesale price curve simulation
# ---------------------------------------------------------------------------
def simulate_caiso_prices() -> np.ndarray:
    """Generate a synthetic CAISO 8760 hourly wholesale price curve ($/kWh).

    Key features modeled from real CAISO day-ahead market patterns:
    - Midday solar trough: prices drop (often negative) 10AM–3PM in spring/summer
      due to solar oversupply on the duck curve belly
    - Evening ramp peak: prices spike 5–9 PM as solar drops and demand rises
    - Winter: less solar, prices stay positive but with morning/evening peaks
    - Random volatility layered on top
    """
    np.random.seed(99)
    prices = np.zeros(HOURS_IN_YEAR)

    for h in range(HOURS_IN_YEAR):
        month = MONTH_ARR[h]
        hour = HOUR_ARR[h]
        day_of_year = DOY_ARR[h]

        # Base price varies by season ($/MWh, will convert to $/kWh at end)
        # Annual cycle: cheaper in spring/summer shoulder, higher in summer peak & winter
        seasonal_base = 35 + 15 * np.sin(2 * np.pi * (day_of_year - 30) / 365)

        # Diurnal shape — the duck curve
        if month in (3, 4, 5, 6, 7, 8, 9):  # Spring + Summer: strong solar
            # Solar depression: deep trough 10AM-3PM
            if 10 <= hour <= 14:
                solar_factor = -45 - 20 * np.sin(np.pi * (hour - 10) / 4)
                # Deeper in spring (Mar-May) when demand is low but solar is high
                if month in (3, 4, 5):
                    solar_factor *= 1.4  # More negative in spring
            elif 15 <= hour <= 16:
                solar_factor = -15  # Transition — still depressed
            # Evening ramp: 5-9 PM spike
            elif 17 <= hour <= 20:
                solar_factor = 40 + 25 * np.sin(np.pi * (hour - 17) / 3)
            # Morning ramp: 6-9 AM
            elif 6 <= hour <= 9:
                solar_factor = 10
            # Overnight: low
            else:
                solar_factor = -5
        else:
            # Winter: no deep solar trough, modest morning/evening peaks
            if 7 <= hour <= 9:
                solar_factor = 15
            elif 17 <= hour <= 20:
                solar_factor = 25
            elif 10 <= hour <= 15:
                solar_factor = 0  # Mild solar effect
            else:
                solar_factor = -5

        # Weekend discount
        dow = DOW_ARR[h]
        weekend_factor = -10 if dow >= 5 else 0

        # Random volatility
        noise = np.random.normal(0, 8)

        # Total $/MWh
        price_mwh = seasonal_base + solar_factor + weekend_factor + noise

        # Convert to $/kWh
        prices[h] = price_mwh / 1000.0

    return prices

# Cache the CAISO price curve (computed once)
CAISO_PRICES = simulate_caiso_prices()


# ---------------------------------------------------------------------------
# Marginal emissions factors (MEF) — synthetic 8760 curves by grid region
# ---------------------------------------------------------------------------
UTILITY_GRID_MAP = {
    "PG&E E-1 (Tiered)": "CAISO",
    "SCE Schedule D (Tiered)": "CAISO",
    "Con Edison Rate I (Seasonal Delivery)": "NYISO",
    "Xcel Energy CO RE-TOU (Time-of-Use)": "WEIS_RMPA",
    "Hawaiian Electric Sch R (Flat)": "HECO",
    "PSEG NJ RS (Flat + Seasonal Supply)": "PJM",
    "CAISO Wholesale (Neg. Pricing)": "CAISO",
}

# Grid region parameters: avg MEF (lbs CO2/kWh), midday min, evening max,
# overnight level, seasonal amplitude (how much summer solar depresses midday)
GRID_REGION_PARAMS = {
    "CAISO": {
        "avg": 0.45, "midday_min": 0.15, "evening_max": 0.85,
        "overnight": 0.55, "seasonal_solar_amp": 0.25,
        "description": "California ISO — heavy solar penetration",
    },
    "NYISO": {
        "avg": 0.55, "midday_min": 0.40, "evening_max": 0.80,
        "overnight": 0.60, "seasonal_solar_amp": 0.12,
        "description": "New York ISO — moderate renewables",
    },
    "WEIS_RMPA": {
        "avg": 0.75, "midday_min": 0.60, "evening_max": 0.95,
        "overnight": 0.78, "seasonal_solar_amp": 0.10,
        "description": "Western EIS / RMPA — coal-heavy baseload",
    },
    "HECO": {
        "avg": 0.70, "midday_min": 0.55, "evening_max": 0.85,
        "overnight": 0.72, "seasonal_solar_amp": 0.10,
        "description": "Hawaiian Electric — oil-dependent grid",
    },
    "PJM": {
        "avg": 0.65, "midday_min": 0.50, "evening_max": 0.80,
        "overnight": 0.68, "seasonal_solar_amp": 0.10,
        "description": "PJM Interconnection — gas/coal mix",
    },
}


def simulate_marginal_emissions(grid_region: str) -> np.ndarray:
    """Return 8760 hourly marginal emissions factors (lbs CO2/kWh) for a grid region.

    Models the diurnal and seasonal variation in grid marginal emissions:
    - Lowest midday (10AM-3PM) when solar/wind is on the margin
    - Highest evening (5-9PM) when gas peakers fire up
    - Moderate overnight
    - Seasonal: summer has deeper midday depression (more solar), winter is flatter
    """
    np.random.seed(77 + hash(grid_region) % 100)
    params = GRID_REGION_PARAMS[grid_region]
    mef = np.zeros(HOURS_IN_YEAR)

    midday_min = params["midday_min"]
    evening_max = params["evening_max"]
    overnight = params["overnight"]
    solar_amp = params["seasonal_solar_amp"]

    for h in range(HOURS_IN_YEAR):
        month = MONTH_ARR[h]
        hour = HOUR_ARR[h]
        day_of_year = DOY_ARR[h]

        # Seasonal factor: summer has more solar depression
        # Peak solar effect around day 172 (summer solstice)
        seasonal_solar = solar_amp * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        seasonal_solar = max(seasonal_solar, 0)  # Only depress in summer half

        # Winter has slightly higher baseline emissions (more heating load = more gas/coal)
        winter_boost = 0.05 * np.cos(2 * np.pi * (day_of_year - 10) / 365)
        winter_boost = max(winter_boost, 0)

        # Diurnal pattern
        if 10 <= hour <= 14:
            # Midday: low emissions (solar/wind on margin)
            # Deeper depression in summer
            base_mef = midday_min - seasonal_solar
            # Smooth shape within the midday window
            midday_shape = np.sin(np.pi * (hour - 10) / 4)
            base_mef = midday_min - seasonal_solar * midday_shape
        elif 15 <= hour <= 16:
            # Transition from midday to evening
            t = (hour - 15) / 2.0
            base_mef = midday_min * (1 - t) + overnight * t
        elif 17 <= hour <= 20:
            # Evening peak: gas peakers on margin
            evening_shape = np.sin(np.pi * (hour - 17) / 3)
            base_mef = overnight + (evening_max - overnight) * evening_shape
        elif 21 <= hour <= 23:
            # Late evening ramp-down
            t = (hour - 21) / 3.0
            base_mef = evening_max * (1 - t) + overnight * t
        elif 6 <= hour <= 9:
            # Morning ramp: transitioning from overnight to approaching midday
            t = (hour - 6) / 4.0
            base_mef = overnight * (1 - t) + (midday_min + 0.10) * t
        else:
            # Overnight (0-5 AM)
            base_mef = overnight

        # Add winter boost
        base_mef += winter_boost

        # Add noise (±10%)
        noise = np.random.normal(1.0, 0.10)
        noise = np.clip(noise, 0.80, 1.20)
        mef[h] = base_mef * noise

    # Clip to reasonable range
    mef = np.clip(mef, 0.0, 1.5)

    # Scale so the annual average matches the target avg
    current_avg = mef.mean()
    if current_avg > 0:
        mef *= params["avg"] / current_avg

    return mef


# Cache all grid region MEF arrays (computed once at module level)
GRID_MEF_CACHE = {region: simulate_marginal_emissions(region) for region in GRID_REGION_PARAMS}


# ---------------------------------------------------------------------------
# Load shape simulation
# ---------------------------------------------------------------------------
def simulate_baseline(climate_key: str, annual_kwh: float, seed: int = 42):
    """Return dict with 8760 arrays for baseload, occupancy, hvac, total."""
    rng = np.random.default_rng(seed)
    temp = simulate_temperature(climate_key)
    prof = CLIMATE_PROFILES[climate_key]

    # --- Baseload: constant ~0.35 kW ---
    baseload = np.full(HOURS_IN_YEAR, 0.35)

    # --- Occupancy: higher 6AM-10PM ---
    occ = np.where((HOUR_ARR >= 6) & (HOUR_ARR <= 22), 0.35, 0.08)

    # --- HVAC: degree-based ---
    cooling = np.maximum(temp - prof["cool_base"], 0) * 0.12
    heating = np.maximum(prof["heat_base"] - temp, 0) * 0.08
    hvac = cooling + heating

    # Combine and scale to target annual total
    raw_total = baseload + occ + hvac
    raw_annual = raw_total.sum()
    scale = annual_kwh / raw_annual
    baseload *= scale
    occ *= scale
    hvac *= scale
    total = baseload + occ + hvac

    # Add noise
    noise = rng.normal(1.0, 0.07, HOURS_IN_YEAR)
    noise = np.clip(noise, 0.8, 1.2)
    total *= noise
    # Redistribute noise proportionally
    frac_base = baseload / (baseload + occ + hvac + 1e-12)
    frac_occ = occ / (baseload + occ + hvac + 1e-12)
    frac_hvac = hvac / (baseload + occ + hvac + 1e-12)
    baseload = total * frac_base
    occ = total * frac_occ
    hvac = total * frac_hvac

    return {
        "baseload": baseload,
        "occupancy": occ,
        "hvac": hvac,
        "total": total,
        "temperature": temp,
    }


# ---------------------------------------------------------------------------
# ECM savings shapes
# ---------------------------------------------------------------------------
def compute_savings(baseline: dict, ecm_type: str, savings_pct: float) -> np.ndarray:
    """Return 8760 hourly savings array (kWh) for the given ECM type."""
    target_savings = baseline["total"].sum() * savings_pct

    if ecm_type == "Lighting retrofit":
        # Saves during occupied hours (6AM-10PM), zero overnight, flat across seasons
        shape = np.where((HOUR_ARR >= 6) & (HOUR_ARR <= 22), 1.0, 0.0)
    elif ecm_type == "HVAC upgrade":
        # Proportional to HVAC load
        shape = baseline["hvac"].copy()
    elif ecm_type == "Envelope improvement":
        # Similar to HVAC but smaller baseline magnitude (handled by scaling)
        shape = baseline["hvac"].copy()
    elif ecm_type == "Behavioral/controls":
        # Small flat baseload reduction + occupancy reduction
        shape = 0.4 * baseline["baseload"] + 0.6 * baseline["occupancy"]
    elif ecm_type == "Custom flat %":
        # Flat percentage every hour — this IS the naive assumption
        shape = baseline["total"].copy()
    else:
        shape = baseline["total"].copy()

    # Scale shape so sum equals target_savings
    shape_sum = shape.sum()
    if shape_sum < 1e-6:
        return np.zeros(HOURS_IN_YEAR)
    savings = shape * (target_savings / shape_sum)
    # Ensure savings don't exceed baseline in any hour
    savings = np.minimum(savings, baseline["total"])
    return savings


# ---------------------------------------------------------------------------
# Tariff engine
# ---------------------------------------------------------------------------
def _monthly_slices():
    """Return list of 12 (start_hour, end_hour) tuples for each month."""
    slices = []
    offset = 0
    for hrs in HOURS_IN_MONTHS:
        slices.append((offset, offset + hrs))
        offset += hrs
    return slices


MONTHLY_SLICES = _monthly_slices()


def _is_summer(month: int, utility: str) -> bool:
    """Return True if the given month (1-12) is in summer for the utility."""
    if utility == "PSEG NJ RS (Flat + Seasonal Supply)":
        return month in (6, 7, 8, 9, 10)  # Jun–Oct
    else:
        return month in (6, 7, 8, 9)  # Jun–Sep for all others


def apply_tariff(hourly_kwh: np.ndarray, utility: str) -> dict:
    """Apply the specified utility tariff to 8760 hourly consumption.

    Returns dict with:
        monthly_bills: list of 12 dicts with component breakdown
        annual_total: float
    """
    monthly_bills = []

    for m_idx in range(12):
        month = m_idx + 1
        s, e = MONTHLY_SLICES[m_idx]
        monthly_kwh_arr = hourly_kwh[s:e]
        monthly_kwh_total = monthly_kwh_arr.sum()
        monthly_hours = HOUR_ARR[s:e]
        monthly_dow = DOW_ARR[s:e]
        summer = _is_summer(month, utility)
        days = DAYS_IN_MONTHS[m_idx]

        bill = {"month": MONTH_NAMES[m_idx], "fixed": 0.0, "energy": 0.0,
                "tier1": 0.0, "tier2": 0.0, "delivery": 0.0, "supply": 0.0,
                "on_peak": 0.0, "off_peak": 0.0, "total": 0.0}

        if utility == "PG&E E-1 (Tiered)":
            bill["fixed"] = 24.0
            baseline_kwh = (8.8 if summer else 10.4) * days
            tier1_kwh = min(monthly_kwh_total, baseline_kwh)
            tier2_kwh = max(monthly_kwh_total - baseline_kwh, 0)
            bill["tier1"] = tier1_kwh * 0.38
            bill["tier2"] = tier2_kwh * 0.47
            bill["energy"] = bill["tier1"] + bill["tier2"]

        elif utility == "SCE Schedule D (Tiered)":
            bill["fixed"] = 0.79 * days
            baseline_kwh = (12.0 if summer else 14.0) * days
            tier1_kwh = min(monthly_kwh_total, baseline_kwh)
            tier2_kwh = max(monthly_kwh_total - baseline_kwh, 0)
            bill["tier1"] = tier1_kwh * 0.30
            bill["tier2"] = tier2_kwh * 0.40
            bill["energy"] = bill["tier1"] + bill["tier2"]

        elif utility == "Con Edison Rate I (Seasonal Delivery)":
            bill["fixed"] = 18.0
            if summer:
                del_first_250 = min(monthly_kwh_total, 250) * 0.141
                del_over_250 = max(monthly_kwh_total - 250, 0) * 0.162
                bill["delivery"] = del_first_250 + del_over_250
            else:
                bill["delivery"] = monthly_kwh_total * 0.141
            bill["supply"] = monthly_kwh_total * 0.09
            bill["energy"] = bill["delivery"] + bill["supply"]

        elif utility == "Xcel Energy CO RE-TOU (Time-of-Use)":
            bill["fixed"] = 7.50
            # On-peak: 5-9 PM (hours 17-20) weekdays
            is_on_peak = ((monthly_hours >= 17) & (monthly_hours <= 20) &
                          (monthly_dow <= 4))  # Mon-Fri = 0-4
            on_peak_kwh = monthly_kwh_arr[is_on_peak].sum()
            off_peak_kwh = monthly_kwh_arr[~is_on_peak].sum()
            if summer:
                bill["on_peak"] = on_peak_kwh * 0.209
                bill["off_peak"] = off_peak_kwh * 0.079
            else:
                bill["on_peak"] = on_peak_kwh * 0.183
                bill["off_peak"] = off_peak_kwh * 0.068
            bill["energy"] = bill["on_peak"] + bill["off_peak"]

        elif utility == "Hawaiian Electric Sch R (Flat)":
            bill["fixed"] = 13.48
            bill["energy"] = monthly_kwh_total * 0.38

        elif utility == "PSEG NJ RS (Flat + Seasonal Supply)":
            bill["fixed"] = 6.00
            bill["delivery"] = monthly_kwh_total * 0.0623
            if summer:
                bill["supply"] = monthly_kwh_total * 0.126236
            else:
                bill["supply"] = monthly_kwh_total * 0.134599
            bill["energy"] = bill["delivery"] + bill["supply"]

        elif utility == "CAISO Wholesale (Neg. Pricing)":
            # No fixed charge at wholesale; pure hourly settlement
            bill["fixed"] = 0.0
            # Multiply each hour's consumption by that hour's price
            hourly_prices = CAISO_PRICES[s:e]
            hourly_cost = (monthly_kwh_arr * hourly_prices).sum()
            bill["energy"] = hourly_cost
            # Track positive vs negative price hours for breakdown
            pos_mask = hourly_prices >= 0
            neg_mask = ~pos_mask
            bill["on_peak"] = (monthly_kwh_arr[pos_mask] * hourly_prices[pos_mask]).sum()
            bill["off_peak"] = (monthly_kwh_arr[neg_mask] * hourly_prices[neg_mask]).sum()

        bill["total"] = bill["fixed"] + bill["energy"]
        monthly_bills.append(bill)

    annual_total = sum(b["total"] for b in monthly_bills)
    return {"monthly_bills": monthly_bills, "annual_total": annual_total}


# ---------------------------------------------------------------------------
# Cost avoidance calculations
# ---------------------------------------------------------------------------
def compute_cost_avoidance(baseline_8760: np.ndarray, actual_8760: np.ndarray,
                           utility: str) -> dict:
    """Compute correct and naive cost avoidance, return comparison dict."""
    result_baseline = apply_tariff(baseline_8760, utility)
    result_actual = apply_tariff(actual_8760, utility)

    cost_baseline = result_baseline["annual_total"]
    cost_actual = result_actual["annual_total"]
    correct_avoidance = cost_baseline - cost_actual

    # Monthly correct avoidance
    monthly_correct = [
        result_baseline["monthly_bills"][i]["total"] -
        result_actual["monthly_bills"][i]["total"]
        for i in range(12)
    ]

    # Naive method — uses blended energy-only rate (excludes fixed charges,
    # since practitioners typically know fixed charges don't change with savings)
    energy_savings = baseline_8760.sum() - actual_8760.sum()
    baseline_energy_cost = sum(b["energy"] for b in result_baseline["monthly_bills"])

    if utility == "CAISO Wholesale (Neg. Pricing)":
        # For wholesale: the naive mistake is using the simple time-average price
        # (what you'd see reported as "average CAISO price") rather than the
        # consumption-weighted price.  A practitioner grabs the average $/MWh
        # from a market report and multiplies by saved kWh.
        avg_rate = CAISO_PRICES.mean()
    else:
        avg_rate = baseline_energy_cost / baseline_8760.sum() if baseline_8760.sum() > 0 else 0

    naive_avoidance = energy_savings * avg_rate

    # Monthly naive (pro-rate by monthly energy savings)
    monthly_naive = []
    for i in range(12):
        s, e = MONTHLY_SLICES[i]
        m_savings = baseline_8760[s:e].sum() - actual_8760[s:e].sum()
        if utility == "CAISO Wholesale (Neg. Pricing)":
            monthly_naive.append(m_savings * CAISO_PRICES[s:e].mean())
        else:
            monthly_naive.append(m_savings * avg_rate)

    error = naive_avoidance - correct_avoidance
    error_pct = (error / correct_avoidance * 100) if abs(correct_avoidance) > 0.01 else 0.0

    return {
        "cost_baseline": cost_baseline,
        "cost_actual": cost_actual,
        "correct_avoidance": correct_avoidance,
        "naive_avoidance": naive_avoidance,
        "error": error,
        "error_pct": error_pct,
        "avg_rate": avg_rate,
        "energy_savings_kwh": energy_savings,
        "monthly_correct": monthly_correct,
        "monthly_naive": monthly_naive,
        "baseline_bills": result_baseline,
        "actual_bills": result_actual,
    }


# ---------------------------------------------------------------------------
# Carbon avoidance calculations
# ---------------------------------------------------------------------------
def compute_carbon_avoidance(baseline_8760: np.ndarray, actual_8760: np.ndarray,
                             utility: str) -> dict:
    """Compute correct and naive carbon avoidance using marginal emissions factors.

    Correct method: sum of (savings_h x MEF_h) for each hour — uses hourly marginal
    emissions factors that reflect what generation source is actually on the margin.

    Naive method: total_kwh_saved x annual_average_MEF — assumes every kWh saved
    displaces the same average amount of carbon regardless of when it occurs.

    Returns dict with lbs CO2 avoided (correct and naive), error, and monthly breakdowns.
    """
    grid_region = UTILITY_GRID_MAP.get(utility, "PJM")
    mef_8760 = GRID_MEF_CACHE[grid_region]

    savings_8760 = baseline_8760 - actual_8760

    # Correct: hourly savings × hourly MEF
    correct_carbon = (savings_8760 * mef_8760).sum()

    # Naive: total savings × annual average MEF
    avg_mef = mef_8760.mean()
    total_kwh_saved = savings_8760.sum()
    naive_carbon = total_kwh_saved * avg_mef

    # Error
    carbon_error = naive_carbon - correct_carbon
    carbon_error_pct = (carbon_error / correct_carbon * 100) if abs(correct_carbon) > 0.01 else 0.0

    # Monthly breakdowns
    monthly_correct_carbon = []
    monthly_naive_carbon = []
    for i in range(12):
        s, e = MONTHLY_SLICES[i]
        m_savings = savings_8760[s:e]
        m_mef = mef_8760[s:e]
        monthly_correct_carbon.append((m_savings * m_mef).sum())
        monthly_naive_carbon.append(m_savings.sum() * avg_mef)

    return {
        "correct_carbon": correct_carbon,
        "naive_carbon": naive_carbon,
        "carbon_error": carbon_error,
        "carbon_error_pct": carbon_error_pct,
        "avg_mef": avg_mef,
        "monthly_correct_carbon": monthly_correct_carbon,
        "monthly_naive_carbon": monthly_naive_carbon,
        "grid_region": grid_region,
    }


# ---------------------------------------------------------------------------
# Explanation text generator
# ---------------------------------------------------------------------------
def get_explanation(utility: str, error: float) -> str:
    """Return a pedagogical explanation of WHY the error occurs for this tariff."""
    direction = "overstates" if error > 0 else "understates"

    explanations = {
        "PG&E E-1 (Tiered)": (
            f"PG&E E-1 is a two-tier inclining block rate. Savings come "
            f"disproportionately from the expensive Tier 2 ($0.47/kWh), because "
            f"the ECM reduces consumption from the top of the usage stack first. "
            f"The naive method uses a blended average rate that mixes cheap Tier 1 "
            f"and expensive Tier 2 kWh, which {direction} the value of those savings."
        ),
        "SCE Schedule D (Tiered)": (
            f"SCE Schedule D has a two-tier structure similar to PG&E. Savings "
            f"reduce Tier 2 consumption ($0.40/kWh) before touching Tier 1 ($0.30/kWh). "
            f"The naive blended rate averages across both tiers, which {direction} the "
            f"true cost avoidance because it misses the marginal value of avoided Tier 2 kWh."
        ),
        "Con Edison Rate I (Seasonal Delivery)": (
            f"Con Edison Rate I has a summer delivery surcharge above 250 kWh/month "
            f"($0.162 vs $0.141/kWh). When savings push summer consumption below the "
            f"250 kWh threshold, the avoided kWh were at the higher rate. The naive method "
            f"uses a year-round blended rate that {direction} the seasonal differential."
        ),
        "Xcel Energy CO RE-TOU (Time-of-Use)": (
            f"Xcel's TOU rate charges $0.209/kWh on-peak vs $0.079/kWh off-peak in "
            f"summer — a 2.6:1 ratio. The naive blended rate assumes every kWh saved "
            f"has the same value regardless of when it occurs. If savings are concentrated "
            f"during off-peak hours (e.g., HVAC running overnight), the naive method "
            f"{direction} their value because those kWh are worth less than the blend."
        ),
        "Hawaiian Electric Sch R (Flat)": (
            f"Hawaiian Electric's Schedule R is a flat volumetric rate with no tiers, "
            f"no TOU differentials, and no seasonal variation. Every kWh has the same "
            f"value ($0.38), so the blended average rate equals the marginal rate. "
            f"The naive method produces near-zero error — this is the one case where "
            f"the shortcut works. But most real tariffs aren't this simple."
        ),
        "PSEG NJ RS (Flat + Seasonal Supply)": (
            f"PSEG NJ has flat distribution but seasonal supply charges (summer $0.126 "
            f"vs non-summer $0.135/kWh — note the inversion: winter supply is more expensive). "
            f"The naive blended rate averages across seasons, which {direction} the value "
            f"of savings depending on their seasonal distribution."
        ),
        "CAISO Wholesale (Neg. Pricing)": (
            f"CAISO wholesale prices follow the duck curve: midday solar oversupply "
            f"drives prices negative (you get *paid* to consume 10AM–3PM in spring/summer), "
            f"while the evening ramp (5–9PM) spikes to $60–90/MWh as solar drops off. "
            f"The naive blended rate averages positive and negative hours into a single "
            f"number, completely obscuring the fact that saving a kWh at noon in April "
            f"has *negative* value (you're avoiding consumption during negative prices — "
            f"that costs you money, not saves it). The naive method {direction} cost "
            f"avoidance because it cannot represent the temporal mismatch between when "
            f"savings occur and what those hours are worth."
        ),
    }
    return explanations.get(utility, "")


# ---------------------------------------------------------------------------
# Carbon explanation text generator
# ---------------------------------------------------------------------------
def get_carbon_explanation(utility: str, error: float) -> str:
    """Return a pedagogical explanation of WHY the carbon error occurs for this grid."""
    grid_region = UTILITY_GRID_MAP.get(utility, "PJM")
    direction = "overstates" if error > 0 else "understates"

    explanations = {
        "CAISO": (
            f"California's grid has massive solar penetration. Midday marginal emissions "
            f"drop to ~0.15 lbs CO2/kWh because solar and wind are on the margin — saving "
            f"a kWh of HVAC at noon displaces almost no fossil generation. But by 6 PM, solar "
            f"drops off and gas peakers fire up, pushing marginal emissions to ~0.85 lbs CO2/kWh. "
            f"The naive method uses the annual average MEF (~0.45 lbs/kWh), which {direction} "
            f"the carbon impact because it ignores this extreme diurnal swing. An HVAC measure "
            f"that saves mostly midday kWh avoids far less carbon than the naive method suggests."
        ),
        "NYISO": (
            f"New York's grid has moderate renewable penetration — enough to create meaningful "
            f"diurnal variation in marginal emissions (midday ~0.40 vs evening ~0.80 lbs CO2/kWh), "
            f"but less extreme than California. The naive average MEF (~0.55 lbs/kWh) {direction} "
            f"carbon avoidance because savings concentrated during midday hours displace cleaner "
            f"generation than the average implies. The error is smaller than CAISO but still "
            f"significant enough to matter for carbon reporting."
        ),
        "WEIS_RMPA": (
            f"Colorado's grid has a coal-heavy baseload that keeps marginal emissions high across "
            f"all hours (~0.60–0.95 lbs CO2/kWh). There is some midday solar relief, but the "
            f"baseline is elevated compared to coastal grids. The naive average MEF (~0.75 lbs/kWh) "
            f"{direction} carbon avoidance. The error is smaller in percentage terms because the "
            f"diurnal variation is less extreme, but the absolute emissions are higher — every kWh "
            f"saved in Colorado avoids more carbon than in California, regardless of time of day."
        ),
        "HECO": (
            f"Hawaii's grid is oil-dependent with emerging solar. Oil-fired generation provides "
            f"a high baseline marginal emissions rate (~0.70 lbs CO2/kWh average), with moderate "
            f"midday relief from growing rooftop and utility-scale solar (~0.55 midday vs ~0.85 "
            f"evening). The naive average MEF {direction} carbon avoidance. Hawaii's grid is "
            f"transitioning — as solar penetration increases, the diurnal MEF swing will widen, "
            f"making the naive method increasingly unreliable."
        ),
        "PJM": (
            f"PJM's gas/coal generation mix produces moderate diurnal variation in marginal "
            f"emissions (~0.50 midday vs ~0.80 evening, averaging ~0.65 lbs CO2/kWh). The naive "
            f"method {direction} carbon avoidance because it cannot distinguish between a kWh "
            f"saved at 2 PM (when gas combined-cycle or renewables may be marginal) versus 7 PM "
            f"(when less efficient gas peakers are dispatched). The error is meaningful for "
            f"carbon accounting and ESG reporting."
        ),
    }
    return explanations.get(grid_region, "")


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLOR_BASELINE = "#1f77b4"
COLOR_ACTUAL = "#2ca02c"
COLOR_SAVINGS = "#ff7f0e"
COLOR_CORRECT = "#2ca02c"
COLOR_NAIVE = "#d62728"
COLOR_ERROR = "#ff7f0e"
COLOR_TIER1 = "#636EFA"
COLOR_TIER2 = "#EF553B"
COLOR_FIXED = "#AB63FA"
COLOR_DELIVERY = "#636EFA"
COLOR_SUPPLY = "#00CC96"
COLOR_ON_PEAK = "#EF553B"
COLOR_OFF_PEAK = "#636EFA"
COLOR_CARBON_CORRECT = "#2d6a4f"
COLOR_CARBON_NAIVE = "#ae2012"
COLOR_CARBON_ERROR = "#e76f51"
COLOR_MEF = "#7209b7"


# ===========================================================================
# STREAMLIT APP
# ===========================================================================

st.markdown(
    """
    <style>
    .metric-card {
        background: #f8f9fa; border-radius: 8px; padding: 16px 20px;
        border-left: 4px solid #1f77b4; margin-bottom: 8px;
    }
    .metric-card.correct { border-left-color: #2ca02c; }
    .metric-card.naive { border-left-color: #d62728; }
    .metric-card.error { border-left-color: #ff7f0e; }
    .metric-card.correct-carbon { border-left-color: #2d6a4f; background: #f0f7f4; }
    .metric-card.naive-carbon { border-left-color: #ae2012; background: #fdf2f0; }
    .metric-card.error-carbon { border-left-color: #e76f51; background: #fef7f0; }
    .metric-label { font-size: 0.85rem; color: #666; margin-bottom: 2px; }
    .metric-value { font-size: 1.6rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Cost & Carbon Avoidance Calculator")

with st.expander("About this tool"):
    st.markdown(
        """
**This is a CMVP training tool** demonstrating a critical concept in M&V:
cost avoidance must be calculated by applying the full rate schedule to
**both** the adjusted baseline consumption **and** the actual (post-retrofit)
consumption independently, then subtracting.

The same principle applies to **carbon avoidance**: you can't multiply saved
kWh by an average grid emissions factor. You need hourly **marginal emissions
factors (MEF)** that reflect what generation source is actually on the margin
at each hour.

This tool generates synthetic 8760 hourly load profiles, applies real tariff
structures from 7 rate schedules, and quantifies the error from the naive
approach — for both **cost** and **carbon** avoidance.

The common shortcut — multiplying energy savings by an average rate — produces
errors because rate schedules are **non-linear**. Tiered rates, time-of-use
rates, seasonal differentials, and wholesale market dynamics all mean that
not every kWh has the same dollar value. The kWh you *avoid* consuming are
typically not at the average cost — they're at the **marginal** cost, which differs.

The same logic applies to emissions: grid marginal emissions vary hourly based
on which generation source is dispatched at the margin. Saving a kWh at noon
in California (when solar is on the margin) avoids almost no carbon, while
saving a kWh at 7 PM (when gas peakers are running) avoids a lot.
"""
    )

with st.expander("Simulation methodology"):
    st.markdown(
        """
#### Load Shape Generation
Baseline consumption is a **synthetic 8760 hourly profile** built from three components:
- **Baseload** (~0.3–0.5 kW constant): refrigeration, standby electronics, always-on loads
- **Occupancy-driven** (6AM–10PM): lighting, plug loads, cooking — modeled as a rectangular
  occupancy schedule scaled to represent typical residential activity
- **Weather-driven (HVAC)**: heating and cooling loads derived from a sinusoidal annual
  temperature model with diurnal swing (±8°F). Cooling activates above a cooling balance
  point, heating below a heating balance point, with magnitude proportional to the
  temperature difference (degree-hour proxy). Climate profiles approximate the 6 utility
  territories plus CAISO central CA.

Random noise (±5–10%) is added for realism. The profile is scaled so the annual total
matches the user-selected consumption level (default 8,000 kWh/year).

#### ECM Savings Shapes
Each ECM type produces a **different temporal savings profile**, which is the key to
why the naive method fails:
- **Lighting retrofit**: saves only during occupied hours (6AM–10PM), zero overnight, flat across seasons
- **HVAC upgrade**: saves proportional to the HVAC load component — large summer savings in
  cooling-dominated climates, large winter savings in heating-dominated climates, minimal in shoulder months
- **Envelope improvement**: similar temporal shape to HVAC (reduces weather-driven load)
- **Behavioral/controls**: blend of baseload reduction (40%) and occupancy reduction (60%)
- **Custom flat %**: uniform reduction every hour — this *is* the naive assumption made visible

All savings shapes are scaled so total annual kWh saved matches the user-selected percentage.

#### Tariff Data
Seven rate structures are encoded from published 2025–2026 tariff sheets:

| Tariff | Structure | Source |
|--------|-----------|--------|
| **PG&E E-1** | 2-tier inclining block, seasonal baseline by territory | PG&E rate schedule, eff. Jan 2026 |
| **SCE Schedule D** | 2-tier inclining block, daily baseline by region | SCE tariff book, eff. Jan 2026 |
| **Con Edison Rate I** | Seasonal delivery tiers (250 kWh breakpoint) + variable supply | NY PSC-approved rates, eff. Jan 2026 |
| **Xcel Energy RE-TOU** | Time-of-use (5–9PM weekday on-peak), no volume tiers | Xcel CO tariff, eff. Nov 2025 |
| **Hawaiian Electric Sch R** | Flat volumetric (base + ECRC + surcharges) | HECO Schedule R, PUC D&O 38164 |
| **PSEG NJ RS** | Flat delivery + seasonal BGS supply | PSEG tariff + BGS auction, eff. Jul 2025 |
| **CAISO Wholesale** | Simulated 8760 hourly LMP with duck-curve negative pricing | Synthetic curve modeled on CAISO SP15 day-ahead patterns |

The CAISO wholesale curve is **not historical data** — it is a synthetic hourly price model
calibrated to reproduce key features of real CAISO markets: negative midday prices in
spring/summer (solar oversupply, −$20 to −$60/MWh), evening ramp spikes ($60–100/MWh),
moderate overnight prices, and weekend discounts. Approximately 13% of hours are negative-priced.

#### Blended Rate Calculation (Naive Method)
For **retail tariffs**, the naive blended rate = total baseline energy charges ÷ total baseline kWh.
This is consumption-weighted and excludes fixed charges — it's what a practitioner would calculate
by dividing the bill by usage.

For **CAISO wholesale**, the naive rate = simple time-average of all 8,760 hourly prices
(unweighted). This is what appears in market reports as "average LMP" and is the number
a practitioner would grab to estimate savings value.

#### Marginal Emissions Factors
Grid emissions vary hourly based on which generation source is on the margin. This tool uses
synthetic 8760 marginal emissions factor (MEF) curves modeled on regional grid characteristics:

- **CAISO** (PG&E, SCE): Heavy solar penetration drives midday MEF near zero; evening gas peakers push MEF to ~0.85 lbs CO2/kWh
- **NYISO** (Con Edison): Moderate renewables, less extreme diurnal swing
- **WEIS/RMPA** (Xcel CO): Coal-heavy baseload keeps emissions high; some solar relief midday
- **HECO** (Hawaiian Electric): Oil-dependent grid with emerging solar
- **PJM** (PSEG NJ): Gas/coal mix with moderate diurnal variation

The naive approach uses the annual average MEF (lbs CO2/kWh). The correct approach uses hourly MEFs, which capture the temporal mismatch between when savings occur and what's actually on the margin.
"""
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Configuration")

utility = st.sidebar.selectbox("Utility / Rate Schedule", UTILITY_NAMES)
default_climate = UTILITY_CLIMATE_MAP[utility]
climate_options = list(CLIMATE_PROFILES.keys())
climate = st.sidebar.selectbox(
    "Climate Zone",
    climate_options,
    index=climate_options.index(default_climate),
)

annual_kwh = st.sidebar.slider(
    "Annual Consumption (kWh)",
    min_value=5000, max_value=20000, value=8000, step=500,
)

ecm_type = st.sidebar.selectbox(
    "ECM Type",
    ["Lighting retrofit", "HVAC upgrade", "Envelope improvement",
     "Behavioral/controls", "Custom flat %"],
)

savings_pct = st.sidebar.slider(
    "Savings Magnitude (%)",
    min_value=5, max_value=30, value=15, step=1,
) / 100.0

# ---------------------------------------------------------------------------
# Compute everything
# ---------------------------------------------------------------------------
baseline = simulate_baseline(climate, annual_kwh)
savings = compute_savings(baseline, ecm_type, savings_pct)
baseline_8760 = baseline["total"]
actual_8760 = baseline_8760 - savings

ca = compute_cost_avoidance(baseline_8760, actual_8760, utility)
carbon = compute_carbon_avoidance(baseline_8760, actual_8760, utility)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Load Shapes", "Bill Calculation",
    "Cost Avoidance — The Point", "Carbon Avoidance",
    "Compare All Utilities",
])

# ===================== TAB 1: Load Shapes =====================
with tab1:
    st.subheader("Load Profiles")

    # --- Typical summer and winter day ---
    # Pick a representative summer day (Jul 15 = day 195) and winter day (Jan 15 = day 14)
    summer_day_start = 195 * 24
    winter_day_start = 14 * 24

    col1, col2 = st.columns(2)
    for col, label, day_start in [
        (col1, "Typical Summer Day (Jul 15)", summer_day_start),
        (col2, "Typical Winter Day (Jan 15)", winter_day_start),
    ]:
        with col:
            hours_24 = list(range(24))
            bl_day = baseline_8760[day_start:day_start + 24]
            act_day = actual_8760[day_start:day_start + 24]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours_24, y=bl_day, mode="lines", name="Baseline",
                line=dict(color=COLOR_BASELINE, width=2),
            ))
            fig.add_trace(go.Scatter(
                x=hours_24, y=act_day, mode="lines", name="Actual (post-ECM)",
                line=dict(color=COLOR_ACTUAL, width=2),
            ))
            fig.add_trace(go.Scatter(
                x=hours_24, y=bl_day - act_day, mode="lines", name="Savings",
                line=dict(color=COLOR_SAVINGS, width=1.5, dash="dash"),
                fill="tozeroy", fillcolor="rgba(255,127,14,0.15)",
            ))
            fig.update_layout(
                title=label,
                xaxis_title="Hour of Day",
                yaxis_title="kW",
                height=350,
                margin=dict(l=40, r=20, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, width="stretch")

    # --- Savings Load Shape (8760) ---
    st.subheader("Savings Load Shape (8760 Hours)")
    st.caption(
        "This is the shape of avoided energy across the year. "
        "If this were flat, the naive method would work. It never is."
    )

    savings_8760 = baseline_8760 - actual_8760

    # Color by season for visual clarity
    colors_8760 = np.where(
        np.isin(MONTH_ARR, [6, 7, 8, 9]), "Summer", "Winter"
    )

    fig_sls = go.Figure()

    # Summer hours
    summer_mask = np.isin(MONTH_ARR, [6, 7, 8, 9])
    winter_mask = ~summer_mask
    hours_range = np.arange(HOURS_IN_YEAR)

    fig_sls.add_trace(go.Scattergl(
        x=hours_range[winter_mask],
        y=savings_8760[winter_mask],
        mode="markers",
        marker=dict(size=1.5, color="#1f77b4", opacity=0.4),
        name="Winter savings",
    ))
    fig_sls.add_trace(go.Scattergl(
        x=hours_range[summer_mask],
        y=savings_8760[summer_mask],
        mode="markers",
        marker=dict(size=1.5, color="#d62728", opacity=0.4),
        name="Summer savings",
    ))

    # Add monthly average line
    monthly_avg_savings = []
    monthly_mid_hours = []
    for i in range(12):
        s, e = MONTHLY_SLICES[i]
        monthly_avg_savings.append(savings_8760[s:e].mean())
        monthly_mid_hours.append((s + e) / 2)

    fig_sls.add_trace(go.Scatter(
        x=monthly_mid_hours,
        y=monthly_avg_savings,
        mode="lines+markers",
        name="Monthly avg savings",
        line=dict(color=COLOR_SAVINGS, width=3),
        marker=dict(size=8),
    ))

    # Naive assumption line (flat)
    naive_flat = savings_8760.mean()
    fig_sls.add_hline(
        y=naive_flat,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Naive assumption (flat): {naive_flat:.3f} kW",
        annotation_position="top left",
    )

    fig_sls.update_layout(
        xaxis_title="Hour of Year",
        yaxis_title="Savings (kW)",
        height=450,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # Add month labels on x-axis
    month_starts = [0]
    for i in range(11):
        month_starts.append(month_starts[-1] + HOURS_IN_MONTHS[i])
    fig_sls.update_xaxes(
        tickvals=month_starts,
        ticktext=MONTH_NAMES,
    )

    st.plotly_chart(fig_sls, width="stretch")

    st.markdown(
        f"**Peak savings:** {savings_8760.max():.3f} kW | "
        f"**Min savings:** {savings_8760.min():.3f} kW | "
        f"**Ratio:** {savings_8760.max() / max(savings_8760.min(), 0.001):.1f}× — "
        f"this non-uniformity is why you can't use an average rate."
    )

    st.divider()

    # --- Monthly consumption ---
    st.subheader("Monthly Consumption (kWh)")
    monthly_bl = []
    monthly_act = []
    for i in range(12):
        s, e = MONTHLY_SLICES[i]
        monthly_bl.append(baseline_8760[s:e].sum())
        monthly_act.append(actual_8760[s:e].sum())

    monthly_savings = [b - a for b, a in zip(monthly_bl, monthly_act)]

    fig_m = go.Figure()
    fig_m.add_trace(go.Bar(
        x=MONTH_NAMES, y=monthly_act, name="Actual Consumption",
        marker_color=COLOR_ACTUAL,
    ))
    fig_m.add_trace(go.Bar(
        x=MONTH_NAMES, y=monthly_savings, name="Energy Savings",
        marker_color=COLOR_SAVINGS,
    ))
    fig_m.update_layout(
        barmode="stack",
        yaxis_title="kWh",
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_m, width="stretch")

    st.caption(
        f"Annual baseline: {baseline_8760.sum():,.0f} kWh | "
        f"Annual actual: {actual_8760.sum():,.0f} kWh | "
        f"Energy savings: {baseline_8760.sum() - actual_8760.sum():,.0f} kWh "
        f"({savings_pct*100:.0f}%)"
    )

# ===================== TAB 2: Bill Calculation =====================
with tab2:
    st.subheader("Monthly Bill Comparison")

    bl_bills = ca["baseline_bills"]["monthly_bills"]
    act_bills = ca["actual_bills"]["monthly_bills"]

    # Side-by-side tables
    col_bl, col_act = st.columns(2)

    def _bill_table(bills, title):
        rows = []
        for b in bills:
            rows.append({
                "Month": b["month"],
                "Fixed ($)": f"{b['fixed']:.2f}",
                "Energy ($)": f"{b['energy']:.2f}",
                "Total ($)": f"{b['total']:.2f}",
            })
        rows.append({
            "Month": "ANNUAL",
            "Fixed ($)": f"{sum(b['fixed'] for b in bills):.2f}",
            "Energy ($)": f"{sum(b['energy'] for b in bills):.2f}",
            "Total ($)": f"{sum(b['total'] for b in bills):.2f}",
        })
        df = pd.DataFrame(rows)
        return df

    with col_bl:
        st.markdown("**Adjusted Baseline Bills**")
        st.dataframe(_bill_table(bl_bills, "Baseline"), hide_index=True, width="stretch")

    with col_act:
        st.markdown("**Actual (Post-ECM) Bills**")
        st.dataframe(_bill_table(act_bills, "Actual"), hide_index=True, width="stretch")

    # --- Stacked bar chart by component ---
    st.subheader("Bill Components by Month")

    # Determine which components are relevant for this utility
    def _build_component_fig(bills, title):
        fig = go.Figure()
        months = [b["month"] for b in bills]

        # Fixed charges
        fig.add_trace(go.Bar(
            x=months, y=[b["fixed"] for b in bills],
            name="Fixed", marker_color=COLOR_FIXED,
        ))

        if utility in ("PG&E E-1 (Tiered)", "SCE Schedule D (Tiered)"):
            fig.add_trace(go.Bar(
                x=months, y=[b["tier1"] for b in bills],
                name="Tier 1 Energy", marker_color=COLOR_TIER1,
            ))
            fig.add_trace(go.Bar(
                x=months, y=[b["tier2"] for b in bills],
                name="Tier 2 Energy", marker_color=COLOR_TIER2,
            ))
        elif utility == "Xcel Energy CO RE-TOU (Time-of-Use)":
            fig.add_trace(go.Bar(
                x=months, y=[b["on_peak"] for b in bills],
                name="On-Peak Energy", marker_color=COLOR_ON_PEAK,
            ))
            fig.add_trace(go.Bar(
                x=months, y=[b["off_peak"] for b in bills],
                name="Off-Peak Energy", marker_color=COLOR_OFF_PEAK,
            ))
        elif utility in ("Con Edison Rate I (Seasonal Delivery)",
                         "PSEG NJ RS (Flat + Seasonal Supply)"):
            fig.add_trace(go.Bar(
                x=months, y=[b["delivery"] for b in bills],
                name="Delivery", marker_color=COLOR_DELIVERY,
            ))
            fig.add_trace(go.Bar(
                x=months, y=[b["supply"] for b in bills],
                name="Supply", marker_color=COLOR_SUPPLY,
            ))
        else:
            fig.add_trace(go.Bar(
                x=months, y=[b["energy"] for b in bills],
                name="Energy", marker_color=COLOR_TIER1,
            ))

        fig.update_layout(
            barmode="stack", title=title,
            yaxis_title="$ / month", height=380,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _build_component_fig(bl_bills, "Baseline Bills"),
            width="stretch",
        )
    with c2:
        st.plotly_chart(
            _build_component_fig(act_bills, "Actual Bills"),
            width="stretch",
        )

    st.caption(
        f"Annual baseline bill: ${ca['cost_baseline']:,.2f} | "
        f"Annual actual bill: ${ca['cost_actual']:,.2f} | "
        f"Blended average rate: ${ca['avg_rate']:.4f}/kWh"
    )

# ===================== TAB 3: Cost Avoidance — The Point =====================
with tab3:
    st.subheader("Cost Avoidance Comparison")

    # Big metric cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""<div class="metric-card correct">
            <div class="metric-label">&#10003; Correct Cost Avoidance</div>
            <div class="metric-value">${ca['correct_avoidance']:,.2f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div class="metric-card naive">
            <div class="metric-label">&#10007; Naive Cost Avoidance</div>
            <div class="metric-value">${ca['naive_avoidance']:,.2f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m3:
        err_sign = "+" if ca["error"] >= 0 else ""
        st.markdown(
            f"""<div class="metric-card error">
            <div class="metric-label">&#9888; Error (Naive − Correct)</div>
            <div class="metric-value">{err_sign}${ca['error']:,.2f} ({err_sign}{ca['error_pct']:.1f}%)</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Methodology summary
    mc1, mc2 = st.columns(2)

    if utility == "CAISO Wholesale (Neg. Pricing)":
        naive_rate_description = (
            "Simple time-average of all 8760 hourly CAISO prices "
            "(unweighted — what a market report publishes as 'average LMP')"
        )
    else:
        naive_rate_description = (
            "Baseline energy cost ÷ baseline kWh = consumption-weighted "
            "blended rate (total energy charges from bill ÷ total usage, "
            "excluding fixed charges)"
        )

    with mc1:
        st.markdown(
            f"""**Correct Method** — apply full tariff to each load shape independently:
1. Run adjusted baseline 8760 through tariff → baseline bill: **${ca['cost_baseline']:,.2f}**
2. Run actual 8760 through tariff → actual bill: **${ca['cost_actual']:,.2f}**
3. Cost avoidance = ${ca['cost_baseline']:,.2f} − ${ca['cost_actual']:,.2f} = **${ca['correct_avoidance']:,.2f}**
"""
        )
    with mc2:
        st.markdown(
            f"""**Naive Method** — single blended rate × total kWh saved:
1. Energy savings: {ca['energy_savings_kwh']:,.0f} kWh
2. Blended rate: ${ca['avg_rate']:.4f}/kWh
   *({naive_rate_description})*
3. Naive avoidance = {ca['energy_savings_kwh']:,.0f} × ${ca['avg_rate']:.4f} = **${ca['naive_avoidance']:,.2f}**
"""
        )

    # Monthly comparison chart
    st.subheader("Monthly Cost Avoidance: Correct vs Naive")
    fig_ca = go.Figure()
    fig_ca.add_trace(go.Bar(
        x=MONTH_NAMES, y=ca["monthly_correct"],
        name="Correct", marker_color=COLOR_CORRECT,
    ))
    fig_ca.add_trace(go.Bar(
        x=MONTH_NAMES, y=ca["monthly_naive"],
        name="Naive", marker_color=COLOR_NAIVE, opacity=0.7,
    ))
    fig_ca.update_layout(
        barmode="group",
        yaxis_title="$ / month",
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_ca, width="stretch")

    # Explanation
    st.subheader("Why Does the Error Occur?")
    explanation = get_explanation(utility, ca["error"])
    st.info(explanation)

# ===================== TAB 4: Carbon Avoidance =====================
with tab4:
    st.subheader("Carbon Avoidance Comparison")
    st.caption(
        f"Grid region: **{carbon['grid_region']}** "
        f"({GRID_REGION_PARAMS[carbon['grid_region']]['description']})"
    )

    # Big metric cards — carbon style
    cm1, cm2, cm3 = st.columns(3)
    with cm1:
        st.markdown(
            f"""<div class="metric-card correct-carbon">
            <div class="metric-label">&#10003; Correct Carbon Avoided</div>
            <div class="metric-value">{carbon['correct_carbon']:,.0f} lbs CO&#8322;</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with cm2:
        st.markdown(
            f"""<div class="metric-card naive-carbon">
            <div class="metric-label">&#10007; Naive Carbon Avoided</div>
            <div class="metric-value">{carbon['naive_carbon']:,.0f} lbs CO&#8322;</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with cm3:
        cerr_sign = "+" if carbon["carbon_error"] >= 0 else ""
        st.markdown(
            f"""<div class="metric-card error-carbon">
            <div class="metric-label">&#9888; Error (Naive − Correct)</div>
            <div class="metric-value">{cerr_sign}{carbon['carbon_error']:,.0f} lbs ({cerr_sign}{carbon['carbon_error_pct']:.1f}%)</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Methodology boxes
    cmc1, cmc2 = st.columns(2)
    with cmc1:
        st.markdown(
            f"""**Correct Method** — hourly savings × hourly marginal emissions factor:
1. For each hour h: carbon_avoided_h = savings_h × MEF_h
2. MEF varies from ~{GRID_REGION_PARAMS[carbon['grid_region']]['midday_min']:.2f} lbs/kWh (midday) to ~{GRID_REGION_PARAMS[carbon['grid_region']]['evening_max']:.2f} lbs/kWh (evening)
3. Total = sum of all 8760 hourly values = **{carbon['correct_carbon']:,.0f} lbs CO2**
"""
        )
    with cmc2:
        st.markdown(
            f"""**Naive Method** — total kWh saved × annual average MEF:
1. Energy savings: {ca['energy_savings_kwh']:,.0f} kWh
2. Average MEF: {carbon['avg_mef']:.3f} lbs CO2/kWh
3. Naive carbon avoided = {ca['energy_savings_kwh']:,.0f} × {carbon['avg_mef']:.3f} = **{carbon['naive_carbon']:,.0f} lbs CO2**
"""
        )

    # Monthly carbon avoidance bar chart
    st.subheader("Monthly Carbon Avoidance: Correct vs Naive")
    fig_carb = go.Figure()
    fig_carb.add_trace(go.Bar(
        x=MONTH_NAMES, y=carbon["monthly_correct_carbon"],
        name="Correct (hourly MEF)", marker_color=COLOR_CARBON_CORRECT,
    ))
    fig_carb.add_trace(go.Bar(
        x=MONTH_NAMES, y=carbon["monthly_naive_carbon"],
        name="Naive (avg MEF)", marker_color=COLOR_CARBON_NAIVE, opacity=0.7,
    ))
    fig_carb.update_layout(
        barmode="group",
        yaxis_title="lbs CO2 / month",
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_carb, width="stretch")

    # Carbon explanation
    st.subheader("Why Does the Carbon Error Occur?")
    carbon_explanation = get_carbon_explanation(utility, carbon["carbon_error"])
    st.info(carbon_explanation)

    st.divider()

    # Dual-axis chart: savings load shape vs MEF for a typical summer day
    st.subheader("The Mismatch: Savings Shape vs Marginal Emissions (Typical Summer Day)")
    st.caption(
        "This chart reveals the core problem: when savings peak (midday HVAC), "
        "marginal emissions are at their lowest. The naive method misses this entirely."
    )

    grid_region = carbon["grid_region"]
    mef_8760 = GRID_MEF_CACHE[grid_region]

    # Use Jul 15 as typical summer day
    summer_day_h = 195 * 24
    hours_24 = list(range(24))
    savings_day = savings_8760[summer_day_h:summer_day_h + 24]
    mef_day = mef_8760[summer_day_h:summer_day_h + 24]

    fig_dual = go.Figure()

    # Savings on primary y-axis
    fig_dual.add_trace(go.Bar(
        x=hours_24, y=savings_day,
        name="Savings (kW)",
        marker_color=COLOR_SAVINGS,
        opacity=0.7,
        yaxis="y",
    ))

    # MEF on secondary y-axis
    fig_dual.add_trace(go.Scatter(
        x=hours_24, y=mef_day,
        mode="lines+markers",
        name="Marginal Emissions Factor (lbs CO2/kWh)",
        line=dict(color=COLOR_MEF, width=3),
        marker=dict(size=6),
        yaxis="y2",
    ))

    # Average MEF line
    fig_dual.add_trace(go.Scatter(
        x=hours_24, y=[carbon["avg_mef"]] * 24,
        mode="lines",
        name=f"Annual Avg MEF ({carbon['avg_mef']:.3f})",
        line=dict(color=COLOR_MEF, width=2, dash="dash"),
        yaxis="y2",
    ))

    fig_dual.update_layout(
        xaxis_title="Hour of Day",
        yaxis=dict(
            title=dict(text="Savings (kW)", font=dict(color=COLOR_SAVINGS)),
            tickfont=dict(color=COLOR_SAVINGS),
        ),
        yaxis2=dict(
            title=dict(text="MEF (lbs CO2/kWh)", font=dict(color=COLOR_MEF)),
            tickfont=dict(color=COLOR_MEF),
            anchor="x",
            overlaying="y",
            side="right",
            rangemode="tozero",
        ),
        height=450,
        margin=dict(l=60, r=60, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        barmode="overlay",
    )

    st.plotly_chart(fig_dual, width="stretch")

    st.markdown(
        f"**Key insight:** On this summer day, savings peak during midday hours "
        f"when the MEF is at its lowest (~{mef_day[12]:.2f} lbs/kWh at noon). "
        f"The naive method values those kWh at the annual average "
        f"({carbon['avg_mef']:.3f} lbs/kWh), overstating the carbon impact of "
        f"midday savings and understating the impact of evening savings."
    )

# ===================== TAB 5: Compare All Utilities =====================
with tab5:
    st.subheader("Cross-Utility Comparison")
    st.caption(
        "Same load profile and ECM applied across all 7 rate schedules."
    )

    comparison_rows = []
    for u_name in UTILITY_NAMES:
        u_ca = compute_cost_avoidance(baseline_8760, actual_8760, u_name)
        u_carbon = compute_carbon_avoidance(baseline_8760, actual_8760, u_name)
        comparison_rows.append({
            "Utility": u_name,
            "Correct Avoidance ($)": f"{u_ca['correct_avoidance']:,.2f}",
            "Naive Avoidance ($)": f"{u_ca['naive_avoidance']:,.2f}",
            "Cost Error (%)": f"{u_ca['error_pct']:+.1f}%",
            "Carbon Correct (lbs)": f"{u_carbon['correct_carbon']:,.0f}",
            "Carbon Naive (lbs)": f"{u_carbon['naive_carbon']:,.0f}",
            "Carbon Error (%)": f"{u_carbon['carbon_error_pct']:+.1f}%",
            "_error_pct": u_ca["error_pct"],
            "_correct": u_ca["correct_avoidance"],
            "_naive": u_ca["naive_avoidance"],
            "_error": u_ca["error"],
            "_carbon_error_pct": u_carbon["carbon_error_pct"],
            "_carbon_correct": u_carbon["correct_carbon"],
            "_carbon_naive": u_carbon["naive_carbon"],
        })

    df_comp = pd.DataFrame(comparison_rows)

    # Display table (without internal columns)
    st.dataframe(
        df_comp[["Utility", "Correct Avoidance ($)", "Naive Avoidance ($)",
                 "Cost Error (%)", "Carbon Correct (lbs)", "Carbon Naive (lbs)",
                 "Carbon Error (%)"]],
        hide_index=True,
        width="stretch",
    )

    # Bar chart of cost error %
    st.subheader("Cost Error Magnitude by Utility")
    fig_err = go.Figure()

    short_names = [n.split("(")[0].strip() for n in UTILITY_NAMES]

    fig_err.add_trace(go.Bar(
        x=short_names,
        y=df_comp["_error_pct"].tolist(),
        marker_color=[
            COLOR_ERROR if abs(e) > 1 else COLOR_CORRECT
            for e in df_comp["_error_pct"]
        ],
        text=[f"{e:+.1f}%" for e in df_comp["_error_pct"]],
        textposition="outside",
    ))
    fig_err.update_layout(
        yaxis_title="Cost Error (%)",
        height=400,
        margin=dict(l=40, r=20, t=20, b=80),
    )
    fig_err.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_err, width="stretch")

    # Bar chart of carbon error %
    st.subheader("Carbon Error Magnitude by Utility")
    fig_cerr = go.Figure()

    fig_cerr.add_trace(go.Bar(
        x=short_names,
        y=df_comp["_carbon_error_pct"].tolist(),
        marker_color=[
            COLOR_CARBON_ERROR if abs(e) > 1 else COLOR_CARBON_CORRECT
            for e in df_comp["_carbon_error_pct"]
        ],
        text=[f"{e:+.1f}%" for e in df_comp["_carbon_error_pct"]],
        textposition="outside",
    ))
    fig_cerr.update_layout(
        yaxis_title="Carbon Error (%)",
        height=400,
        margin=dict(l=40, r=20, t=20, b=80),
    )
    fig_cerr.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_cerr, width="stretch")

    # Correct vs Naive grouped bar — cost
    st.subheader("Cost Avoidance: Correct vs Naive")
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        x=short_names,
        y=df_comp["_correct"].tolist(),
        name="Correct",
        marker_color=COLOR_CORRECT,
    ))
    fig_cmp.add_trace(go.Bar(
        x=short_names,
        y=df_comp["_naive"].tolist(),
        name="Naive",
        marker_color=COLOR_NAIVE,
        opacity=0.7,
    ))
    fig_cmp.update_layout(
        barmode="group",
        yaxis_title="Annual Cost Avoidance ($)",
        height=400,
        margin=dict(l=40, r=20, t=20, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_cmp, width="stretch")

    # Correct vs Naive grouped bar — carbon
    st.subheader("Carbon Avoidance: Correct vs Naive")
    fig_ccmp = go.Figure()
    fig_ccmp.add_trace(go.Bar(
        x=short_names,
        y=df_comp["_carbon_correct"].tolist(),
        name="Correct (hourly MEF)",
        marker_color=COLOR_CARBON_CORRECT,
    ))
    fig_ccmp.add_trace(go.Bar(
        x=short_names,
        y=df_comp["_carbon_naive"].tolist(),
        name="Naive (avg MEF)",
        marker_color=COLOR_CARBON_NAIVE,
        opacity=0.7,
    ))
    fig_ccmp.update_layout(
        barmode="group",
        yaxis_title="Annual Carbon Avoidance (lbs CO2)",
        height=400,
        margin=dict(l=40, r=20, t=20, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_ccmp, width="stretch")

    st.info(
        "Hawaiian Electric (flat rate, no tiers, no TOU, no seasonal variation) "
        "shows near-zero cost error because the blended average rate equals the marginal "
        "rate at every hour. All other tariffs with tiers, TOU differentials, or "
        "seasonal pricing produce measurable cost errors — the naive method is unreliable "
        "whenever the rate schedule is non-linear. Carbon errors follow a different pattern: "
        "they depend on the grid's generation mix and renewable penetration, not the tariff structure. "
        "CAISO shows the largest carbon error due to extreme midday solar depression of marginal emissions."
    )
