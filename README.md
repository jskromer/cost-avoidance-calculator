# Cost Avoidance Calculator — CMVP Training Tool

A Streamlit app demonstrating why cost avoidance must be calculated by applying the full rate schedule to **both** the adjusted baseline consumption **and** the actual consumption independently — not by multiplying energy savings by a blended average rate.

**Live:** [cost-avoidance-calculator on Streamlit Cloud](https://cost-avoidance-calculator-m2n5xq9kudmg6htwtutsmn.streamlit.app)

## The Teaching Point

In M&V, cost avoidance = what you *would have paid* minus what you *actually paid*. The correct calculation requires running both load profiles through the full tariff. The common shortcut — `saved kWh x average $/kWh` — produces errors because rate schedules are non-linear:

- **Tiered rates**: savings come off the top (expensive) tier, but the blended rate averages across all tiers
- **Time-of-use rates**: the value of a saved kWh depends on *when* it was saved
- **Wholesale markets**: negative midday prices mean some "savings" actually cost money
- **Flat rates**: the one case where the shortcut works (included as a control)

## Features

- **Synthetic 8760 load shapes** built from baseload + occupancy + weather-driven (HVAC) components with climate-specific temperature models
- **5 ECM savings profiles** with distinct temporal shapes (lighting, HVAC, envelope, behavioral, flat %)
- **7 rate structures** from published 2025-2026 tariffs:

| Tariff | Type | Key Feature |
|--------|------|-------------|
| PG&E E-1 | 2-tier inclining block | Seasonal baseline by territory |
| SCE Schedule D | 2-tier inclining block | Daily baseline by region |
| Con Edison Rate I | Seasonal delivery tiers | 250 kWh summer breakpoint |
| Xcel Energy RE-TOU | Time-of-use | 5-9 PM weekday on-peak |
| Hawaiian Electric Sch R | Flat | No tiers/TOU/seasons (control) |
| PSEG NJ RS | Flat + seasonal supply | BGS auction pricing |
| CAISO Wholesale | Hourly LMP (synthetic) | Negative midday pricing |

- **Side-by-side comparison**: correct cost avoidance vs. naive method with error quantification
- **Cross-utility comparison**: all 7 tariffs at once showing error magnitude
- **Savings load shape visualization**: 8760 scatter plot revealing the non-uniformity that drives the error

## Running Locally

```bash
cd cost-avoidance-calculator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tabs

1. **Load Shapes** — baseline vs. actual daily profiles (summer/winter), savings load shape (8760), monthly consumption
2. **Bill Calculation** — monthly bill tables and component breakdowns for both load profiles
3. **Cost Avoidance — The Point** — correct vs. naive cost avoidance, error metrics, methodology descriptions, and tariff-specific explanations
4. **Compare All Utilities** — all 7 tariffs side-by-side with error bar charts

## Context

Built as a training exercise for the Certified Measurement & Verification Professional (CMVP) program. The tool reinforces IPMVP cost avoidance principles: savings are determined by comparing what energy *would have cost* under adjusted baseline conditions to what it *actually cost*, using the full applicable rate schedule for each.

## Author

Steve Kromer, PE, CMVP #1 — [Counterfactual Designs](https://counterfactualdesigns.com)
