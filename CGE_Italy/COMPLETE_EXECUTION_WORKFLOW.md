# Complete CGE Model Execution Workflow for Italy (2021-2025)

## Overview
This workflow explains how to run the Italy CGE model with real Italian data calibration for 2021-2025 recursive dynamics, including ETS policy scenarios.

## Step-by-Step Execution Process

### 1. Data Calibration Phase

#### Real Italian Data (2021 Base Year):
- **GDP**: €1,782 trillion (nominal GDP 2021)
- **Population**: 59.13 million people
- **Sectors**: 11 economic sectors (Agriculture, Mining, Manufacturing, etc.)
- **Regions**: 5 NUTS-1 macro regions (North West, North East, Center, South, Islands)

#### SAM Table Calibration:
The Social Accounting Matrix (SAM.xlsx) contains:
- Inter-sectoral flows
- Household income and expenditure by region
- Government revenues and expenditures
- Trade flows (imports/exports)
- Investment and savings
- Energy flows and prices

#### Elasticity Parameters:
- **Substitution Elasticities**: Between energy types (electricity, gas, other)
- **Armington Elasticities**: Between domestic and imported goods
- **Income Elasticities**: Household demand responsiveness
- **Price Elasticities**: Energy demand responsiveness

### 2. Model Configuration

#### ETS Policy Scenarios:
1. **Baseline**: No ETS policy
2. **ETS1**: Current EU ETS coverage (Power, Industry, Aviation)
3. **ETS2**: Expanded ETS coverage (Buildings, Transport, excluding Rail)

#### Time Horizon:
- **Base Year**: 2021
- **Simulation Period**: 2021-2025 (5 years recursive dynamics)

### 3. Key Model Components

#### Sectors Modeled:
1. Agriculture, Forestry, Fishing
2. Mining and Quarrying
3. Manufacturing
4. Electricity, Gas, Steam
5. Water Supply, Waste Management
6. Construction
7. Trade, Transport, Accommodation
8. Information and Communication
9. Financial and Insurance
10. Real Estate, Professional Services
11. Public Administration, Other Services

#### Regional Structure:
1. North West (Lombardy, Piedmont, etc.)
2. North East (Veneto, Emilia-Romagna, etc.)
3. Center (Tuscany, Lazio, etc.)
4. South (Campania, Calabria, etc.)
5. Islands (Sicily, Sardinia)

### 4. Output Components Generated

#### A. Sectoral Demand (Monetary Values):
- Final demand by sector and year
- Intermediate demand flows
- Regional sectoral output

#### B. Energy Prices:
- **Electricity**: €/MWh by region and year
- **Natural Gas**: €/MWh by region and year
- **Other Energy**: Coal, oil products, renewables

#### C. CO2 Emissions:
- **Quantity**: Tons CO2 by sector and region
- **EU ETS Prices**: €/ton CO2 (EU policy-based)
- **Shadow Prices**: Model-derived carbon prices

#### D. Sectoral Energy Demand:
- **Electricity**: MWh by sector and region
- **Natural Gas**: MWh by sector and region
- Physical quantities and monetary values

#### E. Household Energy Demand (by Region):
- **Electricity**: MWh per household by NUTS-1 region
- **Natural Gas**: MWh per household by region
- Energy expenditure shares

#### F. Macroeconomic Indicators:
- **CPI**: Consumer Price Index (2021=100)
- **PPI**: Producer Price Index (2021=100)
- GDP deflator
- Regional price indices

### 5. Technical Implementation

#### Pyomo Optimization:
The model uses Pyomo mathematical programming with:
- **Solver**: IPOPT (Interior Point Optimizer)
- **Variables**: Prices, quantities, welfare
- **Constraints**: Market clearing, budget constraints, technology constraints
- **Objective**: Social welfare maximization

#### Recursive Dynamics:
Each year builds on the previous:
- Capital stock evolution
- Population growth
- Technology progress
- Policy implementation

### 6. Execution Commands

#### Prerequisites:
```python
# Install required packages
pip install pyomo
pip install ipopt  # or use conda-forge
pip install xlsxwriter openpyxl
```

#### Main Execution:
```python
# Run comprehensive simulation
python run_comprehensive_cge.py
```

### 7. Output Files Generated

#### Excel Workbook:
`Italy_CGE_Comprehensive_Results_[timestamp].xlsx` contains:

1. **Sectoral_Demand**: Monetary values by sector/year/scenario
2. **Energy_Prices**: Electricity, gas, other energy prices
3. **CO2_Emissions**: Quantities and prices by scenario
4. **Sectoral_Energy**: Energy demand by sector
5. **Household_Energy**: Regional household energy demand
6. **Macroeconomic**: CPI, PPI, GDP components
7. **Summary**: Key indicators comparison

#### JSON Files:
- `italy_results_baseline.json`
- `italy_results_ets1.json`
- `italy_results_ets2.json`

### 8. Model Validation

#### Calibration Checks:
- SAM balancing (row sums = column sums)
- GDP consistency with national accounts
- Energy balance validation
- Price level consistency

#### Policy Impact Validation:
- ETS1 vs Baseline: Moderate carbon price impact
- ETS2 vs ETS1: Expanded coverage effects
- Regional differential impacts
- Sectoral reallocation patterns

### 9. Interpretation Guide

#### Reading Results:
- **Positive values**: Increases relative to baseline
- **Negative values**: Decreases relative to baseline
- **Regional variations**: Reflect economic structure differences
- **Temporal patterns**: Show dynamic adjustment paths

#### Key Policy Insights:
- Carbon pricing effects on sectors
- Regional distributional impacts
- Energy transition pathways
- Macroeconomic adjustment costs

### 10. Advanced Analysis Options

#### Sensitivity Analysis:
- Elasticity parameter variations
- Carbon price scenarios
- Technology cost assumptions

#### Extended Scenarios:
- Different ETS coverage options
- Alternative policy instruments
- International cooperation scenarios

## Ready to Execute

The model is now fully configured with real Italian data and ready for comprehensive execution covering all requested outputs: sectoral demand, energy prices, CO2 emissions, household energy demand by region, and macroeconomic indicators.
