# Recursive Dynamic CGE Model for Italy

This folder contains a **Recursive Dynamic Computable General Equilibrium (CGE) model** specifically designed for Italy's economy using **Pyomo optimization framework**. The model simulates economic impacts of EU Emissions Trading System (ETS) policies with period-by-period recursive dynamics.

## Files Structure

```
CGE_Italy/
├── main.py                      # Main model execution with runner function
├── recursive_cge_pyomo.py       # Core recursive dynamic CGE using Pyomo
├── run_model.py                 # Easy-to-use script to run scenarios
├── view_results.py              # Results viewer and analyzer
├── calibrate.py                 # Model calibration and data setup
├── simpleCGE.py                 # CGE equilibrium solver
├── firms.py                     # Firms' behavior module
├── household.py                 # Household behavior module
├── government.py                # Government sector module
├── aggregates.py                # Economic aggregates calculations
├── export_parameters.py         # Export all model parameters to Excel
├── export_math_parameters.py    # Export mathematical parameters
├── export_computation_params.py # Export core computation coefficients
├── view_parameters.py           # Quick parameter viewer
├── data/
│   └── SAM.xlsx                 # Social Accounting Matrix for Italy (22×22)
└── results/                     # All model outputs are saved here
    ├── cge_dynamic_results_*.xlsx     # Detailed dynamic results (Excel)
    ├── cge_summary_*.xlsx             # Summary results (Excel)
    ├── cge_model_parameters_*.xlsx    # Complete model parameters
    ├── cge_mathematical_parameters_*.xlsx  # Mathematical coefficients
    └── cge_computation_parameters_*.xlsx   # Core computation parameters
```

## How to Run the Model

### Option 1: Run All Scenarios (Recommended)

```bash
python main.py
```

This runs all three ETS scenarios automatically:

- **Business as Usual**: No additional climate measures
- **ETS1**: EU ETS for power and industry sectors  
- **ETS2**: EU ETS extension to transport sectors (from 2027)

### Option 2: Run Individual Scenarios

```bash
python run_model.py business_as_usual    # Run business as usual scenario
python run_model.py ets1                 # Run ETS1 (power + industry)
python run_model.py ets2                 # Run ETS2 (transport from 2027)
```

### Option 3: Run Recursive Dynamic Model Directly

```bash
python recursive_cge_pyomo.py
```

### Option 4: Run from Python Code

```python
from main import runner

# Run business as usual scenario
results = runner(sam_path='data/SAM.xlsx', scenario='business_as_usual', verbose=True, final_year=2050)

# Run ETS1 scenario  
results = runner(sam_path='data/SAM.xlsx', scenario='ets1', verbose=True, final_year=2050)

# Run ETS2 scenario
results = runner(sam_path='data/SAM.xlsx', scenario='ets2', verbose=True, final_year=2050)

## ETS Scenarios

The model implements three policy scenarios based on EU Emissions Trading System (ETS) extensions:

1. **Business as Usual**: Current policies continuation without additional climate measures
   - No carbon pricing
   - Natural economic growth with existing regulations

2. **ETS1 - Power and Industry**: EU ETS for electricity and industrial sectors
   - **Covered sectors**: Electricity, Industry, Other Energy
   - **Start year**: 2021 (immediate implementation)
   - **Carbon pricing**: €50/tCO2 initially, growing 5%/year
   - **Target**: 3% annual emissions reduction

3. **ETS2 - Transport Extension**: EU ETS extension to transport sectors
   - **Covered sectors**: Road Transport, Rail Transport, Air Transport, Water Transport
   - **Start year**: 2027 (delayed implementation)
   - **Carbon pricing**: €50/tCO2 from 2027, growing 4%/year
   - **Target**: 2.5% annual emissions reduction in transport

## Model Features

### **Technical Specifications**
- **Model Type**: Recursive Dynamic CGE with Pyomo optimization
- **Solver**: IPOPT (Interior Point Optimizer) for nonlinear programming
- **Solution Method**: Sequential period-by-period solving with state updates
- **Objective Function**: Social welfare maximization

### **Economic Structure**
- **11 Economic Sectors**: Agriculture, Industry, Electricity, Gas, Other Energy, Road Transport, Rail Transport, Air Transport, Water Transport, Other Transport, Other Sectors (14)
- **5 Italian Regions**: NW (Northwest), NE (Northeast), Centre, South, Islands (NUTS-1)
- **2 Production Factors**: Labour, Capital
- **Time Horizon**: 2021-2050 (30 periods)

### **Base Year Calibration (2021)**
- **GDP**: €1,782,000 million (€1.78 trillion)
- **Population**: 59.13 million people
- **GDP per capita**: €30,137
- **Labor Force**: 25.8 million workers
- **Capital Stock**: €8.5 trillion

### **Mathematical Framework**
- **Production Function**: CES (Constant Elasticity of Substitution)
  - `Yi = Ai[αLi×Li^ρ + αKi×Ki^ρ]^(1/ρ)` where σ=0.8, ρ=-0.25
- **Utility Function**: Cobb-Douglas for household preferences
- **Trade Specification**: Armington differentiation (σA=2.0)
- **Dynamic Evolution**: 
  - Capital: `K(t+1) = (1-δ)K(t) + I(t)` with δ=5%
  - Technology: `A(t) = A(0)(1+gA)^t` with gA=1.5%
  - Emissions: `Ei(t) = ci×Zi(t)×(1-γ)^t` with γ=2%

## Output Files

All results are automatically saved in the `results/` folder with timestamps:

### **Dynamic Results Files**
- **`cge_dynamic_results_[scenario]_[timestamp].xlsx`**: Complete time series (2021-2050)
  - Annual GDP, emissions, sectoral output
  - Regional economic indicators
  - Factor prices and quantities
  - Carbon prices and ETS impacts

### **Summary Files**  
- **`cge_summary_[scenario]_[timestamp].xlsx`**: Key indicators summary
  - Scenario comparison table
  - Growth rates and policy impacts
  - Final year outcomes

### **Parameter Documentation**
- **`cge_model_parameters_[timestamp].xlsx`**: Complete model parameters
- **`cge_mathematical_parameters_[timestamp].xlsx`**: Mathematical coefficients  
- **`cge_computation_parameters_[timestamp].xlsx`**: Core computation matrices

## Viewing Results

### Interactive Results Viewer
```bash
python view_results.py
```

### Parameter Viewer

```bash
python view_parameters.py
```

### Export Current Parameters

```bash
python export_parameters.py          # All model parameters
python export_math_parameters.py     # Mathematical parameters  
python export_computation_params.py  # Core computation coefficients
```

## Model Output Variables

The recursive dynamic CGE model generates comprehensive economic and environmental indicators:

### **Economic Variables**

- **GDP trajectories** by scenario (2021-2050)
- **Sectoral output** changes over time
- **Regional economic impacts** across Italian NUTS-1 regions
- **Factor prices** (wages, capital returns)
- **Investment and savings** dynamics
- **Consumption patterns** by sector and region

### **Environmental Variables**

- **Carbon emissions** trajectories by sector
- **Carbon prices** under ETS scenarios
- **Emissions intensity** changes
- **ETS compliance costs** and revenues

### **Policy Impact Variables**

- **ETS coverage** effects by sector
- **Carbon leakage** between sectors
- **Economic efficiency** of carbon pricing
- **Distributional impacts** across regions

## Requirements

The model requires the following Python packages:

- **pyomo** (≥6.0): Optimization framework
- **scipy**: Scientific computing
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **matplotlib**: Plotting and visualization
- **openpyxl**: Excel file handling

Install requirements:

```bash
pip install pyomo scipy numpy pandas matplotlib openpyxl
```

### **IPOPT Solver**

The model uses IPOPT solver for nonlinear optimization. This is typically installed with:

```bash
conda install -c conda-forge ipopt
```

## Technical Notes

### **Model Architecture**

- **Recursive Dynamics**: Each period solved sequentially using previous period as starting point
- **State Variables**: Capital stock, technology levels, emissions accumulate over time
- **Policy Implementation**: ETS constraints activated based on scenario and year
- **Convergence**: IPOPT solver with tolerance 1e-8 for robust solutions

### **Computational Features**

- **Mathematical Framework**: Nonlinear complementarity problem (NCP)
- **Variable Bounds**: Prevents negative quantities and ensures economic realism
- **Constraint Handling**: Automatic constraint activation for ETS scenarios
- **Error Handling**: Robust fallback methods for numerical issues

### **File Management**

- All file operations constrained to the `CGE_Italy` folder
- Results timestamped to avoid overwriting previous runs
- Automatic directory creation for outputs
- Excel format for easy analysis and sharing

## Quick Start Guide

1. **Navigate to CGE_Italy folder**
2. **Run the complete model**:

   ```bash
   python main.py
   ```

3. **Check results** in the `results/` folder
4. **View parameters** with:

   ```bash
   python view_parameters.py
   ```

## Example Results

After running the model, you'll find:

- **6 Excel result files** (2 per scenario)
- **3 parameter documentation files**
- **Complete mathematical coefficient matrices**
- **Time series data** for economic and environmental variables

## Model Integration

This CGE model is designed to integrate with the broader AdOpT-NET0 framework, providing economic impacts of energy system transformations and climate policies for Italy.

---

**Model Version**: Recursive Dynamic CGE v2.0  
**Last Updated**: September 2025  
**Framework**: Pyomo + IPOPT  
**Target**: EU ETS Policy Analysis for Italy
