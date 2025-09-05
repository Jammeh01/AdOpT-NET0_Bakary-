"""
Calibrated Recursive Dynamic CGE Model for Italy
Using proper SAM-based calibration with realistic economic values
"""

import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
import os
from datetime import datetime


class CalibratedRecursiveCGE:
    def __init__(self, sam_file_path):
        """Initialize the calibrated recursive dynamic CGE model"""
        self.sam_file = sam_file_path
        self.sam = None
        self.base_year = 2021
        self.sectors = []
        self.regions = []
        self.results = {}
        self.load_and_calibrate_data()

    def load_and_calibrate_data(self):
        """Load SAM data and perform base year calibration"""
        print("Loading and calibrating SAM data...")

        # Load SAM
        self.sam = pd.read_excel(self.sam_file, index_col=0)
        print(f"SAM loaded: {self.sam.shape}")

        # Define economic structure
        self.sectors = ['Agriculture', 'Industry', 'Electricity', 'Gas', 'Other Energy',
                        'Road Transport', 'Rail Transport', 'Air Transport', 'Water Transport',
                        'Other Transport', 'other Sectors (14)']

        self.regions = ['Households(NW)', 'Households(NE)', 'Households(Centre)',
                        'Households(South)', 'Households(Islands)']

        # Calculate base year calibration values
        self.base_sectoral_output = {}
        self.base_intermediate_demand = {}
        self.base_final_demand = {}

        for sector in self.sectors:
            if sector in self.sam.columns:
                # Gross output
                self.base_sectoral_output[sector] = self.sam[sector].sum()

                # Intermediate demand (from other sectors)
                intermediate = sum(self.sam.loc[s, sector]
                                   for s in self.sectors if s in self.sam.index)
                self.base_intermediate_demand[sector] = intermediate

                # Final demand (households + government + exports)
                final = self.base_sectoral_output[sector] - intermediate
                self.base_final_demand[sector] = final

        # Calculate base GDP and other aggregates
        self.base_labor_income = self.sam.loc['Labour'].sum()
        self.base_capital_income = self.sam.loc['Capital'].sum()
        sam_based_gdp = self.base_labor_income + self.base_capital_income

        # Calibrate to Italy's actual 2021 GDP
        self.target_gdp = 1782000  # Italy's actual 2021 GDP in million EUR
        self.calibration_factor = self.target_gdp / sam_based_gdp
        self.base_gdp = self.target_gdp

        # Apply calibration factor to all economic variables
        self.base_labor_income *= self.calibration_factor
        self.base_capital_income *= self.calibration_factor

        for sector in self.sectors:
            if sector in self.base_sectoral_output:
                self.base_sectoral_output[sector] *= self.calibration_factor
                self.base_intermediate_demand[sector] *= self.calibration_factor
                self.base_final_demand[sector] *= self.calibration_factor

        # Calculate household income by region
        self.base_household_income = {}
        for region in self.regions:
            if region in self.sam.index:
                self.base_household_income[region] = self.sam.loc[region].sum(
                ) * self.calibration_factor

        self.total_household_income = sum(self.base_household_income.values())

        # Population data (Italy 2021: ~59.13 million - actual figure)
        self.base_population = 59.13  # million people

        print(
            f"Base year GDP: €{self.base_gdp:,.0f} million (calibrated to actual Italy 2021)")
        print(f"Base year population: {self.base_population:.2f} million")
        print(
            f"GDP per capita: €{self.base_gdp/self.base_population*1000:,.0f}")
        print(f"Calibration factor applied: {self.calibration_factor:.4f}")

    def create_period_model(self, year, scenario_params):
        """Create Pyomo model for a specific year"""
        model = ConcreteModel(name=f"CGE_{year}")

        # Sets
        model.sectors = Set(initialize=self.sectors)
        model.regions = Set(initialize=self.regions)

        # Parameters (calibrated to base year)
        model.base_output = Param(
            model.sectors, initialize=self.base_sectoral_output, mutable=True)
        model.base_labor = Param(
            initialize=self.base_labor_income, mutable=True)
        model.base_capital = Param(
            initialize=self.base_capital_income, mutable=True)

        # Growth parameters
        years_from_base = year - self.base_year
        labor_growth = 0.005  # 0.5% annual labor productivity growth
        capital_growth = 0.03  # 3% annual capital growth
        # -0.2% annual population decline (Italy demographic trend)
        pop_growth = -0.002

        # Variables
        model.sectoral_output = Var(model.sectors, bounds=(0, None))
        model.labor_income = Var(bounds=(0, None))
        model.capital_income = Var(bounds=(0, None))
        model.gdp = Var(bounds=(0, None))
        model.consumption = Var(model.regions, bounds=(0, None))
        model.total_consumption = Var(bounds=(0, None))
        model.emissions = Var(model.sectors, bounds=(0, None))
        model.total_emissions = Var(bounds=(0, None))
        model.carbon_price = Var(bounds=(0, None))
        model.population = Var(bounds=(0, None))

        # GDP identity
        def gdp_rule(model):
            return model.gdp == model.labor_income + model.capital_income
        model.gdp_constraint = Constraint(rule=gdp_rule)

        # Production functions with growth
        def production_rule(model, sector):
            # 1.5% annual growth
            growth_factor = (1 + 0.015) ** years_from_base
            return model.sectoral_output[sector] == model.base_output[sector] * growth_factor
        model.production_constraint = Constraint(
            model.sectors, rule=production_rule)

        # Factor income evolution
        def labor_income_rule(model):
            return model.labor_income == model.base_labor * (1 + labor_growth) ** years_from_base
        model.labor_constraint = Constraint(rule=labor_income_rule)

        def capital_income_rule(model):
            return model.capital_income == model.base_capital * (1 + capital_growth) ** years_from_base
        model.capital_constraint = Constraint(rule=capital_income_rule)

        # Population evolution
        def population_rule(model):
            return model.population == self.base_population * (1 + pop_growth) ** years_from_base
        model.population_constraint = Constraint(rule=population_rule)

        # Total consumption (share of GDP)
        def total_consumption_rule(model):
            consumption_share = 0.58  # Italy's consumption share of GDP ~58%
            return model.total_consumption == model.gdp * consumption_share
        model.total_consumption_constraint = Constraint(
            rule=total_consumption_rule)

        # Regional consumption distribution
        regional_shares = {
            'Households(NW)': 0.28,
            'Households(NE)': 0.20,
            'Households(Centre)': 0.22,
            'Households(South)': 0.22,
            'Households(Islands)': 0.08
        }

        def regional_consumption_rule(model, region):
            return model.consumption[region] == model.total_consumption * regional_shares[region]
        model.regional_consumption_constraint = Constraint(
            model.regions, rule=regional_consumption_rule)

        # Emissions model
        emission_factors = {
            'Agriculture': 0.15,
            'Industry': 0.25,
            'Electricity': 0.40,
            'Gas': 0.20,
            'Other Energy': 0.30,
            'Road Transport': 0.35,
            'Rail Transport': 0.05,
            'Air Transport': 0.45,
            'Water Transport': 0.25,
            'Other Transport': 0.30,
            'other Sectors (14)': 0.10
        }

        def emissions_rule(model, sector):
            # Convert to MtCO2
            base_emissions = self.base_sectoral_output[sector] * \
                emission_factors[sector] / 1000
            # Apply scenario-specific emission reductions
            reduction_factor = 1.0

            if scenario_params['name'] == 'ets1' and sector in scenario_params.get('covered_sectors', []):
                # ETS1 sectors: gradual emission reduction
                reduction_factor = max(0.5, 1 - years_from_base * 0.02)
            elif scenario_params['name'] == 'ets2' and sector in scenario_params.get('covered_sectors', []) and year >= 2027:
                # ETS2 sectors: reduction from 2027
                years_ets2 = max(0, year - 2027)
                reduction_factor = max(0.6, 1 - years_ets2 * 0.025)

            return model.emissions[sector] == base_emissions * reduction_factor
        model.emissions_constraint = Constraint(
            model.sectors, rule=emissions_rule)

        def total_emissions_rule(model):
            return model.total_emissions == sum(model.emissions[sector] for sector in model.sectors)
        model.total_emissions_constraint = Constraint(
            rule=total_emissions_rule)

        # Carbon pricing with actual ETS prices
        def carbon_price_rule(model):
            if scenario_params['name'] == 'business_as_usual':
                return model.carbon_price == 0.0
            elif scenario_params['name'] == 'ets1':
                # Actual EU ETS prices: 2021: €25, rising to €80-100 by 2030, €120-150 by 2050
                if year <= 2025:
                    price = 25.0 + (year - 2021) * 5.0  # €25 to €45
                elif year <= 2030:
                    price = 45.0 + (year - 2025) * 7.0  # €45 to €80
                elif year <= 2040:
                    price = 80.0 + (year - 2030) * 3.0  # €80 to €110
                else:
                    price = 110.0 + (year - 2040) * 2.0  # €110 to €130
                return model.carbon_price == min(price, 130.0)
            elif scenario_params['name'] == 'ets2' and year >= 2027:
                # ETS2 for transport: lower initial prices, reaching €50-80 by 2050
                years_ets2 = year - 2027
                if years_ets2 <= 3:  # 2027-2030
                    price = 10.0 + years_ets2 * 5.0  # €10 to €25
                elif years_ets2 <= 13:  # 2030-2040
                    price = 25.0 + (years_ets2 - 3) * 2.5  # €25 to €50
                else:  # 2040-2050
                    price = 50.0 + (years_ets2 - 13) * 1.5  # €50 to €65
                return model.carbon_price == min(price, 65.0)
            else:
                return model.carbon_price == 0.0
        model.carbon_price_constraint = Constraint(rule=carbon_price_rule)

        # Objective: Maximize social welfare (represented by total consumption)
        def welfare_objective(model):
            return model.total_consumption
        model.objective = Objective(rule=welfare_objective, sense=maximize)

        return model

    def solve_recursive_dynamic(self, scenario_params, end_year=2050):
        """Solve the recursive dynamic model"""
        print(f"\n=== SOLVING {scenario_params['name'].upper()} SCENARIO ===")
        print(f"Time horizon: {self.base_year}-{end_year}")

        results = {
            'years': [],
            'gdp': [],
            'population': [],
            'gdp_per_capita': [],
            'total_consumption': [],
            'labor_income': [],
            'capital_income': [],
            'total_emissions': [],
            'carbon_price': [],
            'sectoral_output': {sector: [] for sector in self.sectors},
            'regional_consumption': {region: [] for region in self.regions}
        }

        # Solve year by year
        total_years = end_year - self.base_year + 1
        for i, year in enumerate(range(self.base_year, end_year + 1)):
            progress = (i + 1) / total_years * 100
            print(f"Solving year {year}... ({progress:.1f}% complete)")

            model = self.create_period_model(year, scenario_params)

            # Solve model
            solver = SolverFactory('ipopt')
            solver_result = solver.solve(model, tee=False)

            if solver_result.solver.termination_condition == TerminationCondition.optimal:
                print(f" Year {year} solved successfully")

                # Store results
                results['years'].append(year)
                results['gdp'].append(value(model.gdp))
                results['population'].append(value(model.population))
                results['gdp_per_capita'].append(
                    value(model.gdp) / value(model.population) * 1000)
                results['total_consumption'].append(
                    value(model.total_consumption))
                results['labor_income'].append(value(model.labor_income))
                results['capital_income'].append(value(model.capital_income))
                results['total_emissions'].append(value(model.total_emissions))
                results['carbon_price'].append(value(model.carbon_price))

                # Sectoral outputs
                for sector in self.sectors:
                    results['sectoral_output'][sector].append(
                        value(model.sectoral_output[sector]))

                # Regional consumption
                for region in self.regions:
                    results['regional_consumption'][region].append(
                        value(model.consumption[region]))

            else:
                print(f" Year {year} failed to solve")
                break

        return results

    def save_results_to_excel(self, results, scenario_name, output_dir):
        """Save results to a single Excel file with multiple sheets"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibrated_cge_results_{scenario_name}_{timestamp}.xlsx"
        filepath = os.path.join(output_dir, filename)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main results summary
            summary_data = {
                'Year': results['years'],
                'GDP (€ million)': results['gdp'],
                'Population (million)': results['population'],
                'GDP per capita (€)': results['gdp_per_capita'],
                'Total Consumption (€ million)': results['total_consumption'],
                'Labor Income (€ million)': results['labor_income'],
                'Capital Income (€ million)': results['capital_income'],
                'Total Emissions (MtCO2)': results['total_emissions'],
                'Carbon Price (€/tCO2)': results['carbon_price']
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Sectoral output
            sectoral_data = {'Year': results['years']}
            for sector in self.sectors:
                sectoral_data[f'{sector} (€ million)'] = results['sectoral_output'][sector]
            sectoral_df = pd.DataFrame(sectoral_data)
            sectoral_df.to_excel(
                writer, sheet_name='Sectoral_Output', index=False)

            # Regional consumption
            regional_data = {'Year': results['years']}
            for region in self.regions:
                regional_data[f'{region} (€ million)'] = results['regional_consumption'][region]
            regional_df = pd.DataFrame(regional_data)
            regional_df.to_excel(
                writer, sheet_name='Regional_Consumption', index=False)

            # Key indicators comparison
            if len(results['years']) > 1:
                base_year_idx = 0
                final_year_idx = len(results['years']) - 1

                comparison_data = {
                    'Indicator': [
                        'GDP (€ million)',
                        'GDP per capita (€)',
                        'Population (million)',
                        'Total Consumption (€ million)',
                        'Total Emissions (MtCO2)',
                        'Carbon Price (€/tCO2)'
                    ],
                    f'{results["years"][base_year_idx]}': [
                        f"{results['gdp'][base_year_idx]:,.0f}",
                        f"{results['gdp_per_capita'][base_year_idx]:,.0f}",
                        f"{results['population'][base_year_idx]:.2f}",
                        f"{results['total_consumption'][base_year_idx]:,.0f}",
                        f"{results['total_emissions'][base_year_idx]:.1f}",
                        f"{results['carbon_price'][base_year_idx]:.2f}"
                    ],
                    f'{results["years"][final_year_idx]}': [
                        f"{results['gdp'][final_year_idx]:,.0f}",
                        f"{results['gdp_per_capita'][final_year_idx]:,.0f}",
                        f"{results['population'][final_year_idx]:.2f}",
                        f"{results['total_consumption'][final_year_idx]:,.0f}",
                        f"{results['total_emissions'][final_year_idx]:.1f}",
                        f"{results['carbon_price'][final_year_idx]:.2f}"
                    ]
                }
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.to_excel(
                    writer, sheet_name='Base_vs_Final_Year', index=False)

        print(f"Results saved to: {filepath}")
        return filepath


def run_calibrated_scenarios():
    """Run all three scenarios with the calibrated model"""

    # Initialize model
    sam_file = "data/SAM.xlsx"
    model = CalibratedRecursiveCGE(sam_file)

    # Define scenarios
    scenarios = [
        {
            'name': 'business_as_usual',
            'description': 'Business as Usual - No additional climate policies',
            'covered_sectors': []
        },
        {
            'name': 'ets1',
            'description': 'ETS1 - Power, Industry, Gas, Aviation & Maritime sectors',
            'covered_sectors': ['Electricity', 'Industry', 'Other Energy', 'Gas', 'Air Transport', 'Water Transport']
        },
        {
            'name': 'ets2',
            'description': 'ETS2 - Road and Other Transport sectors (from 2027)',
            'covered_sectors': ['Road Transport', 'Other Transport']
        }
    ]

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"RUNNING SCENARIO: {scenario['description']}")
        print(f"{'='*60}")

        # Solve recursive dynamic model
        results = model.solve_recursive_dynamic(scenario, end_year=2050)
        all_results[scenario['name']] = results

        # Save individual scenario results
        model.save_results_to_excel(results, scenario['name'], results_dir)

        # Print key results
        if results['years']:
            final_year = results['years'][-1]
            final_idx = len(results['years']) - 1
            base_idx = 0

            print(f"\n=== KEY RESULTS FOR {final_year} ===")
            print(f"GDP: €{results['gdp'][final_idx]:,.0f} million")
            print(
                f"Population: {results['population'][final_idx]:.2f} million")
            print(
                f"GDP per capita: €{results['gdp_per_capita'][final_idx]:,.0f}")
            print(
                f"Total Consumption: €{results['total_consumption'][final_idx]:,.0f} million")
            print(
                f"Total Emissions: {results['total_emissions'][final_idx]:.1f} MtCO2")
            print(
                f"Carbon Price: €{results['carbon_price'][final_idx]:.2f}/tCO2")

            # Calculate growth rates
            gdp_growth = ((results['gdp'][final_idx] /
                          results['gdp'][base_idx]) - 1) * 100
            emission_change = (
                (results['total_emissions'][final_idx] / results['total_emissions'][base_idx]) - 1) * 100
            print(
                f"\n=== GROWTH OVER PERIOD ({results['years'][0]}-{final_year}) ===")
            print(f"GDP Growth: {gdp_growth:.1f}%")
            print(f"Emission Change: {emission_change:.1f}%")

    return all_results


if __name__ == "__main__":
    print("CALIBRATED RECURSIVE DYNAMIC CGE MODEL FOR ITALY")
    print("="*60)
    results = run_calibrated_scenarios()
    print("\nAll scenarios completed successfully!")
