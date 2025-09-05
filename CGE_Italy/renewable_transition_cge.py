"""
RENEWABLE ENERGY TRANSITION CGE MODEL - ITALY
Complete energy system redesign with realistic renewable transition dynamics
Electricity = Renewable Energy, Gas = Natural Gas, Other Fuels = Fossil Fuels
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import warnings
warnings.filterwarnings('ignore')


class RenewableTransitionCGE:
    def __init__(self):
        """Initialize the renewable energy transition CGE model for Italy"""

        # Time periods
        self.years = list(range(2021, 2051))
        self.base_year = 2021

        # Sectors (aligned with Italian economy)
        self.sectors = [
            'Agriculture', 'Industry', 'Electricity_Gen', 'Gas_Supply',
            'Oil_Refining', 'Road_Transport', 'Rail_Transport', 'Air_Transport',
            'Water_Transport', 'Other_Transport', 'Services', 'Households'
        ]

        # Italian NUTS-1 regions
        self.regions = ['North_West', 'North_East',
                        'Centre', 'South', 'Islands']

        # Energy carriers redefined
        self.energy_carriers = {
            'renewable_electricity': {
                'name': 'Renewable Electricity',
                'description': 'Solar PV, wind, hydro, biomass electricity',
                'co2_factor': 0.0,  # Zero emissions for renewables
                'base_price_2021': 90,  # EUR/MWh
                'learning_curve': 0.85,  # 15% cost reduction per doubling
                'resource_potential': 'High'
            },
            'natural_gas': {
                'name': 'Natural Gas',
                'description': 'Pipeline and LNG natural gas',
                'co2_factor': 0.202,  # tCO2/MWh
                'base_price_2021': 65,  # EUR/MWh
                'import_dependency': 0.95,  # 95% imported
                'volatility': 'High'
            },
            'fossil_fuels': {
                'name': 'Fossil Fuels',
                'description': 'Oil products, coal, diesel, gasoline',
                'co2_factor': 0.315,  # tCO2/MWh (weighted average)
                'base_price_2021': 95,  # EUR/MWh
                'import_dependency': 0.92,  # 92% imported
                'phase_out_target': 2050
            }
        }

        # Policy scenarios with realistic renewable transition
        self.scenarios = {
            'business_as_usual': {
                'carbon_price': self._get_carbon_price_trajectory('low'),
                'renewable_targets': self._get_renewable_targets('slow'),
                'fossil_phase_out': self._get_phase_out_schedule('gradual'),
                'energy_efficiency': 0.015,  # 1.5% per year
                'description': 'Slow transition, minimal policy intervention'
            },
            'ets_phase1': {
                'carbon_price': self._get_carbon_price_trajectory('medium'),
                'renewable_targets': self._get_renewable_targets('medium'),
                'fossil_phase_out': self._get_phase_out_schedule('moderate'),
                'energy_efficiency': 0.025,  # 2.5% per year
                'description': 'ETS coverage for power and industry'
            },
            'ets_full_coverage': {
                'carbon_price': self._get_carbon_price_trajectory('high'),
                'renewable_targets': self._get_renewable_targets('aggressive'),
                'fossil_phase_out': self._get_phase_out_schedule('rapid'),
                'energy_efficiency': 0.035,  # 3.5% per year
                'description': 'Full ETS coverage including transport and buildings'
            }
        }

        print(" Renewable Energy Transition CGE Model Initialized")
        print("Electricity = Renewable Energy (Solar, Wind, Hydro)")
        print(" Gas = Natural Gas (Fossil Fuel with CO₂ emissions)")
        print("Other Fuels = Fossil Fuels (Oil, Coal, Diesel)")

    def _get_carbon_price_trajectory(self, intensity):
        """Get realistic carbon price trajectories based on EU ETS"""
        base_prices = {
            'low': 25,      # EUR/tCO2 in 2021
            'medium': 35,   # EUR/tCO2 in 2021
            'high': 45      # EUR/tCO2 in 2021
        }

        growth_rates = {
            'low': 0.03,    # 3% annual growth
            'medium': 0.05,  # 5% annual growth
            'high': 0.07    # 7% annual growth
        }

        base_price = base_prices[intensity]
        growth_rate = growth_rates[intensity]

        trajectory = {}
        for year in self.years:
            years_from_base = year - 2021
            trajectory[year] = base_price * \
                (1 + growth_rate) ** years_from_base

            # Cap at realistic maximum
            if intensity == 'high' and year > 2040:
                trajectory[year] = min(
                    trajectory[year], 150)  # Max 150 EUR/tCO2

        return trajectory

    def _get_renewable_targets(self, ambition):
        """Get renewable electricity share targets"""
        # Italy's current renewable electricity share ~42% (2021)
        targets = {
            'slow': {2021: 0.42, 2030: 0.55, 2040: 0.68, 2050: 0.75},
            'medium': {2021: 0.42, 2030: 0.65, 2040: 0.80, 2050: 0.90},
            'aggressive': {2021: 0.42, 2030: 0.72, 2040: 0.88, 2050: 0.95}
        }

        target_points = targets[ambition]
        trajectory = {}

        # Interpolate between target points
        for year in self.years:
            if year in target_points:
                trajectory[year] = target_points[year]
            else:
                # Linear interpolation
                years_sorted = sorted(target_points.keys())
                for i in range(len(years_sorted)-1):
                    y1, y2 = years_sorted[i], years_sorted[i+1]
                    if y1 <= year <= y2:
                        share1, share2 = target_points[y1], target_points[y2]
                        trajectory[year] = share1 + \
                            (share2 - share1) * (year - y1) / (y2 - y1)
                        break

        return trajectory

    def _get_phase_out_schedule(self, speed):
        """Get fossil fuel phase-out schedules"""
        # Coal phase-out by 2030, oil reduction, gas transition
        schedules = {
            'gradual': {
                'coal_phase_out': 2035,
                'oil_reduction_rate': 0.02,  # 2% per year
                'gas_peak_year': 2030,
                'gas_decline_rate': 0.01    # 1% per year after peak
            },
            'moderate': {
                'coal_phase_out': 2030,
                'oil_reduction_rate': 0.035,  # 3.5% per year
                'gas_peak_year': 2027,
                'gas_decline_rate': 0.025   # 2.5% per year after peak
            },
            'rapid': {
                'coal_phase_out': 2028,
                'oil_reduction_rate': 0.05,  # 5% per year
                'gas_peak_year': 2025,
                'gas_decline_rate': 0.04    # 4% per year after peak
            }
        }

        return schedules[speed]

    def initialize_energy_demands(self):
        """Initialize sectoral energy demands by carrier (2021 baseline)"""

        # Italy's total final energy consumption ~260 TWh = 260,000,000 MWh (2021)
        total_energy_2021 = 260_000_000  # MWh

        # Realistic sectoral energy shares based on Italian energy statistics
        sectoral_shares = {
            'Industry': 0.28,           # Manufacturing, steel, chemicals
            'Road_Transport': 0.25,     # Cars, trucks, buses
            'Households': 0.22,         # Residential heating, appliances
            'Services': 0.12,           # Commercial buildings, offices
            'Air_Transport': 0.05,      # Aviation fuel
            'Agriculture': 0.03,        # Farm equipment, processing
            'Water_Transport': 0.02,    # Shipping, ports
            'Rail_Transport': 0.01,     # Trains (mostly electric)
            'Other_Transport': 0.01,    # Other transport modes
            'Electricity_Gen': 0.008,   # Power plant auxiliaries
            'Gas_Supply': 0.007,        # Gas infrastructure
            'Oil_Refining': 0.005       # Refinery operations
        }

        # Energy carrier shares by sector (2021 baseline)
        carrier_shares_2021 = {
            'Industry': {'renewable_electricity': 0.35, 'natural_gas': 0.38, 'fossil_fuels': 0.27},
            'Road_Transport': {'renewable_electricity': 0.02, 'natural_gas': 0.03, 'fossil_fuels': 0.95},
            'Households': {'renewable_electricity': 0.25, 'natural_gas': 0.45, 'fossil_fuels': 0.30},
            'Services': {'renewable_electricity': 0.42, 'natural_gas': 0.35, 'fossil_fuels': 0.23},
            'Air_Transport': {'renewable_electricity': 0.01, 'natural_gas': 0.01, 'fossil_fuels': 0.98},
            'Agriculture': {'renewable_electricity': 0.28, 'natural_gas': 0.12, 'fossil_fuels': 0.60},
            'Water_Transport': {'renewable_electricity': 0.02, 'natural_gas': 0.08, 'fossil_fuels': 0.90},
            'Rail_Transport': {'renewable_electricity': 0.78, 'natural_gas': 0.02, 'fossil_fuels': 0.20},
            'Other_Transport': {'renewable_electricity': 0.15, 'natural_gas': 0.15, 'fossil_fuels': 0.70},
            'Electricity_Gen': {'renewable_electricity': 0.42, 'natural_gas': 0.38, 'fossil_fuels': 0.20},
            'Gas_Supply': {'renewable_electricity': 0.25, 'natural_gas': 0.65, 'fossil_fuels': 0.10},
            'Oil_Refining': {'renewable_electricity': 0.15, 'natural_gas': 0.25, 'fossil_fuels': 0.60}
        }

        # Regional distribution (based on population and economic activity)
        regional_shares = {
            'North_West': 0.26,  # Lombardy, Piedmont, Liguria, Valle d'Aosta
            'North_East': 0.23,  # Veneto, Trentino-Alto Adige, Friuli-Venezia Giulia, Emilia-Romagna
            'Centre': 0.20,      # Tuscany, Umbria, Marche, Lazio
            'South': 0.23,       # Abruzzo, Molise, Campania, Puglia, Basilicata, Calabria
            'Islands': 0.08      # Sicily, Sardinia
        }

        # Initialize multi-dimensional energy demand matrix
        self.energy_demands = {}

        for scenario in self.scenarios.keys():
            self.energy_demands[scenario] = {}

            for year in self.years:
                # Calculate year-specific total energy with efficiency gains
                efficiency_factor = (
                    1 - self.scenarios[scenario]['energy_efficiency']) ** (year - 2021)
                total_energy_year = total_energy_2021 * efficiency_factor

                # GDP growth effect (Italy ~1.2% annual growth projected)
                gdp_growth_factor = (1.012) ** (year - 2021)
                total_energy_year *= gdp_growth_factor

                self.energy_demands[scenario][year] = pd.DataFrame(
                    index=pd.MultiIndex.from_product(
                        [self.sectors, self.regions]),
                    columns=['renewable_electricity',
                             'natural_gas', 'fossil_fuels'],
                    dtype=float
                )

                for sector in self.sectors:
                    sector_total = total_energy_year * sectoral_shares[sector]

                    for region in self.regions:
                        regional_total = sector_total * regional_shares[region]

                        # Apply renewable transition dynamics
                        renewable_target = self.scenarios[scenario]['renewable_targets'][year]
                        transition_factor = self._calculate_transition_factor(
                            year, sector, scenario)

                        # Calculate carrier shares with transition
                        base_shares = carrier_shares_2021[sector].copy()
                        adjusted_shares = self._apply_renewable_transition(
                            base_shares, renewable_target, transition_factor, sector, year
                        )

                        # Distribute energy across carriers
                        for carrier in ['renewable_electricity', 'natural_gas', 'fossil_fuels']:
                            self.energy_demands[scenario][year].loc[(sector, region), carrier] = \
                                regional_total * adjusted_shares[carrier]

        print("Energy demands initialized with renewable transition dynamics")
        print(f"Total 2021 baseline: {total_energy_2021:,.0f} MWh")
        print(f"Current renewable electricity share: 42%")
        print(f"2050 renewable targets: BAU 75%, ETS1 90%, ETS2 95%")

    def _calculate_transition_factor(self, year, sector, scenario):
        """Calculate sector-specific renewable transition factor"""

        # Sector transition speeds (relative to economy average)
        sector_speeds = {
            'Electricity_Gen': 1.5,    # Fastest transition
            'Rail_Transport': 1.3,     # Already mostly electric
            'Services': 1.2,           # Commercial buildings
            'Households': 1.0,         # Average speed
            'Industry': 0.9,           # Industrial processes
            'Road_Transport': 0.8,     # Electric vehicles
            'Agriculture': 0.7,        # Farm equipment
            'Other_Transport': 0.6,    # Mixed transport
            'Gas_Supply': 0.5,         # Gas infrastructure
            'Water_Transport': 0.4,    # Shipping (hardest to electrify)
            'Air_Transport': 0.3,      # Aviation (hardest to electrify)
            'Oil_Refining': 0.2        # Oil refining processes
        }

        base_speed = sector_speeds.get(sector, 1.0)

        # Policy intensity multipliers
        policy_multipliers = {
            'business_as_usual': 0.7,
            'ets_phase1': 1.0,
            'ets_full_coverage': 1.4
        }

        policy_mult = policy_multipliers[scenario]

        # Time progression (S-curve for technology adoption)
        years_from_start = year - 2021
        max_years = 2050 - 2021
        time_factor = 1 / \
            (1 + np.exp(-0.15 * (years_from_start - max_years/2)))

        return base_speed * policy_mult * time_factor

    def _apply_renewable_transition(self, base_shares, renewable_target, transition_factor, sector, year):
        """Apply renewable energy transition to carrier shares"""

        # Start with base shares
        new_shares = base_shares.copy()

        # Current renewable share for this sector
        current_renewable = base_shares['renewable_electricity']

        # Calculate target renewable share for sector
        if sector in ['Air_Transport', 'Water_Transport', 'Oil_Refining']:
            # Hard-to-electrify sectors: lower renewable targets
            sector_renewable_target = renewable_target * 0.3
        elif sector in ['Road_Transport', 'Rail_Transport']:
            # Transport: high electrification potential
            sector_renewable_target = renewable_target * 1.2
        elif sector in ['Services', 'Households']:
            # Buildings: medium-high potential
            sector_renewable_target = renewable_target * 1.0
        else:
            # Other sectors: follow general trend
            sector_renewable_target = renewable_target * 0.8

        # Limit to realistic maximum
        sector_renewable_target = min(sector_renewable_target, 0.95)

        # Apply transition with speed factor
        renewable_increase = (sector_renewable_target -
                              current_renewable) * transition_factor
        new_renewable_share = current_renewable + renewable_increase
        new_renewable_share = max(0, min(new_renewable_share, 0.95))

        # Redistribute remaining shares between gas and fossil fuels
        remaining_share = 1 - new_renewable_share

        if remaining_share > 0:
            # Apply phase-out schedules
            current_gas_share = base_shares['natural_gas']
            current_fossil_share = base_shares['fossil_fuels']
            current_total_fossil = current_gas_share + current_fossil_share

            if current_total_fossil > 0:
                # Natural gas transitions slower than other fossil fuels
                gas_preference = 0.6 if year < 2035 else 0.3

                new_gas_share = remaining_share * gas_preference
                new_fossil_share = remaining_share * (1 - gas_preference)
            else:
                new_gas_share = 0
                new_fossil_share = 0
        else:
            new_gas_share = 0
            new_fossil_share = 0

        # Final shares
        new_shares['renewable_electricity'] = new_renewable_share
        new_shares['natural_gas'] = new_gas_share
        new_shares['fossil_fuels'] = new_fossil_share

        # Normalize to ensure sum = 1
        total = sum(new_shares.values())
        if total > 0:
            for carrier in new_shares:
                new_shares[carrier] /= total

        return new_shares

    def calculate_energy_prices(self):
        """Calculate energy prices with carbon pricing and renewable learning curves"""

        self.energy_prices = {}

        for scenario in self.scenarios.keys():
            self.energy_prices[scenario] = {}
            carbon_prices = self.scenarios[scenario]['carbon_price']

            for year in self.years:
                carbon_price = carbon_prices[year]

                self.energy_prices[scenario][year] = pd.DataFrame(
                    index=pd.MultiIndex.from_product(
                        [self.sectors, self.regions]),
                    columns=['renewable_electricity',
                             'natural_gas', 'fossil_fuels'],
                    dtype=float
                )

                # Base prices with different dynamics
                for region in self.regions:
                    # Regional price factors
                    regional_multipliers = {
                        'North_West': 1.02,   # Industrial demand premium
                        'North_East': 1.00,   # Reference region
                        'Centre': 1.08,       # Rome premium
                        'South': 0.95,        # Lower demand/income
                        'Islands': 1.18       # Insularity costs
                    }

                    regional_mult = regional_multipliers[region]

                    for sector in self.sectors:
                        # Renewable electricity: learning curve + grid costs
                        years_experience = year - 2021
                        # 8% cost reduction per year
                        learning_factor = (1 - 0.08) ** years_experience
                        renewable_price = self.energy_carriers['renewable_electricity'][
                            'base_price_2021'] * learning_factor * regional_mult

                        # Natural gas: base price + carbon cost + supply security premium
                        gas_base = self.energy_carriers['natural_gas']['base_price_2021']
                        gas_carbon_cost = carbon_price * \
                            self.energy_carriers['natural_gas']['co2_factor']
                        # 15% premium by 2050 due to security
                        supply_premium = 1 + 0.15 * (year - 2021) / 29
                        gas_price = (gas_base * supply_premium +
                                     gas_carbon_cost) * regional_mult

                        # Fossil fuels: base price + carbon cost + scarcity premium
                        fossil_base = self.energy_carriers['fossil_fuels']['base_price_2021']
                        fossil_carbon_cost = carbon_price * \
                            self.energy_carriers['fossil_fuels']['co2_factor']
                        scarcity_premium = 1 + 0.25 * \
                            (year - 2021) / 29  # 25% premium by 2050
                        fossil_price = (
                            fossil_base * scarcity_premium + fossil_carbon_cost) * regional_mult

                        # Assign prices
                        self.energy_prices[scenario][year].loc[(
                            sector, region), 'renewable_electricity'] = renewable_price
                        self.energy_prices[scenario][year].loc[(
                            sector, region), 'natural_gas'] = gas_price
                        self.energy_prices[scenario][year].loc[(
                            sector, region), 'fossil_fuels'] = fossil_price

        print(
            "Energy prices calculated with renewable learning curves and carbon pricing")

    def calculate_co2_emissions(self):
        """Calculate CO2 emissions by sector, region, and carrier"""

        self.co2_emissions = {}

        for scenario in self.scenarios.keys():
            self.co2_emissions[scenario] = {}

            for year in self.years:
                self.co2_emissions[scenario][year] = pd.DataFrame(
                    index=pd.MultiIndex.from_product(
                        [self.sectors, self.regions]),
                    columns=['renewable_electricity',
                             'natural_gas', 'fossil_fuels'],
                    dtype=float
                )

                for sector in self.sectors:
                    for region in self.regions:
                        for carrier in ['renewable_electricity', 'natural_gas', 'fossil_fuels']:
                            energy_use = self.energy_demands[scenario][year].loc[(
                                sector, region), carrier]
                            emission_factor = self.energy_carriers[carrier]['co2_factor']

                            # Apply emission factor improvements over time
                            if carrier == 'fossil_fuels':
                                # Fossil fuel efficiency improvements
                                # 2% improvement per year
                                efficiency_improvement = 0.98 ** (year - 2021)
                                emission_factor *= efficiency_improvement
                            elif carrier == 'natural_gas':
                                # Natural gas efficiency improvements
                                # 1.5% improvement per year
                                efficiency_improvement = 0.985 ** (year - 2021)
                                emission_factor *= efficiency_improvement

                            # Calculate emissions in tCO2
                            emissions = energy_use * emission_factor / 1000  # Convert to tCO2

                            self.co2_emissions[scenario][year].loc[(
                                sector, region), carrier] = emissions

        print("CO₂ emissions calculated with emission factor improvements")

    def run_scenario_analysis(self):
        """Run complete scenario analysis for all three scenarios"""

        print("\nRUNNING RENEWABLE ENERGY TRANSITION ANALYSIS")
        print("="*70)

        # Initialize all components
        self.initialize_energy_demands()
        self.calculate_energy_prices()
        self.calculate_co2_emissions()

        # Print scenario results
        for scenario in self.scenarios.keys():
            self.print_scenario_results(scenario)

        print("\nRENEWABLE TRANSITION ANALYSIS COMPLETED")

    def print_scenario_results(self, scenario):
        """Print detailed results for a scenario"""

        scenario_name = scenario.replace('_', ' ').title()
        print(f"\n {scenario_name}:")
        print("-" * 50)

        # 2050 results
        year = 2050

        # Total energy consumption by carrier
        total_renewable = 0
        total_gas = 0
        total_fossil = 0
        total_emissions = 0

        for sector in self.sectors:
            for region in self.regions:
                total_renewable += self.energy_demands[scenario][year].loc[(
                    sector, region), 'renewable_electricity']
                total_gas += self.energy_demands[scenario][year].loc[(
                    sector, region), 'natural_gas']
                total_fossil += self.energy_demands[scenario][year].loc[(
                    sector, region), 'fossil_fuels']

                for carrier in ['renewable_electricity', 'natural_gas', 'fossil_fuels']:
                    total_emissions += self.co2_emissions[scenario][year].loc[(
                        sector, region), carrier]

        total_energy = total_renewable + total_gas + total_fossil

        print(f"2050 Energy Consumption:")
        print(
            f"  Renewable Electricity: {total_renewable:,.0f} MWh ({100*total_renewable/total_energy:.1f}%)")
        print(
            f"   Natural Gas: {total_gas:,.0f} MWh ({100*total_gas/total_energy:.1f}%)")
        print(
            f"  Fossil Fuels: {total_fossil:,.0f} MWh ({100*total_fossil/total_energy:.1f}%)")
        print(f"  Total Energy: {total_energy:,.0f} MWh")
        print(f"   Total CO₂: {total_emissions/1000000:.1f} MtCO₂")

        # Energy prices
        avg_renewable_price = np.mean([
            self.energy_prices[scenario][year].loc[(
                sector, region), 'renewable_electricity']
            for sector in self.sectors for region in self.regions
        ])
        avg_gas_price = np.mean([
            self.energy_prices[scenario][year].loc[(
                sector, region), 'natural_gas']
            for sector in self.sectors for region in self.regions
        ])
        avg_fossil_price = np.mean([
            self.energy_prices[scenario][year].loc[(
                sector, region), 'fossil_fuels']
            for sector in self.sectors for region in self.regions
        ])

        print(f"2050 Average Prices:")
        print(f"  Renewable Electricity: {avg_renewable_price:.0f} EUR/MWh")
        print(f"   Natural Gas: {avg_gas_price:.0f} EUR/MWh")
        print(f"  Fossil Fuels: {avg_fossil_price:.0f} EUR/MWh")


def main():
    """Main execution function"""

    print("ITALY RENEWABLE ENERGY TRANSITION CGE MODEL")
    print("=" * 60)
    print("Electricity = Renewable Energy (Solar, Wind, Hydro)")
    print(" Gas = Natural Gas (Fossil with CO₂)")
    print("Other Fuels = Fossil Fuels (Oil, Coal, Diesel)")
    print(" Carbon taxes drive transition to renewables")
    print("=" * 60)

    # Initialize and run model
    model = RenewableTransitionCGE()
    model.run_scenario_analysis()

    print(
        f"\n ANALYSIS COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Complete renewable energy transition simulation")
    print(" Realistic policy-driven shift from fossils to renewables")


if __name__ == "__main__":
    main()
