# main.py - CGE Main Execution
import scipy.optimize as opt
import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
import json
import os

# Import CGE modules
from calibrate import model_data, parameters
from simpleCGE import solve_threeme_equilibrium, run_policy_simulation, extract_linking_variables
import government as gov


def check_square(sam):
    """Test whether the SAM is a square matrix."""
    if hasattr(sam, 'shape'):
        if sam.shape[0] != sam.shape[1]:
            raise ValueError(
                f"SAM is not square. It has {sam.shape[0]} rows and {sam.shape[1]} columns")
    print("SAM is square: PASSED")


def row_col_equal(sam):
    """Test whether row sums and column sums of the SAM are equal."""
    if hasattr(sam, 'sum'):
        row_sum = sam.sum(axis=0)
        col_sum = sam.sum(axis=1)
        np.testing.assert_allclose(row_sum, col_sum, rtol=1e-3)
        print("Row/column sums equal: PASSED")


def runner(sam_path=None, scenario='business_as_usual', verbose=True, final_year=2050):
    """
    Main CGE model runner with recursive dynamics and Pyomo optimization.

    Args:
        sam_path: Path to SAM Excel file
        scenario: Policy scenario ('business_as_usual', 'ets1', 'ets2')
        verbose: Print detailed output
        final_year: Final simulation year

    Returns:
        dict: Complete simulation results
    """

    print("=" * 60)
    print("RECURSIVE DYNAMIC CGE MODEL FOR ITALY")
    print("=" * 60)
    print(f"Scenario: {scenario}")
    print(f"Time horizon: 2021-{final_year}")

    # Load SAM data
    if sam_path and os.path.exists(sam_path):
        sam = pd.read_excel(sam_path, index_col=0, header=0)
        if verbose:
            print(f"SAM loaded from: {sam_path}")
            print(f"SAM dimensions: {sam.shape}")
    else:
        # Create mock SAM structure
        sam = pd.DataFrame()
        if verbose:
            print("Using default SAM structure (no file provided)")

    # Define model structure based on actual SAM data
    if not sam.empty:
        # Extract sectors from SAM (excluding factors and institutions)
        all_accounts = sam.index.tolist()
        ind = [acc for acc in all_accounts if acc in [
            'Agriculture', 'Industry', 'Electricity', 'Gas', 'Other Energy',
            'Road Transport', 'Rail Transport', 'Air Transport', 'Water Transport',
            'Other Transport', 'Services'  # Changed from 'other Sectors (14)'
        ]]
    else:
        ind = ['Agriculture', 'Industry', 'Electricity', 'Gas', 'Other Energy',
               'Road Transport', 'Rail Transport', 'Air Transport', 'Water Transport',
               'Other Transport', 'Services']  # Changed from 'other Sectors (14)'

    h = ['Labour', 'Capital']
    regions = ['NW', 'NE', 'Centre', 'South', 'Islands']
    
    # Institutional agents (separate from economic sectors)
    institutions = ['Households', 'Government', 'Investment', 'Rest_of_World']
    
    # Household regional structure for ETS2 impact analysis
    household_regions = {
        'NW': {'population_share': 0.261, 'energy_intensity': 1.05},    # North-West
        'NE': {'population_share': 0.193, 'energy_intensity': 0.95},    # North-East  
        'Centre': {'population_share': 0.198, 'energy_intensity': 1.02}, # Centre
        'South': {'population_share': 0.234, 'energy_intensity': 0.88},  # South
        'Islands': {'population_share': 0.114, 'energy_intensity': 0.75} # Islands
    }

    # ETS sector definitions based on real EU ETS coverage
    ets1_sectors = ['Electricity', 'Industry', 'Other Energy',
                    'Air Transport', 'Water Transport']  # Power, industry, aviation, maritime
    ets2_sectors = ['Road Transport', 'Rail Transport', 'Other Transport', 
                    'Services']  # Remaining transport + commercial buildings from 2027
    
    # Household energy consumption patterns by ETS2 coverage with switching options
    household_energy_impact = {
        'direct_consumption': 0.22,  # 22% of total energy consumption
        'transport_share': 0.45,     # 45% of household energy for transport
        'heating_share': 0.35,       # 35% for heating (affected by ETS2 fuel prices)
        'electricity_share': 0.20,   # 20% for appliances and electricity
        'ets2_exposure': 0.65,       # 65% of household energy affected by ETS2
        'switching_preferences': {   # NEW: Fuel switching based on cost and preferences
            'transport': {
                'current_mix': {'fossil': 0.85, 'gas': 0.10, 'renewable': 0.05},
                'target_2050_bau': {'fossil': 0.65, 'gas': 0.15, 'renewable': 0.20},
                'target_2050_ets2': {'fossil': 0.15, 'gas': 0.15, 'renewable': 0.70},
                'switching_elasticity': 0.8,  # High responsiveness to cost differences
                'switching_barriers': ['Vehicle costs', 'Charging infrastructure', 'Range anxiety'],
                'policy_support_needed': ['Purchase incentives', 'Charging network', 'Scrappage schemes']
            },
            'heating': {
                'current_mix': {'fossil': 0.35, 'gas': 0.45, 'renewable': 0.20},
                'target_2050_bau': {'fossil': 0.25, 'gas': 0.50, 'renewable': 0.25},
                'target_2050_ets2': {'fossil': 0.05, 'gas': 0.25, 'renewable': 0.70},
                'switching_elasticity': 0.6,  # Medium responsiveness (retrofit barriers)
                'switching_barriers': ['Upfront costs', 'Building characteristics', 'Heat pump efficiency'],
                'policy_support_needed': ['Heat pump subsidies', 'Retrofit programs', 'Technical support']
            },
            'electricity': {
                'current_mix': {'fossil': 0.20, 'gas': 0.38, 'renewable': 0.42},
                'target_2050_bau': {'fossil': 0.15, 'gas': 0.35, 'renewable': 0.50},
                'target_2050_ets2': {'fossil': 0.02, 'gas': 0.15, 'renewable': 0.83},
                'switching_elasticity': 0.9,  # Very high (grid-level switching)
                'switching_barriers': ['Grid stability', 'Storage', 'Intermittency'],
                'policy_support_needed': ['Grid modernization', 'Storage investment', 'Smart grid technology']
            }
        }
    }
    
    # Energy carrier switching model with ETS-driven costs
    energy_carriers_switching = {
        'Renewable Electricity': {
            'base_price_2021': 85,      # EUR/MWh
            'co2_factor': 0.0,          # tCO2/MWh
            'ets_exempt': True,
            'switching_preference': 1.0,  # Highest preference
            'learning_rate': 0.08,       # 8% annual cost reduction
            'max_share_2050': 0.95,     # 95% technical potential
            'switching_speed': 'Fast',   # Quick adoption when cost-competitive
            'infrastructure_req': 'High' # Grid modernization needed
        },
        'Natural Gas': {
            'base_price_2021': 65,      # EUR/MWh
            'co2_factor': 0.202,        # tCO2/MWh
            'ets_subject': True,
            'switching_preference': 0.6,  # Medium preference (transition fuel)
            'learning_rate': 0.02,       # 2% annual cost change
            'max_share_2050': 0.40,     # Transition fuel role
            'switching_speed': 'Medium', # Moderate switching
            'infrastructure_req': 'Low'  # Existing pipeline network
        },
        'Fossil Fuels': {
            'base_price_2021': 95,      # EUR/MWh
            'co2_factor': 0.315,        # tCO2/MWh
            'ets_subject': True,
            'switching_preference': 0.2,  # Low preference
            'learning_rate': -0.01,      # -1% (increasing costs)
            'max_share_2050': 0.05,     # Phase-out to 5%
            'switching_speed': 'Fast',   # Quick switching away when expensive
            'infrastructure_req': 'Low'  # Existing but declining
        },
        'Green Hydrogen': {
            'base_price_2021': 150,     # EUR/MWh (emerging)
            'co2_factor': 0.0,          # tCO2/MWh
            'ets_exempt': True,
            'switching_preference': 0.8,  # High for hard-to-abate sectors
            'learning_rate': 0.12,       # 12% annual cost reduction
            'max_share_2050': 0.25,     # 25% potential
            'switching_speed': 'Slow',   # Technology development needed
            'infrastructure_req': 'Very High' # New infrastructure needed
        }
    }

    # Initialize model
    if verbose:
        print("\nInitializing recursive dynamic CGE model...")
        print(f"Economic Sectors: {len(ind)}")
        print(f"Institutional Agents: {len(institutions)}")
        print(f"ETS1 sectors (covered now): {ets1_sectors}")
        print(f"ETS2 sectors (from 2027): {ets2_sectors}")
        print("Aviation & Maritime now in ETS1 for immediate carbon pricing")
        print(f"Household ETS2 exposure: {household_energy_impact['ets2_exposure']*100:.0f}% of energy consumption")
        print(f"Energy carriers with switching: {len(energy_carriers_switching)}")
        print("Dynamic fuel switching based on ETS costs and preferences enabled")

    d = model_data(sam, h, ind, regions)
    p = parameters(d, ind)

    # Add switching parameters to model data
    d.switching_preferences = household_energy_impact['switching_preferences']
    d.energy_carriers = energy_carriers_switching
    
    if verbose:
        print(f"Base year GDP: €{d.base_gdp:,.0f} million")
        print(f"Number of sectors: {len(ind)}")
        print(f"Number of regions: {len(regions)}")
        print(f"Household final consumption: 58% of GDP")
        print(f"Household energy consumption: 22% of total energy")
        print("Fuel switching elasticities: Transport (0.8), Heating (0.6), Electricity (0.9)")

    # Define the three policy scenarios with household ETS2 impact and fuel switching
    policy_scenarios = {
        'business_as_usual': {
            'name': 'Business as Usual',
            'description': 'Current policies continuation with limited switching incentives',
            'carbon_price_2021': 25,  # €/tCO2
            'carbon_price_growth': 0.02,  # 2% annual growth
            'ets_coverage': [],  # No additional ETS coverage
            'emission_reduction_target': 0.20,  # 20% reduction by 2050
            'switching_policy': 'Limited',
            'renewable_target_2050': 0.50,  # 50% renewable electricity
            'household_impact': {
                'energy_price_increase': 0.15,  # 15% increase by 2050
                'transport_cost_increase': 0.10,  # 10% increase
                'heating_cost_increase': 0.12,   # 12% increase
                'welfare_impact': -0.05,          # -5% welfare change
                'switching_support': 'Minimal',
                'fuel_switching': {
                    'transport_electrification': 0.20,  # 20% by 2050
                    'heating_electrification': 0.25,   # 25% by 2050
                    'renewable_adoption': 0.50        # 50% grid renewable by 2050
                }
            }
        },
        'ets1': {
            'name': 'ETS1 - Power, Industry, Aviation & Maritime with Switching',
            'description': 'EU ETS extension with targeted fuel switching incentives for covered sectors',
            'carbon_price_2021': 50,  # €/tCO2
            'carbon_price_growth': 0.05,  # 5% annual growth
            'ets_coverage': ets1_sectors,
            'emission_reduction_target': 0.55,  # 55% reduction by 2050
            'start_year': 2021,  # Already covered
            'aviation_maritime_priority': True,  # High-emission transport sectors
            'switching_policy': 'Targeted',
            'renewable_target_2050': 0.66,  # 66% renewable electricity
            'household_impact': {
                'energy_price_increase': 0.25,  # 25% increase by 2050 (indirect)
                'transport_cost_increase': 0.18,  # 18% increase (aviation fuel)
                'heating_cost_increase': 0.22,   # 22% increase (electricity/gas)
                'welfare_impact': -0.08,          # -8% welfare change
                'adaptation_support': 0.15,       # 15% of carbon revenue for households
                'switching_support': 'Moderate',
                'fuel_switching': {
                    'transport_electrification': 0.25,  # 25% by 2050 (indirect incentives)
                    'heating_electrification': 0.35,   # 35% by 2050 (electricity price signals)
                    'renewable_adoption': 0.66        # 66% grid renewable by 2050
                }
            },
            'sectoral_switching': {
                'Industry': {
                    'switching_speed': 'Fast',
                    'target_renewable_share': 0.60,
                    'hydrogen_adoption': 0.15,
                    'switching_barriers_addressed': ['Carbon pricing', 'Technology support']
                },
                'Electricity': {
                    'switching_speed': 'Very Fast',
                    'target_renewable_share': 0.80,
                    'storage_investment': 'High',
                    'switching_barriers_addressed': ['Grid modernization', 'Storage deployment']
                },
                'Aviation': {
                    'switching_speed': 'Medium',
                    'sustainable_fuel_share': 0.40,
                    'hydrogen_potential': 0.10,
                    'switching_barriers_addressed': ['SAF production', 'Infrastructure development']
                },
                'Maritime': {
                    'switching_speed': 'Medium',
                    'green_fuel_share': 0.45,
                    'hydrogen_ammonia_potential': 0.35,
                    'switching_barriers_addressed': ['Fuel infrastructure', 'International coordination']
                }
            }
        },
        'ets2': {
            'name': 'ETS2 - Full Economy with Comprehensive Switching Support',
            'description': 'EU ETS extension with comprehensive fuel switching support for all sectors and households',
            'carbon_price_2021': 25,  # €/tCO2 (starts in 2027)
            'carbon_price_2027': 40,  # €/tCO2 (starting price for transport/buildings)
            'carbon_price_growth': 0.08,  # 8% annual growth
            'ets_coverage': ets2_sectors,
            'emission_reduction_target': 0.65,  # 65% reduction by 2050
            'start_year': 2027,  # Transport & buildings ETS starts in 2027
            'household_direct_coverage': True,  # NEW: Households directly affected
            'switching_policy': 'Comprehensive',
            'renewable_target_2050': 0.83,  # 83% renewable electricity
            'household_impact': {
                # Direct ETS2 coverage of household energy consumption with switching support
                'energy_price_increase': 0.45,   # 45% increase by 2050 (direct ETS2)
                'transport_cost_increase': 0.65,  # 65% increase (road transport ETS2)
                'heating_cost_increase': 0.55,   # 55% increase (building fuels ETS2)
                'welfare_impact': -0.18,          # -18% welfare change initially
                'welfare_recovery': 0.12,         # +12% recovery through adaptation
                'switching_support': 'Comprehensive',
                'adaptation_measures': {
                    'energy_efficiency_subsidies': 0.25,    # 25% of carbon revenue
                    'transport_electrification_support': 0.20, # 20% of carbon revenue
                    'heat_pump_incentives': 0.15,          # 15% of carbon revenue
                    'low_income_protection': 0.10          # 10% for vulnerable households
                },
                'behavioral_responses': {
                    'energy_efficiency_improvement': 0.35,  # 35% efficiency gain
                    'transport_modal_shift': 0.28,         # 28% shift to public/active transport
                    'heating_electrification': 0.42,       # 42% switch to heat pumps
                    'renewable_adoption': 0.55             # 55% household renewable adoption
                },
                'fuel_switching': {
                    'transport_electrification': 0.70,  # 70% by 2050 (direct ETS2 + support)
                    'heating_electrification': 0.70,   # 70% by 2050 (heat pumps + support)
                    'renewable_adoption': 0.83        # 83% grid renewable by 2050
                },
                'switching_timeline': {
                    'phase_in_period': 3,      # 3-year phase-in (2027-2030)
                    'full_impact_year': 2030,  # Full ETS2 impact from 2030
                    'adaptation_lag': 5,       # 5-year household adaptation period
                    'switching_acceleration': 2035  # Accelerated switching post-adaptation
                },
                'regional_variation': household_regions,
                'temporal_dynamics': {
                    'phase_in_period': 3,      # 3-year phase-in (2027-2030)
                    'full_impact_year': 2030,  # Full ETS2 impact from 2030
                    'adaptation_lag': 5        # 5-year household adaptation period
                }
            },
            'sectoral_switching': {
                'Road Transport': {
                    'switching_speed': 'Very Fast',
                    'electrification_target': 0.75,
                    'hydrogen_potential': 0.15,
                    'switching_barriers_addressed': ['Purchase incentives', 'Charging infrastructure', 'Total cost of ownership']
                },
                'Rail Transport': {
                    'switching_speed': 'Fast',
                    'electrification_target': 0.90,
                    'hydrogen_potential': 0.10,
                    'switching_barriers_addressed': ['Grid connection', 'Infrastructure upgrade']
                },
                'Services': {
                    'switching_speed': 'Medium',
                    'electrification_target': 0.65,
                    'heat_pump_adoption': 0.55,
                    'switching_barriers_addressed': ['Building retrofits', 'Technology support', 'Financing']
                },
                'Households': {
                    'switching_speed': 'Medium',
                    'transport_electrification': 0.70,
                    'heating_electrification': 0.70,
                    'switching_barriers_addressed': ['Purchase incentives', 'Retrofit support', 'Technical assistance']
                }
            }
        }
    }

    selected_scenario = policy_scenarios.get(
        scenario, policy_scenarios['business_as_usual'])

    if verbose:
        print(f"\nRunning scenario: {selected_scenario['name']}")
        print(f"Description: {selected_scenario['description']}")
        if selected_scenario.get('ets_coverage'):
            print(f"ETS covered sectors: {selected_scenario['ets_coverage']}")
        if selected_scenario.get('start_year'):
            print(f"ETS start year: {selected_scenario['start_year']}")
        if selected_scenario.get('household_direct_coverage'):
            print("HOUSEHOLD DIRECT COVERAGE ENABLED:")
            household_impact = selected_scenario['household_impact']
            print(f"  • Energy price increase: {household_impact['energy_price_increase']*100:.0f}%")
            print(f"  • Transport cost increase: {household_impact['transport_cost_increase']*100:.0f}%")
            print(f"  • Heating cost increase: {household_impact['heating_cost_increase']*100:.0f}%")
            print(f"  • Initial welfare impact: {household_impact['welfare_impact']*100:.0f}%")
            print(f"  • Welfare recovery: {household_impact['welfare_recovery']*100:.0f}%")
            print(f"  • Adaptation measures: {len(household_impact['adaptation_measures'])} programs")
            print(f"  • Behavioral responses: {len(household_impact['behavioral_responses'])} types")
            print(f"  • Regional coverage: {len(household_impact['regional_variation'])} NUTS-1 regions")
        elif selected_scenario.get('household_impact'):
            print("Household indirect impact included")
            household_impact = selected_scenario['household_impact']
            print(f"  • Energy price increase: {household_impact['energy_price_increase']*100:.0f}%")
            print(f"  • Welfare impact: {household_impact['welfare_impact']*100:.0f}%")

    # Use the new recursive dynamic CGE model
    try:
        from recursive_cge_pyomo import RecursiveDynamicCGE

        # Initialize the dynamic CGE model
        cge_model = RecursiveDynamicCGE(
            sam, base_year=2021, final_year=final_year)

        # Run the scenario
        if verbose:
            print(f"\nStarting recursive dynamic simulation...")

        results = cge_model.run_scenario(scenario, save_results=True)

        if verbose and results:
            final_gdp = results['trajectories']['gdp'][-1] if results['trajectories']['gdp'] else 0
            final_emissions = results['trajectories']['total_emissions'][-1] if results['trajectories']['total_emissions'] else 0
            carbon_price = results['trajectories']['carbon_price'][-1] if results['trajectories']['carbon_price'] else 0

            print(f"\n{'='*50}")
            print("SIMULATION RESULTS SUMMARY")
            print(f"{'='*50}")
            print(f"Final Year GDP: €{final_gdp:,.0f} million")
            print(f"Final Emissions: {final_emissions:,.0f} tCO2")
            print(f"Final Carbon Price: €{carbon_price:.2f}/tCO2")
            print(f"Periods simulated: {len(results['periods'])}")

        return results

    except Exception as e:
        print(f"Error in recursive dynamic model: {str(e)}")

        # Fallback to original equilibrium solving
        if verbose:
            print("Falling back to static equilibrium model...")

        equilibrium_results = solve_threeme_equilibrium(
            d, p, ind, h, verbose=verbose)

        if verbose:
            print("Static equilibrium solved successfully!")

        # Prepare results in the new format
        results = {
            'scenario': scenario,
            'scenario_details': selected_scenario,
            'model_type': 'Static Equilibrium (Fallback)',
            'equilibrium': equilibrium_results,
            'metadata': {
                'run_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'final_year': final_year,
                'base_year': d.base_year,
                'regions': regions,
                'sectors': ind,
                'ets1_sectors': ets1_sectors,
                'ets2_sectors': ets2_sectors
            }
        }

        # Save results to Excel format
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        excel_file = os.path.join(
            results_dir, f'cge_results_{scenario}_{timestamp}.xlsx')

        # Save to Excel
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Parameter': ['Scenario', 'Description', 'Model Type', 'Base Year', 'Final Year', 'Run Date'],
                'Value': [
                    selected_scenario['name'],
                    selected_scenario['description'],
                    'Static Equilibrium',
                    d.base_year,
                    final_year,
                    results['metadata']['run_date']
                ]
            }
            pd.DataFrame(summary_data).to_excel(
                writer, sheet_name='Summary', index=False)

            # Sectoral data
            sectoral_data = pd.DataFrame({
                'Sector': ind,
                # Equal distribution as placeholder
                'Base_Output': [d.base_gdp / len(ind)] * len(ind),
                'ETS1_Coverage': [sector in ets1_sectors for sector in ind],
                'ETS2_Coverage': [sector in ets2_sectors for sector in ind]
            })
            sectoral_data.to_excel(
                writer, sheet_name='Sectoral_Data', index=False)

        if verbose:
            print(f"Results saved to: {excel_file}")

        return results

    # Run equilibrium solving
    if verbose:
        print("\nSolving initial equilibrium...")

    try:
        # Solve baseline equilibrium
        equilibrium_results = solve_threeme_equilibrium(d, p, verbose=verbose)

        if verbose:
            print("Initial equilibrium solved successfully!")
            print(
                f"Equilibrium GDP: €{equilibrium_results.get('gdp', 0):,.0f} million")

        # Run policy simulation
        if verbose:
            print(f"\nRunning policy simulation for scenario: {scenario}")

        simulation_results = run_policy_simulation(
            d, p,
            policy_scenario=selected_scenario,
            years_ahead=final_year-d.base_year
        )

        # Extract key linking variables for reporting
        linking_vars = extract_linking_variables(simulation_results, d)

        # Prepare results for saving
        results = {
            'scenario': scenario,
            'scenario_details': selected_scenario,
            'equilibrium': equilibrium_results,
            'simulation': simulation_results,
            'linking_variables': linking_vars,
            'metadata': {
                'run_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'final_year': final_year,
                'base_year': d.base_year,
                'regions': regions,
                'sectors': ind
            }
        }

        # Save results to CGE_Italy/results folder
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)

        # Save detailed results as JSON
        results_file = os.path.join(
            results_dir, f'cge_results_{scenario}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json')

        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                return {k: convert_for_json(v) for k, v in obj.__dict__.items()}
            else:
                return obj

        json_results = convert_for_json(results)

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        if verbose:
            print(f"\nResults saved to: {results_file}")

        # Also save summary Excel file
        excel_file = os.path.join(
            results_dir, f'cge_summary_{scenario}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.xlsx')

        with pd.ExcelWriter(excel_file) as writer:
            # Save key economic indicators
            if 'gdp_trajectory' in simulation_results:
                pd.DataFrame(simulation_results['gdp_trajectory']).to_excel(
                    writer, sheet_name='GDP_Trajectory')

            if 'sectoral_output' in simulation_results:
                pd.DataFrame(simulation_results['sectoral_output']).to_excel(
                    writer, sheet_name='Sectoral_Output')

            # Save linking variables for other models
            if linking_vars:
                pd.DataFrame(linking_vars).to_excel(
                    writer, sheet_name='Linking_Variables')

        if verbose:
            print(f"Summary saved to: {excel_file}")
            print(f"\nSimulation completed successfully!")
            print("=" * 60)

        return results

    except Exception as e:
        print(f"Error during model execution: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


if __name__ == "__main__":
    # Set working directory to CGE_Italy folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Path to SAM data file
    sam_path = os.path.join(script_dir, 'data', 'SAM.xlsx')

    print("RECURSIVE DYNAMIC CGE MODEL FOR ITALY")
    print("Using Pyomo optimization with ETS scenarios")
    print(f"Working directory: {script_dir}")
    print(f"SAM data file: {sam_path}")

    # Run the three new scenarios
    scenarios = ['business_as_usual', 'ets1', 'ets2']

    for scenario in scenarios:
        print("\n" + "="*60)
        print(f"RUNNING {scenario.upper().replace('_', ' ')} SCENARIO")
        print("="*60)

        try:
            results = runner(sam_path=sam_path, scenario=scenario,
                             verbose=True, final_year=2050)

            if results:
                print(
                    f" {scenario.upper().replace('_', ' ')} scenario completed successfully!")
            else:
                print(f" {scenario.upper().replace('_', ' ')} scenario failed!")

        except Exception as e:
            print(f" Error running {scenario} scenario: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("ALL SCENARIOS COMPLETED")
    print("="*60)
    print("Check the 'results' folder for Excel output files.")
    print("ETS1: Power and industry sectors with immediate coverage")
    print("ETS2: Transport sectors starting from 2027")
