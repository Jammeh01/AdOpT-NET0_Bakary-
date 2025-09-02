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
            'Other Transport', 'other Sectors (14)'
        ]]
    else:
        ind = ['Agriculture', 'Industry', 'Electricity', 'Gas', 'Other Energy',
               'Road Transport', 'Rail Transport', 'Air Transport', 'Water Transport',
               'Other Transport', 'other Sectors (14)']

    h = ['Labour', 'Capital']
    regions = ['NW', 'NE', 'Centre', 'South', 'Islands']

    # ETS sector definitions based on real EU ETS coverage
    ets1_sectors = ['Electricity', 'Industry',
                    'Other Energy']  # Power and industry
    ets2_sectors = ['Road Transport', 'Rail Transport',
                    'Air Transport', 'Water Transport']  # Transport from 2027

    # Initialize model
    if verbose:
        print("\nInitializing recursive dynamic CGE model...")
        print(f"Sectors: {len(ind)}")
        print(f"ETS1 sectors (covered now): {ets1_sectors}")
        print(f"ETS2 sectors (from 2027): {ets2_sectors}")

    d = model_data(sam, h, ind, regions)
    p = parameters(d, ind)

    if verbose:
        print(f"Base year GDP: €{d.base_gdp:,.0f} million")
        print(f"Number of sectors: {len(ind)}")
        print(f"Number of regions: {len(regions)}")

    # Define the three new policy scenarios
    policy_scenarios = {
        'business_as_usual': {
            'name': 'Business as Usual',
            'description': 'Current policies continuation without additional climate measures',
            'carbon_price_2021': 25,  # €/tCO2
            'carbon_price_growth': 0.02,  # 2% annual growth
            'ets_coverage': [],  # No additional ETS coverage
            'emission_reduction_target': 0.20  # 20% reduction by 2050
        },
        'ets1': {
            'name': 'ETS1 - Power and Industry',
            'description': 'EU ETS extension to power and industry sectors with stronger carbon pricing',
            'carbon_price_2021': 50,  # €/tCO2
            'carbon_price_growth': 0.05,  # 5% annual growth
            'ets_coverage': ets1_sectors,
            'emission_reduction_target': 0.55,  # 55% reduction by 2050
            'start_year': 2021  # Already covered
        },
        'ets2': {
            'name': 'ETS2 - Transport Sectors',
            'description': 'EU ETS extension to transport sectors starting 2027',
            'carbon_price_2021': 25,  # €/tCO2 (starts in 2027)
            'carbon_price_2027': 40,  # €/tCO2 (starting price for transport)
            'carbon_price_growth': 0.08,  # 8% annual growth
            'ets_coverage': ets2_sectors,
            'emission_reduction_target': 0.42,  # 42% reduction by 2050
            'start_year': 2027  # Transport ETS starts in 2027
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
                    f"✓ {scenario.upper().replace('_', ' ')} scenario completed successfully!")
            else:
                print(f"✗ {scenario.upper().replace('_', ' ')} scenario failed!")

        except Exception as e:
            print(f"✗ Error running {scenario} scenario: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("ALL SCENARIOS COMPLETED")
    print("="*60)
    print("Check the 'results' folder for Excel output files.")
    print("ETS1: Power and industry sectors with immediate coverage")
    print("ETS2: Transport sectors starting from 2027")
