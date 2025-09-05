# Fixed main.py - Italy CGE Model with Working Pyomo Implementation
import os
import pandas as pd
import numpy as np
from clean_recursive_cge_pyomo import RecursivePyomoCGE

def main():
    """
    Main execution for Italy CGE model with Pyomo optimization
    Uses the same working approach as debug_solver.py
    """
    
    print("=" * 80)
    print("ITALY CGE MODEL - MAIN EXECUTION")
    print("Real Italian Data (2021-2025) with Pyomo Optimization")
    print("=" * 80)
    
    # SAM data path
    sam_path = os.path.join(os.path.dirname(__file__), "data", "SAM.xlsx")
    
    # Scenarios to run
    scenarios = {
        'business_as_usual': {
            'description': 'Business as Usual - No ETS expansion',
            'carbon_price_growth': 0.03,
            'base_carbon_price': 25
        },
        'ets1': {
            'description': 'ETS1 - Enhanced Coverage (Gas + Aviation)',  
            'carbon_price_growth': 0.05,
            'base_carbon_price': 50
        },
        'ets2': {
            'description': 'ETS2 - Transport Coverage (Road + Other)',
            'carbon_price_growth': 0.08, 
            'base_carbon_price': 40
        }
    }
    
    all_results = {}
    
    # Run each scenario
    for scenario_name, scenario_params in scenarios.items():
        print(f"\n{'='*60}")
        print(f"RUNNING SCENARIO: {scenario_name.upper()}")
        print(f"Description: {scenario_params['description']}")
        print(f"{'='*60}")
        
        try:
            # Initialize CGE model
            cge = RecursivePyomoCGE(
                sam_file=sam_path,
                base_year=2021,
                final_year=2025,  # Short time horizon for testing
                solver='ipopt'
            )
            
            # Set scenario parameters (using working method signature)
            cge.set_scenario_parameters(
                scenario=scenario_name,
                carbon_price_growth=scenario_params['carbon_price_growth'],
                emission_target=0.5  # Default emission target
            )
            
            # Update base carbon price
            cge.base_carbon_price = scenario_params['base_carbon_price']
            
            print(f"Model initialized successfully")
            print(f"Time horizon: 2021-2025 (5 years)")
            print(f"Base carbon price: €{scenario_params['base_carbon_price']}/tCO2")
            print(f"Carbon price growth: {scenario_params['carbon_price_growth']*100:.1f}%/year")
            
            # Run recursive dynamic simulation
            results = cge.solve_recursive_dynamic(
                scenario_name=scenario_name,
                save_results=True,
                verbose=True
            )
            
            # Store results
            all_results[scenario_name] = results
            
            # Display results summary
            if results.get('solver_status') == 'Optimal':
                final_gdp = results['trajectories']['gdp'][-1] if results['trajectories']['gdp'] else 0
                final_emissions = results['trajectories']['total_emissions'][-1] if results['trajectories']['total_emissions'] else 0
                final_carbon_price = results['trajectories']['carbon_price'][-1] if results['trajectories']['carbon_price'] else 0
                periods_solved = len(results.get('periods', []))
                
                print(f"\n✓ SCENARIO {scenario_name.upper()} - SUCCESS")
                print(f"  Periods solved: {periods_solved}/5")
                print(f"  Final GDP: €{final_gdp:,.0f} million")
                print(f"  Final Emissions: {final_emissions:,.1f} tCO2")
                print(f"  Final Carbon Price: €{final_carbon_price:.2f}/tCO2")
                print(f"  Status: {results.get('solver_status', 'Unknown')}")
            else:
                print(f"\n✗ SCENARIO {scenario_name.upper()} - FAILED")
                print(f"  Status: {results.get('solver_status', 'Failed')}")
                
        except Exception as e:
            print(f"\n✗ ERROR in scenario {scenario_name}: {str(e)}")
            all_results[scenario_name] = {'error': str(e), 'solver_status': 'Error'}
    
    # Final summary
    print(f"\n{'='*80}")
    print("EXECUTION COMPLETED - SUMMARY")
    print(f"{'='*80}")
    
    for scenario_name, results in all_results.items():
        status = results.get('solver_status', 'Unknown')
        periods = len(results.get('periods', [])) if 'periods' in results else 0
        print(f"  {scenario_name:20s}: {status:10s} ({periods}/5 periods)")
    
    print(f"\nResults saved to: results/ folder")
    print(f"Check JSON files: pyomo_cge_results_*.json")


if __name__ == "__main__":
    main()
