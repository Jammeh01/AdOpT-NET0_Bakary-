#!/usr/bin/env python3
"""
Test script for Pyomo-based CGE model
This script tests the basic functionality of the RecursivePyomoCGE class
"""

import os
import json
from clean_recursive_cge_pyomo import RecursivePyomoCGE

def test_pyomo_cge():
    """Test basic functionality of Pyomo CGE model"""
    print("="*60)
    print("TESTING PYOMO-BASED CGE MODEL FOR ITALY")
    print("="*60)
    
    # Load SAM data
    sam_file = "data/SAM.xlsx"
    
    if not os.path.exists(sam_file):
        print(f"Error: SAM file not found at {sam_file}")
        return False
        
    print(f"✓ SAM file found: {sam_file}")
    
    try:
        # Initialize the model
        print("\n1. Initializing Pyomo CGE model...")
        cge_model = RecursivePyomoCGE(
            sam_file=sam_file, 
            base_year=2021, 
            final_year=2030,  # Shorter period for testing
            solver='ipopt'  # Use IPOPT solver
        )
        print("✓ Model initialized successfully")
        
        # Set scenario parameters
        print("\n2. Setting scenario parameters...")
        cge_model.set_scenario_parameters(
            scenario='business_as_usual',
            carbon_price_growth=0.05,
            emission_target=0.5,
            ets_sectors=['Electricity', 'Manufacturing', 'Gas']
        )
        print("✓ Scenario parameters set")
        
        # Test single period solution
        print("\n3. Testing single period solution...")
        capital_stock = {sector: 100000 for sector in cge_model.sectors}  # Simple test values
        
        period_result = cge_model.solve_single_period_pyomo(
            year=2021, 
            capital_stock=capital_stock, 
            verbose=True
        )
        
        if period_result['solved']:
            print("✓ Single period solved successfully")
            print(f"  - GDP: €{period_result['gdp']:,.0f} million")
            print(f"  - Total emissions: {period_result['total_emissions']:,.0f} tCO2")
            print(f"  - Carbon price: €{period_result['carbon_price']:.2f}/tCO2")
        else:
            print("✗ Single period solution failed")
            return False
            
        # Test short recursive solution
        print("\n4. Testing short recursive solution...")
        results = cge_model.solve_recursive_dynamic(
            scenario_name='test_bau',
            save_results=True,
            verbose=True
        )
        
        if results['solver_status'] == 'Optimal':
            print("✓ Recursive solution completed successfully")
            print(f"  - Periods solved: {len(results['periods'])}")
            print(f"  - Final year GDP: €{results['trajectories']['gdp'][-1]:,.0f} million")
            print(f"  - Final year emissions: {results['trajectories']['total_emissions'][-1]:,.0f} tCO2")
        else:
            print("✗ Recursive solution failed")
            return False
            
        print("\n" + "="*60)
        print("ALL TESTS PASSED! PYOMO CGE MODEL IS WORKING")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pyomo_cge()
    if not success:
        exit(1)
