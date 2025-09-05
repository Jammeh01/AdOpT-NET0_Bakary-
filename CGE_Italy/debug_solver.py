# Debug script to identify the solver issue
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clean_recursive_cge_pyomo import RecursivePyomoCGE
import traceback

# Test basic CGE model initialization and single period solve
try:
    # Initialize model with working parameters
    sam_path = r"C:\Users\Jamme002\OneDrive - Universiteit Utrecht\Documents\AdOpT-NET0_Bakary-\CGE_Italy\data\SAM.xlsx"
    
    print("Initializing CGE model...")
    cge = RecursivePyomoCGE(sam_file=sam_path, base_year=2021, final_year=2023)
    
    print("Setting scenario parameters...")
    cge.set_scenario_parameters('business_as_usual', carbon_price_growth=0.03)
    
    print("Testing single period solve for 2021...")
    
    # Initialize capital stock
    initial_capital = {sector: cge.base_output[sector] * 2.5 for sector in cge.sectors}
    
    # Try to solve single period with detailed output
    result = cge.solve_single_period_pyomo(2021, initial_capital, verbose=True)
    
    print(f"Solve result: {result}")
    
    if result.get('solved', False):
        print("✓ Single period solved successfully!")
        print(f"GDP: €{result.get('gdp', 0):,.0f} million")
        print(f"Emissions: {result.get('total_emissions', 0):,.1f} tCO2")
        print(f"Carbon Price: €{result.get('carbon_price', 0):.2f}/tCO2")
    else:
        print("✗ Single period failed to solve")
        print(f"Error: {result.get('error', 'Unknown error')}")
        
except Exception as e:
    print(f"Error during debugging: {str(e)}")
    traceback.print_exc()
