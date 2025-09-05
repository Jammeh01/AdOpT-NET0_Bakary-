#!/usr/bin/env python3
"""
CGE Model for Italy - Execution Script
This script runs the CGE model and ensures all outputs are saved in the CGE_Italy folder.
"""

import os
import sys

# Ensure we're working in the CGE_Italy directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Import the main runner function directly without adding parent to path
try:
    # Try importing from the current directory
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "main", os.path.join(script_dir, "main.py"))
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)
    runner = main_module.runner
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def main():
    """Main execution function"""

    print("CGE Model for Italy - Starting Execution")
    print(f"Working directory: {os.getcwd()}")

    # Path to SAM data file
    sam_path = os.path.join('data', 'SAM.xlsx')

    if not os.path.exists(sam_path):
        print(f"Warning: SAM file not found at {sam_path}")
        print("The model will run with default mock data.")
        sam_path = None
    else:
        print(f"SAM data file found: {sam_path}")

    # Create results directory if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(results_dir)}")

    scenarios_to_run = ['baseline', 'moderate', 'ambitious']

    for scenario in scenarios_to_run:
        print("\n" + "="*60)
        print(f"RUNNING {scenario.upper()} SCENARIO")
        print("="*60)

        try:
            results = runner(sam_path=sam_path, scenario=scenario,
                             verbose=True, final_year=2050)

            if results:
                print(
                    f" {scenario.capitalize()} scenario completed successfully!")
            else:
                print(f" {scenario.capitalize()} scenario failed!")

        except Exception as e:
            print(f" Error running {scenario} scenario: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("MODEL EXECUTION COMPLETED")
    print("="*60)
    print(f"All results saved in: {os.path.abspath('results')}")


if __name__ == "__main__":
    main()
