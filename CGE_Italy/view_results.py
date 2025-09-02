"""
CGE Italy Model - Results Viewer
This script helps you view and analyze the results from the CGE model runs.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def list_result_files():
    """List all result files in the results directory"""
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print("No results directory found. Run the model first.")
        return []

    files = []
    for f in os.listdir(results_dir):
        if f.endswith('.json'):
            files.append(f)

    return sorted(files)


def load_results(filename):
    """Load results from a JSON file"""
    filepath = os.path.join('results', filename)
    with open(filepath, 'r') as f:
        return json.load(f)


def show_summary(results):
    """Show a summary of the results"""
    print("="*60)
    print("SCENARIO SUMMARY")
    print("="*60)

    metadata = results.get('metadata', {})
    scenario_details = results.get('scenario_details', {})

    print(f"Scenario: {scenario_details.get('name', 'Unknown')}")
    print(
        f"Description: {scenario_details.get('description', 'No description')}")
    print(f"Run Date: {metadata.get('run_date', 'Unknown')}")
    print(
        f"Time Period: {metadata.get('base_year', 'Unknown')} - {metadata.get('final_year', 'Unknown')}")
    print(f"Number of Sectors: {len(metadata.get('sectors', []))}")
    print(f"Number of Regions: {len(metadata.get('regions', []))}")

    if 'sectors' in metadata:
        print(f"\nSectors modeled:")
        for i, sector in enumerate(metadata['sectors'], 1):
            print(f"  {i:2d}. {sector}")

    if 'regions' in metadata:
        print(f"\nRegions modeled:")
        for i, region in enumerate(metadata['regions'], 1):
            print(f"  {i}. {region}")


def compare_scenarios():
    """Compare results across different scenarios"""
    result_files = list_result_files()

    if len(result_files) == 0:
        print("No result files found.")
        return

    print("="*60)
    print("SCENARIO COMPARISON")
    print("="*60)

    scenarios = {}

    for file in result_files:
        if 'baseline' in file:
            scenarios['Baseline'] = load_results(file)
        elif 'moderate' in file:
            scenarios['Moderate'] = load_results(file)
        elif 'ambitious' in file:
            scenarios['Ambitious'] = load_results(file)

    if not scenarios:
        print("No recognizable scenario files found.")
        return

    print(f"Found {len(scenarios)} scenarios for comparison:")
    for name in scenarios.keys():
        print(f"  • {name}")

    # You can add more detailed comparison logic here
    print("\nTo view detailed results for a specific scenario, use view_results('filename.json')")


def view_results(filename=None):
    """View detailed results"""
    if filename is None:
        files = list_result_files()
        if not files:
            print("No result files found.")
            return

        print("Available result files:")
        for i, f in enumerate(files, 1):
            print(f"  {i}. {f}")

        try:
            choice = int(input("\nSelect a file (number): ")) - 1
            if 0 <= choice < len(files):
                filename = files[choice]
            else:
                print("Invalid selection.")
                return
        except (ValueError, KeyboardInterrupt):
            print("Invalid input.")
            return

    results = load_results(filename)
    show_summary(results)

    # Show additional details if available
    equilibrium = results.get('equilibrium', {})
    simulation = results.get('simulation', {})

    if equilibrium:
        print(f"\nEquilibrium Results:")
        for key, value in equilibrium.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:,.2f}")

    if simulation:
        print(f"\nSimulation Overview:")
        for key, value in simulation.items():
            if isinstance(value, dict):
                print(f"  {key}: {len(value)} data points")
            else:
                print(f"  {key}: {type(value).__name__}")


def main():
    """Main function"""
    print("CGE Italy Model - Results Viewer")
    print("="*40)

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    while True:
        print("\nOptions:")
        print("1. View specific results")
        print("2. Compare all scenarios")
        print("3. List result files")
        print("4. Exit")

        try:
            choice = input("\nSelect an option (1-4): ").strip()

            if choice == '1':
                view_results()
            elif choice == '2':
                compare_scenarios()
            elif choice == '3':
                files = list_result_files()
                if files:
                    print("\nResult files:")
                    for f in files:
                        print(f"  • {f}")
                else:
                    print("No result files found.")
            elif choice == '4':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-4.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
