"""
CGE Italy Model - Quick Run Script
This script provides an easy way to run the CGE model for Italy with different scenarios.
All outputs will be saved in the CGE_Italy/results folder.
"""

import os
import sys


def setup_environment():
    """Ensure we're in the correct directory and can import modules"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    return script_dir


def run_single_scenario(scenario='business_as_usual', final_year=2050):
    """Run a single scenario"""
    try:
        from main import runner

        script_dir = setup_environment()
        sam_path = os.path.join('data', 'SAM.xlsx')

        print(f"Running {scenario} scenario...")
        print(f"Working directory: {script_dir}")
        print(
            f"Results will be saved to: {os.path.join(script_dir, 'results')}")

        results = runner(sam_path=sam_path, scenario=scenario,
                         verbose=True, final_year=final_year)

        if results:
            print(
                f"✓ {scenario.capitalize().replace('_', ' ')} scenario completed successfully!")
            return True
        else:
            print(f"✗ {scenario.capitalize().replace('_', ' ')} scenario failed!")
            return False

    except Exception as e:
        print(f"Error running {scenario} scenario: {str(e)}")
        return False


def run_all_scenarios(final_year=2050):
    """Run all three scenarios"""
    scenarios = ['business_as_usual', 'ets1', 'ets2']
    results = {}

    setup_environment()

    print("="*60)
    print("RECURSIVE DYNAMIC CGE MODEL - RUNNING ALL SCENARIOS")
    print("="*60)
    print("Scenarios:")
    print("• Business as Usual: Current policies continuation")
    print("• ETS1: Power and industry sectors with enhanced carbon pricing")
    print("• ETS2: Transport sectors with ETS coverage from 2027")
    print("="*60)

    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"RUNNING {scenario.upper().replace('_', ' ')} SCENARIO")
        print('='*50)

        success = run_single_scenario(scenario, final_year)
        results[scenario] = success

    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)

    for scenario, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        scenario_name = scenario.replace('_', ' ').title()
        print(f"{scenario_name:20}: {status}")

    print(f"\nAll Excel results saved in: {os.path.abspath('results')}")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) == 1:
        # No arguments - run all scenarios
        run_all_scenarios()
    elif len(sys.argv) == 2:
        # One argument - run specific scenario
        scenario = sys.argv[1].lower()
        if scenario in ['business_as_usual', 'ets1', 'ets2']:
            run_single_scenario(scenario)
        else:
            print("Error: Invalid scenario. Choose from: business_as_usual, ets1, ets2")
            sys.exit(1)
    else:
        print("Usage:")
        print("  python run_model.py                     # Run all scenarios")
        print("  python run_model.py business_as_usual   # Run business as usual scenario")
        print("  python run_model.py ets1                # Run ETS1 scenario")
        print("  python run_model.py ets2                # Run ETS2 scenario")
        sys.exit(1)
