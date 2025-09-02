#!/usr/bin/env python3
"""
Quick Parameter Viewer - Display key model parameters
"""

from calibrate import model_data
import pandas as pd


def display_key_parameters():
    """Display the most important model parameters"""

    # Load data
    sam_path = 'data/SAM.xlsx'
    sam = pd.read_excel(sam_path, index_col=0)

    h = ['Labour', 'Capital']
    ind = ['Agriculture', 'Industry', 'Electricity', 'Gas', 'Other Energy',
           'Road Transport', 'Rail Transport', 'Air Transport', 'Water Transport',
           'Other Transport', 'other Sectors (14)']

    data = model_data(sam, h, ind)

    print("="*70)
    print("KEY CGE MODEL PARAMETERS FOR ITALY")
    print("="*70)

    print("\n📊 BASE YEAR ECONOMIC DATA (2021):")
    print(
        f"   GDP: €{data.base_gdp:,} million (€{data.base_gdp/1000:.1f} billion)")
    print(f"   Population: {data.base_population} million people")
    print(f"   GDP per capita: €{data.base_gdp/data.base_population:,.0f}")
    print(f"   Labor force: {int(data.Ff0['Labour']):,} workers")
    print(f"   Capital stock: €{int(data.Ff0['Capital']):,} million")

    print("\n🏭 SECTORAL STRUCTURE:")
    total_output = data.Z0.sum()
    for i, sector in enumerate(ind[:5]):  # Show top 5 sectors
        share = data.Z0[sector] / total_output * 100
        print(f"   {sector}: €{data.Z0[sector]:,.0f}M ({share:.1f}%)")
    print(f"   ... and {len(ind)-5} other sectors")
    print(f"   TOTAL OUTPUT: €{total_output:,.0f} million")

    print("\n🌍 REGIONAL DISTRIBUTION:")
    for region, share in data.regional_pop_shares.items():
        pop = data.base_population * share
        print(f"   {region}: {pop:.1f}M people ({share:.1%})")

    print("\n⚡ ETS COVERAGE:")
    ets1_sectors = ['Electricity', 'Industry', 'Other Energy']
    ets2_sectors = ['Road Transport', 'Rail Transport',
                    'Air Transport', 'Water Transport']

    print("   ETS1 (from 2021): Power and Industry")
    for sector in ets1_sectors:
        if sector in ind:
            output = data.Z0[sector]
            print(f"     • {sector}: €{output:,.0f}M")

    print("   ETS2 (from 2027): Transport Sectors")
    for sector in ets2_sectors:
        if sector in ind:
            output = data.Z0[sector]
            print(f"     • {sector}: €{output:,.0f}M")

    print("\n📈 MODEL SPECIFICATIONS:")
    print(f"   Time horizon: {data.base_year}-2050 (30 years)")
    print(f"   Solution method: Recursive dynamic with Pyomo")
    print(f"   Optimization: Social welfare maximization")
    print(f"   Solver: IPOPT (Interior Point Optimizer)")
    print(f"   Production function: CES (Constant Elasticity of Substitution)")
    print(f"   Utility function: Cobb-Douglas")

    print("\n💼 POLICY SCENARIOS:")
    print("   1. Business as Usual: No additional climate policies")
    print("   2. ETS1: EU ETS for power and industry sectors")
    print("   3. ETS2: EU ETS extension to transport (from 2027)")

    print("\n" + "="*70)


if __name__ == "__main__":
    display_key_parameters()
