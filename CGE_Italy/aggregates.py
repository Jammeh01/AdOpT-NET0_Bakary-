# aggregates.py - CGE Aggregates & Market Clearing
import numpy as np
from pandas import Series


def eqSp(ssp, pf, Ff, Fsh, Trf):
    """
    Total household savings.
    Sp = ssp * (sum(pf_h * Ff_h) - Fsh + Trf)
    """
    disposable_income = (pf * Ff).sum() - Fsh + Trf
    Sp = ssp * disposable_income
    return Sp


def eqKd(g, Sp, lam, pq):
    """
    Domestic capital holdings.
    Kd = Sp / (g * sum(lam_i * pq_i))
    """
    investment_price_index = (lam * pq).sum()
    Kd = Sp / (g * investment_price_index)
    return Kd


def eqKf(Kk, Kd):
    """
    Foreign holdings of domestically used capital.
    Kf = Kk - Kd
    """
    Kf = Kk - Kd
    return Kf


def eqKk(pf, Ff, R, lam, pq):
    """
    Capital market clearing equation.
    Kk = (pf_Capital * Ff_Capital) / (R * sum(lam_i * pq_i))
    """
    capital_income = pf["Capital"] * Ff["Capital"]
    investment_price_index = (lam * pq).sum()
    Kk = capital_income / (R * investment_price_index)
    return Kk


def eqSf(g, lam, pq, Kf):
    """
    Net foreign investment/savings.
    Sf = g * Kf * sum(lam_i * pq_i)
    """
    investment_price_index = (lam * pq).sum()
    Sf = g * Kf * investment_price_index
    return Sf


def eqXXv(g, Kk):
    """
    Total investment.
    XXv = g * Kk
    """
    XXv = g * Kk
    return XXv


def eqbop(pWe, pWm, E, M, Sf, Fsh, er):
    """
    Balance of payments.
    sum(pWe_i * E_i) + Sf/er = sum(pWm_i * M_i) + Fsh/er
    """
    export_earnings = (pWe * E).sum()
    import_payments = (pWm * M).sum()
    capital_inflow = Sf / er
    capital_outflow = Fsh / er
    
    bop_balance = export_earnings + capital_inflow - import_payments - capital_outflow
    return bop_balance


def eqResource_constraint(Q, Xp, Xg, Xv, X):
    """
    Resource constraint (market clearing for goods).
    Q_i = Xp_i + Xg_i + Xv_i + sum_j(X_{i,j})
    """
    intermediate_demand = X.sum(axis=1) if hasattr(X, 'sum') else X
    total_demand = Xp + Xg + Xv + intermediate_demand
    resource_balance = Q - total_demand
    return resource_balance


def eqLabor_market_clearing(F, Ff0):
    """
    Labor market clearing.
    sum_i(F_Labour,i) = Ff_Labour
    """
    if 'Labour' in F.index:
        labor_demand = F.loc['Labour'].sum()
        labor_supply = Ff0['Labour']
        labor_balance = labor_supply - labor_demand
    else:
        labor_balance = 0
    return labor_balance


def eqCapital_market_clearing(F, Kk, Kk0, Ff0):
    """
    Capital market clearing.
    sum_i(F_Capital,i) = (Kk/Kk0) * Ff_Capital
    """
    if 'Capital' in F.index:
        capital_demand = F.loc['Capital'].sum()
        capital_supply = (Kk / Kk0) * Ff0['Capital']
        capital_balance = capital_supply - capital_demand
    else:
        capital_balance = 0
    return capital_balance


def eqGDP_expenditure(Xp_total, Xg_total, Xv_total, E_total, M_total):
    """
    GDP from expenditure approach.
    GDP = C + G + I + (X - M)
    """
    GDP = Xp_total + Xg_total + Xv_total + E_total - M_total
    return GDP


def eqGDP_production(Y_total):
    """
    GDP from production approach (value added).
    GDP = sum_i(Y_i)
    """
    GDP = Y_total
    return GDP


def eqEnergy_market_clearing(energy_production, energy_demand_sectors, 
                           energy_demand_households, net_exports):
    """
    Energy market clearing by fuel type.
    Production_fuel = Sectoral_demand + Household_demand + Net_exports
    """
    energy_balance = {}
    
    fuel_types = ['electricity', 'gas', 'other_energy']
    
    for fuel in fuel_types:
        production = energy_production.get(fuel, 0)
        sectoral = energy_demand_sectors.get(fuel, 0)
        household = energy_demand_households.get(fuel, 0)
        exports = net_exports.get(fuel, 0)
        
        if hasattr(sectoral, 'sum'):
            sectoral = sectoral.sum()
        if hasattr(household, 'sum'):  
            household = household.sum()
        
        total_demand = sectoral + household + exports
        energy_balance[fuel] = production - total_demand
    
    return energy_balance


def eqCarbon_budget_constraint(total_emissions, carbon_budget):
    """
    Carbon budget constraint.
    Total_emissions <= Carbon_budget
    """
    carbon_balance = carbon_budget - total_emissions
    return carbon_balance


def eqRegional_GDP(regional_income, regional_production):
    """
    Regional GDP calculation.
    Regional_GDP = Regional_income or Regional_production
    """
    # Can use either income or production approach
    regional_gdp = regional_income  # Using income approach
    return regional_gdp


def eqTrade_balance(pWe, pWm, E, M, er):
    """
    Trade balance in goods and services.
    Trade_balance = er * sum(pWe_i * E_i) - er * sum(pWm_i * M_i)
    """
    export_value = er * (pWe * E).sum()
    import_value = er * (pWm * M).sum()
    trade_balance = export_value - import_value
    return trade_balance


def eqPrice_indices(prices, weights, base_prices=None):
    """
    Aggregate price indices (CPI, PPI).
    """
    if base_prices is None:
        base_prices = Series(1.0, index=prices.index)
    
    # Consumer Price Index (Laspeyres)
    cpi = (weights * prices).sum() / (weights * base_prices).sum()
    
    # Producer Price Index
    ppi = (prices / base_prices).mean()
    
    return {'cpi': cpi, 'ppi': ppi}


def eqWelfare_aggregate(regional_welfare, regional_population):
    """
    National welfare aggregation.
    National_welfare = sum(Regional_welfare * Population_share)
    """
    population_shares = regional_population / regional_population.sum()
    national_welfare = (regional_welfare * population_shares).sum()
    return national_welfare


def eqSavings_investment_balance(Sp, Sg, Sf, XXv):
    """
    Savings-investment balance.
    Sp + Sg + Sf = XXv
    """
    total_savings = Sp + Sg + Sf
    si_balance = total_savings - XXv
    return si_balance


def eqGovernment_budget_balance(revenues, expenditures):
    """
    Government budget balance.
    Budget_balance = Total_revenues - Total_expenditures
    """
    if isinstance(revenues, dict):
        total_revenue = sum(revenues.values())
    else:
        total_revenue = revenues
    
    if isinstance(expenditures, dict):
        total_expenditure = sum(expenditures.values())
    else:
        total_expenditure = expenditures
    
    budget_balance = total_revenue - total_expenditure
    return budget_balance


def eqMacro_consistency_check(aggregates, tolerance=1e-6):
    """
    Check macroeconomic consistency.
    """
    consistency_errors = []
    
    # Check GDP consistency (expenditure vs production)
    if 'GDP_expenditure' in aggregates and 'GDP_production' in aggregates:
        gdp_diff = abs(aggregates['GDP_expenditure'] - aggregates['GDP_production'])
        if gdp_diff > tolerance:
            consistency_errors.append(f"GDP inconsistency: {gdp_diff}")
    
    # Check savings-investment balance
    if all(k in aggregates for k in ['Sp', 'Sg', 'Sf', 'XXv']):
        si_balance = eqSavings_investment_balance(
            aggregates['Sp'], aggregates['Sg'], 
            aggregates['Sf'], aggregates['XXv']
        )
        if abs(si_balance) > tolerance:
            consistency_errors.append(f"S-I imbalance: {si_balance}")
    
    return consistency_errors


def eqAggregate_energy_demand(sectoral_energy, household_energy):
    """
    Aggregate energy demand by fuel type.
    """
    total_energy = {}
    
    for fuel in ['electricity', 'gas', 'other_energy']:
        sectoral = sectoral_energy.get(fuel, 0)
        household = household_energy.get(fuel, 0)
        
        if hasattr(sectoral, 'sum'):
            sectoral = sectoral.sum()
        if hasattr(household, 'sum'):
            household = household.sum()
        
        total_energy[fuel] = sectoral + household
    
    return total_energy


def eqAggregate_emissions(sectoral_emissions, household_emissions=None):
    """
    Aggregate CO2 emissions.
    """
    total_emissions = sectoral_emissions.sum() if hasattr(sectoral_emissions, 'sum') else sectoral_emissions
    
    if household_emissions is not None:
        if hasattr(household_emissions, 'sum'):
            total_emissions += household_emissions.sum()
        else:
            total_emissions += household_emissions
    
    return total_emissions