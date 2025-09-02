# household.py - CGE Household Behavior
import numpy as np
from pandas import Series, DataFrame


def eqI(pf, Ff, Sp, Td, Fsh, Trf):
    """
    Total household disposable income.
    I = sum(pf_h * Ff_h) - Sp - Td - Fsh + Trf
    """
    gross_income = (pf * Ff).sum()
    I = gross_income - Sp - Td - Fsh + Trf
    return I


def eqXp(alpha, I, pq):
    """
    Household consumption demand (Cobb-Douglas utility).
    Xp_i = (alpha_i / pq_i) * I
    """
    Xp = alpha * I / pq
    return Xp


def eqXp_les(gamma, beta, I, pq):
    """
    Household consumption demand (Linear Expenditure System).
    Xp_i = gamma_i + (beta_i / pq_i) * (I - sum_j(pq_j * gamma_j))
    
    Args:
        gamma: Subsistence consumption parameters
        beta: Marginal budget share parameters
        I: Total disposable income
        pq: Consumer prices
    """
    # Calculate subsistence expenditure
    subsistence_expenditure = (pq * gamma).sum()
    
    # Supernumerary income
    supernumerary_income = max(I - subsistence_expenditure, 0)
    
    # LES demand
    Xp = gamma + beta * supernumerary_income / pq
    
    # Ensure non-negative consumption
    Xp = Xp.clip(lower=gamma * 0.1)  # At least 10% of subsistence
    
    return Xp


def eqI_regional(pf, Ff_regional, regional_shares, Sp_total, Td_total, Fsh_total, Trf_total):
    """
    Regional household income distribution.
    
    Args:
        pf: Factor prices
        Ff_regional: Factor endowments by region (DataFrame: factors x regions)
        regional_shares: Income distribution shares by region
        Sp_total: Total private savings
        Td_total: Total direct taxes
        Fsh_total: Total repatriated profits
        Trf_total: Total transfers
    """
    regions = list(regional_shares.keys())
    I_regional = Series(index=regions, dtype=float)
    
    for region in regions:
        # Factor income for this region
        factor_income = 0
        if 'Labour' in Ff_regional.index:
            factor_income += pf['Labour'] * Ff_regional.loc['Labour', region]
        if 'Capital' in Ff_regional.index:
            factor_income += pf['Capital'] * Ff_regional.loc['Capital', region]
        
        # Regional share of taxes, savings, etc.
        share = regional_shares[region]
        regional_sp = Sp_total * share
        regional_td = Td_total * share
        regional_fsh = Fsh_total * share
        regional_trf = Trf_total * share
        
        I_regional[region] = factor_income - regional_sp - regional_td - regional_fsh + regional_trf
    
    return I_regional


def eqXp_regional(alpha_regional, I_regional, pq):
    """
    Regional consumption demands.
    
    Args:
        alpha_regional: Budget shares by good and region (DataFrame: goods x regions)
        I_regional: Income by region
        pq: Consumer prices
    """
    regions = I_regional.index
    goods = alpha_regional.index
    
    Xp_regional = DataFrame(index=goods, columns=regions, dtype=float)
    
    for region in regions:
        for good in goods:
            budget_share = alpha_regional.loc[good, region]
            income = I_regional[region]
            price = pq[good]
            Xp_regional.loc[good, region] = budget_share * income / price
    
    return Xp_regional


def eqEnergy_demand_regional(I_regional, energy_prices, energy_params, population_shares):
    """
    Regional energy demand by households.
    
    Args:
        I_regional: Income by region
        energy_prices: Dict of energy prices {'electricity', 'gas'}
        energy_params: Energy demand parameters
        population_shares: Population shares by region
    """
    regions = I_regional.index
    
    # Initialize energy demands
    electricity_demand = Series(index=regions, dtype=float)
    gas_demand = Series(index=regions, dtype=float)
    
    total_population = 59.13  # Million people
    
    for region in regions:
        # Population in this region
        regional_population = total_population * population_shares[region]
        
        # Income effect (elasticity = 0.8)
        avg_income = I_regional.mean()
        income_effect = (I_regional[region] / avg_income) ** 0.8
        
        # Price effects
        elec_price_effect = (energy_prices['electricity'] / 80) ** (-0.3)  # Base price 80 EUR/MWh
        gas_price_effect = (energy_prices['gas'] / 65) ** (-0.5)  # Base price 65 EUR/MWh
        
        # Base consumption per capita (MWh/person/year)
        base_elec_pc = energy_params['regional_energy_pc']['electricity'][region]
        base_gas_pc = energy_params['regional_energy_pc']['gas'][region]
        
        # Calculate regional demands
        electricity_demand[region] = (regional_population * base_elec_pc * 
                                    income_effect * elec_price_effect)
        
        gas_demand[region] = (regional_population * base_gas_pc * 
                            income_effect * gas_price_effect)
    
    return {'electricity': electricity_demand, 'gas': gas_demand}


def eqHourly_electricity_profile(annual_demand_regional, year=2021):
    """
    Generate 8760-hour electricity demand profiles by region.
    
    Args:
        annual_demand_regional: Annual electricity demand by region (MWh)
        year: Calendar year for profile
    
    Returns:
        DataFrame: Hourly electricity demand (8760 hours x regions)
    """
    hours = 8760
    regions = annual_demand_regional.index
    
    # Create hourly profiles DataFrame
    hourly_profiles = DataFrame(index=range(hours), columns=regions, dtype=float)
    
    for region in regions:
        # Generate regional profile
        profile = _generate_regional_hourly_profile(region, hours)
        
        # Scale to match annual demand
        profile_normalized = profile / profile.mean()
        annual_demand = annual_demand_regional[region]
        hourly_demand = profile_normalized * (annual_demand / hours)
        
        hourly_profiles[region] = hourly_demand
    
    return hourly_profiles


def _generate_regional_hourly_profile(region, hours):
    """
    Generate realistic hourly demand profile for a region.
    
    Args:
        region: Region name
        hours: Number of hours (8760)
    
    Returns:
        numpy.array: Normalized hourly profile
    """
    hour_array = np.arange(hours)
    
    # Daily pattern (24-hour cycle)
    hour_of_day = hour_array % 24
    daily_pattern = 0.7 + 0.3 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    
    # Add evening peak (6-9 PM)
    evening_peak = (hour_of_day >= 18) & (hour_of_day <= 21)
    daily_pattern[evening_peak] *= 1.25
    
    # Reduce night demand (11 PM - 6 AM)
    night_hours = ((hour_of_day >= 23) | (hour_of_day <= 6))
    daily_pattern[night_hours] *= 0.75
    
    # Weekly pattern (lower on weekends)
    day_of_week = (hour_array // 24) % 7
    weekend = (day_of_week >= 5)  # Saturday and Sunday
    weekly_pattern = np.ones(hours)
    weekly_pattern[weekend] *= 0.9
    
    # Seasonal pattern
    day_of_year = (hour_array // 24) % 365
    
    # Regional seasonal adjustments
    if region in ['South', 'Islands']:
        # Higher summer cooling demand
        summer_factor = 1.0 + 0.4 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
        summer_days = (day_of_year >= 152) & (day_of_year <= 243)
        seasonal_pattern = np.ones(hours)
        seasonal_pattern[summer_days] = summer_factor[summer_days]
    else:
        # Higher winter heating demand (heat pumps)
        winter_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 355) / 365)
        winter_days = (day_of_year >= 335) | (day_of_year <= 59)
        seasonal_pattern = np.ones(hours)
        seasonal_pattern[winter_days] = winter_factor[winter_days]
    
    # Combine all patterns
    full_profile = daily_pattern * weekly_pattern * seasonal_pattern
    
    return full_profile


def eqEnergy_expenditure_shares(energy_demands, energy_prices, total_expenditure_regional):
    """
    Calculate energy expenditure shares by region.
    
    Args:
        energy_demands: Dict with 'electricity' and 'gas' Series by region
        energy_prices: Dict with energy prices
        total_expenditure_regional: Total expenditure by region
    """
    regions = energy_demands['electricity'].index
    
    expenditure_shares = DataFrame(
        index=['electricity', 'gas', 'total_energy'], 
        columns=regions, 
        dtype=float
    )
    
    for region in regions:
        # Calculate expenditures
        elec_expenditure = (energy_demands['electricity'][region] * 
                          energy_prices['electricity'])
        gas_expenditure = (energy_demands['gas'][region] * 
                         energy_prices['gas'])
        total_energy_exp = elec_expenditure + gas_expenditure
        
        # Calculate shares of total household expenditure
        total_exp = total_expenditure_regional[region]
        
        expenditure_shares.loc['electricity', region] = elec_expenditure / total_exp
        expenditure_shares.loc['gas', region] = gas_expenditure / total_exp
        expenditure_shares.loc['total_energy', region] = total_energy_exp / total_exp
    
    return expenditure_shares


def eqWelfare_equivalent_variation(consumption_base, consumption_policy, 
                                  prices_base, prices_policy, alpha):
    """
    Calculate welfare change using equivalent variation.
    
    Args:
        consumption_base: Base scenario consumption
        consumption_policy: Policy scenario consumption
        prices_base: Base scenario prices
        prices_policy: Policy scenario prices
        alpha: Budget share parameters
    """
    # Cobb-Douglas indirect utility
    def indirect_utility(income, prices, shares):
        return income / (prices ** shares).prod()
    
    # Calculate incomes
    income_base = (prices_base * consumption_base).sum()
    income_policy = (prices_policy * consumption_policy).sum()
    
    # Calculate utilities
    utility_base = indirect_utility(income_base, prices_base, alpha)
    utility_policy = indirect_utility(income_policy, prices_policy, alpha)
    
    # Equivalent variation
    price_index_base = (prices_base ** alpha).prod()
    income_equivalent = utility_policy * price_index_base
    equivalent_variation = income_equivalent - income_base
    
    # As percentage of base income
    ev_percentage = (equivalent_variation / income_base) * 100
    
    return {'ev_absolute': equivalent_variation, 'ev_percentage': ev_percentage}


def eqLabor_supply_regional(wage, population_regional, participation_rates, 
                          wage_elasticity=0.15):
    """
    Regional labor supply.
    
    Args:
        wage: Regional wage rate
        population_regional: Population by region
        participation_rates: Labor force participation rates by region
        wage_elasticity: Elasticity of labor supply to wage changes
    """
    regions = population_regional.index
    labor_supply = Series(index=regions, dtype=float)
    
    for region in regions:
        base_participation = participation_rates[region]
        population = population_regional[region]
        
        # Wage effect on participation
        wage_effect = wage ** wage_elasticity
        effective_participation = base_participation * wage_effect
        
        labor_supply[region] = population * effective_participation * 1000000  # Convert to persons
    
    return labor_supply


def eqSavings_regional(income_regional, interest_rate, savings_rates):
    """
    Regional savings behavior.
    
    Args:
        income_regional: Income by region
        interest_rate: Real interest rate
        savings_rates: Base savings rates by region
    """
    regions = income_regional.index
    savings_regional = Series(index=regions, dtype=float)
    
    # Interest elasticity of savings
    interest_elasticity = 0.2
    interest_effect = (interest_rate / 0.03) ** interest_elasticity  # Base rate 3%
    
    for region in regions:
        base_rate = savings_rates[region]
        effective_rate = base_rate * interest_effect
        savings_regional[region] = income_regional[region] * effective_rate
    
    return savings_regional