# government.py - CGE Government Behavior
import numpy as np
from pandas import Series


def eqTd(taud, pf, Ff):
    """
    Direct tax revenue.
    Td = taud * sum(pf_h * Ff_h)
    """
    gross_income = (pf * Ff).sum()
    Td = taud * gross_income
    return Td


def eqTrf(tautr, pf, Ff):
    """
    Government transfers to households.
    Trf = tautr * pf_Labour * Ff_Labour
    """
    labor_income = pf['Labour'] * Ff['Labour']
    Trf = tautr * labor_income
    return Trf


def eqTz(tauz, pz, Z):
    """
    Production tax revenue.
    Tz_i = tauz_i * pz_i * Z_i
    """
    Tz = tauz * pz * Z
    return Tz


def eqTm(taum, pm, M):
    """
    Import tariff revenue.
    Tm_i = taum_i * pm_i * M_i
    """
    Tm = taum * pm * M
    return Tm


def eqXg(mu, XXg):
    """
    Government consumption by sector.
    Xg_i = mu_i * XXg
    """
    Xg = mu * XXg
    return Xg


def eqSg(Td, Tz, Tm, Tcarbon, XXg, Trf, pq, mu):
    """
    Government savings (budget balance).
    Sg = (Td + sum(Tz) + sum(Tm) + Tcarbon) - (XXg * sum(mu * pq) + Trf)
    
    Args:
        Td: Direct tax revenue
        Tz: Production tax revenue by sector
        Tm: Import tariff revenue by sector
        Tcarbon: Carbon tax revenue
        XXg: Total government consumption
        Trf: Total transfers
        pq: Consumer prices
        mu: Government consumption shares
    """
    total_revenue = Td + Tz.sum() + Tm.sum() + Tcarbon
    government_consumption_cost = XXg * (mu * pq).sum()
    total_expenditure = government_consumption_cost + Trf
    
    Sg = total_revenue - total_expenditure
    return Sg


def eqCarbon_tax_ets1(carbon_price_ets1, emissions_ets1):
    """
    ETS1 carbon tax revenue (power and industry sectors).
    
    Args:
        carbon_price_ets1: ETS1 carbon price (EUR/tCO2)
        emissions_ets1: CO2 emissions from ETS1 sectors (tCO2)
    """
    Tcarbon_ets1 = carbon_price_ets1 * emissions_ets1.sum()
    return Tcarbon_ets1


def eqCarbon_tax_ets2(carbon_price_ets2, emissions_ets2):
    """
    ETS2 carbon tax revenue (transport and buildings sectors).
    
    Args:
        carbon_price_ets2: ETS2 carbon price (EUR/tCO2)
        emissions_ets2: CO2 emissions from ETS2 sectors (tCO2)
    """
    Tcarbon_ets2 = carbon_price_ets2 * emissions_ets2.sum()
    return Tcarbon_ets2


def eqCarbon_price_path(year, base_year, ets_type, scenario='baseline'):
    """
    Dynamic carbon price trajectory.
    
    Args:
        year: Current year
        base_year: Base year (2021)
        ets_type: 'ets1' or 'ets2'
        scenario: 'baseline', 'moderate', or 'ambitious'
    
    Returns:
        carbon_price: Carbon price for the given year (EUR/tCO2)
    """
    # Define price trajectories by scenario
    price_paths = {
        'baseline': {
            'ets1': {'2021': 50, '2030': 85, '2050': 120},
            'ets2': {'2021': 0, '2026': 25, '2030': 45, '2050': 85}
        },
        'moderate': {
            'ets1': {'2021': 50, '2030': 100, '2050': 180},
            'ets2': {'2021': 0, '2026': 30, '2030': 50, '2050': 120}
        },
        'ambitious': {
            'ets1': {'2021': 50, '2030': 150, '2050': 300},
            'ets2': {'2021': 0, '2026': 40, '2030': 75, '2050': 200}
        }
    }
    
    if scenario not in price_paths:
        scenario = 'baseline'
    
    path = price_paths[scenario][ets_type]
    
    # Linear interpolation between key years
    if year <= base_year:
        return path['2021']
    elif ets_type == 'ets2' and year < 2026:
        return 0.0  # ETS2 starts in 2026
    elif year <= 2026:
        if ets_type == 'ets1':
            # Interpolate 2021-2030 for ETS1
            progress = (year - 2021) / (2030 - 2021)
            return path['2021'] + progress * (path['2030'] - path['2021'])
        else:
            # ETS2 starts at 2026
            return path['2026']
    elif year <= 2030:
        # Interpolate to 2030
        if ets_type == 'ets1':
            progress = (year - 2021) / (2030 - 2021)
            return path['2021'] + progress * (path['2030'] - path['2021'])
        else:
            progress = (year - 2026) / (2030 - 2026)
            return path['2026'] + progress * (path['2030'] - path['2026'])
    elif year <= 2050:
        # Interpolate 2030-2050
        progress = (year - 2030) / (2050 - 2030)
        return path['2030'] + progress * (path['2050'] - path['2030'])
    else:
        # Post-2050
        return path['2050']


def eqCarbon_revenue_recycling(total_carbon_revenue, recycling_scheme='balanced'):
    """
    Carbon revenue recycling policies.
    
    Args:
        total_carbon_revenue: Total carbon tax revenue
        recycling_scheme: 'transfers', 'tax_cuts', 'green_spending', 'balanced'
    
    Returns:
        dict: Revenue recycling allocation
    """
    recycling = {
        'household_transfers': 0.0,
        'labor_tax_cut': 0.0,
        'capital_tax_cut': 0.0,
        'green_investment': 0.0
    }
    
    if recycling_scheme == 'transfers':
        # All revenue as household transfers (progressive)
        recycling['household_transfers'] = total_carbon_revenue
    elif recycling_scheme == 'tax_cuts':
        # Revenue-neutral tax shifts
        recycling['labor_tax_cut'] = total_carbon_revenue * 0.7
        recycling['capital_tax_cut'] = total_carbon_revenue * 0.3
    elif recycling_scheme == 'green_spending':
        # Green investment and some transfers
        recycling['green_investment'] = total_carbon_revenue * 0.6
        recycling['household_transfers'] = total_carbon_revenue * 0.4
    else:  # balanced
        # Mix of all approaches
        recycling['household_transfers'] = total_carbon_revenue * 0.4
        recycling['labor_tax_cut'] = total_carbon_revenue * 0.3
        recycling['green_investment'] = total_carbon_revenue * 0.3
    
    return recycling


def eqDebt_dynamics(debt_current, deficit, interest_rate, gdp_nominal):
    """
    Government debt dynamics.
    
    Args:
        debt_current: Current debt stock
        deficit: Government deficit (negative savings)
        interest_rate: Interest rate on government debt
        gdp_nominal: Nominal GDP
    
    Returns:
        dict: Updated debt and ratios
    """
    # Interest payments
    interest_payments = debt_current * interest_rate
    
    # New debt
    debt_new = debt_current + deficit + interest_payments
    
    # Debt ratios
    debt_to_gdp = debt_new / gdp_nominal
    
    return {
        'debt_new': debt_new,
        'debt_to_gdp': debt_to_gdp,
        'interest_payments': interest_payments
    }


def eqFiscal_rules(debt_to_gdp, deficit_to_gdp):
    """
    EU-style fiscal rules constraints.
    
    Args:
        debt_to_gdp: Government debt-to-GDP ratio
        deficit_to_gdp: Government deficit-to-GDP ratio (positive = deficit)
    
    Returns:
        dict: Fiscal rule compliance and adjustments needed
    """
    # EU fiscal rules
    DEBT_LIMIT = 0.60   # 60% of GDP
    DEFICIT_LIMIT = 0.03  # 3% of GDP
    
    compliance = {
        'debt_compliant': debt_to_gdp <= DEBT_LIMIT,
        'deficit_compliant': deficit_to_gdp <= DEFICIT_LIMIT,
        'overall_compliant': debt_to_gdp <= DEBT_LIMIT and deficit_to_gdp <= DEFICIT_LIMIT
    }
    
    adjustments = {
        'debt_adjustment_needed': max(0, debt_to_gdp - DEBT_LIMIT),
        'deficit_adjustment_needed': max(0, deficit_to_gdp - DEFICIT_LIMIT),
        'fiscal_space': DEBT_LIMIT - debt_to_gdp
    }
    
    return {'compliance': compliance, 'adjustments': adjustments}


def eqRegional_government_transfers(total_transfers, regional_needs_index):
    """
    Distribute government transfers based on regional needs.
    
    Args:
        total_transfers: Total government transfers to distribute
        regional_needs_index: Needs index by region (higher = more needs)
    
    Returns:
        Series: Transfers by region
    """
    regions = regional_needs_index.index
    
    # Base distribution (population-weighted)
    base_shares = Series([0.285, 0.215, 0.225, 0.195, 0.080], 
                        index=['NW', 'NE', 'Centre', 'South', 'Islands'])
    
    # Needs-based redistribution factor
    redistribution_strength = 0.3  # 30% redistribution based on needs
    
    # Normalize needs index
    needs_shares = regional_needs_index / regional_needs_index.sum()
    
    # Combined allocation
    final_shares = ((1 - redistribution_strength) * base_shares + 
                   redistribution_strength * needs_shares)
    
    # Ensure shares sum to 1
    final_shares = final_shares / final_shares.sum()
    
    regional_transfers = total_transfers * final_shares
    return regional_transfers


def eqGreen_investment_impact(green_investment, sectors):
    """
    Green investment productivity impacts.
    
    Args:
        green_investment: Total green investment spending
        sectors: List of sectors
    
    Returns:
        dict: Productivity impacts by sector
    """
    # Green investment has higher productivity multiplier
    base_multiplier = 1.2  # 20% higher than regular investment
    
    # Sector-specific impacts
    green_impacts = {}
    
    for sector in sectors:
        if 'Energy' in sector or sector == 'Electricity':
            # High impact on energy sectors
            green_impacts[sector] = green_investment * base_multiplier * 0.3
        elif 'Transport' in sector:
            # Moderate impact on transport
            green_impacts[sector] = green_investment * base_multiplier * 0.2
        elif sector in ['Industry', 'Agriculture']:
            # Some impact on production sectors
            green_impacts[sector] = green_investment * base_multiplier * 0.1
        else:
            # Small general impact
            green_impacts[sector] = green_investment * base_multiplier * 0.05
    
    return green_impacts