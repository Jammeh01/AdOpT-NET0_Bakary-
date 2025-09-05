# simpleCGE.py -  CGE System Solver
import numpy as np
from pandas import Series
import firms
import household as hh
import government as gov
import aggregates as agg


def cge_system(pvec, p, d, ind, h, Zbar, Qbar, Kdbar, pdbar, Ffbar, R, er):
    """
    ThreeMe CGE system of equations.

    Args:
        pvec: Price vector [py_1, ..., py_n, pf_1, ..., pf_m]
        p: Parameters object
        d: Model data object  
        ind: List of sectors/industries
        h: List of factors
        Zbar: Current output levels
        Qbar: Current Armington quantities
        Kdbar: Current domestic capital
        pdbar: Current domestic prices
        Ffbar: Current factor endowments
        R: Return to capital
        er: Exchange rate

    Returns:
        residuals: System residuals (should be zero at equilibrium)
    """
    n_sectors = len(ind)
    n_factors = len(h)

    # Unpack price vector
    py = Series(pvec[:n_sectors], index=ind)      # Value added prices
    pf = Series(pvec[n_sectors:n_sectors+n_factors], index=h)  # Factor prices

    # World prices (exogenous)
    pe = firms.eqpe(er, d.pWe)
    pm = firms.eqpm(er, d.pWm)

    # Production side
    Y = firms.eqY(p.ay, Zbar)
    X = firms.eqX(p.ax, Zbar)

    # Factor demands
    F = hh.eqF(p.beta, py, Y, pf)

    # Prices
    pq = firms.eqpq(pm, pdbar, p.taum, p.eta, p.deltam, p.deltad, p.gamma)
    pz = firms.eqpz(p.ay, p.ax, py, pq)

    # Capital market
    Kk = agg.eqKk(pf, Ffbar, R, p.lam, pq)

    # Government
    Td = gov.eqTd(p.taud, pf, Ffbar)
    Trf = gov.eqTrf(p.tautr, pf, Ffbar)

    # Foreign sector
    Kf = agg.eqKf(Kk, Kdbar)
    Fsh = firms.eqFsh(R, Kf, er)

    # Household sector
    Sp = agg.eqSp(p.ssp, pf, Ffbar, Fsh, Trf)
    I = hh.eqI(pf, Ffbar, Sp, Td, Fsh, Trf)
    Xp = hh.eqXp(p.alpha, I, pq)

    # Investment
    XXv = agg.eqXXv(d.g, Kk)
    Xv = firms.eqXv(p.lam, XXv)

    # Government consumption
    XXg = d.XXg0  # Exogenous
    Xg = gov.eqXg(p.mu, XXg)

    # Trade (simplified)
    E = firms.eqE(p.theta, p.xie, p.tauz, p.phi, pz, pe, Zbar)
    D = firms.eqDex(p.theta, p.xid, p.tauz, p.phi, pz, pdbar, Zbar)
    M = firms.eqM(p.gamma, p.deltam, p.eta, Qbar, pq, pm, p.taum)

    # System residuals
    residuals = np.zeros(n_sectors + n_factors)

    # Factor market clearing
    residuals[n_sectors] = agg.eqLabor_market_clearing(F, Ffbar)
    residuals[n_sectors +
              1] = agg.eqCapital_market_clearing(F, Kk, d.Kk0, d.Ff0)

    # Goods market clearing
    for i, sector in enumerate(ind):
        intermediate_demand = X[sector].sum() if hasattr(X, 'sum') else 0
        total_demand = Xp[sector] + Xg[sector] + \
            Xv[sector] + intermediate_demand
        residuals[i] = Qbar[sector] - total_demand

    return residuals


def solve_threeme_equilibrium(data, params, time_period=0, method='root', **kwargs):
    """
    Solve ThreeMe equilibrium for a given time period.

    Args:
        data: Model data object
        params: Model parameters object
        time_period: Time period (0 = base year)
        method: Solution method ('root', 'fixed_point')
        **kwargs: Additional solver options

    Returns:
        dict: Equilibrium solution
    """
    import scipy.optimize as opt

    ind = data.sectors if hasattr(data, 'sectors') else list(data.Z0.index)
    h = ['Labour', 'Capital']

    # Initial conditions
    Zbar = data.Z0 * (1 + 0.02) ** time_period  # 2% growth
    Qbar = data.Q0 * (1 + 0.02) ** time_period
    Kdbar = data.Kd0 * (1 + 0.02) ** time_period
    pdbar = Series(1.0, index=ind)
    Ffbar = data.Ff0.copy()
    Ffbar['Labour'] *= (1 + 0.001) ** time_period  # Labor growth

    R = data.R0
    er = 1.0

    # Initial price guess
    initial_prices = np.ones(len(ind) + len(h))

    # Solver options
    solver_options = {
        'method': 'lm',
        'options': {'ftol': 1e-8, 'maxiter': 200}
    }
    solver_options.update(kwargs)

    try:
        if method == 'root':
            solution = opt.root(
                cge_system,
                initial_prices,
                args=(params, data, ind, h, Zbar, Qbar,
                      Kdbar, pdbar, Ffbar, R, er),
                **solver_options
            )

            if solution.success:
                # Unpack solution
                py_sol = Series(solution.x[:len(ind)], index=ind)
                pf_sol = Series(solution.x[len(ind):], index=h)

                # Calculate equilibrium quantities
                Y = firms.eqY(params.ay, Zbar)
                F = hh.eqF(params.beta, py_sol, Y, pf_sol)

                # GDP calculation
                GDP = agg.eqGDP_production(Y.sum())

                return {
                    'success': True,
                    'prices': {'py': py_sol, 'pf': pf_sol},
                    'quantities': {'Y': Y, 'Z': Zbar, 'F': F, 'GDP': GDP},
                    'iterations': solution.nfev,
                    'message': 'Converged successfully'
                }
            else:
                return {
                    'success': False,
                    'message': f"Failed to converge: {solution.message}",
                    'iterations': solution.nfev
                }

    except Exception as e:
        return {
            'success': False,
            'message': f"Solver error: {str(e)}",
            'iterations': 0
        }


def run_policy_simulation(data, params, policy_scenario, years_ahead=30):
    """
    Run multi-period policy simulation.

    Args:
        data: Model data
        params: Model parameters
        policy_scenario: Policy scenario parameters
        years_ahead: Number of years to simulate

    Returns:
        dict: Simulation results
    """
    results = {
        'scenario': policy_scenario.get('name', 'default'),
        'years': [],
        'equilibria': [],
        'carbon_prices': {'ets1': [], 'ets2': []},
        'aggregates': {
            'GDP': [], 'consumption': [], 'investment': [],
            'emissions': [], 'welfare': []
        }
    }

    for t in range(years_ahead):
        year = data.base_year + t
        results['years'].append(year)

        # Update carbon prices
        carbon_ets1 = gov.eqCarbon_price_path(year, data.base_year, 'ets1',
                                              policy_scenario.get('carbon_scenario', 'baseline'))
        carbon_ets2 = gov.eqCarbon_price_path(year, data.base_year, 'ets2',
                                              policy_scenario.get('carbon_scenario', 'baseline'))

        results['carbon_prices']['ets1'].append(carbon_ets1)
        results['carbon_prices']['ets2'].append(carbon_ets2)

        # Solve equilibrium
        equilibrium = solve_threeme_equilibrium(data, params, t)
        results['equilibria'].append(equilibrium)

        if equilibrium['success']:
            # Store aggregates
            results['aggregates']['GDP'].append(
                equilibrium['quantities']['GDP'])

            # Calculate other aggregates (simplified)
            # 60% consumption
            consumption = equilibrium['quantities']['GDP'] * 0.6
            # 20% investment
            investment = equilibrium['quantities']['GDP'] * 0.2
            emissions = 350 * (1 - 0.025) ** t  # 2.5% annual decline

            results['aggregates']['consumption'].append(consumption)
            results['aggregates']['investment'].append(investment)
            results['aggregates']['emissions'].append(emissions)
            results['aggregates']['welfare'].append(
                consumption / data.base_population)
        else:
            # Fill with NaN if failed
            for key in results['aggregates'].keys():
                results['aggregates'][key].append(np.nan)

    return results


def extract_linking_variables(simulation_results, data):
    """
    Extract variables for ADOPT_NET0 linking.

    Args:
        simulation_results: Results from run_policy_simulation
        data: Model data

    Returns:
        dict: ADOPT_NET0 linking variables
    """
    linking_vars = {
        'metadata': {
            'model': 'ThreeMe_Italy',
            'scenario': simulation_results['scenario'],
            'base_year': data.base_year,
            'regions': getattr(data, 'regions', ['NW', 'NE', 'Centre', 'South', 'Islands'])
        },
        'annual_data': {},
        'hourly_electricity_demand': {}
    }

    years = simulation_results['years']

    for i, year in enumerate(years):
        if i < len(simulation_results['aggregates']['GDP']):
            gdp = simulation_results['aggregates']['GDP'][i]
            emissions = simulation_results['aggregates']['emissions'][i]

            # Regional electricity demands (simplified)
            regional_electricity = {}
            regional_gas = {}

            total_elec_demand = 300000  # 300,000 MWh base
            total_gas_demand = 180000   # 180,000 MWh base

            # Growth and efficiency factors
            demand_growth = (1 + 0.005) ** i  # 0.5% annual growth
            # 1.5% annual efficiency improvement
            efficiency_gain = (1 - 0.015) ** i
            net_factor = demand_growth * efficiency_gain

            for region in linking_vars['metadata']['regions']:
                pop_share = data.regional_pop_shares.get(region, 0.2)

                regional_electricity[region] = total_elec_demand * \
                    pop_share * net_factor
                regional_gas[region] = total_gas_demand * pop_share * \
                    net_factor * (1 - 0.02) ** i  # Gas declining

            linking_vars['annual_data'][year] = {
                'gdp_million_eur': gdp,
                'regional_electricity_mwh': regional_electricity,
                'regional_gas_mwh': regional_gas,
                'carbon_price_ets1': simulation_results['carbon_prices']['ets1'][i],
                'carbon_price_ets2': simulation_results['carbon_prices']['ets2'][i],
                'total_emissions_mt': emissions,
                # 1% annual increase
                'electricity_price_eur_mwh': 80 * (1 + 0.01) ** i,
                # 2% annual increase
                'gas_price_eur_mwh': 65 * (1 + 0.02) ** i
            }

    return linking_vars


def validate_solution(equilibrium, tolerance=1e-3):
    """
    Validate equilibrium solution.

    Args:
        equilibrium: Equilibrium solution dict
        tolerance: Validation tolerance

    Returns:
        dict: Validation results
    """
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }

    if not equilibrium.get('success', False):
        validation['is_valid'] = False
        validation['errors'].append("Equilibrium failed to converge")
        return validation

    # Check price positivity
    prices = equilibrium.get('prices', {})
    for price_type, price_series in prices.items():
        if hasattr(price_series, 'min') and price_series.min() <= 0:
            validation['errors'].append(f"Non-positive prices in {price_type}")
            validation['is_valid'] = False

    # Check quantity positivity
    quantities = equilibrium.get('quantities', {})
    for qty_type, qty_series in quantities.items():
        if hasattr(qty_series, 'min') and qty_series.min() < 0:
            validation['warnings'].append(f"Negative quantities in {qty_type}")

    return validation
