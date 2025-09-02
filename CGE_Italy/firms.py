# firms.py - CGE Firm Behavior
import numpy as np
from pandas import Series, DataFrame


def eqpy(b, F, beta, Y):
    """
    Production function (Cobb-Douglas).
    Y_i = b_i * prod(F_{h,i}^{beta_{h,i}})
    """
    py_error = Y - b * (F**beta).prod(axis=0)
    return py_error


def eqX(ax, Z):
    """
    Intermediate input demands.
    X_{h,i} = ax_{h,j} * Z_j
    """
    X = ax * Z
    return X


def eqY(ay, Z):
    """
    Value added.
    Y_i = ay_{i,j} * Z_j
    """
    Y = ay * Z
    return Y


def eqF(beta, py, Y, pf):
    """
    Factor demands (Cobb-Douglas).
    F_{h,j} = beta_{h,j} * (py_j / pf_h) * Y_j
    """
    F = beta.div(pf, axis=0) * Y * py
    return F


def eqEnergy_demand(py, pf_energy, Y, energy_shares, sigma_energy=0.5):
    """
    Energy demand by type (CES function).
    
    Args:
        py: Value added prices
        pf_energy: Energy prices dict {'electricity', 'gas', 'other_energy'}
        Y: Value added by sector
        energy_shares: Energy share parameters
        sigma_energy: Energy substitution elasticity
    """
    eta_energy = (sigma_energy - 1) / sigma_energy
    
    # Energy price index
    pe_composite = ((energy_shares['electricity'] * pf_energy['electricity']**(1-sigma_energy) +
                    energy_shares['gas'] * pf_energy['gas']**(1-sigma_energy) +
                    energy_shares['other_energy'] * pf_energy['other_energy']**(1-sigma_energy))**(1/(1-sigma_energy)))
    
    # Total energy demand
    total_energy = 0.15 * py * Y / pe_composite  # 15% energy share
    
    # Energy demands by type
    electricity_demand = (total_energy * energy_shares['electricity'] * 
                         (pf_energy['electricity'] / pe_composite)**(-sigma_energy))
    
    gas_demand = (total_energy * energy_shares['gas'] * 
                 (pf_energy['gas'] / pe_composite)**(-sigma_energy))
    
    other_energy_demand = (total_energy * energy_shares['other_energy'] * 
                          (pf_energy['other_energy'] / pe_composite)**(-sigma_energy))
    
    return {
        'electricity': electricity_demand,
        'gas': gas_demand,
        'other_energy': other_energy_demand
    }


def eqCO2_emissions(electricity_demand, gas_demand, other_energy_demand, emission_factors):
    """
    Calculate CO2 emissions from energy use.
    """
    emissions = (electricity_demand * emission_factors['electricity'] +
                gas_demand * emission_factors['gas'] +
                other_energy_demand * emission_factors['other_energy'])
    return emissions


def eqpz(ay, ax, py, pq):
    """
    Output price formation.
    pz_i = ay_i * py_i + sum_j(ax_{j,i} * pq_j)
    """
    pz = ay * py + (ax * pq).sum(axis=0)
    return pz


def eqpe(er, pWe):
    """
    Export price in domestic currency.
    pe_i = er * pWe_i
    """
    pe = er * pWe
    return pe


def eqpm(er, pWm):
    """
    Import price in domestic currency.
    pm_i = er * pWm_i
    """
    pm = er * pWm
    return pm


def eqQ(gamma, deltam, deltad, eta, M, D):
    """
    Armington composite good (CES).
    Q_i = gamma_i * [deltam_i * M_i^eta_i + deltad_i * D_i^eta_i]^(1/eta_i)
    """
    Q = gamma * (deltam * M**eta + deltad * D**eta)**(1/eta)
    return Q


def eqM(gamma, deltam, eta, Q, pq, pm, taum):
    """
    Import demand (Armington).
    M_i = [gamma_i^eta_i * deltam_i * pq_i / ((1+taum_i)*pm_i)]^(1/(1-eta_i)) * Q_i
    """
    M = ((gamma**eta * deltam * pq / ((1 + taum) * pm))**(1/(1-eta))) * Q
    return M


def eqD(gamma, deltad, eta, Q, pq, pd):
    """
    Domestic demand from Armington.
    D_i = [gamma_i^eta_i * deltad_i * pq_i / pd_i]^(1/(1-eta_i)) * Q_i
    """
    D = ((gamma**eta * deltad * pq / pd)**(1/(1-eta))) * Q
    return D


def eqpd(gamma, deltad, eta, Q, pq, D):
    """
    Domestic price from Armington.
    pd_i = [gamma_i^eta_i * deltad_i * pq_i] * (D_i/Q_i)^(eta_i-1)
    """
    pd = (gamma**eta * deltad * pq) * (D / Q)**(eta - 1)
    return pd


def eqpq(pm, pd, taum, eta, deltam, deltad, gamma):
    """
    Armington price.
    """
    import_term = ((pm * (1 + taum))**eta / (deltam * gamma**eta))**(1/(eta-1))
    domestic_term = (pd**eta / (deltad * gamma**eta))**(1/(eta-1))
    pq = (import_term + domestic_term)**((eta-1)/eta)
    return pq


def eqZ(theta, xie, xid, phi, E, D):
    """
    Output transformation (CET).
    Z_i = theta_i * [xie_i * E_i^phi_i + xid_i * D_i^phi_i]^(1/phi_i)
    """
    Z = theta * (xie * E**phi + xid * D**phi)**(1/phi)
    return Z


def eqE(theta, xie, tauz, phi, pz, pe, Z):
    """
    Export supply (CET).
    E_i = [theta_i^phi_i * xie_i * (1+tauz_i) * pz_i / pe_i]^(1/(1-phi_i)) * Z_i
    """
    E = ((theta**phi * xie * (1 + tauz) * pz / pe)**(1/(1-phi))) * Z
    return E


def eqDex(theta, xid, tauz, phi, pz, pd, Z):
    """
    Domestic supply for domestic market (CET).
    D_i = [theta_i^phi_i * xid_i * (1+tauz_i) * pz_i / pd_i]^(1/(1-phi_i)) * Z_i
    """
    D = ((theta**phi * xid * (1 + tauz) * pz / pd)**(1/(1-phi))) * Z
    return D


def eqFsh(R, Kf, er):
    """
    Repatriated profits.
    Fsh = R * Kf * er
    """
    Fsh = R * Kf * er
    return Fsh


def eqXv(lam, XXv):
    """
    Investment demand by sector.
    Xv_i = lam_i * XXv
    """
    Xv = lam * XXv
    return Xv


def eqTechnology_change(base_tfp, time_period, carbon_price=0, base_growth=0.012):
    """
    Endogenous technology change.
    
    Args:
        base_tfp: Base total factor productivity
        time_period: Years since base year
        carbon_price: Carbon price (EUR/tCO2)
        base_growth: Base TFP growth rate
    """
    # Autonomous growth
    autonomous_factor = (1 + base_growth) ** time_period
    
    # Carbon-induced innovation (Porter hypothesis)
    carbon_innovation = 1 + 0.001 * carbon_price  # 0.1% per €10/tCO2
    
    new_tfp = base_tfp * autonomous_factor * carbon_innovation
    return new_tfp


def eqEnergy_efficiency(base_efficiency, time_period, carbon_price=0, base_improvement=0.02):
    """
    Energy efficiency improvements over time.
    
    Args:
        base_efficiency: Base year efficiency
        time_period: Years since base year
        carbon_price: Carbon price
        base_improvement: Base efficiency improvement rate
    """
    # Autonomous improvement
    autonomous_improvement = (1 + base_improvement) ** time_period
    
    # Price-induced improvement
    price_improvement = 1 + 0.002 * carbon_price  # 0.2% per €10/tCO2
    
    new_efficiency = base_efficiency * autonomous_improvement * price_improvement
    return new_efficiency


def eqCarbon_cost_passthrough(base_price, carbon_price, emission_intensity):
    """
    Carbon cost pass-through to output prices.
    
    Args:
        base_price: Base output price
        carbon_price: Carbon price (EUR/tCO2)
        emission_intensity: CO2 emissions per unit output (tCO2/unit)
    """
    carbon_cost = carbon_price * emission_intensity
    new_price = base_price + carbon_cost
    return new_price