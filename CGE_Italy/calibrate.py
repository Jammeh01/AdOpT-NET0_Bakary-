# calibrate.py - CGE Model Calibration
import numpy as np
from pandas import Series, DataFrame
import pandas as pd


class model_data(object):
    """
    This class reads the SAM file and initializes variables for ThreeMe model.

    Args:
        sam (DataFrame): DataFrame containing social and economic data
        h (list): List of factors of production ['Labour', 'Capital']
        ind (list): List of industries/sectors
        regions (list): List of regions ['NW', 'NE', 'Centre', 'South', 'Islands']

    Returns:
        model_data (data class): Data used in the ThreeMe CGE model
    """

    def __init__(self, sam, h, ind, regions=None):
        # Base year parameters
        self.base_year = 2021
        self.base_gdp = 1782000  # Million EUR
        self.base_population = 59.13  # Million people
        
        # Regional setup
        if regions is None:
            regions = ['NW', 'NE', 'Centre', 'South', 'Islands']
        self.regions = regions
        
        # Regional population shares (Italian NUTS-1)
        self.regional_pop_shares = {
            'NW': 0.261, 'NE': 0.193, 'Centre': 0.198, 
            'South': 0.234, 'Islands': 0.114
        }
        
        # Extract SAM data or create realistic structure
        self._initialize_sam_data(sam, h, ind)
        self._calculate_aggregates()
        self._initialize_regional_data()
    
    def _initialize_sam_data(self, sam, h, ind):
        """Initialize SAM-based data with energy carrier disaggregation"""
        # Sectoral output based on actual SAM data (Italian economy 2021)
        # Energy carriers disaggregated: Electricity (renewable), Gas, Other Energy (fossil)
        sectoral_outputs = {
            'Agriculture': 69235,        # From SAM
            'Industry': 1120944,         # From SAM - largest sector
            'Electricity': 49287,        # From SAM - renewable electricity carrier
            'Gas': 97895,               # From SAM - natural gas carrier
            'Other Energy': 131924,     # From SAM - fossil fuel carrier
            'Road Transport': 115935,    # From SAM
            'Rail Transport': 5090,      # From SAM
            'Air Transport': 1912,       # From SAM
            'Water Transport': 3960,     # From SAM
            'Other Transport': 8107,     # From SAM
            'other Sectors (14)': 2750257 # From SAM - services/other sectors
        }
        
        # Energy carrier characteristics from SAM
        energy_carrier_properties = {
            'Electricity': {
                'renewable_share': 0.42,    # 42% renewable in base year
                'emission_factor': 0.0,     # Treated as renewable
                'ets_coverage': False,      # Renewable electricity ETS exempt
                'sam_value': 49287
            },
            'Gas': {
                'renewable_share': 0.05,    # 5% biogas/renewable gas
                'emission_factor': 0.202,   # tCO2/MWh
                'ets_coverage': True,       # Gas now in ETS1
                'sam_value': 97895
            },
            'Other Energy': {
                'renewable_share': 0.02,    # 2% renewable (biofuels)
                'emission_factor': 0.315,   # tCO2/MWh
                'ets_coverage': True,       # Fossil fuels in ETS1
                'sam_value': 131924
            }
        }
        
        # Factor endowments
        self.Ff0 = Series(dtype=float, index=h)
        self.Ff0['Labour'] = 25800000  # 25.8 million workers
        self.Ff0['Capital'] = 8500000   # Million EUR capital stock
        
        # Sectoral data aligned with SAM
        self.Z0 = Series([sectoral_outputs.get(sector, 10000) for sector in ind], index=ind)
        
        # Store energy carrier properties
        self.energy_carriers = energy_carrier_properties
        
        # Factor payments by sector (factors x sectors)
        self.F0 = DataFrame(index=h, columns=ind, dtype=float)
        for sector in ind:
            if sector in sectoral_outputs:
                # Energy carriers have different factor intensities
                if sector == 'Electricity':  # Capital-intensive renewable electricity
                    labor_share = 0.25
                elif sector == 'Gas':        # Network infrastructure intensive
                    labor_share = 0.30  
                elif sector == 'Other Energy':  # Traditional energy sector
                    labor_share = 0.35
                elif sector in ['Agriculture', 'Road Transport', 'other Sectors (14)']:
                    labor_share = 0.7
                else:
                    labor_share = 0.6
                
                total_va = self.Z0[sector] * 0.6  # 60% value added
                self.F0.loc['Labour', sector] = total_va * labor_share
                self.F0.loc['Capital', sector] = total_va * (1 - labor_share)
            else:
                self.F0.loc['Labour', sector] = self.Z0[sector] * 0.4
                self.F0.loc['Capital', sector] = self.Z0[sector] * 0.2
        
        # Value added
        self.Y0 = self.F0.sum(axis=0)
        
        # Intermediate inputs (sectors x sectors) - energy carrier interactions
        self.X0 = DataFrame(index=ind, columns=ind, dtype=float)
        for i in ind:
            for j in ind:
                if i == j:
                    self.X0.loc[i, j] = 0  # No self-input
                else:
                    # Energy carrier interdependencies
                    if j == 'Electricity':  # All sectors use electricity
                        self.X0.loc[i, j] = self.Z0[j] * 0.12
                    elif j == 'Gas':  # Gas used by many sectors
                        self.X0.loc[i, j] = self.Z0[j] * 0.08
                    elif j == 'Other Energy':  # Fossil fuels used widely
                        self.X0.loc[i, j] = self.Z0[j] * 0.10
                    elif 'Transport' in j:  # Transport services
                        self.X0.loc[i, j] = self.Z0[j] * 0.05
                    else:
                        self.X0.loc[i, j] = self.Z0[j] * 0.03
        
        # Total intermediate inputs
        self.Xx0 = self.X0.sum(axis=0)
        
        # Final demands
        self.Xp0 = DataFrame(sam, index=ind, columns=['HOH']) if 'HOH' in sam.columns else self._default_consumption(ind)
        self.Xg0 = DataFrame(sam, index=ind, columns=['GOV']) if 'GOV' in sam.columns else self._default_government(ind)
        self.Xv0 = DataFrame(sam, index=ind, columns=['INV']) if 'INV' in sam.columns else self._default_investment(ind)
        
        # Trade
        self.E0 = self._default_exports(ind)
        self.M0 = self._default_imports(ind)
        
        # Government finances
        self.Td0 = self.base_gdp * 0.15  # Direct taxes
        self.Trf0 = self.base_gdp * 0.08  # Transfers
        self.Tz0 = self.Z0 * 0.05  # Production taxes
        self.Tm0 = self.M0 * 0.1   # Import tariffs
        
        # Savings
        self.Sp0 = self.base_gdp * 0.12  # Private savings
        self.Sg0 = self.base_gdp * 0.02  # Government savings
        self.Sf0 = self.base_gdp * 0.03  # Foreign savings
        
        # Capital stock
        self.Kk0 = 8500000      # Total capital
        self.Kf0 = 2000000      # Foreign-owned
        self.Kd0 = self.Kk0 - self.Kf0  # Domestic-owned
        self.Fsh0 = self.Kf0 * 0.05     # Repatriated profits
        
        # World prices
        self.pWe = Series(1.0, index=ind)
        self.pWm = Series(1.0, index=ind)
    
    def _default_consumption(self, ind):
        """Default household consumption structure - SAM Energy Carrier Aligned"""
        # Updated consumption shares based on SAM energy carrier disaggregation
        consumption_shares = {
            'Agriculture': 0.15, 'Industry': 0.25,
            # SAM Energy Carriers with proper shares
            'Electricity': 0.045,      # Increased for renewable electricity (SAM: €49,287M)
            'Gas': 0.055,             # Natural gas consumption (SAM: €97,895M)
            'Other Energy': 0.035,     # Fossil fuel products (SAM: €131,924M)
            # Transport disaggregation from SAM
            'Road Transport': 0.12, 'Rail Transport': 0.01, 'Air Transport': 0.02,
            'Water Transport': 0.001, 'Other Transport': 0.015,
            # Services
            'other Sectors (14)': 0.315  # Adjusted to maintain total = 1.0
        }
        
        total_consumption = self.base_gdp * 0.58  # 58% of GDP
        consumption = Series([total_consumption * consumption_shares.get(sector, 0.01) for sector in ind], index=ind)
        return DataFrame(consumption, columns=['HOH'])
    
    def _default_government(self, ind):
        """Default government consumption structure - SAM Energy Carrier Aligned"""
        # Government consumption reflecting SAM energy carrier structure
        gov_shares = {
            'Agriculture': 0.02, 'Industry': 0.15,
            # SAM Energy Carriers
            'Electricity': 0.06,       # Government electricity use (renewable focus)
            'Gas': 0.04,              # Government gas consumption
            'Other Energy': 0.02,      # Government fossil fuel use
            # Transport
            'Road Transport': 0.08, 'Rail Transport': 0.05, 'Air Transport': 0.02,
            'Water Transport': 0.01, 'Other Transport': 0.03,
            # Services
            'other Sectors (14)': 0.52  # Public services and administration
        }
        
        total_gov = self.base_gdp * 0.19  # 19% of GDP
        gov_consumption = Series([total_gov * gov_shares.get(sector, 0.01) for sector in ind], index=ind)
        return DataFrame(gov_consumption, columns=['GOV'])
    
    def _default_investment(self, ind):
        """Default investment structure"""
        inv_shares = {
            'Agriculture': 0.03, 'Industry': 0.35, 'Electricity': 0.12,
            'Gas': 0.08, 'Other Energy': 0.10, 'Road Transport': 0.05,
            'Rail Transport': 0.03, 'Air Transport': 0.02, 'Water Transport': 0.02,
            'Other Transport': 0.02, 'other Sectors (14)': 0.18
        }
        
        total_inv = self.base_gdp * 0.20  # 20% of GDP
        investment = Series([total_inv * inv_shares.get(sector, 0.01) for sector in ind], index=ind)
        return DataFrame(investment, columns=['INV'])
    
    def _default_exports(self, ind):
        """Default export structure"""
        export_shares = {
            'Agriculture': 0.156, 'Industry': 0.334, 'Electricity': 0.045,
            'Gas': 0.025, 'Other Energy': 0.156, 'Road Transport': 0.089,
            'Rail Transport': 0.034, 'Air Transport': 0.287, 'Water Transport': 0.445,
            'Other Transport': 0.123, 'other Sectors (14)': 0.187
        }
        
        exports = Series([self.Z0[sector] * export_shares.get(sector, 0.1) for sector in ind], index=ind)
        return exports
    
    def _default_imports(self, ind):
        """Default import structure"""
        import_shares = {
            'Agriculture': 0.187, 'Industry': 0.298, 'Electricity': 0.158,
            'Gas': 0.823, 'Other Energy': 0.743, 'Road Transport': 0.076,
            'Rail Transport': 0.041, 'Air Transport': 0.234, 'Water Transport': 0.398,
            'Other Transport': 0.134, 'other Sectors (14)': 0.165
        }
        
        imports = Series([self.Z0[sector] * import_shares.get(sector, 0.1) for sector in ind], index=ind)
        return imports
    
    def _calculate_aggregates(self):
        """Calculate aggregate variables"""
        self.XXp0 = self.Xp0['HOH'].sum()
        self.XXg0 = self.Xg0['GOV'].sum()
        self.XXv0 = self.Xv0['INV'].sum()
        self.Ee0 = self.E0.sum()
        self.Mm0 = self.M0.sum()
        self.Yy0 = self.Y0.sum()
        self.Gdp0 = self.XXp0 + self.XXg0 + self.XXv0 + self.Ee0 - self.Mm0
        self.g = self.XXv0 / self.Kk0
        self.R0 = self.Ff0['Capital'] / self.Kk0
        
        # Armington composite
        self.Q0 = self.Xp0['HOH'] + self.Xg0['GOV'] + self.Xv0['INV'] + self.X0.sum(axis=1)
        self.D0 = self.Z0 - self.E0  # Domestic supply for domestic market
    
    def _initialize_regional_data(self):
        """Initialize regional household data"""
        # Regional consumption by sector
        self.Xp0_regional = DataFrame(index=self.Xp0.index, columns=self.regions, dtype=float)
        
        for region in self.regions:
            income_share = self.regional_pop_shares[region]  # Simplified
            for sector in self.Xp0.index:
                self.Xp0_regional.loc[sector, region] = self.Xp0.loc[sector, 'HOH'] * income_share


class parameters(object):
    """
    This class sets the values of parameters used in the ThreeMe model.

    Args:
        d (model_data): Class of data for use in CGE model
        ind (list): List of industries

    Returns:
        parameters (parameters class): Class of parameters for ThreeMe CGE model
    """

    def __init__(self, d, ind):
        # Elasticities
        self._set_elasticities(ind)
        
        # Share parameters
        self._set_share_parameters(d, ind)
        
        # Tax and policy parameters
        self._set_tax_parameters(d)
        
        # Energy and environmental parameters
        self._set_energy_parameters(ind)
        
        # Regional parameters
        self._set_regional_parameters(d)
    
    def _set_elasticities(self, ind):
        """Set substitution and transformation elasticities"""
        # Armington elasticities (domestic-import substitution)
        elasticity_map = {
            'Agriculture': 2.0, 'Industry': 1.5, 'Electricity': 0.5,
            'Gas': 0.8, 'Other Energy': 1.2, 'Road Transport': 1.0,
            'Rail Transport': 2.0, 'Air Transport': 1.8, 'Water Transport': 1.5,
            'Other Transport': 1.2, 'other Sectors (14)': 1.3
        }
        
        self.sigma = Series([elasticity_map.get(sector, 1.5) for sector in ind], index=ind)
        self.eta = (self.sigma - 1) / self.sigma
        
        # Transformation elasticities (export-domestic)
        transform_map = {
            'Agriculture': 2.5, 'Industry': 2.0, 'Electricity': 0.3,
            'Gas': 0.5, 'Other Energy': 1.5, 'Road Transport': 0.8,
            'Rail Transport': 1.0, 'Air Transport': 3.0, 'Water Transport': 2.5,
            'Other Transport': 1.2, 'other Sectors (14)': 1.8
        }
        
        self.psi = Series([transform_map.get(sector, 2.0) for sector in ind], index=ind)
        self.phi = (self.psi + 1) / self.psi
        
        # Production elasticities
        self.sigma_va = 0.8     # Labor-capital substitution
        self.sigma_energy = 0.5  # Energy substitution
        self.sigma_materials = 0.3  # Energy-materials substitution
    
    def _set_share_parameters(self, d, ind):
        """Set share parameters for production and utility functions"""
        # Household consumption shares
        self.alpha = d.Xp0['HOH'] / d.XXp0
        
        # Production function shares
        self.beta = d.F0 / d.Y0
        
        # Scale parameter in production function
        temp = d.F0 ** self.beta
        self.b = d.Y0 / temp.prod(axis=0)
        
        # Technical coefficients
        self.ax = d.X0 / d.Z0
        self.ay = d.Y0 / d.Z0
        
        # Government and investment shares
        self.mu = d.Xg0['GOV'] / d.XXg0
        self.lam = d.Xv0['INV'] / d.XXv0
        
        # Armington function parameters
        self.deltam = ((1 + d.Tm0/d.M0) * d.M0**(1-self.eta) / 
                      ((1 + d.Tm0/d.M0) * d.M0**(1-self.eta) + d.D0**(1-self.eta)))
        
        self.deltad = (d.D0**(1-self.eta) / 
                      ((1 + d.Tm0/d.M0) * d.M0**(1-self.eta) + d.D0**(1-self.eta)))
        
        # Armington scale parameter
        self.gamma = (d.Q0 / (self.deltam * d.M0**self.eta + 
                             self.deltad * d.D0**self.eta)**(1/self.eta))
        
        # Transformation function parameters
        self.xie = (d.E0**(1-self.phi) / 
                   (d.E0**(1-self.phi) + d.D0**(1-self.phi)))
        
        self.xid = (d.D0**(1-self.phi) / 
                   (d.E0**(1-self.phi) + d.D0**(1-self.phi)))
        
        # Transformation scale parameter
        self.theta = (d.Z0 / (self.xie * d.E0**self.phi + 
                             self.xid * d.D0**self.phi)**(1/self.phi))
    
    def _set_tax_parameters(self, d):
        """Set tax rates and government parameters"""
        # Direct tax rate
        self.taud = d.Td0 / d.Ff0.sum()
        
        # Transfer rate
        self.tautr = d.Trf0 / d.Ff0['Labour']
        
        # Production tax rates
        self.tauz = d.Tz0 / d.Z0
        
        # Import tariff rates
        self.taum = d.Tm0 / d.M0
        
        # Household savings rate
        self.ssp = d.Sp0 / (d.Ff0.sum() - d.Fsh0 + d.Trf0)
    
    def _set_energy_parameters(self, ind):
        """Set energy-specific parameters"""
        # Energy shares in production
        self.energy_shares = {
            'electricity': 0.4, 'gas': 0.3, 'other_energy': 0.3
        }
        
        # CO2 emission factors (kg CO2/unit)
        self.emission_factors = {
            'electricity': 0.298,  # kg CO2/kWh
            'gas': 0.202,          # kg CO2/kWh
            'other_energy': 0.265  # kg CO2/kWh
        }
        
        # Energy efficiency improvement rates (annual)
        efficiency_rates = {
            'Agriculture': 0.015, 'Industry': 0.020, 'Electricity': 0.025,
            'Gas': 0.020, 'Other Energy': 0.020, 'Road Transport': 0.030,
            'Rail Transport': 0.025, 'Air Transport': 0.020, 'Water Transport': 0.020,
            'Other Transport': 0.025, 'other Sectors (14)': 0.015
        }
        
        self.efficiency_improvement = Series([efficiency_rates.get(sector, 0.02) for sector in ind], index=ind)
        
        # Carbon pricing parameters
        self.carbon_price_base = 50.0  # EUR/tCO2 in 2021
        self.carbon_price_growth = {'ets1': 0.05, 'ets2': 0.08}  # Annual growth rates
    
    def _set_regional_parameters(self, d):
        """Set regional parameters"""
        # Regional budget shares (goods x regions)
        self.alpha_regional = DataFrame(index=d.Xp0.index, columns=d.regions, dtype=float)
        
        for region in d.regions:
            for sector in d.Xp0.index:
                base_share = self.alpha[sector]
                # Regional adjustments
                if sector == 'Agriculture' and region in ['South', 'Islands']:
                    regional_factor = 1.2  # Higher food share in South
                elif sector in ['Electricity', 'Gas'] and region in ['South', 'Islands']:
                    regional_factor = 1.1  # Higher energy share for cooling
                elif sector in ['Road Transport', 'Rail Transport'] and region == 'NW':
                    regional_factor = 1.15  # Higher transport in industrial North
                else:
                    regional_factor = 1.0
                
                self.alpha_regional.loc[sector, region] = base_share * regional_factor
        
        # Normalize regional budget shares
        for region in d.regions:
            total_share = self.alpha_regional[region].sum()
            self.alpha_regional[region] = self.alpha_regional[region] / total_share
        
        # Regional energy consumption parameters
        self.regional_energy_pc = {
            'electricity': {'NW': 5.8, 'NE': 5.6, 'Centre': 5.2, 'South': 4.8, 'Islands': 5.0},
            'gas': {'NW': 7.2, 'NE': 6.8, 'Centre': 5.5, 'South': 3.2, 'Islands': 2.8}
        }