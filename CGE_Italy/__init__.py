"""
CGE Model Package
"""
__version__ = "1.0.0"
__author__ = "Bakary Jammeh Model"

# Make key classes and functions available at package level
from .calibrate import model_data, parameters
from .main import runner