# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
API Interface Package

This package provides API endpoints for the Super AI prediction system.
"""

# Import modules to make them available when importing the package
from . import predictor_api
# Import financial API module
from . import financial_api

# Define what gets imported with 'from api_interface import *'
__all__ = ['predictor_api', 'financial_api']
