# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import cartpole
###############################################################################

sys  = cartpole.RotatingCartPole()

# Simultation
sys.xbar[1] = -3.1416
sys.animate_linearized_modes()