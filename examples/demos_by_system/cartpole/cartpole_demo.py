#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:01:30 2023

@author: alex
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import cartpole
###############################################################################

sys  = cartpole.CartPole()

# Simultation
def t2u(t = 0):
    
    u = np.array([ 5 * np.cos( 5 * t) ])
    
    return u

sys.t2u = t2u


    
sys.compute_trajectory()
sys.animate_simulation()