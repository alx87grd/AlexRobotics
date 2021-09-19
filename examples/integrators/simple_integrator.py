# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:13:40 2018

@author: Alexandre
"""
###############################################################################
import numpy as np

###############################################################################
from pyro.dynamic  import integrator
from pyro.analysis import costfunction
###################################
# Simple integrator
###################################

# Simple integrator
sys      = integrator.SimpleIntegrator()

# Default input signal
sys.ubar = np.array([1]) 

###################################
# Analysis
###################################

# Phase plane behavior
sys.plot_phase_plane(0,0) # only one state for two axis!

# Cost function with unit weights
sys.cost_function = costfunction.QuadraticCostFunction(1,1,1)

sys.cost_function.Q[0,0] = 0.1

# Simulation
sys.x0 = np.array([2])

# Plot output
sys.plot_trajectory('xuj')
sys.plot_phase_plane_trajectory(0,0)