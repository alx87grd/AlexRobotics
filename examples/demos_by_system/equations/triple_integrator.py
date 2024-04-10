# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:17:36 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import integrator
from pyro.analysis import costfunction
###################################

# Simple integrator
sys = integrator.TripleIntegrator()

# Simulation params
sys.ubar = np.array([1]) 
sys.x0   = np.array([2,0,0])

# Compute and show results
sys.plot_trajectory('xuj')
sys.plot_phase_plane_trajectory(0, 1)
sys.plot_phase_plane_trajectory_3d()