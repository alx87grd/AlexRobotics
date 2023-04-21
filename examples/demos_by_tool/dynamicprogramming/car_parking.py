#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming 
from pyro.planning import discretizer
from pyro.dynamic  import vehicle

sys  = vehicle.KinematicCarModelwithObstacles()

# Set domain
sys.x_ub = np.array([+35, +3, +3])
sys.x_lb = np.array([-5, -2, -3])

sys.u_ub = np.array([+3, +1])
sys.u_lb = np.array([-3, -1])

# Discrete world
grid_sys = discretizer.GridDynamicSystem(sys, (51, 51, 21), (3, 3), 0.1)

# Cost Function
cf = costfunction.QuadraticCostFunction.from_sys( sys )
cf.xbar = np.array( [0, 0, 0] ) # target
cf.INF  = 1E8
cf.EPS  = 0.00
cf.R    = np.array([[0.1,0],[0,0]])

# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, cf)

dp.alpha = 0.99
dp.solve_bellman_equation()


ctl = dp.get_lookup_table_controller()

ctl.plot_control_law( sys = sys , n = 100)

##############################################################################

# Simulation and animation
cl_sys = ctl + sys
cl_sys.x0   = np.array([30,0,0])
cl_sys.compute_trajectory( 10, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()