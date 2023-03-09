#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import suspension
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming 
from pyro.planning import discretizer

sys  = suspension.QuarterCarOnRoughTerrain()

sys.mass = 0.5
sys.b    = 0.5
sys.k    = 8.0
sys.vx   = 10.0

# Set domain
sys.x_ub = np.array([+12, +1, +40])
sys.x_lb = np.array([-12, -1, +0])

sys.u_ub = np.array([+40])
sys.u_lb = np.array([-40])

# Discrete world
grid_sys = discretizer.GridDynamicSystem(sys, (51, 51, 101), [11], 0.05)

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ 0.0 , 0.0, 20 ]) # target
qcf.INF  = 100000
qcf.EPS  = 0.5

qcf.Q[0,0] = 2.0
qcf.Q[1,1] = 5.0
qcf.Q[2,2] = 0.0

qcf.R[0,0] = 0.1

qcf.S[0,0] = 0.0
qcf.S[1,1] = 0.0
qcf.S[2,2] = 0.0

# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf)

dp.alpha = 0.99
dp.solve_bellman_equation()


ctl = dp.get_lookup_table_controller()

cl_sys = ctl + sys

# Simulation and animation
cl_sys.x0   = np.array([0,0,-60])
cl_sys.compute_trajectory( 10, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()