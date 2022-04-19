#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:53:54 2022

@author: alex
"""

###############################################################################
import numpy as np
###############################################################################
from hybrid_mechanical import TwoSpeedLinearActuator

from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import valueiteration
###############################################################################

sys  = TwoSpeedLinearActuator()

###############################################################################

# Planning

# Set domain
sys.x_ub = np.array([+3, +3])
sys.x_lb = np.array([-3, -3])

sys.u_ub = np.array([+1, +1])
sys.u_lb = np.array([-0, -1])

# Discrete world
grid_sys = discretizer.GridDynamicSystem(sys, (51, 51), (2, 11), 0.1)

# Cost Function
cf = costfunction.QuadraticCostFunction.from_sys( sys )
cf.xbar = np.array( [1, 0] ) # target
cf.INF  = 1E9
cf.EPS  = 0.2
cf.R    = np.array([[0,0],[0,1]])

# VI algo
vi = valueiteration.ValueIteration_ND( grid_sys , cf )

vi.uselookuptable = True
vi.initialize()


vi.compute_steps(100)

vi.assign_interpol_controller()
vi.plot_policy(0)
vi.plot_policy(1)

cl_sys = vi.ctl + sys

cl_sys.x0 = np.array([0, 0])
cl_sys.compute_trajectory( 10, 10001, 'euler')
cl_sys.plot_trajectory('xu')
