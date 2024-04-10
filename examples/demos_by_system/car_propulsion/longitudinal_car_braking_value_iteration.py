#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 13:40:47 2022

@author: alex
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import vehicle_propulsion
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming
from pyro.control  import controller
###############################################################################

sys  = vehicle_propulsion.LongitudinalFrontWheelDriveCarWithWheelSlipInput()

###############################################################################

# Planning

# Set domain
sys.x_ub = np.array([+50, 20,])
sys.x_lb = np.array([0, 0])
sys.u_ub = np.array([0.0])
sys.u_lb = np.array([-0.1])

# Discrete world
grid_sys = discretizer.GridDynamicSystem(sys, (101, 101), (11,), 0.1)

# Cost Function
cf = costfunction.QuadraticCostFunction.from_sys( sys )
cf.xbar = np.array( [0, 0] ) # target
cf.INF  = 1E3
cf.EPS  = 0.00
cf.R[0]   = 1
cf.Q[0,0] = 0
cf.Q[1,1] = 0.01

# VI algo
vi = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys , cf )

vi.solve_bellman_equation()

# Closed-loop Law

vi.plot_cost2go_3D()

ctl = vi.get_lookup_table_controller()

cl_sys = ctl + sys

###############################################################################

## Simulation and animation

x0   = np.array([0, 16])
tf   = 10

cl_sys.x0 = x0
cl_sys.compute_trajectory(tf, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video = 3 )