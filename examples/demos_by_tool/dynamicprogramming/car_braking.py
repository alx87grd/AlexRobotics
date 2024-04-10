#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import vehicle_propulsion
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming 
from pyro.planning import discretizer

sys  = vehicle_propulsion.LongitudinalFrontWheelDriveCarWithWheelSlipInput()

sys.x_ub[1] = 15
sys.x_lb[1] = 0

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [11] , 0.05 )

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys( sys )

qcf.xbar = np.array([ 45 , 0 ]) # target
qcf.Q[0,0] = 0.1
qcf.Q[1,1] = 0.1
qcf.INF  = 1000000

# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf)

dp.solve_bellman_equation( tol = 0.5 , animate_cost2go = True )


ctl = dp.get_lookup_table_controller()

ctl.plot_control_law( sys = sys , n = 100)


# Simulation and animation
cl_sys = ctl + sys
cl_sys.x0   = np.array([0,0])
cl_sys.compute_trajectory( 10, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()