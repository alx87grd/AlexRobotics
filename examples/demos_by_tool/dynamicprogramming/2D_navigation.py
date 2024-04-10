#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import vehicle_steering
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming 
from pyro.planning import discretizer

sys  = vehicle_steering.HolonomicMobileRobotwithObstacles()

#sys.obstacles[1][0] = (5,5)

#sys.x_ub[1] = 15
#sys.x_lb[1] = 0

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [51,51] , [3,3] )

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys( sys )

qcf.xbar = np.array([ 10. , 0. ]) # target
qcf.Q[0,0] = 1.0
qcf.Q[1,1] = 1.0
qcf.R[0,0] = 0.0
qcf.R[1,1] = 0.0
qcf.S[0,0] = 10.0
qcf.S[1,1] = 10.0
qcf.INF    = 8000

# DP algo
#dp = dprog.DynamicProgramming( grid_sys, qcf )
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf)

#dp.plot_cost2go()
#dp.solve_bellman_equation( tol = 0.01 )
dp.solve_bellman_equation( tol = 0.01 , animate_cost2go = True )
#dp.solve_bellman_equation( tol = 0.01 , animate_policy = True )

#grid_sys.plot_grid_value( dp.J_next )

ctl = dp.get_lookup_table_controller()

#ctl.plot_control_law( k=0 , sys = sys , n = 50)
#ctl.plot_control_law( k=1 , sys = sys , n = 50)


##############################################################################

# Simulation and animation
cl_sys      = ctl + sys
cl_sys.x0   = np.array([-8,5])
cl_sys.compute_trajectory( 60, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation( time_factor_video=15.0 )