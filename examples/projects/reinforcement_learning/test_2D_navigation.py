#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import longitudinal_vehicule
from pyro.dynamic  import vehicle

sys  = vehicle.HolonomicMobileRobotwithObstacles()



from pyro.control  import controller
import dynamic_programming as dprog
import discretizer
import costfunction

sys  = vehicle.HolonomicMobileRobotwithObstacles()

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
qcf.INF  = 8000

# DP algo
#dp = dprog.DynamicProgramming( grid_sys, qcf )
dp = dprog.DynamicProgrammingWithLookUpTable( grid_sys, qcf)
dp.plot_cost2go()


dp.solve_bellman_equation( tol = 0.01 )
dp.plot_cost2go()


#grid_sys.plot_grid_value( dp.J_next )

ctl = dprog.LookUpTableController( grid_sys , dp.pi )

#ctl.plot_control_law( k=0 , sys = sys , n = 50)
#ctl.plot_control_law( k=1 , sys = sys , n = 50)


#asign controller
cl_sys = controller.ClosedLoopSystem( sys , ctl )

##############################################################################

# Simulation and animation
cl_sys.x0   = np.array([-8,5])
cl_sys.compute_trajectory( 60, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation( time_factor_video=15.0 )