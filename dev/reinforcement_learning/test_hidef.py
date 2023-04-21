#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import pendulum
from pyro.control  import controller
import dynamic_programming as dprog
import discretizer
import costfunction

sys  = pendulum.SinglePendulum()

sys.x_ub[0] = +2.1
sys.x_lb[0] = -3.3
sys.x_ub[1] = +4.0
sys.x_lb[1] = -5.0

sys.u_ub[0] = +2.5
sys.u_lb[0] = -2.5

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [401,401] , [31] )

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar   = np.array([ -3.14 , 0 ]) # target
qcf.R[0,0] = 10
qcf.INF    = 300

qcf.S[0,0] = 10.0
qcf.S[1,1] = 10.0

# DP algo

dp = dprog.DynamicProgrammingWithLookUpTable( grid_sys, qcf)

dp.solve_bellman_equation( tol = 0.1 )

dp.plot_cost2go()
dp.plot_policy()


#asign controller
ctl    = dp.get_lookup_table_controller()
cl_sys = controller.ClosedLoopSystem( sys , ctl )

##############################################################################

# Simulation and animation
cl_sys.x0   = np.array([0,0])
cl_sys.compute_trajectory( 30, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()