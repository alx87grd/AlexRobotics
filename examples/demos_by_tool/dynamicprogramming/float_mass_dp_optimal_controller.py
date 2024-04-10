#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import massspringdamper
from pyro.control  import controller
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming
from pyro.planning import discretizer

sys  = massspringdamper.FloatingSingleMass()

sys.x_ub[0] =  10.0
sys.x_lb[0] = -10.0
sys.x_lb[1] = -5.0
sys.x_ub[1] = 5.0
sys.u_ub[0] = 5.0
sys.u_lb[0] = -5.0

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [41] , 0.05)

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ -0 , 0 ]) # target
qcf.INF  = 300

qcf.R[0,0] = 10.0

qcf.S[0,0] = 10.0
qcf.S[1,1] = 10.0


# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf)

dp.solve_bellman_equation( tol = 0.5 )

ctl = dp.get_lookup_table_controller()

dp.plot_cost2go()
dp.plot_cost2go_3D()

ctl.plot_control_law( sys = sys , n = 100)


#asign controller
sys.C = np.eye(2)
sys.p = 2
cl_sys = controller.ClosedLoopSystem( sys , ctl )

##############################################################################

# Simulation and animation
cl_sys.x0   = np.array([+5,+3])
cl_sys.compute_trajectory( 20, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()