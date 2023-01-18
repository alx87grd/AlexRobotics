#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import pendulum
from pyro.dynamic  import longitudinal_vehicule

import dynamic_programming as dprog
import discretizer
import costfunction

# sys  = pendulum.SinglePendulum()

sys  = longitudinal_vehicule.LongitudinalFrontWheelDriveCarWithWheelSlipInput()

sys.x_ub[0] = 60
sys.x_lb[0] = 0
sys.x_ub[1] = 20
sys.x_lb[1] = 0

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [3] )

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ 0, 0 ]) # target
qcf.INF  = 1000

qcf.Q[0,0] = 0.0
qcf.Q[1,1] = 0.0
qcf.R[0,0] = 0.0

qcf.S[0,0] = 0.0
qcf.S[1,1] = 10000.0


# DP algo
dp = dprog.DynamicProgrammingWithLookUpTable( grid_sys, qcf)
#dp = dprog.DynamicProgramming2DRectBivariateSpline(grid_sys, qcf)

dp.solve_bellman_equation( animate_cost2go = True )

#dp.compute_steps(200)
# dp.plot_policy()

# dp.solve_bellman_equation( tol = 1 , animate_policy = True )
#dp.plot_cost2go(150)

#dp.animate_cost2go( show = False , save = True )
#dp.animate_policy( show = False , save = True )


# ctl = dp.get_lookup_table_controller()

#ctl.plot_control_law( sys = sys , n = 100)


#asign controller
# cl_sys = controller.ClosedLoopSystem( sys , ctl )
# cl_sys.x0   = np.array([0,0])
# cl_sys.compute_trajectory( 10, 10001, 'euler')
# cl_sys.plot_trajectory('xu')
# cl_sys.plot_phase_plane_trajectory()
# cl_sys.animate_simulation()
