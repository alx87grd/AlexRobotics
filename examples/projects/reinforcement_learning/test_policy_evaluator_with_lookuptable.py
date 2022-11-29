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

sys.x_ub = np.array([+6, +6])
sys.x_lb = np.array([-9,  -6])

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [11] )

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ -3.14 , 0 ]) # target
qcf.INF  = 300

qcf.S[0,0] = 10.0
qcf.S[1,1] = 10.0


# DP algo
dp = dprog.DynamicProgrammingWithLookUpTable( grid_sys, qcf)

dp.solve_bellman_equation( tol = 0.1 )
dp.plot_cost2go()

# Get optimal ctl
ctl = dp.get_lookup_table_controller()


# Evaluate on same grid
#evaluator = dprog.PolicyEvaluatorWithLookUpTable(ctl, grid_sys, qcf)
#evaluator.solve_bellman_equation()
#evaluator.plot_cost2go()

# Evaluate on new grid
grid_sys2 = discretizer.GridDynamicSystem( sys , [301,301] , [11] ,  )

evaluator2 = dprog.PolicyEvaluatorWithLookUpTable(ctl, grid_sys2, qcf)
evaluator2.solve_bellman_equation()
evaluator2.plot_cost2go()


# Evaluate on new grid

# Raise the max torque to avoid hitting the min-max boundary with interpolation
sys.u_ub[0] = +10
sys.u_lb[0] = -10

grid_sys2 = discretizer.GridDynamicSystem( sys , [301,301] , [11] )

evaluator2 = dprog.PolicyEvaluatorWithLookUpTable(ctl, grid_sys2, qcf)
evaluator2.solve_bellman_equation()
evaluator2.plot_cost2go()
