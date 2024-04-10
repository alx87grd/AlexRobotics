#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import pendulum
from pyro.control  import nonlinear
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming 
from pyro.planning import discretizer

sys  = pendulum.SinglePendulum()

sys.x_ub = np.array([+6, +6])
sys.x_lb = np.array([-9,  -6])


# Get optimal ctl
ctl = nonlinear.ComputedTorqueController( sys )
ctl.rbar = np.array([-3.14]) # target

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ ctl.rbar[0] , 0 ]) # target
qcf.INF  = 300

qcf.S[0,0] = 10.0
qcf.S[1,1] = 10.0

# Min/max torque
sys.u_ub[0] = +200
sys.u_lb[0] = -200

# Evaluate on a grid
grid_sys = discretizer.GridDynamicSystem( sys , [301,301] , [11] , 0.05 , False )

evaluator = dynamicprogramming.PolicyEvaluatorWithLookUpTable(ctl, grid_sys, qcf)
evaluator.solve_bellman_equation()
evaluator.plot_cost2go()
