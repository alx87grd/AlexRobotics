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
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf)
#dp = dprog.DynamicProgramming2DRectBivariateSpline(grid_sys, qcf)

dp.solve_bellman_equation( animate_cost2go = True )
