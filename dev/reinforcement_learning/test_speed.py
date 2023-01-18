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

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [51,51] , [11] )

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ -3.14 , 0 ]) # target
qcf.INF  = 1000000


# DP algo
dp1 = dprog.DynamicProgramming( grid_sys, qcf )
dp1.compute_steps(10)

dp2 = dprog.DynamicProgrammingWithLookUpTable( grid_sys, qcf)
dp2.compute_steps(10)

dp4 = dprog.DynamicProgramming2DRectBivariateSpline(grid_sys, qcf)
dp4.compute_steps(10)