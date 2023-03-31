#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:35:09 2023

@author: alex
"""
import numpy as np

from pyro.dynamic  import pendulum
from pyro.planning import discretizer
from pyro.analysis import costfunction
from approximatedynamicprogramming import LinearApproximateDynamicProgramming
from functionapproximation import QuadraticFunctionApproximator
from functionapproximation import MultipleGaussianFunctionApproximator

sys  = pendulum.SinglePendulum()

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [3] )

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ -3.14 , 0 ]) # target
qcf.INF  = 300

# Approx

fa = QuadraticFunctionApproximator( sys.n , x0 = qcf.xbar )

# Discrete world 
grid_sys_gaussian = discretizer.GridDynamicSystem( sys , [11,11] , [3] , 0.05)
X0 = grid_sys_gaussian.state_from_node_id

#fa = MultipleGaussianFunctionApproximator( X0 , 3.0 ) + fa

fa = MultipleGaussianFunctionApproximator( X0 , 1.0 )

# DP algo
dp = LinearApproximateDynamicProgramming( grid_sys, qcf, fa )

dp.alpha = 0.8
dp.gamma = 1.0

dp.w = dp.w + 20

#dp.solve_bellman_equation( tol = 0.1 )
dp.compute_steps(100)

#dp.plot_cost2go()
#dp.plot_cost2go_3D()
dp.animate_cost2go()