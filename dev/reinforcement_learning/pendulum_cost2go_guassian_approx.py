#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import pendulum
from pyro.dynamic  import massspringdamper
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming 

from scipy.linalg  import solve_continuous_are

from functionapproximation import QuadraticFunctionApproximator
from functionapproximation import GaussianFunctionApproximator
from functionapproximation import MultipleGaussianFunctionApproximator

sys  = pendulum.SinglePendulum()

sys.x_ub[0] =  10.0
sys.x_lb[0] = -10.0
sys.x_lb[1] = -5.0
sys.x_ub[1] = 5.0
sys.u_ub[0] = 5.0
sys.u_lb[0] = -5.0

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [201,201] , [11] , 0.05)

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ -3.14 , 0 ]) # target
qcf.INF  = 200

####################
# Value iteration solution
####################

dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf)

dp.solve_bellman_equation( tol = 0.5 )

#ctl = dp.get_lookup_table_controller()

#dp.plot_cost2go()
dp.plot_cost2go_3D()



####################
# Gaussian Approx
####################

# =============================================================================
# qfa1 = GaussianFunctionApproximator( x0 = np.array([0,0]) )
# qfa2 = GaussianFunctionApproximator(  x0 = qcf.xbar )
# qfa3 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([1,0]) )
# qfa4 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([1,1]) )
# qfa5 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([0,1]) )
# qfa6 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([-1,0]) )
# qfa7 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([-1,-1]) )
# qfa8 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([0,-1]) )
# qfa9 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([0.5,0]) )
# qfa10 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([0.5,0.5]) )
# qfa11 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([0,0.5]) )
# 
# qfa = qfa1 + qfa2 + qfa3 + qfa4 + qfa5 + qfa6 + qfa7 + qfa8 + qfa9 + qfa10 + qfa11
# =============================================================================


# Discrete world 
grid_sys_gaussian = discretizer.GridDynamicSystem( sys , [21,21] , [3] , 0.05)
X0 = grid_sys_gaussian.state_from_node_id

qfa = MultipleGaussianFunctionApproximator( X0  ) + QuadraticFunctionApproximator( sys.n )

Xs = grid_sys.state_from_node_id # All state on the grid

P = qfa.compute_all_kernel( Xs )

w , J_hat = qfa.least_square_fit( dp.J , P )

grid_sys.plot_grid_value_3D( J_hat , None , 'Gaussian approx')

grid_sys.plot_grid_value_3D( dp.J , J_hat , 'J vs. J_hat')
