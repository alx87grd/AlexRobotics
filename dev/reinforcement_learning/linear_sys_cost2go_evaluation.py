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

from scipy.linalg  import solve_continuous_are

from functionapproximation import QuadraticFunctionApproximator

sys  = massspringdamper.FloatingSingleMass()

sys.x_ub[0] =  10.0
sys.x_lb[0] = -10.0
sys.x_lb[1] = -5.0
sys.x_ub[1] = 5.0
sys.u_ub[0] = 5.0
sys.u_lb[0] = -5.0

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [11] , 0.05)

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ -0 , 0 ]) # target
qcf.INF  = 300

qcf.R[0,0] = 10.0

qcf.S[0,0] = 10.0
qcf.S[1,1] = 10.0

####################
# Value iteration solution
####################

dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf)

dp.solve_bellman_equation( tol = 0.5 )

ctl = dp.get_lookup_table_controller()

dp.plot_cost2go()
dp.plot_cost2go_3D()

####################
# Quadratic Approx
####################

qfa = QuadraticFunctionApproximator( sys.n , x0 = qcf.xbar )

Xs = grid_sys.state_from_node_id # All state on the grid

P = qfa.compute_all_kernel( Xs )

w , J_hat = qfa.least_square_fit( dp.J , P )

grid_sys.plot_grid_value_3D( J_hat , None , 'Quadratic approx')

grid_sys.plot_grid_value_3D( dp.J , J_hat , 'J vs. J_hat')

####################
# Riccati solution
####################

S = solve_continuous_are( sys.A , sys.B , qcf.Q , qcf.R )

w_riccati = np.array([0, 0, 0, S[0,0], S[1,1], S[0,1]])
J_riccati = P.T @ w_riccati 
grid_sys.plot_grid_value_3D( J_riccati , J_hat , 'J_riccati vs. J_hat')