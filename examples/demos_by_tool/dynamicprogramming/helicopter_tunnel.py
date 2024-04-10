#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import drone
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming 
from pyro.planning import discretizer

sys  = drone.ConstantSpeedHelicopterTunnel()

sys.obstacles = [
                [ (2,2),(4,4)],
                [ (8,5),(10,10)],
                [ (14,0),(16,4)]
                ]

sys.mass = 0.1
sys.vx   = 5.0
sys.width =1.0

# Set domain
sys.x_ub = np.array([+60, 10, +20])
sys.x_lb = np.array([-60, 0, +0])

sys.u_ub = np.array([+20])
sys.u_lb = np.array([-20])

"""
sys.show([0,0])dp.

sys.ubar[0] = 10
sys.animate_simulation()
"""


# Discrete world
grid_sys = discretizer.GridDynamicSystem(sys, (51, 51, 51), [11], 0.05)

# Cost Function
qcf = costfunction.QuadraticCostFunctionWithDomainCheck.from_sys( sys )

qcf.xbar = np.array([ 0.0 , 2.0, 20 ]) # target
qcf.INF  = 100000
qcf.EPS  = 0.2

qcf.Q[0,0] = 2.0
qcf.Q[1,1] = 200.0
qcf.Q[2,2] = 0.0

qcf.R[0,0] = 5.0

qcf.S[0,0] = 20.0
qcf.S[1,1] = 50.0
qcf.S[2,2] = 0.0

# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf)

dp.plot_cost2go(qcf.INF,2,1)

dp.alpha = 0.999
dp.solve_bellman_equation()
    

dp.plot_cost2go(qcf.INF,2,1)


ctl = dp.get_lookup_table_controller()

cl_sys = ctl + sys

# Simulation and animation
cl_sys.x0   = np.array([0,8,-10])
cl_sys.compute_trajectory( 6, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()
