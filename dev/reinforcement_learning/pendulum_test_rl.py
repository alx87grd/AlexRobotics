#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import pendulum
from pyro.control  import controller
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming
from pyro.planning import discretizer

from reinforcementlearning import QLearning

sys  = pendulum.SinglePendulum()

sys.x_ub[0] =  4.0
sys.x_lb[0] = -7.0
sys.x_lb[1] = -6.0
sys.x_ub[1] = 6.0

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [3] , 0.1)

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ -3.14 , 0 ]) # target
qcf.INF  = 300

qcf.R[0,0] = 1.0

qcf.S[0,0] = 10.0
qcf.S[1,1] = 10.0


# DP algo
dp = QLearning( grid_sys , qcf ) 

dp.compute_J_from_Q()

dp.compute_policy_from_Q()

dp.compute_episodes(10)