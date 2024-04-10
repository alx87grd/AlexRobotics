#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import pendulum
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming 
from pyro.planning import discretizer

sys  = pendulum.SinglePendulum()

sys.xbar = np.array([ -3.14 , 0 ]) 

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [3] )

# Cost Function
cf = costfunction.Reachability( sys.isavalidstate , sys.xbar )

# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, cf)

dp.solve_bellman_equation( animate_cost2go = True )
