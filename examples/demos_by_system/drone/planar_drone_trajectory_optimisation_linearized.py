#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np

from pyro.dynamic.drone                   import Drone2D
from pyro.planning.trajectoryoptimisation import DirectCollocationTrajectoryOptimisation
from pyro.dynamic.statespace              import linearize


# Non-linear model
sys = Drone2D()

sys.xbar =  np.array([0,0,0.0,0,0,0])
sys.ubar =  np.array([0.5,0.5]) * sys.mass * sys.gravity
    
# Linear model
ss  = linearize( sys , 0.01 )


planner = DirectCollocationTrajectoryOptimisation( ss )

planner.x_start = np.array([-5,0,0,0,0,0])
planner.x_goal  = np.array([0,0,0,0,0,0])

planner.compute_optimal_trajectory()
planner.show_solution()


