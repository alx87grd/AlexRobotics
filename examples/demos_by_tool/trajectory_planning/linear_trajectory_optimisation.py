#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np

from pyro.dynamic.massspringdamper        import TwoMass
from pyro.planning.trajectoryoptimisation import DirectCollocationTrajectoryOptimisation


sys = TwoMass()

sys.u_ub[0] = +2
sys.u_lb[0] = 0

planner = DirectCollocationTrajectoryOptimisation( sys , 0.1, 40 )

planner.x_start = np.array([0.5,0.5,0,0])
planner.x_goal  = np.array([0,0,0,0])

planner.compute_optimal_trajectory()
planner.show_solution()
planner.animate_solution()


