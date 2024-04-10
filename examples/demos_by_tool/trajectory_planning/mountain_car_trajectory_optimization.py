#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:03:52 2023

@author: alex
"""

##############################################################################
import numpy as np
##############################################################################
from pyro.dynamic  import mountaincar
from pyro.planning import trajectoryoptimisation
from pyro.analysis import costfunction
##############################################################################

sys  = mountaincar.MountainCar()

sys.x_ub = np.array([+0.2,+2.0])
sys.x_lb = np.array([-1.7,-2.0])

sys.u_ub[0] = +1.5
sys.u_lb[0] = -1.5


# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ 0 , 0 ]) # target
qcf.INF  = 30

qcf.R[0,0] = 10.0

qcf.S[0,0] = 10.0
qcf.S[1,1] = 10.0

planner = trajectoryoptimisation.DirectCollocationTrajectoryOptimisation( sys , 0.1, 40 )

planner.x_start = np.array([-1.0,+0.0])
planner.x_goal  = np.array([+0.0,+0.0])

planner.init_dynamic_plot()
planner.set_linear_initial_guest()
planner.compute_optimal_trajectory()
# planner.show_solution()
planner.animate_solution()
