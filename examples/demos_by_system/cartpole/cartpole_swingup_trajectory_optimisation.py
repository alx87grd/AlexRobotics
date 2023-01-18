#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:30:18 2022

@author: alex
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import cartpole
from pyro.planning import trajectoryoptimisation
from pyro.analysis import simulation
###############################################################################


sys  = cartpole.UnderActuatedRotatingCartPole()

sys.cost_function.Q[0,0] = 1.0
sys.cost_function.Q[1,1] = 1.0
sys.cost_function.Q[2,2] = 100
sys.cost_function.Q[3,3] = 1.0

#Max/Min torque
sys.u_ub[0] = +20
sys.u_lb[0] = -20

## Planning

planner = trajectoryoptimisation.DirectCollocationTrajectoryOptimisation( sys , 0.1 , 30 )

# Load an initial guess found with RRT
init_traj = simulation.Trajectory.load('underactuatedcartpole_rrt.npy')
planner.set_initial_trajectory_guest( init_traj )

planner.x_start = np.array([0,-3.14,0,0])
planner.x_goal  = np.array([0,0,0,0])

planner.maxiter = 500
planner.compute_optimal_trajectory()
planner.show_solution()
planner.animate_solution( is_3d = True ) 