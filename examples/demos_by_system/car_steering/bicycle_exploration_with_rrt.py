# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:05:07 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import vehicle_steering
from pyro.planning import randomtree
###############################################################################

sys  = vehicle_steering.KinematicBicyleModel()

###############################################################################

# Planning
planner = randomtree.RRT( sys )

# Set domain
sys.x_ub = np.array([+10,+10,+6.28])
sys.x_lb = np.array([-10,-10,-6.284])

# Planning Problem
planner.x_start = np.array([0,0,0])
planner.x_goal  = np.array([5,5,0])


# Control actions
speed    = 1
steering = 0.5

planner.u_options = [
        np.array([ speed,-steering]),
        np.array([ speed,+steering]),
        np.array([ speed,0]),
        np.array([ -speed,0])
        ]

planner.goal_radius = 1.0
planner.dt          = 0.1
planner.steps       = 5

planner.compute_solution()

planner.plot_tree()
planner.plot_tree_3d()
planner.show_solution()
planner.animate_solution()