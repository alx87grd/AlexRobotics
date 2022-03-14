# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import cartpole
from pyro.planning import randomtree
###############################################################################


sys  = cartpole.UnderActuatedRotatingCartPole()

x_start = np.array([0,-3.14,0,0])
x_goal  = np.array([0,0,0,0])

planner = randomtree.RRT( sys , x_start )

t = 50
    
planner.u_options = [
        np.array([-t]),
        np.array([+t]),
        np.array([ 0])
        ]

planner.goal_radius = 1.5

planner.max_nodes         = 10000
planner.max_solution_time = 1.5
planner.dt                = 0.05
planner.max_distance_compute = 1000
planner.dyna_plot         = False

planner.find_path_to_goal( x_goal )

planner.plot_tree()
planner.plot_open_loop_solution()
planner.animate_solution()