# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""
##############################################################################
import numpy as np
##############################################################################
from pyro.dynamic  import pendulum
from pyro.planning import randomtree
##############################################################################


# Dynamic sysem
sys  = pendulum.SinglePendulum()

# Define planning problem
x_start = np.array([0.1,0])
x_goal  = np.array([-3.14,0])

planner = randomtree.RRT( sys , x_start )

planner.u_options = [
            np.array([-5]),
            np.array([-3]),
            np.array([-1]),
            np.array([ 0]),
            np.array([ 1]),
            np.array([ 3]),
            np.array([ 5])
            ]

planner.goal_radius = 0.2

# Solve Planning Problem
planner.find_path_to_goal( x_goal )

# Print solution
#planner.save_solution('pendulum_rrt.npy')
planner.plot_tree()
planner.show_solution()
planner.animate_solution()
