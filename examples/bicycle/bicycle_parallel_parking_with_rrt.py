# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:20:11 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import vehicle
from pyro.planning import randomtree
###############################################################################

sys  = vehicle.KinematicBicyleModel()

###############################################################################

x_start = np.array([0,0,0])
x_goal  = np.array([0,1,0])

planner = randomtree.RRT( sys , x_start )

speed    = 2
steering = 0.2

planner.u_options = [
        np.array([ speed,-steering]),
        np.array([ speed,+steering]),
        np.array([ speed,0]),
        np.array([-speed,+steering]),
        np.array([-speed,0]),
        np.array([-speed,-steering])
        ]

planner.goal_radius       = 0.5
planner.dt                = 0.1
planner.steps             = 3
planner.max_solution_time = 8.0

planner.find_path_to_goal( x_goal )

planner.plot_tree()
planner.plot_tree_3d()
planner.plot_open_loop_solution()

###############################################################################

sys.dynamic_domain = False
sys.traj = planner.trajectory
sys.animate_simulation()