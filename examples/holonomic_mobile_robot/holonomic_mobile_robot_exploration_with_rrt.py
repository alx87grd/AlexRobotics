# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:05:07 2018

@author: Alexandre
"""

import numpy as np

from pyro.dynamic import vehicle
from pyro.planning import randomtree

sys  = vehicle.HolonomicMobileRobot()

x_start = np.array([0,0])
x_goal  = np.array([8,8])

planner = randomtree.RRT( sys , x_start )

v = 1.0

planner.u_options = [
        np.array([+v,+v]),
        np.array([+v, 0]),
        np.array([+v,-v]),
        np.array([-v,-v]),
        np.array([-v, 0]),
        np.array([-v,+v]),
        np.array([ 0,+v]),
        np.array([ 0,-v]),
        ]

planner.dt          = 0.1
planner.steps       = 5
planner.alpha       = 1.0
planner.beta        = 0.1
planner.goal_radius = 0.01

planner.find_path_to_goal( x_goal )

planner.plot_tree()
planner.plot_open_loop_solution()
planner.animate_solution()