# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:05:07 2018

@author: Alexandre
"""

import numpy as np

from pyro.dynamic  import vehicle_steering
from pyro.planning import randomtree

sys  = vehicle_steering.HolonomicMobileRobotwithObstacles()

sys.obstacles = [
                [ (2,2),(4,10)],
                [ (6,-8),(8,8)],
                [ (-8,-8),(-1,8)]
                ]
        
x_start = np.array([8.0,0])
x_goal  = np.array([0.0,0])

planner = randomtree.RRT( sys , x_start )

v = 1

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
planner.alpha       = 0.9
planner.beta        = 0.1
planner.goal_radius = 1.0

planner.find_path_to_goal( x_goal )

planner.plot_tree()
planner.show_solution()
planner.animate_solution()