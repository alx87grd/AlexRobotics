# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:34:44 2019

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.planning import randomtree
from pyro.dynamic  import manipulator
###############################################################################

torque_controlled_robot = manipulator.TwoLinkManipulatorwithObstacles()
speed_controlled_robot  = manipulator.SpeedControlledManipulator.from_manipulator( torque_controlled_robot )

q_start = np.array([0,0])
q_goal  = np.array([1.57,0])

planner = randomtree.RRT( speed_controlled_robot , q_start )

t = 0.5
    
planner.u_options = [
        np.array([ +t, -t]),
        np.array([ -t, +t]),
        np.array([ 0,+t]),
        np.array([ 0,-t]),
        np.array([+t, 0]),
        np.array([-t, 0])
        ]

planner.goal_radius = 0.1
planner.max_nodes   = 20000

planner.find_path_to_goal( q_goal )

#planner.save_solution('twolinkplan.npy')
planner.plot_tree()
planner.show_solution()
planner.animate_solution()
