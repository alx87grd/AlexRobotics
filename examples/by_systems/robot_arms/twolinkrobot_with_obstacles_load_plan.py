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

planner.load_solution('twolinkplan.npy')

planner.plot_tree()
planner.plot_open_loop_solution()

speed_controlled_robot.traj = planner.trajectory
speed_controlled_robot.animate_simulation( time_factor_video = 50.0 )
