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

torque_controlled_robot = manipulator.FiveLinkPlanarManipulatorwithObstacles()
speed_controlled_robot  = manipulator.SpeedControlledManipulator.from_manipulator( torque_controlled_robot )

q_start = np.array([0,0,0,0,0])
q_goal  = np.array([1.57,0,0,0,0])

planner = randomtree.RRT( speed_controlled_robot , q_start )

t = 0.5
    
planner.u_options = [
        np.array([ +t, -0, +0, +0, +0]),
        np.array([ -t, -0, +0, +0, +0]),
        np.array([ -t, -t, +t, +t, +t]),
        np.array([ +t, +t, -t, -t, -t]),
        np.array([ -t, -t, -t, +t, +t]),
        np.array([ +t, +t, +t, -t, -t])
        ]

planner.goal_radius = 0.4
planner.max_nodes   = 20000
planner.dyna_plot   = False
planner.max_solution_time = 20


planner.find_path_to_goal( q_goal )

planner.save_solution('fivelinkplan.npy')
planner.plot_tree()
planner.plot_open_loop_solution()

speed_controlled_robot.traj = planner.trajectory
speed_controlled_robot.animate_simulation()
