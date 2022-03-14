# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:34:44 2019

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.control  import robotcontrollers
from pyro.dynamic  import manipulator
###############################################################################


# Dynamic model (inputs are motor torques)
torque_controlled_robot    = manipulator.TwoLinkManipulator()

# Kinematic only model (inputs are motor velocities)
speed_controlled_robot  = manipulator.SpeedControlledManipulator.from_manipulator( torque_controlled_robot )


# Kinematic effector position controller
kinematic_controller = robotcontrollers.EndEffectorKinematicController( speed_controlled_robot )

# Controller params
kinematic_controller.gains = np.array([1.0,1.0]) 
kinematic_controller.rbar  = np.array([0.5,0.5]) # Goal position


# Closed-loop system
closed_loop_robot = kinematic_controller + speed_controlled_robot

# Initial position of the robot
closed_loop_robot.x0        = np.array([-0.5,0.2])

# Simulation and plots
closed_loop_robot.plot_trajectory('xu')
closed_loop_robot.animate_simulation()
