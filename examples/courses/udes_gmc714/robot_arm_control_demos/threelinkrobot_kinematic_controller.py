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

torque_controlled_robot = manipulator.ThreeLinkManipulator3D()
speed_controlled_robot  = manipulator.SpeedControlledManipulator.from_manipulator(torque_controlled_robot)

kinematic_controller      = robotcontrollers.EndEffectorKinematicController( speed_controlled_robot , 1 )
kinematic_controller.rbar = np.array([1.0,1.0,1.0])
    
closed_loop_robot     = kinematic_controller + speed_controlled_robot
closed_loop_robot.x0  = np.array([0.1,0.1,0.1])

closed_loop_robot.animate_simulation( is_3d=True )