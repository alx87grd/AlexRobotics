# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:09:55 2019

@author: alxgr
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import manipulator
###############################################################################

torque_controlled_robot      = manipulator.TwoLinkManipulator()

# Settting

#torque_controlled_robot.l1  = 0.5
#torque_controlled_robot.l2  = 0.5
#torque_controlled_robot.lc1 = 0.5
#torque_controlled_robot.lc2 = 0.5

torque_controlled_robot.I1     = 0  
torque_controlled_robot.I2     = 0

torque_controlled_robot.gravity = 9
torque_controlled_robot.d1      = 0.1
torque_controlled_robot.d2      = 0.1
torque_controlled_robot.ubar    = [0,0] 

#torque_controlled_robot.ubar    = [0.5,0] 
#torque_controlled_robot.gravity = 9.8
#torque_controlled_robot.d1      = 0.5
#torque_controlled_robot.d2      = 1


# Simulations

torque_controlled_robot.x0 = np.array([0,0.1,0,0])

torque_controlled_robot.animate_simulation()
#torque_controlled_robot.plot_trajectory('xu')