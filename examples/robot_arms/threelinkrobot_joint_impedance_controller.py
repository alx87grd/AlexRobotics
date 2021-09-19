# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:34:44 2019

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import manipulator
from pyro.control  import robotcontrollers
###############################################################################


# Model
torque_controlled_robot   = manipulator.ThreeLinkManipulator3D()


# Target
q_desired = np.array([0,0,0])

# joint PD 

model = torque_controlled_robot

joint_pd      = robotcontrollers.JointPD(3)
joint_pd.rbar = q_desired
joint_pd.kp   = np.array([100, 100 , 100])
joint_pd.kd   = np.array([100, 100 , 100])

# Closed-loops
robot_with_joint_pd = joint_pd + torque_controlled_robot 

# Simulation
robot_with_joint_pd.x0  = np.array([3.14,-3,2,0,0,0])
robot_with_joint_pd.compute_trajectory()
robot_with_joint_pd.plot_trajectory('x')
robot_with_joint_pd.plot_trajectory('u')
robot_with_joint_pd.animate_simulation(time_factor_video=1, is_3d=True)
