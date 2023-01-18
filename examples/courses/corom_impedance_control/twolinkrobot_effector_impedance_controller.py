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

torque_controlled_robot      = manipulator.TwoLinkManipulator()

# Target
q_desired = np.array([0.5,0.5])
r_desired = torque_controlled_robot.forward_kinematic_effector( q_desired )


# Effector pd 

model = torque_controlled_robot

effector_pd      = robotcontrollers.EndEffectorPD( model )
effector_pd.rbar = r_desired
effector_pd.kp   = np.array([100, 100 ])
effector_pd.kd   = np.array([  0,   0 ])

# Closed-loops

robot_with_effector_pd = effector_pd + torque_controlled_robot 

# Simulations
tf = 4
robot_with_effector_pd.x0 = np.array([0,0,0,0])
robot_with_effector_pd.compute_trajectory( tf )
robot_with_effector_pd.plot_trajectory('xu')
robot_with_effector_pd.animate_simulation()
