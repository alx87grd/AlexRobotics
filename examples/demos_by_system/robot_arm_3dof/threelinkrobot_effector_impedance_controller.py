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
r_desired = torque_controlled_robot.forward_kinematic_effector( q_desired )

# Effector PD 

model = torque_controlled_robot

effector_pd      = robotcontrollers.EndEffectorPD( model )
effector_pd.rbar = r_desired
effector_pd.kp   = np.array([100, 100 , 100])
effector_pd.kd   = np.array([ 10,  10 ,  10])

# Closed-loops
robot_with_effector_pd = effector_pd + torque_controlled_robot 

# Simulation
robot_with_effector_pd.x0  = np.array([3.14,-3,2,0,0,0])
robot_with_effector_pd.compute_trajectory()
robot_with_effector_pd.plot_trajectory('x')
robot_with_effector_pd.plot_trajectory('u')
robot_with_effector_pd.animate_simulation(time_factor_video=1, is_3d=True)
