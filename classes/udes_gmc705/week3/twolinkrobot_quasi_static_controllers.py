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

# Joint PID

dof = 2

joint_pd      = robotcontrollers.JointPD( dof )
joint_pd.rbar = q_desired
joint_pd.kp   = np.array([25, 5 ])
joint_pd.kd   = np.array([ 1, 0 ])


# Effector PID 

model = torque_controlled_robot

effector_pd      = robotcontrollers.EndEffectorPD( model )
effector_pd.rbar = r_desired
effector_pd.kp   = np.array([100, 100 ])
effector_pd.kd   = np.array([  0,   0 ])

# Closed-loops

robot_with_joint_pd    = joint_pd    + torque_controlled_robot 
robot_with_effector_pd = effector_pd + torque_controlled_robot 

# Simulations

robot_with_joint_pd.x0 = np.array([0,0,0,0])
robot_with_joint_pd.animate_simulation()
robot_with_joint_pd.plot_trajectory('xu')

robot_with_joint_pd.x0 = np.array([0,0,0,0])
robot_with_effector_pd.animate_simulation()
robot_with_effector_pd.plot_trajectory('xu')