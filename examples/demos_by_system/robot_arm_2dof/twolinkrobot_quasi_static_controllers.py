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

# Joint PD

dof = 2

joint_PD      = robotcontrollers.JointPD( dof )
joint_PD.rbar = q_desired
joint_PD.kp   = np.array([25, 5 ])
joint_PD.kd   = np.array([ 1, 0 ])


# Effector PD 

model = torque_controlled_robot

effector_PD      = robotcontrollers.EndEffectorPD( model )
effector_PD.rbar = r_desired
effector_PD.kp   = np.array([100, 100 ])
effector_PD.kd   = np.array([  0,   0 ])

# Closed-loops

robot_with_joint_PD    = joint_PD    + torque_controlled_robot 
robot_with_effector_PD = effector_PD + torque_controlled_robot 

# Simulations

x0 = np.array([0,0,0,0])
tf = 5

anims = []

for robot in (robot_with_joint_PD, robot_with_effector_PD):
    robot.x0 = x0
    robot.compute_trajectory(tf)
    anims.append( robot.animate_simulation() )
    robot.plot_trajectory('xu')