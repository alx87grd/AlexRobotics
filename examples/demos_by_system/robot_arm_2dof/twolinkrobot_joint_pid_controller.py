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

# Joint PID

dof = 2

joint_pid      = robotcontrollers.JointPID( dof )
joint_pid.rbar = q_desired
joint_pid.kp   = np.array([25, 5 ])
joint_pid.kd   = np.array([ 5, 2 ])
joint_pid.ki   = np.array([ 8, 4 ])


# Closed-loops

robot_with_joint_pid    = joint_pid + torque_controlled_robot 

# Simulations
tf = 10
robot_with_joint_pid.x0 = np.array([0,0,0,0,0,0])
robot_with_joint_pid.compute_trajectory( tf )
robot_with_joint_pid.plot_trajectory('xu')
robot_with_joint_pid.plot_internal_controller_states()
robot_with_joint_pid.animate_simulation()
