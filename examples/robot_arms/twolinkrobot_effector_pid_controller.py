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

# effector PID

dof = 2

effector_pid      = robotcontrollers.EndEffectorPID( torque_controlled_robot )
effector_pid.rbar = r_desired
effector_pid.kp   = np.array([100, 100 ])
effector_pid.kd   = np.array([  0,   0 ])
effector_pid.ki   = np.array([ 50,  50 ])


# Closed-loops

robot_with_effector_pid    = effector_pid + torque_controlled_robot 

# Simulations
tf = 20
robot_with_effector_pid.x0 = np.array([0,0,0,0,0,0])
robot_with_effector_pid.compute_trajectory( tf )
robot_with_effector_pid.plot_trajectory('xu')
robot_with_effector_pid.plot_internal_controller_states()
robot_with_effector_pid.animate_simulation()
