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
r_desired = np.array([0.5,1,2])

# Effector pid 

model = torque_controlled_robot

effector_pid      = robotcontrollers.EndEffectorPID( model )
effector_pid.rbar = r_desired
effector_pid.kp   = np.array([100, 100 , 100])
effector_pid.kd   = np.array([ 30,  30 ,  30])
effector_pid.ki   = np.array([ 90,  90 ,  90])

# Closed-loops
robot_with_effector_pid = effector_pid + torque_controlled_robot 

# Simulation
robot_with_effector_pid.x0  = np.array([0,0,0,0,0,0,0,0,0])

robot_with_effector_pid.compute_trajectory( 10 )
robot_with_effector_pid.plot_trajectory('x')
robot_with_effector_pid.plot_internal_controller_states()
robot_with_effector_pid.plot_end_effector_trajectory()
robot_with_effector_pid.animate_simulation(time_factor_video=1, is_3d=True)
