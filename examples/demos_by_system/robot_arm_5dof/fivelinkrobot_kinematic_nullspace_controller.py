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

# Dynamic model
torque_controlled_robot = manipulator.FiveLinkPlanarManipulator()
sys  = manipulator.SpeedControlledManipulator.from_manipulator( torque_controlled_robot )

# Controller
ctl = robotcontrollers.EndEffectorKinematicControllerWithNullSpaceTask( sys )

# Main objective
ctl.rbar  = np.array([1.0,1.0])
ctl.gains = np.array([1.0,1.0])

# Secondary objective
ctl.q_d         = np.array([-1,-1,-1,-1,-1])
ctl.gains_null  = np.array([10,10,10,10,10])


# Closed-loop dynamics
cl_sys = ctl + sys

# Simulation and plots
cl_sys.x0        = np.array([0.1,0.1,0.1,0.1,0.1])

cl_sys.plot_trajectory('x')
cl_sys.animate_simulation()
