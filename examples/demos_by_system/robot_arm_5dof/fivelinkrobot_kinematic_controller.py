# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:34:44 2019

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.control.robotcontrollers import EndEffectorKinematicController
from pyro.dynamic.manipulator import FiveLinkPlanarManipulator
from pyro.dynamic.manipulator import SpeedControlledManipulator
###############################################################################

# Dynamic model :
# Five link robot with internal velocity controllers at the joints
sys  = SpeedControlledManipulator.from_manipulator(FiveLinkPlanarManipulator())

# Controller
ctl       = EndEffectorKinematicController( sys )
ctl.rbar  = np.array([1.0,1.0]) # target effector position
ctl.gains = np.array([2.0,2.0]) # gains
    
# Closed-loop dynamics
cl_sys  = ctl + sys

# Simulation
cl_sys.x0 = np.array([0.1,0.1,0.1,0.1,0.1])  # Initial config
cl_sys.plot_trajectory('x')
cl_sys.animate_simulation()
