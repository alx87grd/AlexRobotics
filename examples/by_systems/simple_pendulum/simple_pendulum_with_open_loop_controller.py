# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""
##############################################################################
import numpy as np
##############################################################################
from pyro.dynamic  import pendulum
from pyro.planning import OpenLoopController
##############################################################################

# Dynamic system
sys  = pendulum.SinglePendulum()

# Openloop controller loading RRT solution
ctl = OpenLoopController.load_from_file('pendulum_rrt.npy')

# Closing the loop
cl_sys = ctl + sys

# Simulation
cl_sys.x0 = np.array([0.1,0])
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()