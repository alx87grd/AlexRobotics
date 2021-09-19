# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import pendulum
from pyro.control  import nonlinear
from pyro.analysis import simulation
###############################################################################

sys  = pendulum.DoublePendulum()

traj = simulation.Trajectory.load('double_pendulum_rrt.npy')

# Controller
ctl  = nonlinear.ComputedTorqueController( sys , traj )

# goal
ctl.rbar = np.array([0,0])

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0 = np.array([-3.14,0,0,0])
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()