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

sys  = pendulum.SinglePendulum()


# Controller

traj = simulation.Trajectory.load('pendulum_rrt.npy')

ctl  = nonlinear.SlidingModeController( sys , traj )

ctl.lam  = 5
ctl.gain = 2

# goal
ctl.rbar = np.array([-3.14])

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0 = np.array([0.1,0])
cl_sys.compute_trajectory(tf=5, n=10001, solver='euler') 
# Note: Use "euler" solver when using sliding mode controllers
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()