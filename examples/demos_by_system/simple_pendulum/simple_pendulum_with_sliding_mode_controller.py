# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import pendulum
from pyro.control import nonlinear
###############################################################################

sys  = pendulum.SinglePendulum()


ctl  = nonlinear.SlidingModeController( sys )

ctl.lam  = 4.0
ctl.gain = 25.0

# Set Point
q_target = np.array([3.14])
ctl.rbar = q_target

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0 = np.array([0.1,0])
cl_sys.compute_trajectory(tf=5, n=10001, solver='euler') 
# Note: Use "euler" solver when using sliding mode controllers
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory_closed_loop()
cl_sys.plot_phase_plane()
cl_sys.animate_simulation()