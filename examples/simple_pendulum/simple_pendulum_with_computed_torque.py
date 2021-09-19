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
ctl  = nonlinear.ComputedTorqueController( sys )

# Set Point
q_target = np.array([3.14])
ctl.rbar = q_target

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0  = np.array([-2,0])
cl_sys.compute_trajectory()
cl_sys.plot_phase_plane_trajectory()
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()