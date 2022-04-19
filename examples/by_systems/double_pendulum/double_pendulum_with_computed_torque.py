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

sys = pendulum.DoublePendulum()
ctl  = nonlinear.ComputedTorqueController( sys )

ctl.w0   = 1.5
ctl.zeta = 0.5 
ctl.rbar = np.array([0,0])

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0  = np.array([-3.14,1,0,0])
cl_sys.compute_trajectory( tf = 10 )
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory(0, 2)
cl_sys.animate_simulation()