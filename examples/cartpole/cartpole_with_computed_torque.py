# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import cartpole
from pyro.control  import nonlinear
###############################################################################


sys  = cartpole.RotatingCartPole()

ctl  = nonlinear.ComputedTorqueController( sys )

ctl.w0   = 1.0
ctl.zeta = 0.7

# goal
ctl.rbar = np.array([0,0])

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0 = np.array([-3.14,-3.14,0,0])
cl_sys.compute_trajectory()
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation(time_factor_video=1.0, is_3d=True)