#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:24:14 2020

@author: alex
"""

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


# Custom time-based reference for controller
def t2r( t ):
    r = np.array([0.,0.])
    r[1] = 1.0 * np.sin( 5 * t )
    return r

ctl.t2r = t2r

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0  = np.array([-3.14,1,0,0])
cl_sys.compute_trajectory( tf = 20 )
cl_sys.plot_trajectory()
cl_sys.plot_phase_plane_trajectory(0, 2)
cl_sys.animate_simulation()