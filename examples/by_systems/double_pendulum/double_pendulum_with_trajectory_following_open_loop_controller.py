# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import pendulum
from pyro.planning import plan
###############################################################################

sys  = pendulum.DoublePendulum()

ctl  = plan.OpenLoopController.load_from_file( 'double_pendulum_rrt.npy' )

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0 = np.array([-3.14,0,0,0])
cl_sys.compute_trajectory(10)
cl_sys.plot_phase_plane_trajectory(0,2)
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()