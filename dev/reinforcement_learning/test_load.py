#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 22:42:21 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import pendulum
from pyro.control  import controller
import dynamic_programming as dprog
import discretizer
from pyro.analysis import costfunction

sys  = pendulum.SinglePendulum()

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [3] )

# Cost Function
qcf = sys.cost_function

qcf.xbar = np.array([ -3.14 , 0 ]) # target
qcf.INF  = 1000000


# DP algo
#dp = dprog.DynamicProgramming( grid_sys, qcf )
#dp = dprog.DynamicProgrammingWithLookUpTable( grid_sys, qcf)
dp = dprog.DynamicProgrammingFast2DGrid(grid_sys, qcf)

dp.load_J_next('test2d')
#dp.compute_steps(100)

ctl = dprog.LookUpTableController( grid_sys , dp.pi )

ctl.plot_control_law( sys = sys , n = 100)


#asign controller
cl_sys = controller.ClosedLoopSystem( sys , ctl )

##############################################################################

# Simulation and animation
cl_sys.x0   = np.array([0,0])
cl_sys.compute_trajectory( 10, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()