#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import longitudinal_vehicule


from pyro.control  import controller
import dynamic_programming as dprog
import discretizer
import costfunction

sys  = longitudinal_vehicule.LongitudinalFrontWheelDriveCarWithWheelSlipInput()

sys.x_ub[1] = 15
sys.x_lb[1] = 0

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [11] , 0.05 )

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys( sys )

qcf.xbar = np.array([ 45 , 0 ]) # target
qcf.Q[0,0] = 0.1
qcf.Q[1,1] = 0.1
qcf.INF  = 1000000


# DP algo
#dp = dprog.DynamicProgramming( grid_sys, qcf )
dp = dprog.DynamicProgrammingWithLookUpTable( grid_sys, qcf)


dp.compute_steps(300)
dp.plot_cost2go()


grid_sys.plot_grid_value( dp.J_next )

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