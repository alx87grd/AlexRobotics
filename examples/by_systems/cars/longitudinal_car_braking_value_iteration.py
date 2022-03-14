#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 13:40:47 2022

@author: alex
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import longitudinal_vehicule
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import valueiteration
from pyro.control  import controller
###############################################################################

sys  = longitudinal_vehicule.LongitudinalFrontWheelDriveCarWithWheelSlipInput()

###############################################################################

# Planning

# Set domain
sys.x_ub = np.array([+50, 20,])
sys.x_lb = np.array([0, 0])
sys.u_ub = np.array([0.0])
sys.u_lb = np.array([-0.1])

# Discrete world
grid_sys = discretizer.GridDynamicSystem(sys, (101, 101), (11,), 0.1)

# Cost Function
cf = costfunction.QuadraticCostFunction.from_sys( sys )
cf.xbar = np.array( [0, 0] ) # target
cf.INF  = 1E3
cf.EPS  = 0.00
cf.R[0]   = 1
cf.Q[0,0] = 0
cf.Q[1,1] = 0.01

# VI algo
vi = valueiteration.ValueIteration_ND( grid_sys , cf )

vi.uselookuptable = True
vi.initialize()

vi.load_data('car_braking_test')
#vi.compute_steps(60,True)
#vi.save_data('car_braking_test')

###############################################################################

# Closed-loop Law

vi.assign_interpol_controller()

vi.plot_policy(0)
vi.plot_3D_cost()

cl_sys = controller.ClosedLoopSystem( sys , vi.ctl )

###############################################################################

## Simulation and animation

x0   = np.array([0, 16])
tf   = 10

cl_sys.x0 = x0
cl_sys.compute_trajectory(tf, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video = 3 )