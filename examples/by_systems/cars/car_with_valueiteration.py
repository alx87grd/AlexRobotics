# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:05:07 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import vehicle
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import valueiteration
from pyro.control  import controller
###############################################################################

sys  = vehicle.KinematicCarModelwithObstacles()

###############################################################################

# Planning

# Set domain
sys.x_ub = np.array([+35, +3, +3])
sys.x_lb = np.array([-5, -2, -3])

sys.u_ub = np.array([+3, +1])
sys.u_lb = np.array([-3, -1])

# Discrete world
grid_sys = discretizer.GridDynamicSystem(sys, (51, 51, 21), (3, 3), 0.1)

# Cost Function
cf = costfunction.QuadraticCostFunction.from_sys( sys )
cf.xbar = np.array( [30, 0, 0] ) # target
cf.INF  = 1E8
cf.EPS  = 0.00
cf.R    = np.array([[0.1,0],[0,0]])

# VI algo
vi = valueiteration.ValueIteration_ND( grid_sys , cf )

vi.uselookuptable = True
vi.initialize()

#vi.load_data('car_vi2')
vi.compute_steps(2,True)
#vi.save_data('car_vi2')

###############################################################################

# Closed-loop Law

vi.assign_interpol_controller()

vi.plot_policy(0)
vi.plot_policy(1)

cl_sys = controller.ClosedLoopSystem( sys , vi.ctl )

###############################################################################

## Simulation and animation

x0   = np.array([0, 0, 0])
tf   = 20

cl_sys.x0 = x0
cl_sys.compute_trajectory(tf, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()