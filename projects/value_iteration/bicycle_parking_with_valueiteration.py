# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""

import numpy as np

from pyro.dynamic  import vehicle
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import valueiteration
from pyro.control  import controller

# initialize system
sys  = vehicle.KinematicBicyleModel()

sys.x_ub = np.array( [+2,+2,+0.5] )
sys.x_lb = np.array( [-0,-0,-0.5] )

sys.u_ub = np.array( [+1,+1.0] )
sys.u_lb = np.array( [-1,-1.0] )

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , (41,41,21) , (3,3) , 0.05 )
# Cost Function
cf = costfunction.QuadraticCostFunction.from_sys( sys )
cf.xbar = np.array( [1,1,0])
cf.INF  = 1E4
cf.EPS  = 0
cf.R    = np.array([[0.01,0],[0,0]])

# VI algo

vi = valueiteration.ValueIteration_ND( grid_sys , cf )

vi.uselookuptable = True
vi.initialize()
vi.load_data('bicycle_parking_vi')
#vi.compute_steps(100, maxJ=args.maxJ, plot=has_dynamic_plot)
#vi.save_data('bicycle_parking_vi')

vi.assign_interpol_controller()

vi.plot_cost2go(100)
vi.plot_policy(0)
vi.plot_policy(1)

cl_sys = controller.ClosedLoopSystem( sys , vi.ctl )

## Simulation and animation
tf = 5

cl_sys.x0 = np.array( [0,0,0])
cl_sys.compute_trajectory( tf , 10001 , 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()
