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

sys  = vehicle.KinematicBicyleModel()

sys.x_ub = np.array( [+2,+2,+0.5] )
sys.x_lb = np.array( [-0,-0,-0.5] )

sys.u_ub = np.array( [+1,+1.0] )
sys.u_lb = np.array( [-1,-1.0] )

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , (41,41,21) , (3,3) , 0.05 ) 
# Cost Function
cf = costfunction.QuadraticCostFunction.from_sys(sys)


cf.xbar = np.array( [1,1,0] ) # target
cf.INF  = 1E4
cf.EPS  = 0.2
cf.R    = np.array([[0.01,0],[0,0]])

# VI algo

"""
EXPERIMENTAL TEST NOT WORKING
"""

vi = valueiteration.ValueIteration_ND( grid_sys , cf )

vi.uselookuptable = True
vi.initialize()
vi.load_data('parking_vi')
vi.compute_steps(2)
vi.save_data('parking_vi')

vi.assign_interpol_controller()


#
cl_sys = controller.ClosedLoopSystem( sys , vi.ctl )


## Simulation and animation
cl_sys.x0   = np.array([0.2,0.2,0])
tf   = 5
cl_sys.compute_trajectory( tf , 10001 , 'euler')
