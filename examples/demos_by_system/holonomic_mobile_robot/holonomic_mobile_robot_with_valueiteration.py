# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""

import numpy as np

from pyro.dynamic  import vehicle_steering
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming

sys  = vehicle_steering.HolonomicMobileRobotwithObstacles()

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , (51,51) , (3,3) ) 

# Cost Function
cf = costfunction.QuadraticCostFunction.from_sys( sys )
cf.INF = 1500

# VI algo
vi = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys , cf )

vi.solve_bellman_equation()

# Closed-loop Law

vi.clean_infeasible_set()
vi.plot_cost2go_3D()
#vi.plot_policy(0)
#vi.plot_policy(1)

ctl = vi.get_lookup_table_controller()





# Closed loop
cl_sys = ctl + sys

# Simulation and animation
cl_sys.x0   = np.array([9,0])
cl_sys.compute_trajectory(tf=20)
#cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()