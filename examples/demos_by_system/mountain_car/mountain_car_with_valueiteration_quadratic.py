# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""
##############################################################################
import numpy as np
##############################################################################
from pyro.dynamic  import mountaincar
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming
##############################################################################

sys  = mountaincar.MountainCar()

sys.x_ub = np.array([+0.2,+2.0])
sys.x_lb = np.array([-1.7,-2.0])

sys.u_ub[0] = +0.2
sys.u_lb[0] = -0.2

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [201,201] , [11] )

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ 0 , 0 ]) # target
qcf.INF  = 30

qcf.R[0,0] = 10.0

qcf.S[0,0] = 10.0
qcf.S[1,1] = 10.0


# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf)

dp.solve_bellman_equation( tol = 0.01)
#dp.solve_bellman_equation( tol = 1 , animate_cost2go = True )
#dp.solve_bellman_equation( tol = 1 , animate_policy = True )

#dp.animate_cost2go( show = False , save = True )
#dp.animate_policy( show = False , save = True )

dp.clean_infeasible_set()
dp.plot_cost2go_3D()
dp.plot_policy()

ctl = dp.get_lookup_table_controller()

# Simulation
cl_sys = ctl + sys
cl_sys.x0   = np.array([-1., 0.])
cl_sys.compute_trajectory( 25, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()