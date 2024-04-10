# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""
##############################################################################
import numpy as np
##############################################################################
from pyro.dynamic  import pendulum
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming
##############################################################################


sys  = pendulum.SinglePendulum()

sys.x_ub = np.array([-3.14+6, +6])
sys.x_lb = np.array([-3.14-6, -6])

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [201,201] , [3] )

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys )

# Cost Function

xbar = np.array([-3.14,0])
tcf = costfunction.TimeCostFunction( xbar )
tcf.INF = 5.0
tcf.EPS = 0.1

sys.cost_function = tcf

# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, tcf )

dp.solve_bellman_equation( tol = 0.05 , animate_cost2go = False )


dp.clean_infeasible_set()
dp.plot_cost2go_3D()
dp.plot_policy()

#dp.animate_policy( show = True , save = False )
#dp.animate_cost2go( show = False , save = True )
dp.animate_policy( show = False , save = True )


ctl = dp.get_lookup_table_controller()

#ctl.plot_control_law( sys = sys , n = 100)


#asign controller
cl_sys = ctl + sys
cl_sys.x0   = np.array([0., 0.])
cl_sys.compute_trajectory( 10, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()