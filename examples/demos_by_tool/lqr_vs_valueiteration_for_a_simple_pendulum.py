# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""
##############################################################################
import numpy as np
##############################################################################
from pyro.dynamic  import pendulum
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming 
from pyro.planning import discretizer
from pyro.control  import lqr
##############################################################################

sys  = pendulum.SinglePendulum()

sys.xbar  = np.array([ -3.14 , 0 ]) # target and linearization point

##############################
# Cost Function
##############################

qcf     = costfunction.QuadraticCostFunction.from_sys( sys )
qcf.INF = 300

##############################
# VI algo 
##############################

grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [11] )

dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf)

dp.solve_bellman_equation()
#dp.plot_policy(0)

dp_ctl = dp.get_lookup_table_controller()

##############################
#LQR
##############################

lqr_ctl = lqr.linearize_and_synthesize_lqr_controller( sys, qcf )

##############################
# Policy plots
##############################

dp_ctl.plot_control_law(0,1,0,0,100,sys)
lqr_ctl.plot_control_law(0,1,0,0,100,sys)

##############################
# Simulations
##############################

x0 = np.array([-0 ,0])

cl_sys_lqr =   lqr_ctl + sys 

cl_sys_lqr.x0   = x0
cl_sys_lqr.plot_trajectory('xuj')
cl_sys_lqr.animate_simulation()

cl_sys_vi =   dp_ctl + sys 

cl_sys_vi.x0   = x0
cl_sys_vi.plot_trajectory('xuj')
cl_sys_vi.animate_simulation()
