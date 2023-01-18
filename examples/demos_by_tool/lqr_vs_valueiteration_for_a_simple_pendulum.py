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
from pyro.planning import valueiteration

from pyro.control.lqr           import linearize_and_synthesize_lqr_controller
##############################################################################

sys  = pendulum.SinglePendulum()

sys.xbar  = np.array([ -3.14 , 0 ]) # target and linearization point

##############################################################################

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys( sys )

qcf.INF  = 10000

# VI algo load results

grid_sys = discretizer.GridDynamicSystem( sys )

vi = valueiteration.ValueIteration_2D( grid_sys , qcf )

vi.initialize()
vi.load_data('simple_pendulum_vi')
#vi.compute_steps(110,True)
vi.assign_interpol_controller()
vi.plot_policy(0)

#LQR
lqr_ctl = linearize_and_synthesize_lqr_controller(sys, qcf)

lqr_ctl.plot_control_law(0,1,0,0,100,sys)

##############################################################################

# CLosed-loop behavior

x0 = np.array([-0 ,0])

cl_sys_lqr =   lqr_ctl + sys 

cl_sys_lqr.x0   = x0
cl_sys_lqr.plot_trajectory('xuj')
cl_sys_lqr.animate_simulation()

cl_sys_vi =   vi.ctl + sys 

cl_sys_vi.x0   = x0
cl_sys_vi.plot_trajectory('xuj')
cl_sys_vi.animate_simulation()
