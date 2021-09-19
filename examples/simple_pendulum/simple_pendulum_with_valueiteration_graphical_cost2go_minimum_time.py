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
from pyro.control  import controller
##############################################################################


sys  = pendulum.SinglePendulum()

sys.x_lb[0] = -10
sys.x_ub[0] = +10

##############################################################################

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys )

# Cost Function

xbar = np.array([-3.14,0])
tcf = costfunction.TimeCostFunction( xbar )
tcf.EPS = 0.1
#tcf.INF = 0

sys.cost_function = tcf

# VI algo
vi = valueiteration.ValueIteration_ND( grid_sys , tcf )

vi.initialize()
vi.plot_max_J = 10
#vi.load_data('simple_pendulum_vi_minimum_time')
vi.compute_steps(100,True)
vi.assign_interpol_controller()
vi.plot_policy(0)
vi.plot_cost2go(10)
#vi.save_data('simple_pendulum_vi_minimum_time')

##############################################################################

#asign controller
cl_sys = controller.ClosedLoopSystem( sys , vi.ctl )

##############################################################################

# Simulation and animation
cl_sys.x0   = np.array([0,0])
cl_sys.compute_trajectory()
cl_sys.plot_trajectory('xuj')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()