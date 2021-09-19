# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""

import numpy as np

from pyro.dynamic  import pendulum
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import valueiteration
from pyro.control  import controller

sys  = pendulum.SinglePendulum()

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys )

# Cost Function
qcf = sys.cost_function

qcf.xbar = np.array([ -3.14 , 0 ]) # target
qcf.INF  = 10000

# VI algo
vi = valueiteration.ValueIteration_ND( grid_sys , qcf )

vi.initialize()
#vi.load_data('simple_pendulum_vi')
vi.plot_max_J = 250
vi.compute_steps(100, plot=True)
vi.assign_interpol_controller()
vi.plot_policy(0)
vi.plot_cost2go()
# vi.save_data('simple_pendulum_vi')

#asign controller
cl_sys = controller.ClosedLoopSystem( sys , vi.ctl )
cl_sys.cost_function = None

# Simulation and animation
cl_sys.x0   = np.array([0,0])
tf   = 10
cl_sys.compute_trajectory(tf)
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()

