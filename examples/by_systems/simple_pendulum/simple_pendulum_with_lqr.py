# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic.pendulum      import SinglePendulum
from pyro.analysis.costfunction import QuadraticCostFunction
from pyro.control.lqr           import linearize_and_synthesize_lqr_controller
###############################################################################


sys = SinglePendulum()

# Nominal state to stabilize
sys.xbar[0] = 3.14

# Cost function
cf  = QuadraticCostFunction.from_sys( sys )

cf.Q[0,0] = 1
cf.Q[1,1] = 1

# LQR controller
ctl = linearize_and_synthesize_lqr_controller( sys , cf )

# Simulation Closed-Loop Non-linear with LQR controller
cl_sys = ctl + sys
cl_sys.x0 = np.array([2,0])
cl_sys.compute_trajectory()
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()