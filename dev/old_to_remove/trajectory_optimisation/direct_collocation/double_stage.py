# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import pendulum
from pyro.control  import nonlinear
from pyro.planning import trajectoryoptimisation
from pyro.analysis import simulation
###############################################################################


sys = pendulum.DoublePendulum()

#Max/Min torque
sys.u_ub[0] = +20
sys.u_ub[1] = +20
sys.u_lb[0] = -20
sys.u_lb[1] = -20


tf = 4.0

# Coarce traj optimization
n  = 20
dt = tf/n

planner = trajectoryoptimisation.DirectCollocationTrajectoryOptimisation( sys , dt , n )

planner.x_start = np.array([-3.14,0,0,0])
planner.x_goal  = np.array([0,0,0,0])

planner.maxiter = 500
planner.compute_optimal_trajectory()
planner.show_solution()

# Fine traj optimization
n = 100
dt = tf/n
planner2 = trajectoryoptimisation.DirectCollocationTrajectoryOptimisation( sys , dt , n )

planner2.x_start = np.array([-3.14,0,0,0])
planner2.x_goal  = np.array([0,0,0,0])

planner2.set_initial_trajectory_guest( planner.traj )
planner2.maxiter = 500
planner2.compute_optimal_trajectory()
planner2.show_solution()

planner2.save_solution( 'double_pendulum_directcollocation_hires.npy' )

# Controller
ctl  = nonlinear.ComputedTorqueController( sys , planner2.traj )

ctl.rbar = np.array([0,0])
ctl.w0   = 5
ctl.zeta = 1

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0 = np.array([-3.14,0,0,0])
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()