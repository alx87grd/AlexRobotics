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
from pyro.analysis import simulation
###############################################################################

sys  = pendulum.DoublePendulum()


# ref traj
#traj = simulation.Trajectory.load( 'double_pendulum_rrt.npy' )
#traj = simulation.Trajectory.load( 'double_pendulum_directcollocation.npy' )
traj = simulation.Trajectory.load('double_pendulum_directcollocation_hires.npy')

# Controller
ctl  = nonlinear.SlidingModeController( sys , traj )
#ctl  = nonlinear.SlidingModeController( sys )

# goal
ctl.rbar = np.array([0,0])
ctl.lam  = 1
ctl.gain = 2

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0  = np.array([-3.14,0,0,0])


# Solver param
tf = 10
n  = 10001
solver = 'euler' # necessary for sliding mode

cl_sys.compute_trajectory(tf,n,solver)
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory(0, 2)
cl_sys.animate_simulation()